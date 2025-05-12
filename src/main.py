import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import zmq
from predictor import build_sam2_video_predictor
from protocol import (
    Action,
    DoubleClick,
    DrawBox,
    Event,
    FrameMetadata,
    Prompts,
    SingleClick,
)
from pydantic import ValidationError
from rich import print
from zmq import Context, Socket

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
)

APP_ROOT = str(Path(__file__).parent.parent)

checkpoint = Path(APP_ROOT) / "checkpoints/sam2.1_hiera_small.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"

predictor = build_sam2_video_predictor(config_file=model_cfg, ckpt_path=checkpoint)


def create_subscriber(endpoint: str = "tcp://127.0.0.1:5563") -> Tuple[Context, Socket]:
    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.connect(endpoint)
    sub.subscribe(b"")

    return ctx, sub


def add_points(event: Event, clear_pts: bool) -> Tuple[list, Dict[str, torch.Tensor]]:
    with torch.inference_mode(), torch.autocast("mps", dtype=torch.bfloat16):
        points = np.array([event.points], dtype=np.float32)
        labels = np.array([event.labels], dtype=np.int32)

        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=event.inference_state,
            frame_idx=event.frame_idx,
            obj_id=event.obj_id,
            points=points,
            labels=labels,
            clear_old_points=clear_pts,
        )

        return out_obj_ids, out_mask_logits


def handle(action: Optional[Action]) -> Prompts:
    """
    Extract prompt type (i.e. mouse event), coordinates, and label
    from a multipart message's metadata

    For labels:
        `0` means negative click
        `1` means positive click
        `2` means the top-left corner of the box
        `3` means the bottom-right corner of the box
    """

    if action is None:
        return None

    match action:
        case SingleClick(coords=pt):
            action_name, coords, label = "single_click", pt, 1
        case DoubleClick(coords=pt):
            action_name, coords, label = "double_click", pt, 0
        case DrawBox(coords=pt):
            action_name, coords, label = "draw_box", pt, (2, 3)
        case _:
            raise RuntimeError(f"Unexpected action type {action.type}")

    logger.info("%s %s %s", action_name, coords, label)

    return action_name, coords, label


"""
We want to make propagation actually see "future" frames.
We need to accumulate a sliding window of frames in `inference_state`, 
then tell SAM2 to propagate from the prompt frame onward through that window

    > On each new frame, we append into the buffer tensor of shape (T, 3, H, W)
    > When we get a click/box prompt on the current frame at local index T-1, we call `add_new_points_or_box`
    > We propagate through all later frames in the buffer

    For example:

    1. At t0, we recieve f0 -> buffer[f0] -> prompt on f0 (prompt_idx=0) -> no propagation yet
    2. At t1, we receive f1 -> buffer[f0,f1] -> prompt remains at prompt_idx=0 -> propagate yields mask for i=1 (f1)
    3. If we prompt again on f2, update prompt_idx=2 and then propagate into frames f3, f4, ...
    ...

        Time            t0          t1          t2          t3
        Frames          f0          f1          f2          f3 
        Buffer idxes    b0          b1          b2          b3
        Prompt on       [f0]        [f0,f1]     [f0,f1,f2]  [f0,f1,f2,f3]
        Propagate       -           i=1         i=1,i=2     i=1,i=2,i=3
"""


def consumer_loop(sub: Socket):
    # Track the first click for a given `obj_id`
    # We need to know whether this is the first click for a given `obj_id` (so we clear old points)
    # or a refinement click (so we merge points)
    # Every time we finish one object's propagation and clear buffer, we call `click_counts.clear()`
    # to reset the counter so our next click starts a brand-new object (obj_id=0 again)
    # Without it, we wouldn't be able to group clicks into separate "objects" versus "refinements on the same object"
    click_counts: Dict[int, int] = {}

    inference_state = None
    prompt_idx = -1
    buffer_size = 10

    while True:
        try:
            with torch.inference_mode(), torch.autocast("mps", dtype=torch.bfloat16):
                # We receive one video frame at a time in `jpg_bytes` along with its metadata
                metadata_bytes, jpg_bytes = sub.recv_multipart()
                metadata = FrameMetadata.model_validate_json(
                    metadata_bytes.decode("utf-8"),
                )

                if inference_state is None:
                    inference_state = predictor.init_state(
                        video_height=metadata.height, video_width=metadata.width
                    )

                # Always concatenate new frame to existing frames frist
                # We trim later to fit into `buffer_size`
                predictor.update_state_with_frame(inference_state, jpg_bytes)

                prompt = handle(metadata.action)

                # On click -> Seed SAM2
                if prompt is not None:
                    _, coords, labels = prompt

                    # As we just appended the frame, we get its index `prompt_idx` at n - 1
                    # where n is the total number of frames so far
                    prompt_idx = inference_state["num_frames"] - 1

                    obj_id = len(click_counts)
                    click_counts.setdefault(obj_id, 0)

                    # Clear previous points on first click
                    clear_pts = click_counts[obj_id] == 0
                    click_counts[obj_id] += 1

                    event = Event(
                        inference_state=inference_state,
                        points=coords,
                        labels=labels,
                        frame_idx=prompt_idx,
                        obj_id=obj_id,
                    )

                    # Add click/box to the current frame
                    out_obj_ids, out_mask_logits = add_points(event, clear_pts)
                    inference_state["obj_ids"] = out_obj_ids.copy()

                # Always try to propagate if we have at least one frame beyond last prompt
                # For example, if num_frames=5, frame_idx=4, prompt_idx=4,
                # we wouldn't propagate as we are already at buffer's last index
                # However, num_frames=5, frame_idx=4, prompt_idx=3 works because 5 > 3 + 1
                if prompt_idx >= 0 and inference_state["num_frames"] > prompt_idx + 1:
                    # `video_segments` contains the per-frame segmentation results
                    video_segments = {}
                    for (
                        out_frame_idx,
                        out_obj_ids,
                        out_mask_logits,
                    ) in predictor.propagate_in_video(
                        inference_state,
                        start_frame_idx=prompt_idx,
                        max_frame_num_to_track=buffer_size,
                        reverse=False,  # Real-time video capturing only goes forward
                    ):
                        print(out_frame_idx, out_obj_ids, out_mask_logits)

                        video_segments[out_frame_idx] = {
                            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                            for i, out_obj_id in enumerate(out_obj_ids)
                        }

                    print("Video Segments >", video_segments)

                    # TODO: Move this to rust
                    import cv2

                    # Right after the `propagate_in_video` loop:
                    if out_frame_idx == inference_state["num_frames"] - 1:
                        # Decode the same `jpg_bytes` we just recv’d
                        img = cv2.imdecode(
                            np.frombuffer(jpg_bytes, np.uint8), cv2.IMREAD_COLOR
                        )
                        # Get binary mask from the logits; `out_mask_logits` has shape (1, num_objs, H, W)
                        mask = (out_mask_logits[0, 0] > 0).cpu().numpy()  # H×W bool

                        # If the model works at a different resolution than the camera, resize:
                        mask_resized = cv2.resize(
                            mask.astype(np.uint8),
                            (img.shape[1], img.shape[0]),
                            interpolation=cv2.INTER_NEAREST,
                        )

                        # Make a green overlay
                        overlay = np.zeros_like(img)
                        overlay[mask_resized == 1] = (0, 255, 0)

                        # Blend and display
                        blended = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
                        cv2.imshow("segmented", blended)
                        cv2.waitKey(1)

                    # Discard all frames preceding the prompt
                    # We'll use the current prompt frame as the base state to propagate through all future frames
                    inference_state["images"] = inference_state["images"][prompt_idx:]
                    inference_state["num_frames"] = inference_state["images"].shape[0]
                    prompt_idx = 0

                    click_counts.clear()

                if inference_state["num_frames"] > buffer_size and prompt_idx == -1:
                    # If we never received a prompt, we can safely trim it to only keep the latest frame
                    predictor.reset_images(inference_state)

        except ValidationError as e:
            print("Bad metadata JSON:", e)
        except Exception as e:
            import traceback

            traceback.print_exc()
            print("Unexpected error in consumer_loop:", e)
            continue


def main() -> None:
    ctx, sub = create_subscriber()
    try:
        consumer_loop(sub)
    except:
        sub.close()
        ctx.term()


if __name__ == "__main__":
    main()
