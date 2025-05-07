import logging
import uuid
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


def add_points(event: Event) -> Tuple[list, Dict[str, torch.Tensor]]:
    points = np.array([event.points], dtype=np.float32)
    labels = np.array([event.labels], np.int32)

    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=event.inference_state,
        frame_idx=event.frame_idx,
        obj_id=event.obj_id,
        points=points,
        labels=labels,
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
    ...

        Time            t0          t1          t2          t3
        Frames          f0          f1          f2          f3 
        Buffer idxes    b0          b1          b2          b3
        Prompt on       [f0]        [f0,f1]     [f0,f1,f2]  [f0,f1,f2,f3]
        Propagate       -           i=1         i=1,i=2     i=1,i=2,i=3
"""


def consumer_loop(sub: Socket):
    inference_state = None

    while True:
        try:
            # We receive one video frame in `jpg_bytes` at a time along with its metadata
            metadata_bytes, jpg_bytes = sub.recv_multipart()
            metadata = FrameMetadata.model_validate_json(
                metadata_bytes.decode("utf-8"),
            )

            if inference_state is None:
                inference_state = predictor.init_state(
                    video_height=metadata.height, video_width=metadata.width
                )

            predictor.update_state_with_frame(inference_state, jpg_bytes)

            result = handle(metadata.action)
            if result is None:
                continue

            obj_id = uuid.uuid4()
            _, coords, labels = result

            prompt_idx = inference_state["num_frames"] - 1
            event = Event(
                inference_state=inference_state,
                points=coords,
                labels=labels,
                frame_idx=prompt_idx,
                obj_id=obj_id,
            )

            out_mask_logits, out_obj_ids = add_points(event)

            # Video_segments contains the per-frame segmentation results
            video_segments = {}
            for (
                out_frame_idx,
                out_obj_ids,
                out_mask_logits,
            ) in predictor.propagate_in_video(
                inference_state,
                start_frame_idx=prompt_idx,
                max_frame_num_to_track=10,
                reverse=False,
            ):
                print(out_frame_idx, out_obj_ids, out_mask_logits)
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }

        except ValidationError as e:
            print("Bad metadata JSON:", e)
        except Exception as e:
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
