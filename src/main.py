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


def consumer_loop(sub: Socket):
    metadata_bytes, jpg_bytes = sub.recv_multipart()
    metadata = FrameMetadata.model_validate_json(metadata_bytes.decode("utf-8"))
    inference_state = predictor.init_state(
        video_height=metadata.height, video_width=metadata.width
    )

    while True:
        try:
            metadata_bytes, jpg_bytes = sub.recv_multipart()
            metadata = FrameMetadata.model_validate_json(metadata_bytes.decode("utf-8"))
            predictor.update_state_with_frame(inference_state, jpg_bytes)

            result = handle(metadata.action)
            if result is None:
                continue

            frame_idx = metadata.frame_id
            obj_id = uuid.uuid4()

            _, coords, labels = result

            event = Event(
                inference_state=inference_state,
                points=coords,
                labels=labels,
                frame_idx=frame_idx,
                obj_id=obj_id,
            )

            out_mask_logits, out_obj_ids = add_points(event)

            print(out_mask_logits)
            print(out_obj_ids)

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
