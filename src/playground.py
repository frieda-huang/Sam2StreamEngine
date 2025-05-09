import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor
from pathlib import Path

APP_ROOT = str(Path(__file__).parent.parent)

checkpoint = Path(APP_ROOT) / "checkpoints/sam2.1_hiera_small.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"

predictor = build_sam2_video_predictor(model_cfg, checkpoint, device="mps")

with torch.inference_mode(), torch.autocast("mps", dtype=torch.bfloat16):
    state = predictor.init_state(
        "/Users/friedahuang/Documents/Sam2StreamEngine/src/01_dog.mp4"
    )

    ann_frame_idx = 0
    ann_obj_id = 1

    points = np.array([[82, 410]], dtype=np.float32)
    labels = np.array([0], np.int32)

    frame_idx, object_ids, masks = predictor.add_new_points_or_box(
        state, ann_frame_idx, ann_obj_id, points, labels
    )

    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        state
    ):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    print(video_segments)
