# Code in this file is adapted from **https://github.com/facebookresearch/sam2

import logging
from collections import OrderedDict

import cv2
import numpy as np
import torch
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from sam2.sam2_video_predictor import SAM2VideoPredictor

if not GlobalHydra.instance().is_initialized():
    initialize_config_module("sam2", version_base="1.2")


def _load_checkpoint(model, ckpt_path):
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"]
        missing_keys, unexpected_keys = model.load_state_dict(sd)
        if missing_keys:
            logging.error(missing_keys)
            raise RuntimeError()
        if unexpected_keys:
            logging.error(unexpected_keys)
            raise RuntimeError()
        logging.info("Loaded checkpoint sucessfully")


def build_sam2_video_predictor(
    config_file,
    ckpt_path=None,
    device="mps",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    vos_optimized=False,
    **kwargs,
) -> "SAM2VideoStreamingPredictor":
    hydra_overrides = [
        "++model._target_=predictor.SAM2VideoStreamingPredictor",
    ]
    if vos_optimized:
        hydra_overrides = [
            "++model._target_=sam2.sam2_video_predictor.SAM2VideoPredictorVOS",
            "++model.compile_image_encoder=True",  # Let sam2_base handle this
        ]

    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            # the sigmoid mask logits on interacted frames with clicks in the memory encoder so that the encoded masks are exactly as what users see from clicking
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            # fill small holes in the low-res masks up to `fill_hole_area` (before resizing them to the original video resolution)
            "++model.fill_hole_area=8",
        ]
    hydra_overrides.extend(hydra_overrides_extra)

    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def preprocess_frame_from_bytes(
    jpg_bytes: bytes,
    img_mean=(0.485, 0.456, 0.406),
    img_std=(0.229, 0.224, 0.225),
    image_size=512,
    compute_device=torch.device("mps"),
):
    img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
    img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]

    # Decode jpeg into a 1-D NumPy uint8 array -> (H, W, 3) BGR uint8
    array = np.frombuffer(jpg_bytes, dtype=np.uint8)

    # Decompress the array into a 2-D image with 3 channels
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)

    # Resize each image to square
    frame = cv2.resize(frame, (image_size, image_size))

    # BGR -> RGB
    image = frame[:, :, ::-1]

    # (H, W, C) -> (C, H, W) for PyTorch convention
    image = image.transpose(2, 0, 1).astype(np.float32) / 255.0

    # Shape (1, 3, H, W)
    tensor = torch.from_numpy(image).unsqueeze(0)

    # Normalize by mean and std
    mean = img_mean.view(1, -1, 1, 1)
    std = img_std.view(1, -1, 1, 1)
    tensor = (tensor - mean) / std

    return tensor.to(compute_device)


class SAM2VideoStreamingPredictor(SAM2VideoPredictor):
    def __init__(
        self,
        fill_hole_area=0,
        non_overlap_masks=False,
        clear_non_cond_mem_around_input=False,
        add_all_frames_to_correct_as_cond=False,
        **kwargs,
    ):
        super().__init__(
            fill_hole_area,
            non_overlap_masks,
            clear_non_cond_mem_around_input,
            add_all_frames_to_correct_as_cond,
            **kwargs,
        )

    def init_state(
        self,
        video_height: int,
        video_width: int,
        offload_video_to_cpu=False,
        offload_state_to_cpu=False,
    ) -> dict:
        compute_device = self.device  # device of the model
        inference_state = {}
        inference_state["images"] = torch.empty(0, 3, self.image_size, self.image_size)
        inference_state["num_frames"] = 0

        # whether to offload the video frames to CPU memory
        # turning on this option saves the GPU memory with only a very small overhead
        inference_state["offload_video_to_cpu"] = offload_video_to_cpu
        # whether to offload the inference state to CPU memory
        # turning on this option saves the GPU memory at the cost of a lower tracking fps
        # (e.g. in a test case of 768x768 model, fps dropped from 27 to 24 when tracking one object
        # and from 24 to 21 when tracking two objects)
        inference_state["offload_state_to_cpu"] = offload_state_to_cpu
        # the original video height and width, used for resizing final output scores
        inference_state["video_height"] = video_height
        inference_state["video_width"] = video_width
        inference_state["device"] = compute_device
        if offload_state_to_cpu:
            inference_state["storage_device"] = torch.device("cpu")
        else:
            inference_state["storage_device"] = compute_device
        # inputs on each frame
        inference_state["point_inputs_per_obj"] = {}
        inference_state["mask_inputs_per_obj"] = {}
        # visual features on a small number of recently visited frames for quick interactions
        inference_state["cached_features"] = {}
        # values that don't change across frames (so we only need to hold one copy of them)
        inference_state["constants"] = {}
        # mapping between client-side object id and model-side object index
        inference_state["obj_id_to_idx"] = OrderedDict()
        inference_state["obj_idx_to_id"] = OrderedDict()
        inference_state["obj_ids"] = []
        # Slice (view) of each object tracking results, sharing the same memory with "output_dict"
        inference_state["output_dict_per_obj"] = {}
        # A temporary storage to hold new outputs when user interact with a frame
        # to add clicks or mask (it's merged into "output_dict" before propagation starts)
        inference_state["temp_output_dict_per_obj"] = {}
        # Frames that already holds consolidated outputs from click or mask inputs
        # (we directly use their consolidated outputs during tracking)
        # metadata for each tracking frame (e.g. which direction it's tracked)
        inference_state["frames_tracked_per_obj"] = {}
        # Warm up the visual backbone and cache the image feature on frame 0
        # self._get_image_feature(inference_state, frame_idx=0, batch_size=1)
        return inference_state

    def update_state_with_frame(self, inference_state: dict, jpg_bytes: bytes) -> None:
        image_tensor = preprocess_frame_from_bytes(
            jpg_bytes,
            image_size=self.image_size,
        )

        buf = inference_state["images"].to(self.device)
        inference_state["images"] = torch.cat([buf, image_tensor], dim=0)
        inference_state["num_frames"] = inference_state["images"].shape[0]

    def _get_image_feature(self, inference_state, frame_idx, batch_size):
        """Compute the image features on a given frame."""
        # Look up in the cache first
        image, backbone_out = inference_state["cached_features"].get(
            frame_idx, (None, None)
        )
        if backbone_out is None:
            # Cache miss -- we will run inference on a single image
            device = inference_state["device"]

            # We use frame_idx = 0 since `images` is of the shape (1, 3, H, W) when streaming the video
            image = inference_state["images"][0].to(device).float().unsqueeze(0)
            backbone_out = self.forward_image(image)
            # TODO: we can use an LRU cache for more frames in the future.
            inference_state["cached_features"][frame_idx] = (image, backbone_out)

        # Expand the features to have the same dimension as the number of objects
        expanded_image = image.expand(batch_size, -1, -1, -1)
        expanded_backbone_out = {
            "backbone_fpn": backbone_out["backbone_fpn"].copy(),
            "vision_pos_enc": backbone_out["vision_pos_enc"].copy(),
        }
        for i, feat in enumerate(expanded_backbone_out["backbone_fpn"]):
            expanded_backbone_out["backbone_fpn"][i] = feat.expand(
                batch_size, -1, -1, -1
            )
        for i, pos in enumerate(expanded_backbone_out["vision_pos_enc"]):
            pos = pos.expand(batch_size, -1, -1, -1)
            expanded_backbone_out["vision_pos_enc"][i] = pos

        features = self._prepare_backbone_features(expanded_backbone_out)
        features = (expanded_image,) + features
        return features
