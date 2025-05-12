# Sam2StreamEngine

Sam2StreamEngine attempts to apply [Segment Anything Model 2 (SAM 2)](https://github.com/facebookresearch/sam2) on live videos.

Given that SAM 2 allows users to use a click, box, or mask as the input to select an object on any frame of video, we use `single_click` for a positive prompt, telling the model to generate a segmentation mask for the object containing that point. `double_click` marks areas to exclude from segmentation. `draw_box` via a drag mouse event specifies a rectangular region containing the object to segment, resulting in a precise segmentation mask for the object inside.

### How to run the code?

1. Make sure you have Rust, [uv](https://github.com/astral-sh/uv), and sam2 installed. It currently uses `sam2.1_hiera_small.pt` model checkpoint and `sam2_hiera_s.yaml` for model configurations.

2. Run the following commands

```bash
cd video_streamer
cargo run

cd src
python main.py
```

3. You will see a GUI pop up; simply click on any object on the video frame, it will create a green mask predicted by SAM 2.

## Limitations

Currently, video freezes after the first 2 clicks despite successfully generating the green mask.

## Progress

-   [live demo #1](https://x.com/JYFHuang/status/1921794553117360574)
