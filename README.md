# Video-Stitching-Perspective-Transition
Homography-based perspective video transition using [LoFTR](https://kornia.readthedocs.io/en/latest/models/loftr.html) feature matching and DeepLabV3 person segmentation for seamless foreground-aware stitching between clips.

## Overview
This project generates a perspective-aligned transition between two videos so that a hard cut appears as a continuous camera movement.

It works by:
- Matching features between the last frame of Video A and a reference frame of Video B using [LoFTR](https://kornia.readthedocs.io/en/latest/models/loftr.html)
- Estimating a homography with RANSAC
- Warping Video B into Video A’s perspective
- Segmenting people using DeepLabV3
- Blending foreground and background with temporal fade control
- Rendering a final stitched output

## Requirments
- Python 3.10+
```
torch>=2.0
torchvision>=0.15
opencv-python>=4.8
kornia>=0.7
numpy>=1.23
```
### Install with:
```
pip install -r requirements.txt
```
### CUDA Support
<br> Install PyTorch with CUDA from:
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
### Running the script
```
python video_stitcher.py -a path/to/videoA.mp4 -b path/to/videoB.mp4 -o output.mp4 --overlap 42 --loftr-max-dim 1152 --fade-in 10 --fade-out 10
```
OR simply go with default command:
```
python video_stitcher.py -a path/to/videoA.mp4 -b path/to/videoB.mp4 -o output.mp4
```
### Arguments
| Argument           | Short | Type   | Default           | Description                                                        |
|--------------------|-------|--------|-------------------|--------------------------------------------------------------------|
| `--video-a`        | `-a`  | str    | `pre.mp4`        | Path to first video clip (Outgoing)                                |
| `--video-b`        | `-b`  | str    | `post.mp4`       | Path to second video clip (Incoming)                               |
| `--output`         | `-o`  | str    | `transition.mp4`  | Output video path                                                  |
| `--overlap`        |       | int    | `40`              | Number of frames to overlap/transition between videos              |
| `--loftr-max-dim`  |       | int    | `1152`            | Maximum dimension for LoFTR feature matching                       |
| `--fade-in`        |       | int    | `10`              | Number of frames to fade in the foreground at the start            |
| `--fade-out`       |       | int    | `10`              | Number of frames to fade out the pre-clip at the end               |
### Fix download issues:
```
urllib.error.URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1000)>
```
If you encounter this error while downloading `loftr_indoor_ds_new.ckpt`, simply download it manually instead (this happens due to some HTTP issues).

Steps
- Download the `indoor_ds_new.ckpt` from this drive link: [Google Drive](https://drive.google.com/drive/folders/1xu2Pq6mZT5hmFgiYMBT9Zt8h1yO-3SIp)
- Rename the file from `indoor_ds_new.ckpt` to  `loftr_indoor_ds_new.ckpt`
- Paste the file in `C:\Users\user\.cache\torch\hub\checkpoints`
That's it, it will solve the error. (This is not an issue with other models)

### Example


### License: MIT
<br> Copyright (c) 2026 Akash Bora
**Get more video effects at [www.akascape.com](https://www.akascape.com) 👈**

