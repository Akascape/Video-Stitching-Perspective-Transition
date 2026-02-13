"""
Video Stitcher with Smooth Transitions using LoFTR and DeepLabV3
Author: Akash Bora (@akascape on GitHub)
License: MIT
Copyright (c) 2026 Akascape
Repo URL: https://github.com/Akascape/Video-Stitching-Perspective-Transition
"""
import cv2
import torch
import kornia
from kornia.feature import LoFTR
import numpy as np
import sys
import argparse
from torchvision import models, transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- SEGMENTATION USING DeepLabV3 ---
def get_segmentation_model(device):
    print("Loading DeepLabV3...")
    model = models.segmentation.deeplabv3_resnet101(weights=models.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT)
    model.to(device)
    model.eval()
    return model

def get_person_mask(frame_bgr, model, device):
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(520), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(frame_bgr).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    output_predictions = output.argmax(0)
    # Class 15 = Person
    mask = (output_predictions == 15).byte().cpu().numpy()
    h, w = frame_bgr.shape[:2]
    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return (mask_resized * 255).astype(np.uint8)

# --- LOFTR ---
def load_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
    cap.release()
    return frames, fps, width, height

def preprocess_for_loftr(img_bgr, device, max_dim):
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = img_gray.shape
    scale = max_dim / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = cv2.resize(img_gray, (new_w, new_h))
    img_tensor = torch.from_numpy(img_resized).float()[None, None] / 255.0
    return img_tensor.to(device), scale

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Video stitcher with smooth transitions using LoFTR feature matching and DeepLabV3 segmentation"
    )
    parser.add_argument(
        "-a", "--video-a",
        default="pre.mp4",
        help="Path to first video clip (Outgoing) (default: pre2.mp4)"
    )
    parser.add_argument(
        "-b", "--video-b",
        default="post.mp4",
        help="Path to second video clip (Incoming) (default: post2.mp4)"
    )
    parser.add_argument(
        "-o", "--output",
        default="transition.mp4",
        help="Output video path (default: transition.mp4)"
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=40,
        help="Number of frames to overlap/transition between videos (default: 42)"
    )
    parser.add_argument(
        "--loftr-max-dim",
        type=int,
        default=1152,
        help="Maximum dimension for LoFTR feature matching (higher = better quality, more compute) (default: 1152)"
    )
    parser.add_argument(
        "--fade-in",
        type=int,
        default=10,
        help="Number of frames to fade in the foreground at the start (default: 10)"
    )
    parser.add_argument(
        "--fade-out",
        type=int,
        default=10,
        help="Number of frames to fade out the pre-clip at the end (default: 10)"
    )
    
    args = parser.parse_args()
    
    # Setup variables from arguments
    VIDEO_A_PATH = args.video_a
    VIDEO_B_PATH = args.video_b
    OUTPUT_PATH = args.output
    OVERLAP_FRAMES = args.overlap
    LOFTR_MAX_DIM = args.loftr_max_dim
    FADE_IN_FRAMES = args.fade_in
    FADE_OUT_FRAMES = args.fade_out
    
    # Setup
    matcher = LoFTR(pretrained="indoor_new").to(DEVICE)
    matcher.eval()
    seg_model = get_segmentation_model(DEVICE)

    print("Loading videos...")
    frames_a, fps, w, h = load_video_frames(VIDEO_A_PATH)
    frames_b, _, _, _ = load_video_frames(VIDEO_B_PATH)

    if not frames_a or not frames_b:
        print("Error loading videos.")
        sys.exit(1)
    
    if len(frames_b) <= OVERLAP_FRAMES:
        print(f"Error: Video B is too short. Needs at least {OVERLAP_FRAMES+1} frames.")
        sys.exit(1)

    print(f"Computing alignment: Video A (End) <-> Video B (Frame {OVERLAP_FRAMES})...")
    
    img_a_ref = frames_a[-1] 
    img_b_ref = frames_b[OVERLAP_FRAMES] 

    tensor_a, scale_a = preprocess_for_loftr(img_a_ref, DEVICE, LOFTR_MAX_DIM)
    tensor_b, scale_b = preprocess_for_loftr(img_b_ref, DEVICE, LOFTR_MAX_DIM)

    with torch.no_grad():
        correspondences = matcher({"image0": tensor_a, "image1": tensor_b})
        mkpts0 = correspondences['keypoints0'].cpu().numpy() / scale_a
        mkpts1 = correspondences['keypoints1'].cpu().numpy() / scale_b

    if len(mkpts0) < 4:
        print("Error: Not enough matches.")
        sys.exit(1)

    H, _ = cv2.findHomography(mkpts1, mkpts0, cv2.RANSAC, 5.0)

    # Render
    print("Rendering...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (w, h))

    # Calculate where the transition starts in Video A
    transition_start_idx = len(frames_a) - OVERLAP_FRAMES

    # --- RENDER VIDEO A (+ GHOST B) ---
    for i, frame_a in enumerate(frames_a):
        
        if i >= transition_start_idx:
            b_idx = i - transition_start_idx
            frame_b = frames_b[b_idx]

            # Get Mask
            mask = get_person_mask(frame_b, seg_model, DEVICE)
            warped_b = cv2.warpPerspective(frame_b, H, (w, h), borderMode=cv2.BORDER_REPLICATE)
            warped_mask = cv2.warpPerspective(mask, H, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

            # Apply fade-in effect to the foreground
            fade_alpha = warped_mask.astype(float) / 255.0
            if b_idx < FADE_IN_FRAMES:
                # Linear fade-in: starts at 0, ends at 1
                fade_factor = b_idx / FADE_IN_FRAMES
                fade_alpha = fade_alpha * fade_factor
            
            fade_alpha = cv2.merge([fade_alpha, fade_alpha, fade_alpha]) # 3 channels

            # Apply fade-out effect to Video A (pre-clip) at the end
            pre_clip_alpha = 1.0
            post_clip_alpha = 0.0
            frames_from_end = len(frames_a) - i
            if frames_from_end <= FADE_OUT_FRAMES:
                # Linear fade-out: ends at 0
                fade_factor = frames_from_end / FADE_OUT_FRAMES
                pre_clip_alpha = fade_factor
                post_clip_alpha = 1.0 - fade_factor

            foreground = cv2.multiply(fade_alpha, warped_b.astype(float))
            pre_clip_bg = cv2.multiply((1.0 - fade_alpha) * pre_clip_alpha, frame_a.astype(float))
            post_clip_bg = cv2.multiply((1.0 - fade_alpha) * post_clip_alpha, warped_b.astype(float))
            background = cv2.add(pre_clip_bg, post_clip_bg)
            combined = cv2.add(foreground, background).astype(np.uint8)
            
            out.write(combined)
        else:
            frame_out = frame_a.copy().astype(float)
            frames_from_end = len(frames_a) - i
            if frames_from_end <= FADE_OUT_FRAMES:
                # Linear fade-out: ends at 0
                fade_factor = frames_from_end / FADE_OUT_FRAMES
                frame_out = frame_out * fade_factor
            
            out.write(frame_out.astype(np.uint8))

    # --- RENDER VIDEO B  ---
    print(f"Cutting to full Video B (Skipping first {OVERLAP_FRAMES} frames)...")
    
    remaining_b = frames_b[OVERLAP_FRAMES:]

    for i, frame_b in enumerate(remaining_b):
        warped_b = cv2.warpPerspective(frame_b, H, (w, h), borderMode=cv2.BORDER_REPLICATE)
        out.write(warped_b)
        
        if i % 30 == 0:
            print(f"Post-clip progress: {i}/{len(remaining_b)}")

    out.release()
    print(f"Done! Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()