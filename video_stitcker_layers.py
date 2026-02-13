"""
Video Stitcher with Smooth Transitions 
Author: Akash Bora (@akascape on GitHub)
Outputs synced layers for compositing
License: MIT
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
    parser = argparse.ArgumentParser(description="Multi-pass video stitcher for NLE compositing")
    parser.add_argument("-a", "--video-a", default="pre2.mov", help="Path to first video clip")
    parser.add_argument("-b", "--video-b", default="post2.mov", help="Path to second video clip")
    parser.add_argument("--overlap", type=int, default=40, help="Number of frames to overlap/transition")
    parser.add_argument("--loftr-max-dim", type=int, default=1152, help="Max dim for LoFTR")
    args = parser.parse_args()
    
    VIDEO_A_PATH = args.video_a
    VIDEO_B_PATH = args.video_b
    OVERLAP_FRAMES = args.overlap
    LOFTR_MAX_DIM = args.loftr_max_dim
    
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

    print(f"Computing alignment: Video A (End) <-> Video B (Overlap End)...")
    
    img_a_ref = frames_a[-1] 
    img_b_ref = frames_b[OVERLAP_FRAMES - 1] 

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
    
    H_BA = H / H[2, 2]
    H_AB = np.linalg.inv(H_BA)
    H_AB = H_AB / H_AB[2, 2]
    I = np.eye(3)

    print("Rendering Multi-Pass Tracks...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # 3 Separate Output Files
    out_a = cv2.VideoWriter("layer_a.mp4", fourcc, fps, (w, h))
    out_b = cv2.VideoWriter("layer_b.mp4", fourcc, fps, (w, h))
    out_mask = cv2.VideoWriter("mask_b.mp4", fourcc, fps, (w, h))

    transition_start_idx = len(frames_a) - OVERLAP_FRAMES
    black_frame = np.zeros((h, w, 3), dtype=np.uint8)

    # --- 1. RENDER PRE-TRANSITION ---
    for i in range(transition_start_idx):
        out_a.write(frames_a[i])
        out_b.write(black_frame)
        out_mask.write(black_frame)

    # --- 2. RENDER OVERLAP TRANSITION ---
    for i in range(OVERLAP_FRAMES):
        t = i / max(1, (OVERLAP_FRAMES - 1))
        
        H_A_t = (1 - t) * I + t * H_AB
        H_B_t = (1 - t) * H_BA + t * I
        
        frame_a = frames_a[transition_start_idx + i]
        frame_b = frames_b[i]

        # Geometric Warps with Constant Black Borders
        warped_a = cv2.warpPerspective(frame_a, H_A_t, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
        warped_b = cv2.warpPerspective(frame_b, H_B_t, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
        
        # Track Matte Generator
        mask = get_person_mask(frame_b, seg_model, DEVICE)
        warped_mask = cv2.warpPerspective(mask, H_B_t, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mask_3ch = cv2.merge([warped_mask, warped_mask, warped_mask])

        out_a.write(warped_a)
        out_b.write(warped_b)
        out_mask.write(mask_3ch)
        
        print(f"Overlap progress: {i+1}/{OVERLAP_FRAMES}")

    # --- 3. RENDER POST-TRANSITION ---
    print("Writing Post-Transition...")
    for i in range(OVERLAP_FRAMES, len(frames_b)):
        out_a.write(black_frame)
        out_b.write(frames_b[i])
        
        # Continue generating masks for the rest of Video B just in case you want to use them
        mask = get_person_mask(frames_b[i], seg_model, DEVICE)
        mask_3ch = cv2.merge([mask, mask, mask])
        out_mask.write(mask_3ch)

    out_a.release()
    out_b.release()
    out_mask.release()
    print("Done! Saved layer_a.mp4, layer_b.mp4, and mask_b.mp4")

if __name__ == "__main__":
    main()
