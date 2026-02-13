"""
Video Stitcher with Smooth Transitions using LoFTR and DeepLabV3
Author: Akash Bora (@akascape on GitHub)
Modified: Dynamic Perspective Interpolation, Clean Border Compositing & Fast Foreground Removal
License: MIT
Version: 2
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
    parser = argparse.ArgumentParser(description="Video stitcher with smooth transitions using LoFTR and DeepLabV3")
    parser.add_argument("-a", "--video-a", default="pre.mp4", help="Path to first video clip")
    parser.add_argument("-b", "--video-b", default="post.mp4", help="Path to second video clip")
    parser.add_argument("-o", "--output", default="transition.mp4", help="Output video path")
    parser.add_argument("--overlap", type=int, default=40, help="Number of frames to overlap/transition")
    parser.add_argument("--loftr-max-dim", type=int, default=1152, help="Max dim for LoFTR")
    parser.add_argument("--fade-in", type=int, default=10, help="Frames to fade in foreground")
    parser.add_argument("--fade-out", type=int, default=10, help="Frames to fade out pre-clip")
    args = parser.parse_args()
    
    VIDEO_A_PATH, VIDEO_B_PATH, OUTPUT_PATH = args.video_a, args.video_b, args.output
    OVERLAP_FRAMES = args.overlap
    LOFTR_MAX_DIM = args.loftr_max_dim
    FADE_IN_FRAMES, FADE_OUT_FRAMES = args.fade_in, args.fade_out
    
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

    # H maps Video B -> Video A
    H, _ = cv2.findHomography(mkpts1, mkpts0, cv2.RANSAC, 5.0)
    
    # Normalize matrices
    H_BA = H / H[2, 2]
    H_AB = np.linalg.inv(H_BA)
    H_AB = H_AB / H_AB[2, 2]
    I = np.eye(3)

    print("Rendering...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (w, h))

    transition_start_idx = len(frames_a) - OVERLAP_FRAMES
    dummy_white_frame = np.ones((h, w), dtype=np.uint8) * 255
    erosion_kernel = np.ones((3,3), np.uint8)

    # --- RENDER PRE-TRANSITION ---
    for i in range(transition_start_idx):
        out.write(frames_a[i])

    # --- RENDER OVERLAP TRANSITION ---
    for i in range(OVERLAP_FRAMES):
        t = i / max(1, (OVERLAP_FRAMES - 1))
        
        # Interpolate matrices
        H_A_t = (1 - t) * I + t * H_AB
        H_B_t = (1 - t) * H_BA + t * I
        
        frame_a = frames_a[transition_start_idx + i]
        frame_b = frames_b[i]

        # Get Segmentation Mask
        mask = get_person_mask(frame_b, seg_model, DEVICE)

        # Fast Background Inpainting (Foreground Remover)
        # Scales down to inpaint instantly, then scales up and replaces only the masked area
        scale = 4
        small_b = cv2.resize(frame_b, (w // scale, h // scale))
        small_mask = cv2.resize(mask, (w // scale, h // scale), interpolation=cv2.INTER_NEAREST)
        small_clean = cv2.inpaint(small_b, small_mask, 3, cv2.INPAINT_TELEA)
        frame_b_clean_fast = cv2.resize(small_clean, (w, h))

        mask_3ch_f = cv2.merge([mask, mask, mask]).astype(float) / 255.0
        frame_b_clean = (frame_b.astype(float) * (1.0 - mask_3ch_f) + frame_b_clean_fast.astype(float) * mask_3ch_f).astype(np.uint8)

        # Warping
        warped_b_clean = cv2.warpPerspective(frame_b_clean, H_B_t, (w, h), borderMode=cv2.BORDER_REPLICATE)
        warped_b_full = cv2.warpPerspective(frame_b, H_B_t, (w, h), borderMode=cv2.BORDER_REPLICATE)
        warped_a_clean = cv2.warpPerspective(frame_a, H_A_t, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

        # Masks Warping
        warped_mask = cv2.warpPerspective(mask, H_B_t, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        mask_a_valid = cv2.warpPerspective(dummy_white_frame, H_A_t, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mask_a_valid = cv2.erode(mask_a_valid, erosion_kernel, iterations=1)
        mask_a_valid_f = cv2.merge([mask_a_valid, mask_a_valid, mask_a_valid]).astype(float) / 255.0

        # Alphas
        fade_alpha = warped_mask.astype(float) / 255.0
        if i < FADE_IN_FRAMES:
            fade_alpha *= (i / FADE_IN_FRAMES)
        fade_alpha_3ch = cv2.merge([fade_alpha, fade_alpha, fade_alpha])

        pre_clip_alpha = 1.0
        post_clip_alpha = 0.0
        frames_from_end = OVERLAP_FRAMES - i
        if frames_from_end <= FADE_OUT_FRAMES:
            fade_factor = frames_from_end / FADE_OUT_FRAMES
            pre_clip_alpha = fade_factor
            post_clip_alpha = 1.0 - fade_factor

        # Composite Background (A and Clean B)
        bg_a_part = warped_a_clean.astype(float) * pre_clip_alpha
        bg_b_part = warped_b_clean.astype(float) * post_clip_alpha
        pure_bg_blend = bg_a_part + bg_b_part

        bg_final = pure_bg_blend * mask_a_valid_f + warped_b_clean.astype(float) * (1.0 - mask_a_valid_f)

        # Composite Foreground (Person strictly overlays over Background)
        final_frame = warped_b_full.astype(float) * fade_alpha_3ch + bg_final * (1.0 - fade_alpha_3ch)
        
        out.write(final_frame.astype(np.uint8))
        print(f"Transition progress: {i+1}/{OVERLAP_FRAMES}")

    # --- RENDER POST-TRANSITION ---
    print(f"Cutting to full Video B (Skipping first {OVERLAP_FRAMES} frames)...")
    for i in range(OVERLAP_FRAMES, len(frames_b)):
        out.write(frames_b[i])
        if (i - OVERLAP_FRAMES) % 30 == 0:
            print(f"Post-clip progress: {i - OVERLAP_FRAMES}/{len(frames_b) - OVERLAP_FRAMES}")

    out.release()
    print(f"Done! Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
