# import cv2
# import matplotlib.pyplot as plt
# import numpy as np

# from src.datatools.reader import read_annot
# from src.datatools.intersections import get_intersections



# sample_id = '00000'
# img_path = f'./data/valid/{sample_id}.jpg'
# annot_path = f'./data/valid/{sample_id}.json'

# IMG_SIZE = (960, 540)

# LINE_COLORS = {
#     "Big rect. left bottom": (45, 175, 80),
#     "Big rect. left main": (40, 175, 170),
#     "Big rect. left top": (255, 175, 65),
#     "Big rect. right bottom": (80, 0, 50),
#     "Big rect. right main": (70, 0, 40),
#     "Big rect. right top": (255, 0, 50),
#     "Goal left crossbar": (0, 0, 255),
#     "Goal left post left": (0, 127, 255),
#     "Goal left post right": (127, 0, 255),
#     "Goal right crossbar": (0, 0, 255),
#     "Goal right post left": (0, 255, 255),
#     "Goal right post right": (255, 0, 255),
#     "Middle line": (0, 0, 0),
#     "Side line bottom": (65, 85, 150),
#     "Side line left": (50, 255, 255),
#     "Side line right": (85, 60, 0),
#     "Side line top": (255, 85, 65),
#     "Small rect. left bottom": (0, 127, 127),
#     "Small rect. left main": (127, 127, 127),
#     "Small rect. left top": (127, 127, 0),
#     "Small rect. right bottom": (0, 127, 127),
#     "Small rect. right main": (127, 127, 127),
#     "Small rect. right top": (127, 127, 0)
# }

# def create_binary_mask(annot, img_shape):
#     h, w = img_shape[:2]
#     mask = np.zeros((h, w), dtype=np.uint8)
    
#     for cls in annot:
#         points = annot[cls]
#         if len(points) > 1:  # Only draw if it's a line (multiple points)
#             points = [(int(round(point[0] * w)), int(round(point[1] * h))) for point in points]
#             for i in range(len(points) - 1):
#                 cv2.line(mask, points[i], points[i + 1], 255, 1, cv2.LINE_AA)
    
#     # Note: We skip drawing any points, intersections, or single-point annotations
#     # Goals are included as they are lines (e.g., crossbar, posts)
    
#     return mask

# def show_img(img):
#     plt.figure(dpi=150)
#     plt.imshow(img[:, :, ::-1])  # BGR2RGB
#     plt.show()

# def show_mask(mask):
#     plt.figure(dpi=150)
#     plt.imshow(mask, cmap='gray')
#     plt.show()

# def overlay_mask_on_image(img, mask, color=(0, 255, 0), alpha=0.5):
#     overlay = img.copy()
#     colored_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
#     colored_mask[mask > 0] = color
#     cv2.addWeighted(colored_mask, alpha, overlay, 1 - alpha, 0, overlay)
#     return overlay

# # Load image and annotations
# img = cv2.imread(img_path)
# annot = read_annot(annot_path)

# # Create binary mask (same size as image, only lines and goals)
# mask = create_binary_mask(annot, img.shape)

# # Show the binary mask
# show_mask(mask)

# # Optionally, overlay the mask on the original image and show
# overlaid_img = overlay_mask_on_image(img, mask)
# show_img(overlaid_img)

# # Optionally, save the mask and overlaid image
# cv2.imwrite(f'./masks{sample_id}_mask.jpg', mask)
# cv2.imwrite(f'./{sample_id}_overlaid.png', overlaid_img)



#!/usr/bin/env python3
import os
from pathlib import Path
import argparse
import cv2
import numpy as np
from tqdm import tqdm

# your project utils
from src.datatools.reader import read_annot  # expects same basename .json next to image

# --------------------------- helpers ---------------------------

def _xy_from_point(p):
    """Accept [x,y] or {'x':..., 'y':...} (normalized 0..1)."""
    if isinstance(p, dict):
        return float(p["x"]), float(p["y"])
    return float(p[0]), float(p[1])

def create_binary_mask(annot: dict, img_shape, thickness: int = 2) -> np.ndarray:
    """Draw only polylines (no single points) into a uint8 mask, same size as image."""
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    for cls, points in annot.items():
        if not points or len(points) < 2:
            continue
        # denorm to pixels
        pts_xy = []
        for p in points:
            x, y = _xy_from_point(p)
            px = int(round(x * w))
            py = int(round(y * h))
            pts_xy.append((px, py))
        # draw consecutive segments
        for i in range(len(pts_xy) - 1):
            cv2.line(mask, pts_xy[i], pts_xy[i + 1], color=255, thickness=thickness, lineType=cv2.LINE_AA)
    return mask

def overlay_mask(img_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    overlay = img_bgr.copy()
    color_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    color_mask[mask > 0] = (0, 255, 0)   # green overlay
    cv2.addWeighted(color_mask, alpha, overlay, 1 - alpha, 0, overlay)
    return overlay

# --------------------------- main ---------------------------

def main(
    images_dir: str = r"c:\Users\ASIA LAPTOP.COM\Desktop\codes\PnLCalib\dataset\soccernet-calibration-sportlight\valid",
    masks_dir: str = r"c:\Users\ASIA LAPTOP.COM\Desktop\codes\PnLCalib\dataset\soccernet-calibration-sportlight\masks",
    save_overlays: bool = True,
    overlays_dir: str = r"c:\Users\ASIA LAPTOP.COM\Desktop\codes\PnLCalib\dataset\soccernet-calibration-sportlight\overlays",
    thickness: int = 2
):
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    overlays_dir = Path(overlays_dir)

    masks_dir.mkdir(parents=True, exist_ok=True)
    if save_overlays:
        overlays_dir.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    img_paths = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in exts])
    print(img_paths)
    saved, skipped = 0, 0
    for img_path in tqdm(img_paths, desc="Building masks"):
        stem = img_path.stem
        annot_path = images_dir / f"{stem}.json"
        if not annot_path.exists():
            skipped += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            skipped += 1
            continue

        try:
            annot = read_annot(str(annot_path))  # should return dict[str, list[[x,y] or {'x','y'}]]
        except Exception:
            skipped += 1
            continue

        mask = create_binary_mask(annot, img.shape, thickness=thickness)

        out_mask_path = masks_dir / f"{stem}_mask.png"
        cv2.imwrite(str(out_mask_path), mask)

        if save_overlays:
            over = overlay_mask(img, mask, alpha=0.5)
            cv2.imwrite(str(overlays_dir / f"{stem}_overlay.jpg"), over)

        saved += 1

    print(f"Done. Saved: {saved}  |  Skipped (missing/bad): {skipped}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create binary line masks for all images in a folder.")
    parser.add_argument("--images-dir", type=str, default=r"c:\Users\ASIA LAPTOP.COM\Desktop\codes\PnLCalib\dataset\soccernet-calibration-sportlight\valid", help="Folder containing images + matching JSONs")
    parser.add_argument("--masks-dir", type=str, default=r"c:\Users\ASIA LAPTOP.COM\Desktop\codes\PnLCalib\dataset\soccernet-calibration-sportlight\masks", help="Output folder for mask PNGs")
    parser.add_argument("--save-overlays", action="store_true", help="Also save image+mask overlays")
    parser.add_argument("--overlays-dir", type=str, default=r"c:\Users\ASIA LAPTOP.COM\Desktop\codes\PnLCalib\dataset\soccernet-calibration-sportlight\overlays", help="Output folder for overlays")
    parser.add_argument("--thickness", type=int, default=2, help="Line thickness in pixels (mask)")
    args = parser.parse_args()

    main(
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        save_overlays=args.save_overlays,
        overlays_dir=args.overlays_dir,
        thickness=args.thickness,
    )
