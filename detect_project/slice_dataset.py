import os
import cv2
import glob
from pathlib import Path

def slice_image_and_labels(image_path, label_path, out_img_dir, out_l_dir, slice_size=640, overlap=0.2):
    img = cv2.imread(str(image_path))
    if img is None: return
    h, w, _ = img.shape
    
    boxes = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f.readlines():
                c, x, y, bw, bh = map(float, line.strip().split())
                # convert relative to absolute
                x1 = (x - bw/2) * w
                y1 = (y - bh/2) * h
                x2 = (x + bw/2) * w
                y2 = (y + bh/2) * h
                boxes.append([int(c), x1, y1, x2, y2])
                
    step = int(slice_size * (1 - overlap))
    
    y_starts = list(range(0, h, step))
    x_starts = list(range(0, w, step))
    
    if y_starts[-1] + slice_size > h: y_starts[-1] = max(h - slice_size, 0)
    if x_starts[-1] + slice_size > w: x_starts[-1] = max(w - slice_size, 0)
    
    base_name = Path(image_path).stem
    
    for y0 in y_starts:
        for x0 in x_starts:
            y1 = min(y0 + slice_size, h)
            x1 = min(x0 + slice_size, w)
            
            crop_img = img[y0:y1, x0:x1]
            crop_h, crop_w, _ = crop_img.shape
            
            if crop_h < 10 or crop_w < 10: continue
            
            out_boxes = []
            for box in boxes:
                c, bx1, by1, bx2, by2 = box
                # Intersection
                ix1 = max(x0, bx1)
                iy1 = max(y0, by1)
                ix2 = min(x1, bx2)
                iy2 = min(y1, by2)
                
                # Check if valid box remains
                if ix2 > ix1 and iy2 > iy1:
                    # check if center is in crop (to avoid heavily truncated edge boxes)
                    center_x = (bx1 + bx2) / 2
                    center_y = (by1 + by2) / 2
                    if x0 <= center_x <= x1 and y0 <= center_y <= y1:
                        # Convert to relative of crop
                        rel_cx = ((ix1 + ix2) / 2 - x0) / crop_w
                        rel_cy = ((iy1 + iy2) / 2 - y0) / crop_h
                        rel_bw = (ix2 - ix1) / crop_w
                        rel_bh = (iy2 - iy1) / crop_h
                        out_boxes.append(f"{c} {rel_cx:.6f} {rel_cy:.6f} {rel_bw:.6f} {rel_bh:.6f}")
                        
            # Save only if there's an object OR randomly keep some negative backgrounds (10%)
            import random
            if len(out_boxes) > 0 or random.random() < 0.1:
                crop_name = f"{base_name}_{y0}_{x0}"
                cv2.imwrite(os.path.join(out_img_dir, f"{crop_name}.jpg"), crop_img)
                if len(out_boxes) > 0:
                    with open(os.path.join(out_l_dir, f"{crop_name}.txt"), 'w') as f:
                        f.write("\n".join(out_boxes) + "\n")

def main():
    # Directories
    src_dir = r"D:\code\ultralytics-v8.2.100\detect_project\yolo_dataset_aug400"
    out_dir = r"D:\code\ultralytics-v8.2.100\detect_project\yolo_dataset_sliced"
    
    out_img = os.path.join(out_dir, "images", "train")
    out_lbl = os.path.join(out_dir, "labels", "train")
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lbl, exist_ok=True)
    
    # Process all subset images
    for subset in ['train', 'val', 'test']:
        img_paths = glob.glob(os.path.join(src_dir, "images", subset, "*.jpg")) + glob.glob(os.path.join(src_dir, "images", subset, "*.bmp"))
        for p in img_paths:
            lbl_path = p.replace("images", "labels").rsplit(".", 1)[0] + ".txt"
            slice_image_and_labels(p, lbl_path, out_img, out_lbl, slice_size=640, overlap=0.2)
            
    # YAML config
    yaml_content = f"""path: {out_dir}
train: images/train
val: images/train
names:
  0: NG
"""
    with open(os.path.join(out_dir, "dataset.yaml"), "w") as f:
        f.write(yaml_content)
    print("Slicing complete! Dataset is ready at:", out_dir)

if __name__ == "__main__":
    main()

