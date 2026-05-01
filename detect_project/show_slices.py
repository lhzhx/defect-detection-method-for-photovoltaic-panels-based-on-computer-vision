import cv2
import matplotlib.pyplot as plt
import numpy as np

def main():
    image_path = r"D:\code\ultralytics-v8.2.100\detect_project\images\20221121_084125244.bmp"
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    h, w, _ = img.shape
    
    slice_h, slice_w = 512, 512
    overlap_h = int(slice_h * 0.2)
    overlap_w = int(slice_w * 0.2)
    
    step_h = slice_h - overlap_h
    step_w = slice_w - overlap_w
    
    # Calculate grid size
    grid_rows = (h - overlap_h) // step_h + (1 if (h - overlap_h) % step_h != 0 else 0)
    grid_cols = (w - overlap_w) // step_w + (1 if (w - overlap_w) % step_w != 0 else 0)
    
    fig = plt.figure(figsize=(20, 10))
    
    # Left: Original Image with grid lines
    ax1 = fig.add_subplot(1, 2, 1)
    img_with_grid = img_rgb.copy()
    
    slices = []
    
    for y in range(0, h, step_h):
        for x in range(0, w, step_w):
            y_end = min(h, y + slice_h)
            x_end = min(w, x + slice_w)
            
            # Start position might need adjustment if we hit the edge to keep slice size constant if possible
            y_start = max(0, y_end - slice_h) if y_end == h else y
            x_start = max(0, x_end - slice_w) if x_end == w else x
            
            cv2.rectangle(img_with_grid, (x_start, y_start), (x_end, y_end), (255, 0, 0), 5)
            
            slice_img = img_rgb[y_start:y_end, x_start:x_end]
            slices.append(slice_img)
            
            if x_end == w:
                break
        if y_end == h:
            break
            
    ax1.imshow(img_with_grid)
    ax1.set_title("Original Image with Slice Grid")
    ax1.axis("off")
    
    # Right: Slices shown in a grid
    n_slices = len(slices)
    cols = min(4, int(np.ceil(np.sqrt(n_slices))))
    rows = int(np.ceil(n_slices / cols))
    
    gap = 20
    canvas_h = rows * slice_h + (rows - 1) * gap
    canvas_w = cols * slice_w + (cols - 1) * gap
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
    
    for i, slc in enumerate(slices):
        r = i // cols
        c = i % cols
        y_pos = r * (slice_h + gap)
        x_pos = c * (slice_w + gap)
        
        sh, sw, _ = slc.shape
        # Sometimes dimensions don't match slice_w/slice_h exactly if it's the edge. 
        # But we handled crop logic earlier so they should be exact unless h/w < slice size
        canvas[y_pos:y_pos+sh, x_pos:x_pos+sw] = slc

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(canvas)
    ax2.set_title(f"Individual Slices ({n_slices} total)")
    ax2.axis("off")
    
    output_path = "D:\\code\\ultralytics-v8.2.100\\detect_project\\image_slicing_effect.jpg"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()

