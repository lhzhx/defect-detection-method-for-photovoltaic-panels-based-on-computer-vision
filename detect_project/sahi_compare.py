import cv2
import matplotlib.pyplot as plt
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction
from sahi.utils.cv import read_image
from ultralytics import YOLO

def main():
    model_path = r"D:\code\ultralytics-v8.2.100\runs\detect\pv_yolov11n_custom_pretrained5\weights\best.pt"
    image_path = r"D:\code\ultralytics-v8.2.100\detect_project\images\20221121_084125244.bmp"

    # SAHI detection model
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=model_path,
        confidence_threshold=0.2,
        device="cuda:0", # or 'cpu'
    )

    # Standard Prediction
    result_standard = get_prediction(image_path, detection_model)
    result_standard.export_visuals(export_dir="detect_project/sahi_out", file_name="standard")
    
    # Sliced Prediction
    result_sliced = get_sliced_prediction(
        image_path,
        detection_model,
        slice_height=512,
        slice_width=512,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2
    )
    result_sliced.export_visuals(export_dir="detect_project/sahi_out", file_name="sliced")

    img_standard = cv2.imread("detect_project/sahi_out/standard.png")
    img_sliced = cv2.imread("detect_project/sahi_out/sliced.png")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    axes[0].imshow(cv2.cvtColor(img_standard, cv2.COLOR_BGR2RGB) if img_standard is not None else img_standard)
    axes[0].set_title("Standard Inference")
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(img_sliced, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Sliced Inference (SAHI)")
    axes[1].axis("off")

    output_path = "slice_compare.jpg"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Comparison image saved to {output_path}")

if __name__ == "__main__":
    main()
