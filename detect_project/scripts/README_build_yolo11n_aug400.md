# Build YOLOv11n Dataset (40 -> 400) from LabelMe JSON

This script builds a new YOLO dataset from:

- `images/*.bmp` (40 source images)
- `annotations/*.json` (LabelMe rectangles)

and expands it to 400 images with these augmentations:

- small-angle rotation
- horizontal flip
- scale + translation
- Gaussian noise
- HSV jitter

## Script

- `scripts/build_yolo11n_aug400_from_json.py`

## Quick Run

```powershell
python .\detect_project\scripts\build_yolo11n_aug400_from_json.py --images-dir .\detect_project\images --annotations-dir .\detect_project\annotations --output-dir .\detect_project\yolo_dataset_aug400 --target-total 400 --val-ratio 0.2 --clean
```

## Output Layout

- `yolo_dataset_aug400/images/train`
- `yolo_dataset_aug400/images/val`
- `yolo_dataset_aug400/labels/train`
- `yolo_dataset_aug400/labels/val`
- `yolo_dataset_aug400/dataset.yaml`
- `yolo_dataset_aug400/build_manifest.json`

## Validate

```powershell
python .\detect_project\scripts\validate_yolo_dataset.py --dataset-dir .\detect_project\yolo_dataset_aug400 --num-classes 1
```

## Train YOLOv11n

```powershell
python .\detect_project\scripts\train_yolov11n.py --data .\detect_project\yolo_dataset_aug400\dataset.yaml --model yolo11n.pt --epochs 120 --imgsz 640 --batch 16 --workers 0 --name pv_yolo11n_aug400
```

