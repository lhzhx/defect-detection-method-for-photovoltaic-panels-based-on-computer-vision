import os
import numpy as np
from ultralytics import YOLO


def main():
    # 1. 更新为切片训练由于生成的最新模型路径
    model_path = r"D:\code\ultralytics-v8.2.100\runs\detect\pv_yolov11n_custom_sliced_trained\weights\best.pt"

    # 2. 更新为切片数据集的配置文件路径
    data_path = r"D:\code\ultralytics-v8.2.100\detect_project\yolo_dataset_sliced\dataset.yaml"

    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)

    thresholds = np.arange(0.05, 0.65, 0.05)
    results_list = []

    print("Starting threshold scan...")
    for conf in thresholds:
        print(f"\n--- Scanning conf={conf:.2f} ---")
        metrics = model.val(
            data=data_path,
            imgsz=640,  # 3. 注意：切片数据集的训练分辨率为640，此处必须对齐
            conf=conf,
            split='val',
            verbose=False,
            plots=False
        )

        p = metrics.box.mp
        r = metrics.box.mr
        map50 = metrics.box.map50

        f1 = 2 * (p * r) / (p + r + 1e-16)

        results_list.append({
            'conf': conf,
            'p': p,
            'r': r,
            'f1': f1,
            'map50': map50
        })
        print(f"Result for conf={conf:.2f}: P={p:.4f}, R={r:.4f}, F1={f1:.4f}, mAP50={map50:.4f}")

    print("\n" + "=" * 50)
    print("Threshold Scan Summary (Sorted by F1-Score)")
    print("=" * 50)

    results_list.sort(key=lambda x: x['f1'], reverse=True)

    print(f"{'Rank':<5} | {'Conf':<6} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'mAP50':<10}")
    print("-" * 65)
    for i, res in enumerate(results_list):
        print(
            f"{i + 1:<5} | {res['conf']:<6.2f} | {res['p']:<10.4f} | {res['r']:<10.4f} | {res['f1']:<10.4f} | {res['map50']:<10.4f}")


if __name__ == "__main__":
    main()
