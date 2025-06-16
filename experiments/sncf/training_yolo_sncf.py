# File: train_yolo.py

from ultralytics import YOLO


def main():
    model_name = "yolo11x.pt"

    # 1. Load a pre-trained model
    model = YOLO(model_name)

    # 2. Train the model with specific data augmentations
    results = model.train(
        # Required arguments
        data="/datasets/shared_datasets/SNCF/DATASET_etat_feu/sncf_dataset.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        name=f"{model_name.split(".pt")[0]}_sncf",
        # --- Data Augmentation Arguments ---
        # Geometric Augmentations
        degrees=15.0,  # Random rotation (-15 to +15 degrees)
        translate=0.1,  # Random translation (-10% to +10%)
        scale=0.5,  # Random scaling (-50% to +50%)
        shear=5.0,  # Shear angle in degrees
        perspective=0.0005,  # Random perspective transform
        # Flip Augmentations
        flipud=0.5,  # Probability of vertical flip (upside down)
        fliplr=0.5,  # Probability of horizontal flip (left-right, default is 0.5)
        # Color Space Augmentations
        hsv_h=0.015,  # Hue augmentation (fraction)
        hsv_s=0.7,  # Saturation augmentation (fraction)
        hsv_v=0.4,  # Value (brightness) augmentation (fraction)
        # Advanced Augmentations (enabled by default, can be turned off)
        mosaic=1.0,  # Probability of applying mosaic (combining 4 images)
        mixup=0.1,  # Probability of applying mixup (mixing two images/labels)
        copy_paste=0.1,  # Probability of applying copy-paste augmentation
    )

    print("Training finished.")
    print(f"Best model weights saved at: {results.save_dir}/weights/best.pt")


if __name__ == "__main__":
    main()
