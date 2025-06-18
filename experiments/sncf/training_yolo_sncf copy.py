# File: train_yolo.py
from ultralytics import YOLO

import wandb
from wandb.integration.ultralytics import add_wandb_callback


def main():
    model_name = "yolo12x.pt"
    # model_name = "rtdetr-x.pt"

    # 1. Load a pre-trained model
    model = YOLO(model_name)
    # model = RTDETR(model_name)

    # wandb.init(
    add_wandb_callback(model, enable_model_checkpointing=True)

    # add_wandb_callback(model, enable_model_checkpointing=True)

    # 2. Train the model with specific data augmentations
    results = model.train(
        # Required arguments
        data="/datasets/shared_datasets/SNCF/DATASET_etat_feu/sncf_dataset.yaml",
        epochs=300,
        imgsz=640,  # Considerer plus grand TODO
        batch=18,  # 24 passe mais beaucoup plus lent! # 8,
        device=[0, 1],
        # autoanchor=True,
        optimizer="auto",
        pretrained=True,  # "./runs/detect/yolov8_sncf_augmented_training4/weights/best.pt",
        # multi_scale=True,
        amp=True,
        name=f"{model_name.split('.pt')[0]}_sncf",
        project="cods-sncf",
        # --- Data Augmentation Arguments ---
        # Geometric Augmentations
        degrees=12.0,  # Random rotation (-15 to +15 degrees)
        translate=0.2,  # Random translation (-10% to +10%)
        scale=0.5,  # Random scaling (-50% to +50%)
        shear=5.0,  # Shear angle in degrees
        perspective=0.002,  # Random perspective transform
        # Flip Augmentations
        flipud=0.5,  # Probability of vertical flip (upside down)
        fliplr=0.5,  # Probability of horizontal flip (left-right, default is 0.5)
        # Color Space Augmentations
        hsv_h=0.03,  # Hue augmentation (fraction)
        hsv_s=0.8,  # Saturation augmentation (fraction)
        hsv_v=0.5,  # Value (brightness) augmentation (fraction)
        # Advanced Augmentations (enabled by default, can be turned off)
        mosaic=0.2,  # Probability of applying mosaic (combining 4 images)
        mixup=0.2,  # Probability of applying mixup (mixing two images/labels)
        copy_paste=0.1,  # Probability of applying copy-paste augmentation
    )

    print("Training finished.")
    print(f"Best model weights saved at: {results.save_dir}/weights/best.pt")

    wandb.finish()


if __name__ == "__main__":
    main()
