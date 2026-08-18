"""
Albumentations pipelines for LeafDoc.

Train pipeline mirrors realistic phone-camera variation (rotation, lighting,
mild blur) without distorting the leaf shape/color signal the classifier
relies on -- so no heavy color jitter or channel shuffling, since disease
symptoms are literally defined by color/texture changes.
"""
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(cfg: dict) -> A.Compose:
    img_size = cfg["data"]["image_size"]
    aug = cfg["augmentation"]
    return A.Compose(
        [
            A.RandomResizedCrop(size=(img_size, img_size), scale=(0.8, 1.0)),
            A.HorizontalFlip(p=aug["h_flip_p"]),
            A.VerticalFlip(p=aug["v_flip_p"]),
            A.Rotate(limit=aug["rotate_limit"], p=aug["rotate_p"]),
            A.RandomBrightnessContrast(p=aug["brightness_contrast_p"]),
            A.GaussianBlur(blur_limit=(3, 5), p=aug["blur_p"]),
            A.Normalize(mean=aug["mean"], std=aug["std"]),
            ToTensorV2(),
        ]
    )


def get_val_transforms(cfg: dict) -> A.Compose:
    img_size = cfg["data"]["image_size"]
    aug = cfg["augmentation"]
    return A.Compose(
        [
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=aug["mean"], std=aug["std"]),
            ToTensorV2(),
        ]
    )
