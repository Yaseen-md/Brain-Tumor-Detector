import os

DATASET_PATH = "../dataset"
SPLITS = ["train", "val", "test"]
CLASSES = ["glioma", "meningioma", "pituitary", "non-tumor"]

def validate_dataset():
    for split in SPLITS:
        split_path = os.path.join(DATASET_PATH, split)
        assert os.path.exists(split_path), f"Missing {split} folder!"
        for cls in CLASSES:
            cls_path = os.path.join(split_path, cls)
            assert os.path.exists(cls_path), f"Missing class folder: {cls_path}"
            assert len(os.listdir(cls_path)) > 0, f"{cls_path} is empty!"

print("✅ Dataset structure validated!")