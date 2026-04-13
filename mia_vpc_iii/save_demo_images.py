"""Save sample images from the tf_flowers dataset for the Streamlit demo.

Downloads the original tf_flowers archive from TensorFlow's hosting and
extracts 2 images per class into data/demo/.
"""

import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from mia_vpc_iii.config import DATA_DIR

DEMO_DIR = DATA_DIR / "demo"
IMAGES_PER_CLASS = 2
FLOWERS_URL = (
    "https://storage.googleapis.com/download.tensorflow.org/"
    "example_images/flower_photos.tgz"
)
CLASS_NAMES = ["dandelion", "daisy", "tulips", "sunflowers", "roses"]


def main() -> None:
    DEMO_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        archive_path = Path(tmpdir) / "flower_photos.tgz"
        print(f"Downloading tf_flowers from {FLOWERS_URL} ...")
        urllib.request.urlretrieve(FLOWERS_URL, archive_path)

        print("Extracting archive ...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(tmpdir, filter="data")

        flowers_root = Path(tmpdir) / "flower_photos"

        for class_name in CLASS_NAMES:
            class_dir = flowers_root / class_name
            if not class_dir.is_dir():
                print(f"Warning: class directory not found: {class_dir}")
                continue

            images = sorted(class_dir.iterdir())
            for i, img_path in enumerate(images[:IMAGES_PER_CLASS]):
                dest = DEMO_DIR / f"{class_name}_{i}.jpg"
                shutil.copy2(img_path, dest)
                print(f"Saved {dest.name}")

    print(f"\nDone — images saved to {DEMO_DIR}")


if __name__ == "__main__":
    main()
