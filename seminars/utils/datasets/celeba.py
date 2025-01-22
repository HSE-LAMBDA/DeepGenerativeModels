import os
import zipfile
import torch
import gdown
from torch.utils.data import Dataset
from torchvision import transforms
import re
from PIL import Image

try:
    from natsort import natsorted
    SORT_FN = natsorted
except ImportError:
    # fallback to regular sorted if natsort is not available
    SORT_FN = sorted


class CelebADataset(Dataset):
    """
    Custom Dataset class for the CelebA dataset. Automatically downloads
    from Google Drive if data is not found in the specified root_dir.

    Args:
        root_dir (str): The directory to store (or locate) the CelebA data.
        transform (callable, optional): Optional transform to be applied
            on a PIL Image sample.
    """

    def __init__(self, 
                 root_dir: str = "../data/celeba",
                 transform: transforms.Compose = None):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform

        # The unzipped folder will be <root_dir>/img_align_celeba
        self.dataset_folder = os.path.join(root_dir, "img_align_celeba")
        if not os.path.isdir(self.dataset_folder):
            # If folder doesn't exist, attempt to download & unzip
            self._download_celeba()

        # Load file names
        self.filenames = os.listdir(self.dataset_folder)
        # Ensure consistent ordering (important if you rely on index-based consistency)
        self.filenames = SORT_FN(self.filenames)

        # Check for attribute file
        attr_file_path = os.path.join(root_dir, "list_attr_celeba.txt")
        if not os.path.isfile(attr_file_path):
            raise FileNotFoundError(
                f"Could not find '{attr_file_path}'. "
                "This file should have been included in the CelebA download."
            )

        # Load attributes
        self.annotations = []
        with open(attr_file_path, "r") as f:
            lines = f.read().splitlines()

        # First line has the number of images, second line has the attribute names
        # The rest lines each correspond to one image
        # E.g.:
        # 202599
        # 5_o_Clock_Shadow Arched_Eyebrows ...
        # 000001.jpg -1 -1 ...
        # ...
        header = None
        for i, line in enumerate(lines):
            # line might have variable spaces, so split robustly
            line = re.sub(r"\s+", " ", line.strip())
            if i == 0:
                continue  # number of images
            elif i == 1:
                # header line with attribute names
                header = line.split(" ")
            else:
                parts = line.split(" ")
                filename = parts[0]
                # the rest are attribute labels
                attr_vals = [int(val) for val in parts[1:]]
                self.annotations.append((filename, attr_vals))

        # Convert to a dict: filename -> attribute array
        # so we can quickly look up attributes by filename
        self.attr_map = {
            fn: torch.tensor(attr_vals, dtype=torch.long)
            for fn, attr_vals in self.annotations
        }

    def _download_celeba(self):
        """
        If the CelebA folder isn't found, download and extract from Google Drive.
        """
        os.makedirs(self.root_dir, exist_ok=True)
        zip_path = os.path.join(self.root_dir, "img_align_celeba.zip")

        if not os.path.isfile(zip_path):
            # This is the Google Drive link from the original CelebA dataset
            download_url = "https://drive.google.com/uc?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM"
            print("Downloading CelebA dataset from Google Drive (1.3GB). This might take a while...")
            gdown.download(download_url, zip_path, quiet=False)
        else:
            print(f"Found existing ZIP file at {zip_path}. Skipping download.")

        print(f"Extracting '{zip_path}'...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(self.root_dir)
        print("Extraction finished.")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): index of the sample

        Returns:
            tuple: (image, info_dict) where:
                image is the transformed PIL image,
                info_dict is a dictionary containing:
                    - 'filename': str
                    - 'idx': int
                    - 'attributes': torch.Tensor of shape (#attributes,)
        """
        img_name = self.filenames[idx]
        img_path = os.path.join(self.dataset_folder, img_name)

        # Load the image
        img = Image.open(img_path).convert("RGB")

        # Apply transforms
        if self.transform is not None:
            img = self.transform(img)

        # Fetch attributes (if they exist in the attr_map; some extra files might appear)
        attributes = self.attr_map.get(img_name, torch.zeros(40, dtype=torch.long))

        # Return both
        info_dict = {
            "filename": img_name,
            "idx": idx,
            "attributes": attributes
        }
        return img, info_dict
