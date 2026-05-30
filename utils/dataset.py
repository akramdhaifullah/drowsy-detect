import os

from PIL import Image
from torch.utils.data import Dataset


class MRLEyeDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        image_path = self.image_paths[idx]

        image = Image.open(image_path).convert("RGB")

        filename = os.path.basename(image_path)
        filename = filename.replace(".png", "")

        parts = filename.split("_")

        label = int(parts[4])

        if self.transform:
            image = self.transform(image)

        return image, label
