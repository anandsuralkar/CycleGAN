# ImageDataset.py

import os
import glob
import random
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from utils import add_watermark_tensor, add_noise_patch # <-- import from utils.py

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train',
                 watermark_tensor=None, wm_alpha=0.5, wm_pos=(1.0, 1.0)):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.watermark = watermark_tensor
        self.pos = wm_pos

        self.files_A = sorted(glob.glob(os.path.join(root, f'{mode}A', '*.*')))
        self.files_B = sorted(glob.glob(os.path.join(root, f'{mode}B', '*.*')))

    def __getitem__(self, index):
        img_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]).convert("RGB"))

        if self.unaligned:
            img_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]).convert("RGB"))
        else:
            img_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]).convert("RGB"))
        
        size_w = tuple(self.watermark.shape[1:])

        img_A_clean = img_A.clone()
        img_B_clean = img_B.clone()
        img_A_wm = add_watermark_tensor(image = img_A_clean, watermark = self.watermark, pos = self.pos, size = size_w)
        img_B_wm = add_watermark_tensor(image = img_B_clean, watermark = self.watermark, pos = self.pos, size = size_w)

        img_A_new = img_A.clone()
        img_B_new = img_B.clone()
        img_A_noise = add_noise_patch(image = img_A_new, pos = self.pos, size = size_w)
        img_B_noise = add_noise_patch(image = img_B_new, pos = self.pos, size = size_w)

        return {
            'A': img_A,
            'B': img_B,
            'wA': img_A_wm,
            'wB': img_B_wm,
            'nA': img_A_noise,
            'nB': img_B_noise,
        }

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
