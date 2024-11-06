import logging
from os import listdir
from os.path import splitext
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image, ImageOps
from torch.utils.data import Dataset


class BasicDataset(Dataset):
    def __init__(
        self, images_dir: str, masks_dir: str, scale: float = 1.0, mask_suffix: str = ""
    ) -> None:
        self.images_dir: Path = Path(images_dir)
        self.masks_dir: Path = Path(masks_dir)
        assert 0 < scale <= 1, "Scale must be between 0 and 1"
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids: List[str] = [
            splitext(file)[0]
            for file in listdir(images_dir)
            if not file.startswith(".")
        ]
        if not self.ids:
            raise RuntimeError(
                f"No input file found in {images_dir}, make sure you put your images there"
            )
        logging.info(f"Creating dataset with {len(self.ids)} examples")

    def __len__(self) -> int:
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img: Image.Image, scale: float, is_mask: bool) -> NDArray:
        w, h = (
            2048,
            2048,
        )  # pil_img.size # change this if your images are sometimes bigger than 2048x2048px or not square
        newW, newH = int(scale * w), int(scale * h)
        assert (
            newW > 0 and newH > 0
        ), "Scale is too small, resized images would have no pixel"
        pil_img = pil_img.resize(
            (newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC
        )
        img_ndarray: NDArray = np.asarray(pil_img)

        if img_ndarray.ndim == 2 and not is_mask:
            img_ndarray = img_ndarray[np.newaxis, ...]
        elif not is_mask:
            img_ndarray = img_ndarray.transpose((2, 0, 1))

        if not is_mask:
            img_ndarray = img_ndarray / 255

        if is_mask:
            if [np.max(img_ndarray), np.min(img_ndarray)] != [1, 0]:
                img_ndarray = img_ndarray / 255
        return img_ndarray

    @classmethod
    def load(cls, filename: Union[str, Path]) -> Image.Image:
        ext: str = splitext(filename)[1]
        if ext in [".npz", ".npy"]:
            return Image.fromarray(np.load(str(filename)))
        elif ext in [".pt", ".pth"]:
            return Image.fromarray(torch.load(str(filename)).numpy())
        else:
            img: Image.Image = Image.open(filename)
            img.load()  # required for png.split()

            if len(np.array(img).shape) > 2:
                background: Image.Image = Image.new("RGB", img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])  # 3 is the alpha channel
                img = ImageOps.grayscale(background)

            return img

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        name: str = self.ids[idx]
        mask_file: List[Path] = list(
            self.masks_dir.glob(name + self.mask_suffix + ".*")
        )
        img_file: List[Path] = list(self.images_dir.glob(name + ".*"))

        assert (
            len(mask_file) == 1
        ), f"Either no mask or multiple masks found for the ID {name}: {mask_file}"
        assert (
            len(img_file) == 1
        ), f"Either no image or multiple images found for the ID {name}: {img_file}"

        mask: Image.Image = self.load(mask_file[0])
        img: Image.Image = self.load(img_file[0])

        assert (
            img.size == mask.size
        ), f"Image and mask {name} should be the same size, but are {img.size} and {mask.size}"

        img_processed: NDArray = self.preprocess(img, self.scale, is_mask=False)
        mask_processed: NDArray = self.preprocess(mask, self.scale, is_mask=True)

        return {
            "image": torch.as_tensor(img_processed.copy()).float().contiguous(),
            "mask": torch.as_tensor(mask_processed.copy()).long().contiguous(),
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir: str, masks_dir: str, scale: float = 1) -> None:
        super().__init__(images_dir, masks_dir, scale, mask_suffix="_mask")
