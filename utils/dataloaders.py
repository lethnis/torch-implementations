import os
from pathlib import Path
from typing import Tuple
import numpy as np

import matplotlib.pyplot as plt

from PIL import Image

import torch
from torch.utils.data import Dataset

import torchvision
from torchvision import transforms


class ImageClassificationDataset(Dataset):
    def __init__(self, directory: Path, transform: torchvision.transforms = None, name: str = "train"):
        """Warning: use one of ImageClassificationDataset.from_splitted
        or ImageClassificationDataset.from_full methods.

        Note:
            1. full transforms will be only applied for 'train' dataset.
            2. for 'val' and 'test' will be applied only Resize, Normalize and ToTensor.

        Args:
            directory (Path): path to the directory with classes
            transform (transforms, optional): transformations that will be applied. Defaults to None.
            name (str, optional): name of the dataset. Needed for sanity check. Defaults to "train".

        Attributes:
            directory (Path): path to the directory with classes.
            paths (list[Path]): list of paths to the images.
            name (str): name of the dataset.
            transform (torchvision.transforms): list of transformations that will be applied.
            classes (list[str]): list of classes.
            class_to_idx (dict[str, int]): dict of pairs class and their index.
            idx_to_class (dict[int, str]): dict of pairs index and their class.
        """

        self.directory = directory
        self.paths = list(directory.glob("*/*"))

        self.name = name
        self.transform = self._check_transform(transform)

        self.classes = [i.name for i in sorted(self.directory.glob("*/"))]
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))
        self.idx_to_class = dict(zip(range(len(self.classes)), self.classes))

        self.targets = torch.tensor([self.class_to_idx[i.parent.name] for i in self.paths])

    @classmethod
    def from_splitted(cls, directory: str, transform: torchvision.transforms = None):
        """Generates Image Classification Datasets from directory with the
        'train', 'test', 'val' folders.

        Note:
            1. full transforms will be only applied for 'train' dataset.
            2. for 'val' and 'test' will be applied only Resize, Normalize and ToTensor.

        Args:
            directory (str): base directory with splitted data.
            transform (transforms): transforms for data preparation and augmentation. Defaults to None.

        Returns:
            tuple: a collection of datasets based on subdirectories in provided directory. Tries to return in the following order: "train", "val", "test".
        """

        directory = Path(directory)

        # find what folders exactly in main directory
        parts = np.array(["train", "val", "valid", "validation", "test"])
        indexes = np.intersect1d(np.asarray(os.listdir(directory)), parts, return_indices=True)[2]

        if not len(parts) > 0:
            print(f"Didn't found any of {parts} in {directory}")
            exit()

        parts = parts[sorted(indexes)]
        return (ImageClassificationDataset(directory / i, transform, name=i) for i in parts)

    @classmethod
    def from_full(cls, directory: str, transform: torchvision.transforms = None, split: float = 0.1, seed: int = 0):
        """Generates 'train' and 'test' datasets from folder with classes.

        Note:
            1. full transforms will be only applied for 'train' dataset.
            2. for 'test' will be applied only Resize, Normalize and ToTensor.

        Args:
            directory (str): base directory with classes not splitted.
            transform (transforms): transforms that will be applied to datasets. Defaults to None.
            split (float, optional): train-test split factor. Defaults to 0.1.
            seed (int, optional): random seed for reproducibility. Defaults to 0.

        Returns:
            tuple: tuple of 2 ImageClassificationDataset for 'train' and 'test'.
        """

        # store classes as full paths
        directory = Path(directory)
        classes = sorted(directory.glob("*/"))

        # initialize empty lists for full paths to images
        train_paths = []
        test_paths = []

        # set seed for reproducibility
        np.random.seed(seed)

        # for every class generate mask that will take samples for train and test
        for class_ in classes:
            # full paths to images as np.array
            paths = np.array(list(class_.glob("*")))
            # store number of how much to take for test dataset
            test_part = int(len(list(paths)) * split)

            # get random indexes from all images paths
            indexes = np.random.choice(np.arange(len(list(paths))), test_part, replace=False)
            # mask will look like [False False False] for every image
            mask = np.zeros_like(paths, dtype=bool)
            # values of mask will be set to True for images we will take
            mask[indexes] = 1

            # apply mask for all paths and add chosen paths to test_paths
            test_paths.extend(paths[mask])
            # apply inverse mask for all paths
            train_paths.extend(paths[~mask])

        # now generate datasets and change all paths to paths for a specific dataset
        train_ds = ImageClassificationDataset(directory, transform, "train")
        train_ds.paths = train_paths
        train_ds.targets = torch.tensor([train_ds.class_to_idx[i.parent.name] for i in train_ds.paths])

        test_ds = ImageClassificationDataset(directory, transform, "test")
        test_ds.paths = test_paths
        test_ds.targets = torch.tensor([test_ds.class_to_idx[i.parent.name] for i in test_ds.paths])

        return train_ds, test_ds

    def display_samples(self, rows: int = 4, cols: int = 4):
        """Displays images on the screen.

        Args:
            rows (int, optional): number of rows. Defaults to 4.
            cols (int, optional): number of columns. Defaults to 4.
        """

        _ = self._draw_samples(rows, cols)
        plt.show()

    def save_samples(self, rows: int = 4, cols: int = 4, filepath: str = "samples.png"):
        """Save images from the dataset.

        Args:
            rows (int, optional): number of rows. Defaults to 4.
            cols (int, optional): number of columns. Defaults to 4.
            filepath (str, optional): where to save a plot. Defaults to "samples.png".
        """
        _ = self._draw_samples(rows, cols)
        plt.savefig(filepath)

    def display_transformed(self, rows: int = 4, cols: int = 4):
        """Displays transformed images on the screen.

        Args:
            rows (int, optional): number of rows. Defaults to 4.
            cols (int, optional): number of columns. Defaults to 4.
        """

        _ = self._draw_transformed(rows, cols)
        plt.show()

    def save_transformed(self, rows: int = 4, cols: int = 4, filepath: str = "samples.png"):
        """Save transformed images from the dataset.

        Args:
            rows (int, optional): number of rows. Defaults to 4.
            cols (int, optional): number of columns. Defaults to 4.
            filepath (str, optional): where to save a plot. Defaults to "samples.png".
        """

        _ = self._draw_transformed(rows, cols)
        plt.savefig(filepath)

    def _draw_samples(self, rows: int, cols: int) -> plt.figure:
        """Helper function for display_samples and save_samples.
        Draws images on a figure along with class index and class name.

        Args:
            rows (int): number of rows
            cols (int): number of columns

        Returns:
            plt.figure: figure with images drawn on it.
        """
        # how many images will be drawn
        num_samples = rows * cols
        # needed for random choosing from all images
        total_images = len(self.paths)
        # set up the figure size
        fig = plt.figure(figsize=(cols * 2, rows * 2))

        for i in range(num_samples):
            plt.subplot(rows, cols, i + 1)

            # get random index
            idx = np.random.randint(0, total_images)
            # get name of the class
            class_ = self.paths[idx].parent.name

            plt.imshow(self.load_image(idx))
            plt.title(f"Class {self.class_to_idx[class_]}: {class_}")
            plt.axis("off")
            plt.tight_layout()

        return fig

    def _draw_transformed(self, rows: int, cols: int) -> plt.figure:
        """Helper function for display_transformed and save_transformed.
        Draws transformed images on a figure along with class index and class name.

        Args:
            rows (int): number of rows
            cols (int): number of columns

        Returns:
            plt.figure: figure with images drawn on it.
        """
        # how many images will be drawn
        num_samples = rows * cols
        # needed for random choosing from all images
        total_images = len(self.paths)
        # set up the figure size
        fig = plt.figure(figsize=(cols * 2, rows * 2))

        for i in range(num_samples):
            plt.subplot(rows, cols, i + 1)

            # get random index
            idx = np.random.randint(0, total_images)

            img, class_idx = self[idx]

            img = img.permute(1, 2, 0)

            plt.imshow(img)
            plt.title(f"Class {class_idx}: {self.idx_to_class[class_idx]}")
            plt.axis("off")
            plt.tight_layout()

        return fig

    def _check_transform(self, transform: torchvision.transforms):
        """For 'train' split transforms will remain the same.
        For 'test' and 'val' splits method will leave only Resize, Normalize, ToTensor.

        Args:
            transform (transforms): list of transforms

        Returns:
            transforms: list of new transforms
        """

        if transform is None:
            return None

        # if split is for training don't change anything
        if self.name == "train":
            return transform

        # add to the empty list Resize and Normalize transforms
        new_transform = []
        for i in transform.transforms:
            if i.__class__.__qualname__ in ["Resize", "Normalize", "ToTensor"]:
                new_transform.append(i)

        return transforms.Compose(new_transform)

    def load_image(self, index: int) -> Image.Image:
        image_path = self.paths[index]
        return Image.open(image_path).convert("RGB")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img = self.load_image(index)
        class_name = self.paths[index].parent.name
        class_idx = self.class_to_idx[class_name]

        if self.transform:
            return self.transform(img), class_idx
        return img, class_idx
