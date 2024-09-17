# -*- coding: utf-8 -*-
import math
import torch
from torch import Tensor
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import v2 as transforms
from typing import Tuple
from .DatasetWrapper import DatasetWrapper
import numpy as np
from numpy import power
import random  # 导入标准库 random

class CIFAR10_(DatasetWrapper[Tuple[Tensor, int]]):
    num_classes = 10
    mean = (0.49139967861519607843, 0.48215840839460784314, 0.44653091444546568627)
    std = (0.21117028181572183225, 0.20857934290628859220, 0.21205155387102001073)
    basic_transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean, std, inplace=True),
            transforms.ToPureTensor(),
        ]
    )
    augment_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, 4),
            transforms.RandomHorizontalFlip(),
            transforms.TrivialAugmentWide(
                interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean, std, inplace=True),
            transforms.ToPureTensor(),
        ]
    )

    def __init__(
        self,
        root: str,
        train: bool,
        base_ratio: float,
        num_phases: int,
        augment: bool = False,
        inplace_repeat: int = 1,
        shuffle_seed: int | None = None,
    ) -> None:
        self.dataset = CIFAR10(root, train=train, download=True)
        super().__init__(
            self.dataset.targets,
            base_ratio,
            num_phases,
            augment,
            inplace_repeat,
            shuffle_seed,
        )


class CIFAR100_(DatasetWrapper[Tuple[Tensor, int]]):
    num_classes = 100
    mean = (0.50707515923713235294, 0.48654887331495098039, 0.44091784336703431373)
    std = (0.26733428848992695514, 0.25643846542136995765, 0.27615047402246589731)
    # std = (0.21103932286924015314, 0.20837755491382136483, 0.21551368222930648019)
    basic_transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean, std, inplace=True),
            transforms.ToPureTensor(),
        ]
    )
    augment_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, 4),
            transforms.RandomHorizontalFlip(),
            transforms.TrivialAugmentWide(
                interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean, std, inplace=True),
            transforms.ToPureTensor(),
        ]
    )

    def __init__(
        self,
        root: str,
        train: bool,
        base_ratio: float,
        num_phases: int,
        augment: bool = False,
        inplace_repeat: int = 1,
        shuffle_seed: int | None = None,
    ) -> None:
        self.dataset = CIFAR100(root, train=train, download=True)

        super().__init__(
            self.dataset.targets,
            base_ratio,
            num_phases,
            augment,
            inplace_repeat,
            shuffle_seed,
        )



class LongTailedCIFAR100_(DatasetWrapper[Tuple[Tensor, int]]):
    num_classes = 100
    mean = (0.50707515923713235294, 0.48654887331495098039, 0.44091784336703431373)
    std = (0.26733428848992695514, 0.25643846542136995765, 0.27615047402246589731)
    basic_transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean, std, inplace=True),
            transforms.ToPureTensor(),
        ]
    )
    augment_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, 4),
            transforms.RandomHorizontalFlip(),
            transforms.TrivialAugmentWide(
                interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean, std, inplace=True),
            transforms.ToPureTensor(),
        ]
    )

    def __init__(
        self,
        root: str,
        train: bool,
        base_ratio: float,
        num_phases: int,
        augment: bool = False,
        inplace_repeat: int = 1,
        shuffle_seed: int | None = None,
    ) -> None:
        # Load the original CIFAR-100 dataset
        self.dataset = CIFAR100(root, train=train, download=True)
        
        # If train is True, modify the dataset to create the long-tail distribution
        if train:
            # Set random seed for reproducibility if needed
            if shuffle_seed is not None:
                np.random.seed(shuffle_seed)
                torch.manual_seed(shuffle_seed)

            # Build class indices (list of sample indices for each class)
            targets_np = np.array(self.dataset.targets)
            data_np = np.array(self.dataset.data)
            
            class_indices = [[] for _ in range(self.num_classes)]
            for idx, label in enumerate(targets_np):
                class_indices[label].append(idx)

            # Calculate the number of classes in base and incremental tasks
            base_size = 10  # e.g., base_size = 10
            incremental_size = 90       # e.g., incremental_size = 90
            phase_size = 10        # e.g., phase_size = 90 // 9 = 10
            num_phases=9
            # Compute sample sizes per task
            max_samples = 500  # Samples per class in base task
            r = (1 / max_samples) ** (1 / num_phases)  # Decay factor

            samples_per_task = []
            # Base task sample size
            samples_per_task.append(max_samples)

            # Incremental tasks sample sizes
            for t in range(1, num_phases + 1):  # t from 1 to num_phases
                n_t = int(round(max_samples * (r ** t)))
                n_t = max(n_t, 1)
                samples_per_task.append(n_t)

            # Now, select samples for each class according to the computed sample sizes
            selected_indices = []
            # Base task classes
            base_classes = list(range(0, base_size))
            s_base = samples_per_task[0]
            for c in base_classes:
                idxs = class_indices[c]
                np.random.shuffle(idxs)
                selected_idxs = idxs[:s_base]
                selected_indices.extend(selected_idxs)

            # Incremental tasks
            for t in range(1, num_phases + 1):
                s_t = samples_per_task[t]
                start_class = base_size + (t - 1) * phase_size
                end_class = base_size + t * phase_size
                task_classes = list(range(start_class, end_class))
                for c in task_classes:
                    idxs = class_indices[c]
                    np.random.shuffle(idxs)
                    selected_idxs = idxs[:s_t]
                    selected_indices.extend(selected_idxs)

            # Update self.dataset.data and self.dataset.targets with the selected samples
            self.dataset.data = data_np[selected_indices]
            self.dataset.targets = [targets_np[idx] for idx in selected_indices]
        
        # Call the parent class constructor with the (possibly modified) targets
        super().__init__(
            self.dataset.targets,
            base_ratio,
            num_phases,
            augment,
            inplace_repeat,
            shuffle_seed,
        )



