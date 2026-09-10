from typing import Sequence, Union, List, Optional
import torch
from torch.utils.data import Dataset, Subset
import numpy as np


class CognitiveLoadDataset(Dataset):
    def __init__(
        self,
        dynamic_data: Union[np.ndarray, torch.Tensor],
        static_data: Union[np.ndarray, torch.Tensor],
        labels: Union[np.ndarray, torch.Tensor],
        subject_ids: Sequence[Union[int, str]],
    ) -> None:
        self.X_dyn = torch.as_tensor(dynamic_data, dtype=torch.float32)

        static_tensor = torch.as_tensor(static_data, dtype=torch.long)
        if static_tensor.ndim == 1:
            static_tensor = static_tensor.unsqueeze(1)
        self.X_stat = static_tensor

        self.y = torch.as_tensor(labels, dtype=torch.float32)
        self.subject_id = list(subject_ids)

        if not (
            self.X_dyn.shape[0] == self.X_stat.shape[0]
            and self.X_stat.shape[0] == self.y.shape[0]
            and self.y.shape[0] == len(self.subject_id)
        ):
            raise ValueError("All inputs must have the same number of samples")

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return (
            self.X_dyn[idx],
            self.X_stat[idx],
            self.y[idx],
            self.subject_id[idx],
        )

    @property
    def num_dynamic_features(self) -> int:
        return self.X_dyn.shape[1] if self.X_dyn.ndim > 1 else 1

    @property
    def num_static_features(self) -> int:
        return self.X_stat.shape[1] if self.X_stat.ndim > 1 else 1

    def get_indices_by_subjects(
        self, subject_ids: Sequence[Union[int, str]]
    ) -> List[int]:
        subject_set = set(subject_ids)
        return [
            idx for idx, subject_id in enumerate(self.subject_id)
            if subject_id in subject_set
        ]

    def get_subset_by_subjects(
        self, subject_ids: Sequence[Union[int, str]]
    ) -> Subset:
        indices = self.get_indices_by_subjects(subject_ids)
        return Subset(self, indices)

    def get_subset_by_mask(
        self, mask: Union[np.ndarray, Sequence[bool]]
    ) -> Subset:
        if isinstance(mask, np.ndarray):
            if mask.ndim != 1 or len(mask) != len(self):
                raise ValueError(
                    "Mask must be a 1D boolean array of the same length as the dataset"
                )
            indices = np.nonzero(mask)[0].tolist()
        else:
            mask_list = list(mask)
            if len(mask_list) != len(self):
                raise ValueError("Mask length must match dataset length")
            indices = [idx for idx, m in enumerate(mask_list) if bool(m)]

        return Subset(self, indices)
