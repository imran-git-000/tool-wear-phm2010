from typing import TYPE_CHECKING

import numpy as np
import torch
from loguru import logger
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from tool_wear_phm2010.data_types.models import Cut

Sample = tuple[torch.Tensor, torch.Tensor, int]


class Phm2010Dataset(Dataset[Sample]):
    """Overlapping windows cut inside each cut, never across cuts.

    Returns (window, wear, cut_position). window is (n_channels,
    window_size); wear is per-flute wear, NaN for unlabelled cuts;
    cut_position indexes `cuts` and aggregates window outputs back to
    cut level.
    """

    def __init__(self, cuts: list[Cut], window_size: int, hop: int) -> None:
        if hop < 1:
            raise ValueError(f'hop must be >= 1, got {hop}')
        shortest = min(len(cut.signals) for cut in cuts)
        if not 1 <= window_size <= shortest:
            raise ValueError(
                f'window_size must be in [1, {shortest}], got {window_size}'
            )

        # per cut: (n_windows, n_channels, window_size), a view, no copy
        self.windows = [
            sliding_window_view(cut.signals, window_size, axis=0)[::hop]
            for cut in cuts
        ]
        self.wear = [
            np.full(3, np.nan, dtype=np.float32)
            if cut.wear is None
            else cut.wear
            for cut in cuts
        ]
        # end index of each cut's windows in the flat index space
        self.ends = np.cumsum([len(w) for w in self.windows])

    def __len__(self) -> int:
        return int(self.ends[-1])

    def __getitem__(self, index: int) -> Sample:
        cut = int(np.searchsorted(self.ends, index, side='right'))
        start = self.ends[cut - 1] if cut else 0
        # copy: torch cannot wrap a strided view
        window = np.array(self.windows[cut][index - start], dtype=np.float32)
        return (
            torch.from_numpy(window),
            torch.from_numpy(self.wear[cut]),
            cut,
        )


class DataLoader:
    """Turn a list of cuts into a batched torch DataLoader of windows."""

    def __init__(
        self,
        window_size: int,
        hop: int,
        num_workers: int = 0,
        pin_memory: bool | None = None,
        seed: int | None = 0,
    ) -> None:
        self.logger = logger.bind(class_name=self.__class__.__name__)
        self.window_size = window_size
        self.hop = hop
        # num_workers > 0 pickles the cuts into every worker (spawn on
        # Windows): gigabytes per worker for a full cutter
        self.num_workers = num_workers
        self.pin_memory = (
            (num_workers > 0 and torch.cuda.is_available())
            if pin_memory is None
            else pin_memory
        )
        self.seed = seed

    def get_dataloader(
        self,
        cuts: list[Cut],
        batch_size: int,
        train: bool,
    ) -> TorchDataLoader:
        """Shuffle and drop the last partial batch only for training."""
        dataset = Phm2010Dataset(
            cuts=cuts,
            window_size=self.window_size,
            hop=self.hop,
        )
        if train and len(dataset) < batch_size:
            raise ValueError(
                f'{len(dataset)} windows is fewer than batch_size '
                f'{batch_size}; with drop_last the loader would be empty'
            )

        self.logger.info(
            'Built {split} loader: {n_cuts} cuts, {n} windows',
            split='train' if train else 'eval',
            n_cuts=len(cuts),
            n=len(dataset),
        )
        generator = None
        if train and self.seed is not None:
            generator = torch.Generator().manual_seed(self.seed)
        return TorchDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=train,
            drop_last=train,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            generator=generator,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers else None,
        )
