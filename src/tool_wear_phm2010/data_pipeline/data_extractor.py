from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from loguru import logger

from tool_wear_phm2010.data_types.models import Cut

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


class DataExtractor:
    """Load PHM 2010 cuts one at a time."""

    def __init__(self, data_dir: Path) -> None:
        self.logger = logger.bind(class_name=self.__class__.__name__)
        self.data_dir = data_dir

    def load_wear(self, cutter: str) -> np.ndarray | None:
        path = self.data_dir / cutter / f'{cutter}_wear.csv'
        if not path.exists():
            return None
        wear = pd.read_csv(path, index_col='cut')
        return wear.to_numpy(dtype=np.float32)

    def iter_cuts(self, cutter: str) -> Iterator[Cut]:
        wear = self.load_wear(cutter)
        paths = sorted((self.data_dir / cutter / cutter).glob('*.csv'))
        self.logger.info(f'Loading {cutter}: {len(paths)} cuts')
        for i, path in enumerate(paths):
            signals = pd.read_csv(
                path,
                header=None,
                engine='pyarrow',
                dtype='float32',
            ).to_numpy()
            yield Cut(
                cutter=cutter,
                index=i + 1,
                signals=signals,
                wear=None if wear is None else wear[i],
            )
