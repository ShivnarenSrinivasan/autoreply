from enum import Enum
from pathlib import Path

import pandas as pd

from . import (
    constants as C,
)


class Type(str, Enum):
    """Dataset type."""

    TRAIN = 'train'
    DEV = 'dev'
    TEST = 'test'


class Dataset(str, Enum):
    """Dataset enumeration."""

    GROCERY = 'grocery'
    MUSIC = 'music'
    QA = 'qa'


def load(ds: Dataset, type_: Type = Type.TRAIN, root: Path = C.DATA) -> pd.DataFrame:
    path = root.joinpath(type_.value, filename(ds))
    return pd.read_csv(path)


def filename(ds: Dataset) -> str:
    return f'{ds.value}.csv'
