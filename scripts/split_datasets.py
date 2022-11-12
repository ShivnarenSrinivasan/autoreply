"""Split and save source dataset."""
from __future__ import annotations
import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple, Mapping

import numpy as np
import pandas as pd
from sklearn import (
    model_selection,
)

from autoreply import (
    dataset,
    misc_data,
)

_SEED = 42


def main(root: Path) -> None:
    """Split and save source dataset."""
    data = root.joinpath('data')

    for ds in dataset.Dataset:
        logging.debug(ds)
        _split = split(_load_raw(_raw_file_path(ds, data)))
        save_data(data, _split, ds)

    misc = misc_data.load()
    dbs = split_misc(misc)
    save_misc(data, dbs)


@dataclass(repr=False, frozen=True, kw_only=True, slots=True)
class Data:
    train: pd.DataFrame
    dev: pd.DataFrame
    test: pd.DataFrame

    def __getitem__(self, key: dataset.Type) -> pd.DataFrame:
        return getattr(self, key.value)


def _raw_file_path(ds: dataset.Dataset, root: Path) -> Path:
    paths = {
        dataset.Dataset.GROCERY: 'grocery_questions.csv',
        dataset.Dataset.MUSIC: 'music_questions.csv',
        dataset.Dataset.QA: 'qa_dataset.csv',
        # Dataset.VIDEO_GAME: 'video_game_qa.csv',
        # `video_game_qa.csv` has questions duplicated as answers
    }
    return root.joinpath('raw', paths[ds])


def save_data(root: Path, data: Data, ds: dataset.Dataset) -> None:
    for type_ in dataset.Type:
        dir_ = root.joinpath(type_.value)
        dir_.mkdir(exist_ok=True)
        file = dir_.joinpath(dataset.filename(ds))
        data[type_].to_csv(file, index=False)


def save_misc(root: Path, dbs: SplitMisc) -> None:
    for type_ in dataset.Type:
        dir_ = root.joinpath(type_.value)
        dir_.mkdir(exist_ok=True)
        dbs[type_].to_csv(dir_)


def _load_raw(file: Path) -> pd.DataFrame:
    # `qa_dataset` has a � char (U+FFFD, the official replacement char)
    # It is not possible to parse this, apparently--the only way is to set errors to ignore
    df = pd.read_csv(file, encoding=None, encoding_errors='ignore')

    return df.loc[:, ~df.columns.str.match(r'^Unnamed: \d')]


def split(df: pd.DataFrame, dev_frac: float = 0.1, test_frac: float = 0.2) -> Data:
    train_frac = 1 - (dev_frac + test_frac)

    X_train, _X_rest = model_selection.train_test_split(
        df, train_size=train_frac, random_state=_SEED
    )
    X_dev, X_test = model_selection.train_test_split(
        _X_rest, train_size=dev_frac / (dev_frac + test_frac), random_state=_SEED
    )

    assert isinstance(X_train, pd.DataFrame), 'Not a dataframe'
    assert isinstance(X_dev, pd.DataFrame), 'Not a dataframe'
    assert isinstance(X_test, pd.DataFrame), 'Not a dataframe'

    for type_, _df in (
        ('src', df),
        ('train', X_train),
        ('dev', X_dev),
        ('test', X_test),
    ):
        _log_frame_size(_df, type_)

    return Data(train=X_train, dev=X_dev, test=X_test)


SplitMisc = Mapping[dataset.Type, misc_data.DBResult]


def split_misc(raw: misc_data.MiscSchema) -> SplitMisc:
    data = raw['data']
    db = misc_data.arrays(data)
    summary = misc_data.Summary(
        misc_data.summary(data).join(db.title.set_index('title'))
    )

    title_ids = _gen_indices(summary)

    def create_db(df: pd.DataFrame, type_: dataset.Type) -> misc_data.DBResult:
        return misc_data.arrays(
            [d for d in data if d['title'] in set(title_ids[type_].index)]
        )

    return {type_: create_db(title_ids[type_], type_) for type_ in dataset.Type}


class _Indices(NamedTuple):
    train: pd.Series[int]
    dev: pd.Series[int]
    test: pd.Series[int]

    def __getitem__(self, key: dataset.Type) -> pd.DataFrame:
        return getattr(self, key.value)


def _gen_indices(summary: misc_data.Summary) -> _Indices:
    n = len(summary)
    indexer = np.arange(n)

    c1 = indexer % 9 == 0
    c2 = indexer % 3 == 0

    dev_mask = c1
    test_mask = c2 & ~c1
    train_mask = ~c2

    def _subset(mask: np.ndarray[bool, Any]) -> pd.Series[int]:
        return summary.iloc[mask]['title_id']

    train = _subset(train_mask)
    dev = _subset(dev_mask)
    test = _subset(test_mask)

    def _log():
        n_train = len(train)
        n_dev = len(dev)
        n_test = len(test)
        tot = n_dev + n_test + n_train

        def pct(x: int) -> str:
            return f'{x/tot:.2%}'

        msg = [
            f'n_train = {n_train} ({pct(n_train)})',
            f'n_dev = {n_dev} ({pct(n_dev)})',
            f'n_test = {n_test} ({pct(n_test)})',
        ]
        print('\n'.join(msg))

    _log()

    return _Indices(
        *(summary.iloc[mask]['title_id'] for mask in (train_mask, dev_mask, test_mask))
    )


def _log_frame_size(df: pd.DataFrame, type_: str) -> None:
    logging.debug(f'{type_}: {len(df)} records')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path, metavar='', help='path to root dir')
    _ARGS = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    main(_ARGS.root)
