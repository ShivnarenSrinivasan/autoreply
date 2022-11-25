import pandas as pd
from typing import Iterable

from autoreply import (
    misc_data,
)


def summary(db: misc_data.DBResult) -> pd.DataFrame:
    para = db.paragraph.groupby('title_id')['para_id'].count()
    ques = db.ques_ans.groupby('title_id')['question'].count()

    return pd.concat([db.title.set_index('title_id'), para, ques], axis=1)


def get_context(db: misc_data.DBResult, title_id: int, para_id: int) -> str:
    ser: pd.Series[str] = db.paragraph.loc[
        lambda df: (df['title_id'] == title_id) & (df['para_id'] == para_id), 'context'
    ]
    assert len(ser) == 1, f'n_entries = {len(ser)}'
    return ser.iat[0]


def filter_db(db: misc_data.DBResult, title_ids: Iterable[int]) -> misc_data.DBResult:
    def _filter(df: pd.DataFrame, col: str = 'title_id') -> pd.DataFrame:
        return df.loc[df[col].isin(title_ids)]

    title = _filter(db.title)
    para = _filter(db.paragraph)
    ques_ans = _filter(db.ques_ans)
    _ques_ids = ques_ans['ques_ans_id']
    answer = db.answer.loc[lambda df: df['ques_ans_id'].isin(_ques_ids)]

    return misc_data.DBResult(title, para, ques_ans, answer)
