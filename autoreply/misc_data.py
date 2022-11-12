"""Processing for train_data file."""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TypeVar,
    NewType,
    Sequence,
    Iterable,
)
from typing_extensions import TypedDict

import pandas as pd

from autoreply import dataset

from . import (
    constants as C,
)


class MiscSchema(TypedDict):
    data: Sequence[Datum]


class Datum(TypedDict):
    title: str
    paragraphs: Sequence[Paragraph]


Data = Sequence[Datum]


class Paragraph(TypedDict):
    qas: Sequence[QuestionAnswer]
    context: str


class QuestionAnswer(TypedDict):
    question: str
    id: str
    answers: Sequence[Answer]
    is_impossible: bool


class Answer(TypedDict):
    text: str
    answer_start: int


def load(root: Path = C.DATA) -> MiscSchema:
    file = root.joinpath('raw', 'train_data.txt')

    with open(file) as fp:
        json_ = json.load(fp)
    return MiscSchema(**json_)


Summary = NewType('Summary', pd.DataFrame)


def summary(data: Data) -> Summary:
    df = pd.DataFrame(
        {
            artist['title']: {
                'questions': sum(len(para['qas']) for para in artist['paragraphs']),
                'paragraphs': len(artist['paragraphs']),
                'len_para': sum(len(para['context']) for para in artist['paragraphs']),
            }
            for artist in data
        }
    ).T.sort_values(by=['questions', 'paragraphs'], ascending=False)
    return Summary(df)


_T = TypeVar('_T', bound='DBResult')


@dataclass(repr=False, frozen=True, kw_only=True, slots=True)
class DBResult:
    title: pd.DataFrame
    paragraph: pd.DataFrame
    ques_ans: pd.DataFrame
    answer: pd.DataFrame

    @classmethod
    def keys(cls) -> Iterable[str]:
        return [field.name for field in dataclasses.fields(cls)]

    @classmethod
    def load(cls: type[_T], data: Path, type_: dataset.Type = dataset.Type.TRAIN) -> _T:
        path = data.joinpath(type_.value)
        return cls(
            **{file: pd.read_csv(path.joinpath(f'{file}.csv')) for file in cls.keys()}
        )

    def to_csv(self, path: Path) -> None:
        frames = dataclasses.asdict(self)
        for name, frame in frames.items():
            frame.to_csv(path.joinpath(f'{name}.csv'))


def arrays(data: Data) -> DBResult:
    title = []
    paragraph = []
    question_answer = []
    answer = []

    for title_id, datum in enumerate(data):
        title_ = datum['title']
        title.append((title_id, title_))

        for para_id, para in enumerate(datum['paragraphs']):
            paragraph.append((title_id, para_id, para['context']))

            for ques_ans_ in para['qas']:
                question_answer.append(
                    (
                        title_id,
                        para_id,
                        ques_ans_['id'],
                        ques_ans_['question'],
                        ques_ans_['is_impossible'],
                    )
                )

                for ans_id, ans_ in enumerate(ques_ans_['answers']):
                    answer.append(
                        (ques_ans_['id'], ans_id, ans_['text'], ans_['answer_start'])
                    )

    return DBResult(
        title=pd.DataFrame(title, columns=['title_id', 'title']),
        paragraph=pd.DataFrame(paragraph, columns=['title_id', 'para_id', 'context']),
        ques_ans=pd.DataFrame(
            question_answer,
            columns=['title_id', 'para_id', 'ques_ans_id', 'question', 'is_impossible'],
        ),
        answer=pd.DataFrame(
            answer, columns=['ques_ans_id', 'ans_id', 'answer', 'answer_start']
        ),
    )
