from typing import NamedTuple, Tuple
import numpy as np
from sklearn import metrics, feature_extraction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from autoreply import (
    misc_data,
)


class Result(NamedTuple):
    accuracy: float
    n_paras: int
    n_ques: int


def predict(db: misc_data.DBResult) -> Result:

    para = db.paragraph
    ques = db.ques_ans['question']

    vec = feature_extraction.text.TfidfVectorizer(
        strip_accents='ascii', ngram_range=(1, 2), max_df=0.7
    )

    X = vec.fit_transform(para.context)
    y = vec.transform(ques)

    res = [np.argmin(metrics.pairwise_distances(X, _y)) for _y in y]

    acc = metrics.accuracy_score(db.ques_ans.para_id, res)
    # acc = metrics.accuracy_score(y_true, y_pred)

    return Result(acc, len(para), len(ques))


def alt_predict(db: misc_data.DBResult) -> Tuple[Result, TfidfVectorizer]:

    para = db.paragraph
    ques = db.ques_ans['question']

    vec = feature_extraction.text.TfidfVectorizer(
        strip_accents='ascii', ngram_range=(1, 2), max_df=0.7
    )

    X = vec.fit_transform(para.context)
    y = vec.transform(ques)

    res = [np.argmin(metrics.pairwise_distances(X, _y)) for _y in y]

    y_true = [
        int(f'{titel}{para}')
        for titel, para in db.ques_ans.filter(['title_id', 'para_id']).to_numpy()
    ]
    y_pred = [
        int(f'{titel}{para}')
        for titel, para in db.paragraph.filter(['title_id', 'para_id'])
        .iloc[res]
        .to_numpy()
    ]
    # print(y_true, y_pred)
    acc = metrics.accuracy_score(y_true, y_pred)

    return Result(acc, len(para), len(ques)), vec


def svd(db: misc_data.DBResult):

    para = db.paragraph
    ques = db.ques_ans['question']

    vec = feature_extraction.text.TfidfVectorizer(
        strip_accents='ascii', ngram_range=(1, 2), max_df=0.7
    )
    svd = TruncatedSVD(100)

    X = vec.fit_transform(para.context)

    y = vec.transform(ques)

    X_svd = svd.fit_transform(X)
    y_svd = svd.transform(y)

    return vec, svd, X_svd, y_svd
