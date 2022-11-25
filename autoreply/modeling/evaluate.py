import multiprocessing as mp
from collections.abc import (
    Collection,
    Iterable,
    Sequence,
)

from typing import (
    NamedTuple,
    Tuple,
    List,
)
import more_itertools as mit
from gensim.models.doc2vec import Doc2Vec
from . import (
    preprocess,
)


class Key(NamedTuple):
    title: int
    para: int
    q_id: int


# result: dict[Key, list[Score]] = {Key(ques.title_id, ques.para_id, ques.ques_ans_id): infer(ques.question) for ques in db.ques_ans.head(1000).itertuples()}

# a = list(result.values())[0]
# a
# c = Counter(_a.tag[0] for _a in a)
# c.most_common(2)

Tag = Tuple[int, int]


class Score(NamedTuple):
    tag: Tag
    similarity: float


def infer(txt: str, doc_model: Doc2Vec, epochs: int) -> List[Score]:
    tokens = preprocess.main(txt)
    vec = doc_model.infer_vector(tokens, epochs=epochs)
    return [Score(*vec) for vec in doc_model.dv.most_similar(vec, topn=5)]


def results(
    X: Collection[str], doc_model: Doc2Vec, *, epochs: int = 5, n_workers: int = 1
):
    if n_workers == 1:
        return [infer(x, doc_model, epochs)[0].tag for x in X]
    else:
        return _multiprocess(n_workers, X, doc_model, epochs)


def _multiprocess(n: int, contexts: Collection[str], model: Doc2Vec, epochs: int):
    sections = mit.distribute(n, contexts)
    with mp.Pool(n) as pool:
        res = list(mit.flatten(pool.map(Eval(model, epochs), sections)))
    return res


class Eval:
    def __init__(self, model: Doc2Vec, epochs: int = 10) -> None:
        self._model = model
        self._epochs = 10

    def __call__(self, contexts: Iterable[str]) -> Sequence[int]:
        return results(contexts, self._model, epochs=self._epochs)


# def result_summary(result: dict[Key, Score]):
#     N = len(result)
#     title = 0
#     para = 0
#     for key, score in result.items():
#         c = Counter(_a.tag[0] for _a in score)
#         common = c.most_common(2)
