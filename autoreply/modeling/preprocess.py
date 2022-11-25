import functools as ft
from typing import (
    Sequence,
)

import gensim
import unidecode
from nltk import corpus


_STOPWORDS = gensim.parsing.preprocessing.STOPWORDS | set(
    corpus.stopwords.words('english')
)


def main(txt: str) -> Sequence[str]:
    rem_short = ft.partial(gensim.parsing.preprocessing.remove_short_tokens, minsize=2)
    rem_stop = ft.partial(
        gensim.parsing.preprocessing.remove_stopword_tokens, stopwords=_STOPWORDS
    )

    return rem_stop(
        rem_short(gensim.utils.simple_preprocess((unidecode.unidecode(txt))))
    )
