"""
Define a lightweight data structure to store and look up the indices belonging to arbitrary tokens.
Originally based on the [diagnnose](https://github.com/i-machine-think/diagnnose) W2I class.
"""

import abc
from collections import defaultdict
from functools import wraps
from typing import Dict, List, Union, Iterable, Optional, Callable, Any

# Custom types
# TODO: More fine-grained types after implementing decorators
Corpus = Union[str, Iterable[str]]
IndexedCorpus = [Iterable[int], Iterable[Iterable[int]]]


# TODO
# - Proper doc
# - __repr__
# - type checks / exceptions
# - Make compatible with numpy arrays / pytorch tensors / tensorflow tensors
# - Add option to not create new entries for unknown words to limit memory size
# - Build documentation
# - Write README
# - Polishing, fancy README tags
# - GitHub repo description
# - Release on PIP
# - Release to i-machine-think
# - General release


class IncrementingDefaultdict(dict):
    """
    A defaultdict where the value return value for an unknown key is the number of entries.
    Therefore, after every lookup of a new token, this value is incremented by one.
    """
    def __getitem__(self, item):
        if item not in self:
            self[item] = len(self)

        return super().__getitem__(item)


class T2IMeta(defaultdict, abc.ABC):
    """
    T2I superclass, mostly to provide an informative return type annotation for build() and extend() (you cannot
    annotate the return type of a static function with the class it was defined in).
    """
    @property
    @abc.abstractmethod
    def t2i(self):
        ...

    @staticmethod
    @abc.abstractmethod
    def build(corpus: Corpus, delimiter: str, unk_token: str, eos_token: str):
        ...

    @abc.abstractmethod
    def extend(self, corpus: Corpus, delimiter: str):
        ...

    @abc.abstractmethod
    def index(self, corpus: Corpus, delimiter: str):
        ...

    @abc.abstractmethod
    def unindex(self, indexed_corpus: IndexedCorpus, joiner: Optional[str]):
        ...

    @abc.abstractmethod
    def __call__(self, corpus: Corpus, delimiter: str):
        ...


def indexing_consistency(func: Callable) -> Callable:
    # TODO: Docstrings
    @wraps(func)
    def with_indexing_consistency(self: T2IMeta, corpus: Corpus, *args, **kwargs) -> IndexedCorpus:
        if type(corpus) == str:
            corpus = [corpus]
            indexed_corpus = func(self, corpus, *args, **kwargs)

            return indexed_corpus[0]

        else:
            return func(self, corpus, *args, **kwargs)

    return with_indexing_consistency


def unindexing_consistency(func: Callable) -> Callable:
    # TODO: Docstrings
    @wraps(func)
    def with_unindexing_consistency(self: T2IMeta, indexed_corpus: IndexedCorpus, *args, **kwargs) -> Corpus:
        if all([type(el) == int for el in indexed_corpus]):
            indexed_corpus = [indexed_corpus]
            unindexed_corpus = func(self, indexed_corpus, *args, **kwargs)

            return unindexed_corpus[0]

        else:
            return func(self, indexed_corpus, *args, **kwargs)

    return with_unindexing_consistency


class T2I(T2IMeta):
    """
    Provides vocab functionality mapping words to indices.

    Non-existing tokens are mapped to the id of an unk token that should
    be present in the vocab file.

    @TODO
    """
    def __init__(self, t2i: Dict[str, int], unk_token: str = "<unk>", eos_token: str = "<eos>") -> None:
        if unk_token not in t2i:
            t2i[unk_token] = len(t2i)
        if eos_token not in t2i:
            t2i[eos_token] = len(t2i)

        super().__init__(lambda: self.unk_idx, t2i)

        self.unk_idx = t2i[unk_token]
        self.unk_token = unk_token
        self.eos_token = eos_token
        self.i2t = dict([(v, k) for k, v in self.items()])
        self.i2t[self[self.unk_token]] = self.unk_token  # Make sure there is always an index associated with <unk>

    @property
    def t2i(self) -> Dict[str, int]:
        return self

    @staticmethod
    def build(corpus: Corpus, delimiter: str = " ", unk_token: str = "<unk>",
              eos_token: str = "<eos>") -> T2IMeta:
        """
        Build token index from scratch on a corpus.

        @TODO: Docstring
        """
        t2i = T2I._create_index(corpus, delimiter)

        return T2I(t2i, unk_token, eos_token)

    def extend(self, corpus: Corpus, delimiter: str = " ") -> T2IMeta:
        """
        Extend an existing T2I with tokens from a new tokens.

        @TODO: Docstring
        """
        raw_t2i = T2I._create_index(corpus, delimiter, seed_dict=dict(self))

        t2i = T2I(raw_t2i, self.unk_token, self.eos_token)

        return t2i

    @staticmethod
    def _create_index(corpus: Corpus, delimiter: str = " ", seed_dict: dict = {}) -> Dict[str, int]:
        """
        Create a simple dictionary, mapping every type in a Corpus to a unique index.

        @TODO: Docstring
        """
        t2i = IncrementingDefaultdict(seed_dict)

        if type(corpus) == str:
            corpus = [corpus]  # Avoid code redundancy in case of single string

        for sentence in corpus:
            tokens = sentence.strip().split(delimiter)
            [t2i[token] for token in tokens]

        return dict(t2i)

    @indexing_consistency
    def index(self, corpus: Corpus, delimiter: str = " ") -> IndexedCorpus:
        """
        Assign indices to a sentence or a series of sentences.

        @TODO: Docstring
        """
        return self.__call__(corpus, delimiter=delimiter)

    @unindexing_consistency
    def unindex(self, indexed_corpus: IndexedCorpus, joiner: Optional[str] = " ") -> Corpus:
        """
        Convert indices back to their original words.

        @TODO: Docstring
        """
        corpus = []
        for sequence in indexed_corpus:
            tokens = list(map(self.i2t.__getitem__, sequence))

            if joiner is not None:
                tokens = joiner.join(tokens)

            corpus.append(tokens)

        return corpus

    @indexing_consistency
    def __call__(self, corpus: Corpus, delimiter: str = " ") -> IndexedCorpus:
        """
        Assign indices to a sentence or a series of sentences.

        @TODO: Docstring
        """
        indexed_corpus = []

        for sentence in corpus:
            indexed_corpus.append(list(map(self.t2i.__getitem__, sentence.strip().split(delimiter))))

        return indexed_corpus

    def __repr__(self) -> str:
        """ Return a string representation of the T2I object. """
        ...  # TODO
