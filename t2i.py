"""
Define a lightweight data structure to store and look up the indices belonging to arbitrary tokens.
Originally based on the [diagnnose](https://github.com/i-machine-think/diagnnose) W2I class.
"""

from collections import defaultdict
from typing import Dict, List, Union, Iterable, Optional

# Custom types
Corpus = Union[str, List[str]]
IndexedCorpus = [Iterable[int], Iterable[Iterable[int]]]


# TODO
# - Proper doc
# - Unit tests
# - __repr__
# - type checks / exceptions
# - More elegant and consistent handling of sequence vs. sequence of sequences with decorator?
# - Make compatible with numpy arrays / pytorch tensors / tensorflow tensors


class T2I(defaultdict):
    """
    Provides vocab functionality mapping words to indices.

    Non-existing tokens are mapped to the id of an unk token that should
    be present in the vocab file.

    @TODO
    """
    def __init__(self, t2i: Dict[str, int], unk_token: str="<unk>", eos_token: str="<eos>") -> None:
        if unk_token not in t2i:
            t2i[unk_token] = len(t2i)
        if eos_token not in t2i:
            t2i[eos_token] = len(t2i)

        super().__init__(lambda: self.unk_idx, t2i)

        self.unk_idx = t2i[unk_token]
        self.unk_token = unk_token
        self.eos_token = eos_token
        self.i2t = list(t2i.keys())

    @property
    def t2i(self) -> Dict[str, int]:
        return self

    @staticmethod
    def build(corpus: Corpus, delimiter: str=" ", unk_token: str="<unk>",
              eos_token: str="<eos>") -> defaultdict:
        """
        Build token index from scratch on a corpus.

        @TODO
        """
        t2i = defaultdict(lambda f: len(t2i))

        if type(corpus) == str:
            corpus = [corpus]  # Avoid code redundancy in case of single string

        for sentence in corpus:
            tokens = sentence.strip().split(delimiter)
            map(t2i.__getitem__, tokens)

        return T2I(dict(t2i), unk_token, eos_token)

    def index(self, corpus: Corpus, delimiter=" ") -> IndexedCorpus:
        """
        Assign indices to a sentence or a series of sentences.

        @TODO
        """
        self.__call__(corpus, delimiter)

    def unindex(self, indexed_corpus: IndexedCorpus, joiner: Optional[str]=None) -> Corpus:
        """
        Convert indices back to their original words.

        @TODO
        """
        if type(indexed_corpus[0]) == int:
            indexed_corpus = [indexed_corpus]  # Avoid code redundancy in case of single string

        corpus = []
        for sequence in indexed_corpus:
            tokens = list(map(self.i2t.__getitem__, sequence))

            if joiner is not None:
                tokens = joiner.join(tokens)

            corpus.append(tokens)

        return corpus

    def __call__(self, corpus: Union[str, List[str]], delimiter: str=" ") -> IndexedCorpus:
        """
        Assign indices to a sentence or a series of sentences.

        @TODO
        """
        if type(corpus) == str:
            corpus = [corpus]  # Avoid code redundancy in case of single string

        indexed_corpus = []

        for sentence in corpus:
            indexed_corpus.append(list(map(self.t2i.__getitem__, sentence.strip().split(" "))))

        return indexed_corpus

    def __repr__(self) -> str:
        """ Return a string representation of the T2I object. """
        ...  # TODO
