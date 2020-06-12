"""
Define a lightweight data structure to store and look up the indices belonging to arbitrary tokens.
Originally based on the [diagnnose](https://github.com/i-machine-think/diagnnose) W2I class.
"""

import abc
from collections import defaultdict
from functools import wraps
from typing import Dict, Union, Iterable, Optional, Callable, Any, Hashable

# Custom types
Index = Dict[str, int]
Corpus = Union[str, Iterable[str]]
IndexedCorpus = [Iterable[int], Iterable[Iterable[int]]]


# TODO
# - type checks / exceptions
# - Build from vocab file
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
    (Technically) A defaultdict where the value return value for an unknown key is the number of entries. However, it
    doesn't inherit from defaultdict, because functions returning the value for missing keys can only return a constant
    value. In this case, after every lookup of a new token, this value for an unknown is incremented by one.
    """
    def __getitem__(self, key: Hashable) -> Any:
        """
        Return value corresponding to key. If key doesn't exist yet, return CURRENT size of defaultdict.

        Parameters
        ----------
        key: Hashable
            Key to be looked up.

        Returns
        -------
        value: Any
            Value associated key or current length of dict in case of a new key.
        """
        if key not in self:
            self[key] = len(self)  # TODO: Replace len with largest index in case of arbitrary indices from seed_index

        return super().__getitem__(key)


class T2IMeta(defaultdict, abc.ABC):
    """
    T2I superclass, mostly to provide an informative return type annotation for build() and extend() (you cannot
    annotate the return type of a static function with the class it was defined in).
    """
    @property
    @abc.abstractmethod
    def t2i(self):
        """
        Return the dictionary mapping tokens to unique indices.

        Returns
        -------
        t2i:Index
            Dictionary mapping from tokens to indices.
        """
        ...

    @staticmethod
    @abc.abstractmethod
    def build(corpus: Corpus, delimiter: str, unk_token: str, eos_token: str):
        """
        Build token index from scratch on a corpus.

        Parameters
        ----------
        corpus: Corpus
            Corpus that is being used to build the index.
        delimiter: str
            Delimiter between tokens. Default is a whitespace ' '.
        unk_token: str
            Token that should be used for unknown words. Default is '<unk>'.
        eos_token: str
            Token that marks the end of a sequence. Default is '<eos>'.

        Returns
        -------
        t2i: T2I
            New T2I object.
        """
        ...

    @abc.abstractmethod
    def extend(self, corpus: Corpus, delimiter: str):
        """
        Extend an existing T2I with tokens from a new tokens and build indices for them.

        Parameters
        ----------
        corpus: Corpus
            Corpus that is being used to extend the index.
        delimiter: str
            Delimiter between tokens. Default is a whitespace ' '.

        Returns
        -------
        t2i: T2I
            New T2I object.
        """
        ...

    @abc.abstractmethod
    def index(self, corpus: Corpus, delimiter: str):
        """
        Assign indices to a sentence or a series of sentences.

        Parameters
        ----------
        corpus: Corpus
            Corpus that is being indexed.
        delimiter: str
            Delimiter between tokens. Default is a whitespace ' '.

        Returns
        -------
        indexed_corpus: IndexedCorpus
            Indexed corpus.
        """
        ...

    @abc.abstractmethod
    def unindex(self, indexed_corpus: IndexedCorpus, joiner: Optional[str]):
        """
        Convert indices back to their original tokens. A joiner can be specified to determine how tokens are pieced
        back together. If the joiner is None, the tokens are not joined and are simply returned as a list.

        Parameters
        ----------
        indexed_corpus: IndexedCorpus
            An indexed corpus.
        joiner: Optional[str]
            String used to join tokens. Default is a whitespace ' '. If the value is None, tokens are not joined and a
            list of tokens is returned.

        Returns
        -------
        corpus: Corpus
            Un-indexed corpus.
        """
        ...

    @abc.abstractmethod
    def __call__(self, corpus: Corpus, delimiter: str):
        """
        Assign indices to a sentence or a series of sentences.

        Parameters
        ----------
        corpus: Corpus
            Corpus that is being indexed.
        delimiter: str
            Delimiter between tokens. Default is a whitespace ' '.

        Returns
        -------
        indexed_corpus: IndexedCorpus
            Indexed corpus.
        """
        ...

    def __repr__(self) -> str:
        """ Return a string representation of the core dict inside a T2I object. """
        # This is a way to call the __repr__ function of the grandparent class dict
        # dict -> defaultdict -> T2IMeta -> T2I
        return dict.__repr__(self)


def indexing_consistency(func: Callable) -> Callable:
    """
    Make T2I.index() and T2I.__call__() agnostic to whether the input is a string or a list of strings, i.e.

    str -> List[int]
    List[str] -> List[List[str]]

    This is achieved by simply putting a single sentence into a list. This way, the above methods always process lists
    of lists.

    Parameters
    ----------
    func: Callable
        Indexing function to be decorated.

    Returns
    -------
    func: Callable
        Decorated indexing function.
    """
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
    """
    Make T2I.unindex() agnostic to whether the input is a list of ints or a list of lists of ints, i.e.

    List[int] -> str or List[str]
    List[List[int]] -> List[str]

    This is achieved by simply putting a single indexed sentence into a list. This way, the above methods always process
    lists of lists.

    Parameters
    ----------
    func: Callable
        Indexing function to be decorated.

    Returns
    -------
    func: Callable
        Decorated indexing function.
    """
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
    Provides vocab functionality mapping tokens to indices. After building an index, sentences or a corpus of sentences
    can be mapped to the tokens' assigned indices. There are special tokens for the end of a sentence (eos_token) and
    for tokens that were not added to the index during the build phase (unk_token).
    """
    def __init__(self, t2i: Index, unk_token: str = "<unk>", eos_token: str = "<eos>") -> None:
        """
        Initialize the T2I class.

        Parameters
        ----------
        t2i:Index
            Dictionary mapping from tokens to indices.
        unk_token: str
            Token for unknown words not contained in t2i. Default is '<unk>'.
        eos_token: str
            End-of-sequence token. Default is '<eos>'.
        """
        # TODO: Assert t2i indices are unique

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
    def t2i(self) ->Index:
        """
        Return the dictionary mapping tokens to unique indices.

        Returns
        -------
        t2i:Index
            Dictionary mapping from tokens to indices.
        """
        return self

    @staticmethod
    def build(corpus: Corpus, delimiter: str = " ", unk_token: str = "<unk>", eos_token: str = "<eos>") -> T2IMeta:
        """
        Build token index from scratch on a corpus.

        Parameters
        ----------
        corpus: Corpus
            Corpus that is being used to build the index.
        delimiter: str
            Delimiter between tokens. Default is a whitespace ' '.
        unk_token: str
            Token that should be used for unknown words. Default is '<unk>'.
        eos_token: str
            Token that marks the end of a sequence. Default is '<eos>'.

        Returns
        -------
        t2i: T2I
            New T2I object.
        """
        t2i = T2I._create_index(corpus, delimiter)

        return T2I(t2i, unk_token, eos_token)

    def extend(self, corpus: Corpus, delimiter: str = " ") -> T2IMeta:
        """
        Extend an existing T2I with tokens from a new tokens and build indices for them.

        Parameters
        ----------
        corpus: Corpus
            Corpus that is being used to extend the index.
        delimiter: str
            Delimiter between tokens. Default is a whitespace ' '.

        Returns
        -------
        t2i: T2I
            New T2I object.
        """
        raw_t2i = T2I._create_index(corpus, delimiter, seed_index=dict(self))

        t2i = T2I(raw_t2i, self.unk_token, self.eos_token)

        return t2i

    @staticmethod
    def _create_index(corpus: Corpus, delimiter: str = " ", seed_index: Optional[Index] = None) ->Index:
        """
        Create a simple dictionary, mapping every type in a Corpus to a unique index.

        Parameters
        ----------
        corpus: Corpus
            Corpus that is being used to build or extend the index.
        delimiter: str
            Delimiter between tokens. Default is a whitespace ' '.
        seed_index: dict
            Index coming from another source that is being extended.

        Returns
        -------
        t2i: Index
            Newly build or extended index, mapping words to indices.
        """
        if seed_index is None:
            seed_index = {}

        t2i = IncrementingDefaultdict(seed_index)

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

        Parameters
        ----------
        corpus: Corpus
            Corpus that is being indexed.
        delimiter: str
            Delimiter between tokens. Default is a whitespace ' '.

        Returns
        -------
        indexed_corpus: IndexedCorpus
            Indexed corpus.
        """
        indexed_corpus = self.__call__(corpus, delimiter=delimiter)

        return indexed_corpus

    @unindexing_consistency
    def unindex(self, indexed_corpus: IndexedCorpus, joiner: Optional[str] = " ") -> Corpus:
        """
        Convert indices back to their original tokens. A joiner can be specified to determine how tokens are pieced
        back together. If the joiner is None, the tokens are not joined and are simply returned as a list.

        Parameters
        ----------
        indexed_corpus: IndexedCorpus
            An indexed corpus.
        joiner: Optional[str]
            String used to join tokens. Default is a whitespace ' '. If the value is None, tokens are not joined and a
            list of tokens is returned.

        Returns
        -------
        corpus: Corpus
            Un-indexed corpus.
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

        Parameters
        ----------
        corpus: Corpus
            Corpus that is being indexed.
        delimiter: str
            Delimiter between tokens. Default is a whitespace ' '.

        Returns
        -------
        indexed_corpus: IndexedCorpus
            Indexed corpus.
        """
        indexed_corpus = []

        for sentence in corpus:
            indexed_corpus.append(list(map(self.t2i.__getitem__, sentence.strip().split(delimiter))))

        return indexed_corpus

    def __repr__(self) -> str:
        """ Return a string representation of a T2I object. """
        return f"T2I(Size: {len(self.t2i)}, unk_token: {self.unk_token}, eos_token: {self.eos_token}, "\
                f"{super().__repr__()})"
