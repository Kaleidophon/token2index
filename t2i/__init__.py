"""
Define a lightweight data structure to store and look up the indices belonging to arbitrary tokens.
Originally based on the [diagnnose](https://github.com/i-machine-think/diagnnose) W2I class.
"""

from __future__ import annotations
import codecs
import sys
import pickle
from typing import Dict, Union, Iterable, Optional, Any, Hashable, Tuple

# LIB
from t2i.decorators import indexing_consistency, unindexing_consistency

# Custom types
Corpus = Union[str, Iterable[str]]
IndexedCorpus = [Iterable[int], Iterable[Iterable[int]]]

# Restrict direct imports from t2i.decorators module
sys.modules["t2i.decorators"] = None
__all__ = ["T2I", "Index", "Corpus", "IndexedCorpus"]


# TODO
# - Update i2t after extend
# - Determine compatibility with Python version
# - Don't inherit from dict
# - Index tests
# - type checks / exceptions
# - Build documentation
# - Write README
# - GitHub repo description
# - Release on PIP
# - Release to i-machine-think
# - General release


class Index(dict):
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
            self[key] = self.highest_idx + 1

        return super().__getitem__(key)

    @property
    def highest_idx(self) -> int:
        """
        Return the currently highest index in the index. Return -1 if the index is empty.
        """
        return max(max(self.values()), len(self) - 1) if len(self) > 0 else -1


class T2I(dict):
    """
    Provides vocab functionality mapping tokens to indices. After building an index, sentences or a corpus of sentences
    can be mapped to the tokens' assigned indices. There are special tokens for the end of a sentence (eos_token) and
    for tokens that were not added to the index during the build phase (unk_token).
    """

    def __init__(
        self,
        index: Union[Dict[str, int], Index],
        unk_token: str = "<unk>",
        eos_token: str = "<eos>",
        *special_tokens: Tuple[str],
    ) -> None:
        """
        Initialize the T2I class.

        Parameters
        ----------
        index:Index
            Dictionary mapping from tokens to indices.
        unk_token: str
            Token for unknown words not contained in t2i. Default is '<unk>'.
        eos_token: str
            End-of-sequence token. Default is '<eos>'.
        special_tokens: Tuple[str]
            An arbitrary number of additional special tokens, given as unnamed arguments.
        """
        assert len(set(index.values())) == len(
            index.values()
        ), "Index must only contain unique keys."

        for special_token in [unk_token, eos_token] + list(special_tokens):
            index[special_token] = (
                max(max(index.values()) + 1, len(index)) if len(index) > 0 else 0
            )

        super().__init__(index)
        self.unk_idx = index[unk_token]
        self.unk_token = unk_token
        self.eos_token = eos_token
        self.i2t = dict([(v, k) for k, v in self.items()])
        self.i2t[
            self[self.unk_token]
        ] = self.unk_token  # Make sure there is always an index associated with <unk>

        # torchtext vocab compatability
        self.itos = self.i2t
        self.stoi = self.t2i

        self.pickled = None  # See __setitem__

    @property
    def t2i(self) -> Index:
        """
        Return the dictionary mapping tokens to unique indices.

        Returns
        -------
        t2i: Index
            Dictionary mapping from tokens to indices.
        """
        return Index(self)

    @staticmethod
    def build(
        corpus: Corpus,
        delimiter: str = " ",
        unk_token: str = "<unk>",
        eos_token: str = "<eos>",
    ) -> T2I:
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

    @staticmethod
    def from_file(
        vocab_path: str,
        encoding: str = "utf-8",
        delimiter: str = "\t",
        unk_token: str = "<unk>",
        eos_token: str = "<eos>",
    ) -> T2I:
        """
        Generate a T2I object from a file. This file can have two possible formats:

        1. One token per line (in which case the index is the line number)
        2. A token and its corresponding index, separated by some delimiter (default is "\t"):

        Parameters
        ----------
        vocab_path: str
            Path to vocabulary file.
        encoding: str
            Encoding of vocabulary file (default is 'utf-8').
        delimiter: str
            Delimiter in case the format is token <delimiter> index. Default is '\t'.
        unk_token: str
            Token that should be used for unknown words. Default is '<unk>'.
        eos_token: str
            Token that marks the end of a sequence. Default is '<eos>'.


        Returns
        -------
        t2i: T2I
            T2I object built from vocal file.
        """
        # TODO: Check file format?

        with codecs.open(vocab_path, "r", encoding) as vocab_file:
            entries = [line.strip() for line in vocab_file.readlines()]

            # Infer file format
            # Format token <delimiter> index
            if len(entries[0].split(delimiter)) == 2:
                index = {}

                for entry in entries:
                    token, idx = entry.split(delimiter)
                    index[token] = int(idx)

            # Format: One token per line
            else:
                index = dict(zip(entries, range(len(entries))))

        return T2I(index, unk_token, eos_token)

    def extend(self, corpus: Corpus, delimiter: str = " ") -> T2I:
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
        raw_t2i = T2I._create_index(corpus, delimiter, index=Index(self))

        t2i = T2I(raw_t2i, self.unk_token, self.eos_token)

        return t2i

    @staticmethod
    def _create_index(
        corpus: Corpus, delimiter: str = " ", index: Optional[Index] = None
    ) -> Index:
        """
        Create a simple dictionary, mapping every type in a Corpus to a unique index.

        Parameters
        ----------
        corpus: Corpus
            Corpus that is being used to build or extend the index.
        delimiter: str
            Delimiter between tokens. Default is a whitespace ' '.
        index: dict
            Index coming from another source that is being extended.

        Returns
        -------
        t2i: Index
            Newly build or extended index, mapping words to indices.
        """
        if index is None:
            index = Index()

        if type(corpus) == str:
            corpus = [corpus]  # Avoid code redundancy in case of single string

        for sentence in corpus:
            tokens = sentence.strip().split(delimiter)
            [index[token] for token in tokens]

        return index

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
    def unindex(
        self, indexed_corpus: IndexedCorpus, joiner: Optional[str] = " "
    ) -> Corpus:
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
            indexed_corpus.append(
                list(map(self.__getitem__, sentence.strip().split(delimiter)))
            )

        return indexed_corpus

    def __missing__(self, key: str) -> int:
        """ Return the unk token index in case of a missing entry. """
        return self.unk_idx

    def __setitem__(self, key: str, value: int) -> None:
        """ Don't allow to set new indices after class was initialized. """
        # TODO: Find a better way to do this
        # TODO: Fix this by not inheriting from dict and instead using this as a wrapper class and not implementing
        # TODO: __setitem__()
        # This line is one of the most adventurous line I have ever written and I hate it:
        # I want to have __setitem__ raise this exception here so that the index cannot be manipulated directly, e.g.
        # by doing 't2i["hello"] = 46'. However, during loading a serialized version of this object, pickle is using
        # ___setitem__ to rebuild the T2I object. Thus, this function raises an error during unpickling and the
        # object cannot be un-serialized. For some reason, the attribute with the name "pickled" is current being loaded
        # last, so checking for its existence is an indicator for the progress of the unpickling. Naturally this is not
        # good code, as I don't know WHY it is loaded last and this behavior might break with new, future attributes.
        if hasattr(self, "pickled"):
            raise NotImplementedError(
                "Setting of new indices not possible after initialization. Use extend() instead."
            )

        else:
            super().__setitem__(key, value)

    def save(self, path: str) -> None:
        """ Save T2I object as pickle. """
        with open(path, "wb") as f:
            self.pickled = True
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> T2I:
        """ Load serialized T2I object. """
        with open(path, "rb") as f:
            t2i = pickle.load(f)

        return t2i

    def __setstate__(self, state):
        self.pickled = False

    def __repr__(self) -> str:
        """ Return a string representation of a T2I object. """
        return (
            f"T2I(Size: {len(self.t2i)}, unk_token: {self.unk_token}, eos_token: {self.eos_token}, "
            f"{dict.__repr__(self)})"
        )
