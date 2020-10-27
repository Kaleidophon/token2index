"""
Define a lightweight data structure to store and look up the indices belonging to arbitrary tokens.
Originally based on the `diagnnose <https://github.com/i-machine-think/diagnnose>`_ W2I class.
"""

import codecs
from collections import Counter, Iterable as IterableClass  # Distinguish from typing.Iterable
import sys
import pickle
from typing import Dict, Union, Iterable, Optional, Any, Hashable, Tuple, Iterator
import warnings

# LIB
from t2i.decorators import indexing_consistency, unindexing_consistency

# Constants
# Define the standard unk and eos token here
STD_UNK = "<unk>"
STD_EOS = "<eos>"
STD_PAD = "<pad>"

# Custom types
Corpus = Union[str, Iterable[str], Iterable[Iterable[str]]]
IndexedCorpus = [Iterable[int], Iterable[Iterable[int]]]

# Restrict direct imports from t2i.decorators module
sys.modules["t2i.decorators"] = None
__all__ = ["T2I", "Index", "Corpus", "IndexedCorpus", "STD_EOS", "STD_UNK", "STD_PAD"]
__version__ = "1.0.2"
__author__ = "Dennis Ulmer"


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

    def items(self) -> Iterable[Tuple[str, int]]:
        """
        The same as a usual dict items(), except that the entries are sorted by index (this has otherwise proven to
        create problems in Python < 3.6).
        """
        return ((token, idx) for token, idx in sorted(super().items(), key=lambda tpl: tpl[1]))

    @property
    def highest_idx(self) -> int:
        """
        Return the currently highest index in the index. Return -1 if the index is empty.
        """
        return max(max(self.values()), len(self) - 1) if len(self) > 0 else -1


class T2I:
    """
    Provides vocab functionality mapping tokens to indices. After building an index, sentences or a corpus of sentences
    can be mapped to the tokens' assigned indices. There are special tokens for the end of a sentence (eos_token) and
    for tokens that were not added to the index during the build phase (unk_token).
    """

    def __init__(
        self,
        index: Optional[Union[Dict[str, int], Index]] = None,
        counter: Optional[Counter] = None,
        max_size: Optional[int] = None,
        min_freq: int = 1,
        unk_token: str = STD_UNK,
        eos_token: str = STD_EOS,
        pad_token: str = STD_PAD,
        special_tokens: Iterable[str] = tuple(),
    ) -> None:
        """
        Initialize the T2I class.

        Parameters
        ----------
        index: Optional[Union[Dict[str, int], Index]]
            Dictionary mapping from tokens to indices.
        counter: Optional[Counter]
            Counter with token frequencies in corpus. Default is None.
        max_size: Optional[int]
            Maximum size of T2I index. Default is None, which means no maximum size.
        min_freq: int
            Minimum frequency of a token for it to be included in the index. Default is 1.
        unk_token: str
            Token for unknown words not contained in t2i. Default is '<unk>'.
        eos_token: str
            End-of-sequence token. Default is '<eos>'.
        pad_token: str
            Padding token. Default is '<pad>'.
        special_tokens: Iterable[str]
            An arbitrary number of additional special tokens.
        """
        assert max_size is None or max_size > 2, "max_size has to be larger than 2, {} given.".format(max_size)
        assert min_freq > 0, "min_freq has to be at least 1, {} given.".format(min_freq)

        if counter is not None and min_freq == 1:
            warnings.warn("Token frequencies were given but min_freq is still set to 1?")

        if index is None:
            index = {}

        if type(index) == dict:
            index = Index(index)

        assert len(set(index.values())) == len(index.values()), "Index must only contain unique keys."

        all_special_tokens = [unk_token, eos_token, pad_token] + list(special_tokens)

        assert len(all_special_tokens) == len(set(all_special_tokens)), (
            "Unknown, end-of-sequence and padding token must not be specified via 'special_tokens', use corresponding "
            "key-word arguments ('unk_token', 'eos_token', 'pad_token') instead."
        )

        # Make sure that special tokens always come first by deleting them first if they already occur in index
        for special_token in all_special_tokens:
            if special_token in index:
                del index[special_token]

        # Build index
        self._index = Index()
        for token, idx in index.items():
            if max_size is not None:
                if len(self._index) >= max_size - len(all_special_tokens):
                    break

            if counter is None or counter[token] >= min_freq:
                self._index[token] = idx

        for special_token in all_special_tokens:
            self._index[special_token] = self._index.highest_idx + 1

        self.counter = counter
        self.max_size = max_size
        self.min_freq = min_freq
        self.special_tokens = special_tokens
        self.unk_idx = index[unk_token]
        self.unk_token = unk_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self._build_i2t()

        # torchtext vocab compatibility
        self.itos = self.i2t
        self.stoi = self.t2i

    def _build_i2t(self) -> None:
        """
        (Re-)Build the index-to-token mapping.
        """
        self.i2t = dict([(v, k) for k, v in self._index.items()])
        self.i2t[self[self.unk_token]] = self.unk_token  # Make sure there is always an index associated with eos token

    @property
    def t2i(self) -> Index:
        """
        Return the dictionary mapping tokens to unique indices.

        Returns
        -------
        t2i: Index
            Dictionary mapping from tokens to indices.
        """
        return self._index

    @staticmethod
    def build(
        corpus: Corpus,
        delimiter: str = " ",
        counter: Optional[Counter] = None,
        max_size: Optional[int] = None,
        min_freq: int = 1,
        unk_token: str = STD_UNK,
        eos_token: str = STD_EOS,
        pad_token: str = STD_PAD,
        special_tokens: Iterable[str] = tuple(),
    ):
        """
        Build token index from scratch on a corpus.

        Parameters
        ----------
        corpus: Corpus
            Corpus that is being used to build the index.
        delimiter: str
            Delimiter between tokens. Default is a whitespace ' '.
        counter: Optional[Counter]
            Counter with token frequencies in corpus. Default is None.
        max_size: Optional[int]
            Maximum size of T2I index. Default is None, which means no maximum size.
        min_freq: int
            Minimum frequency of a token for it to be included in the index. Default is 1.
        unk_token: str
            Token that should be used for unknown words. Default is 'STD_UNK'.
        eos_token: str
            Token that marks the end of a sequence. Default is '<eos>'.
        pad_token: str
            Padding token. Default is '<pad>'.
        special_tokens: Iterable[str]
            An arbitrary number of additional special tokens, given as unnamed arguments.

        Returns
        -------
        t2i: T2I
            New T2I object.
        """
        assert max_size is None or max_size > 2, "max_size has to be larger than 2, {} given.".format(max_size)
        assert min_freq > 0, "min_freq has to be at least 1, {} given.".format(min_freq)

        T2I._check_corpus(corpus)
        t2i = T2I._create_index(corpus, delimiter)

        return T2I(t2i, counter, max_size, min_freq, unk_token, eos_token, pad_token, special_tokens)

    @staticmethod
    def from_file(
        vocab_path: str,
        encoding: str = "utf-8",
        delimiter: str = "\t",
        counter: Optional[Counter] = None,
        max_size: Optional[int] = None,
        min_freq: int = 1,
        unk_token: str = STD_UNK,
        eos_token: str = STD_EOS,
        pad_token: str = STD_PAD,
        special_tokens: Iterable[str] = tuple(),
    ):
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
        counter: Optional[Counter]
            Counter with token frequencies in corpus. Default is None.
        max_size: Optional[int]
            Maximum size of T2I index. Default is None, which means no maximum size.
        min_freq: int
            Minimum frequency of a token for it to be included in the index. Default is 1.
        unk_token: str
            Token that should be used for unknown words. Default is 'STD_UNK'.
        eos_token: str
            Token that marks the end of a sequence. Default is '<eos>'.
        pad_token: str
            Padding token. Default is '<pad>'.
        special_tokens: Iterable[str]
            An arbitrary number of additional special tokens.

        Returns
        -------
        t2i: T2I
            T2I object built from vocab file.
        """
        assert max_size is None or max_size > 2, "max_size has to be larger than 2, {} given.".format(max_size)
        assert min_freq > 0, "min_freq has to be at least 1, {} given.".format(min_freq)

        def _get_file_format(line: str) -> int:
            """ Infer the vocab file format based on a a line. """
            return len(line.split(delimiter))

        with codecs.open(vocab_path, "r", encoding) as vocab_file:
            entries = [line.strip() for line in vocab_file.readlines()]

            # Infer file format
            file_format = _get_file_format(entries[0])

            # Format token <delimiter> index
            index = {}

            if file_format == 2:

                for entry in entries:
                    if _get_file_format(entry) != file_format:
                        raise ValueError("Line in vocab file had unexpected format.")

                    token, idx = entry.split(delimiter)

                    # Ignore special tokens, they will be added later
                    if token in [unk_token, eos_token] + list(special_tokens):
                        continue

                    index[token] = int(idx)

            # Format: One token per line
            elif file_format == 1:
                for idx, token in enumerate(entries):
                    if _get_file_format(token) != file_format:
                        raise ValueError("Line in vocab file had unexpected format.")

                    # Ignore special tokens, they will be added later
                    if token in [unk_token, eos_token] + list(special_tokens):
                        continue

                    index[token] = idx

            else:
                raise ValueError("Vocab file has an unrecognized format.")

        return T2I(index, counter, max_size, min_freq, unk_token, eos_token, pad_token, special_tokens)

    def extend(self, corpus: Corpus, delimiter: str = " "):
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
        self._check_corpus(corpus)
        raw_t2i = T2I._create_index(corpus, delimiter, index=Index(self._index))

        t2i = T2I(
            raw_t2i,
            unk_token=self.unk_token,
            eos_token=self.eos_token,
            pad_token=self.pad_token,
            special_tokens=self.special_tokens,
        )

        return t2i

    @staticmethod
    def _create_index(corpus: Corpus, delimiter: str = " ", index: Optional[Index] = None) -> Index:
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

        # If type(corpus) is str
        if type(corpus) == str:
            corpus = [corpus]  # Avoid code redundancy in case of single string

        for sentence in corpus:
            # If type(corpus) is Iterable[str]
            if type(sentence) == str:
                tokens = sentence.strip().split(delimiter)

            # If type(corpus) is Iterable[Iterable[str]]
            else:
                tokens = sentence

            # Perform lookup
            [index[token] for token in tokens]

        return index

    @indexing_consistency
    def index(self, corpus: Corpus, delimiter: str = " ", pad_to: Optional[Union[str, int]] = None) -> IndexedCorpus:
        """
        Assign indices to a sentence or a series of sentences.

        Parameters
        ----------
        corpus: Corpus
            Corpus that is being indexed.
        delimiter: str
            Delimiter between tokens. Default is a whitespace ' '.
        pad_to: Optional[Union[str, int]]
            Indicate whether shorter sequences in this corpus should be padded up to the length of the longest sequence
            ('max') or to a fixed length (any positive integer) or not not at all (None). Default is None.

        Returns
        -------
        indexed_corpus: IndexedCorpus
            Indexed corpus.
        """
        indexed_corpus = self(corpus, delimiter=delimiter, pad_to=pad_to)

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
    def __call__(self, corpus: Corpus, delimiter: str = " ", pad_to: Optional[Union[str, int]] = None) -> IndexedCorpus:
        """
        Assign indices to a sentence or a series of sentences.

        Parameters
        ----------
        corpus: Corpus
            Corpus that is being indexed.
        delimiter: str
            Delimiter between tokens. Default is a whitespace ' '.
        pad_to: Optional[Union[str, int]]
            Indicate whether shorter sequences in this corpus should be padded up to the length of the longest sequence
            ('max') or to a fixed length (any positive integer) or not not at all (None). Default is None.

        Returns
        -------
        indexed_corpus: IndexedCorpus
            Indexed corpus.
        """
        self._check_corpus(corpus)
        indexed_corpus = []

        # Resolve pad_up_to
        if type(pad_to) == str:
            assert pad_to == "max", "'pad_up_to' can only used with strings when 'max', '{}' found.".format(pad_to)
            max_seq_len = max(len(seq.strip().split(delimiter)) for seq in corpus)

        elif type(pad_to) == int:
            assert pad_to > 0, "Length sequences are being padded to has to be greater than 0, {} specified".format(
                pad_to
            )

            max_seq_len = pad_to

        elif pad_to is None:
            max_seq_len = -1

        else:
            raise TypeError("'pad_up_to' has to be 'max', a positive int or None, {} found.".format(pad_to))

        # Index
        for sentence in corpus:

            # If type(corpus) is Iterable[str]
            if type(sentence) == str:
                split_sentence = sentence.strip().split(delimiter)

            # If type(corpus) is Iterable[Iterable[str]]
            else:
                split_sentence = sentence

            if len(split_sentence) < max_seq_len:
                split_sentence += [self.pad_token] * (max_seq_len - len(split_sentence))

            elif len(split_sentence) > max_seq_len > 0:
                warnings.warn(
                    "Sentence '{}' is longer than specified padding length (specified {}, found {}).".format(
                        sentence, len(split_sentence), max_seq_len
                    )
                )

            indexed_corpus.append(list(map(self.__getitem__, split_sentence)))

        return indexed_corpus

    @staticmethod
    def _check_corpus(corpus: Corpus) -> None:
        """ Check whether the current corpus is a proper instance of Corpus. """

        # Type is str
        if type(corpus) == str:
            return

        elif isinstance(corpus, IterableClass):
            sample = corpus[0]

            # Type is Iterable[str]
            if type(sample) == str:
                return

            elif isinstance(sample, IterableClass):
                ssample = sample[0]

                # Type is Iterable[Iterable[str]]
                if type(ssample) == str:
                    return

        raise AssertionError(
            "'corpus' argument has to be of type str, Iterable[str] or Iterable[Iterable[str]], different type found."
        )

    def __getitem__(self, token: str) -> int:
        """ Return the index corresponding to a token. """
        return self._index.get(token, self.unk_idx)

    def __contains__(self, token: str) -> bool:
        """ Return whether token exists in index. """
        return token in self._index

    def __len__(self) -> int:
        """ Return length of index. """
        return len(self._index)

    def __eq__(self, other) -> bool:
        """ Compare this T2I to another object. """
        if not isinstance(other, T2I):
            return False

        return self._index == other._index

    def __iter__(self) -> Iterator[Tuple[str, int]]:
        """ Iterate over this T2I index by iterating over tokens and their corresponding indices. """
        for token, idx in self._index.items():
            yield token, idx

    def tokens(self) -> Tuple[str, ...]:
        """ Return all token in this T2I object. """
        return tuple(token for token, _ in self._index.items())

    def indices(self) -> Tuple[int, ...]:
        """ Return all indices in this T2I object. """
        return tuple(idx for _, idx in self._index.items())

    def save(self, path: str) -> None:
        """ Save T2I object as pickle. """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str):
        """ Load serialized T2I object. """
        with open(path, "rb") as f:
            t2i = pickle.load(f)

        return t2i

    def __repr__(self) -> str:
        """ Return a string representation of a T2I object. """
        return "T2I(Size: {}, unk_token: {}, eos_token: {}, pad_token: {}, {})".format(
            len(self.t2i), self.unk_token, self.eos_token, self.pad_token, self._index.__repr__()
        )
