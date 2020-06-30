"""
Module with function decorators, enabling the processing of items or iterables of items in the same function.
Likewise, it also enables compatibility with Numpy, Pytorch and Tensorflow.
"""

# STD
from collections import defaultdict
from functools import wraps
from typing import Callable

# CONST
# This defaultdict is used to apply conversions to data types defined in other frameworks (torch.Tensor /
# tf.Tensor). This is done by mapping the type of the sequence to a function converting it into an easier type.
# Thus, in the normal case, a sequence based on a Python Iterable is just kept as is.
CONVERSIONS = defaultdict(lambda: lambda seq: seq)

# Ensure compatibility with Numpy
# This is in a try/except block in case the user hasn't installed numpy
try:
    import numpy as np

    CONVERSIONS[np.array] = CONVERSIONS[np.ndarray] = lambda t: t.tolist()
except:
    pass

# Ensure compatibility with Pytorch
# This is in a try/except block in case the user hasn't installed torch
try:
    import torch

    CONVERSIONS[torch.Tensor] = lambda t: t.type(torch.LongTensor).tolist()
    CONVERSIONS[torch.LongTensor] = lambda t: t.tolist()
except:
    pass

# Ensure compatibility with Tensorflow
# This is in a try/except block in case the user hasn't installed tensorflow
try:
    import tensorflow as tf

    CONVERSIONS[tf.Tensor] = lambda t: tf.make_ndarray(tf.cast(t, tf.int32)).tolist()

    from tensorflow.python.framework.ops import EagerTensor

    CONVERSIONS[EagerTensor] = lambda t: tf.cast(t, tf.int32).numpy().tolist()

except:
    pass


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
    def with_indexing_consistency(self, corpus, *args, **kwargs):
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
    def with_unindexing_consistency(self, indexed_corpus, *args, **kwargs):
        indexed_corpus = CONVERSIONS[type(indexed_corpus)](indexed_corpus)

        if all([type(el) == int for el in indexed_corpus]):
            indexed_corpus = [indexed_corpus]
            unindexed_corpus = func(self, indexed_corpus, *args, **kwargs)

            return unindexed_corpus[0]

        else:
            return func(self, indexed_corpus, *args, **kwargs)

    return with_unindexing_consistency
