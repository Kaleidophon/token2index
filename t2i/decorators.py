"""
Module with function decorators, enabling
"""

# STD
from functools import wraps
from typing import Callable


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
        if all([type(el) == int for el in indexed_corpus]):
            indexed_corpus = [indexed_corpus]
            unindexed_corpus = func(self, indexed_corpus, *args, **kwargs)

            return unindexed_corpus[0]

        else:
            return func(self, indexed_corpus, *args, **kwargs)

    return with_unindexing_consistency
