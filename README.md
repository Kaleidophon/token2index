# :zap: :card_index: token2index: A lightweight but powerful library for token indexing

[![Build](https://travis-ci.org/Kaleidophon/token2index.svg?branch=master)](https://travis-ci.org/github/Kaleidophon/token2index/builds)
[![Documentation Status](https://readthedocs.org/projects/token2index/badge/?version=latest)](https://token2index.readthedocs.io/en/latest/?badge=latest)
[![Coverage Status](https://coveralls.io/repos/github/Kaleidophon/token2index/badge.svg?branch=master)](https://coveralls.io/github/Kaleidophon/token2index?branch=master)
[![Compatibility](https://img.shields.io/badge/Python-3.5%20%7C%203.6%20%7C%203.7%20%7C%203.8-blue)]()
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)

``token2index`` is a small yet powerful library facilitating the fast and easy creation of a data structure mapping 
tokens to indices, primarily aimed at applications for Natural Language Processing. The library is fully tested, and 
does not require any additional requirements. The documentation can be found [here](https://token2index.readthedocs.io/en/latest/), some feature highlights are 
shown below.

**Who / what is this for?**

This class is written to be used for NLP applications where we want to assign an index to every word in a sequence e.g. to be later used to look up corresponding 
word embeddings. Building an index and indexing batches of sequences for Deep Learning models using frameworks like PyTorch or Tensorflow are common steps but are often written from 
scratch every time. This package provides a ready-made package combining many useful features, like reading vocabulary files, building indices from a corpus or indexing entire batches in one single
function call, all while being fully tested.

### :sparkles: Feature Highlights

* **Building and extending vocab**

    One way to build the index from a corpus is using the build() function:

    ```python
    >>> from t2i import T2I
    >>> t2i = T2I.build(["colorless green ideas dream furiously", "the horse raced past the barn fell"])
    >>> t2i
    T2I(Size: 13, unk_token: <unk>, eos_token: <eos>, pad_token: <pad>, {'colorless': 0, 'green': 1, 'ideas': 2, 'dream': 3, 'furiously': 4, 'the': 5, 'horse': 6, 'raced': 7, 'past': 8, 'parn': 9, 'fell': 10, '<unk>': 11, '<eos>': 12, '<pad>': 13})
    ```
  
    The index can always be extended again later using `extend()`:
    
    ```python
    >>> t2i = t2i.extend("completely new words")
    T2I(Size: 16, unk_token: <unk>, eos_token: <eos>, pad_token: <pad>, {'colorless': 0, 'green': 1, 'ideas': 2, 'dream': 3, 'furiously': 4, 'the': 5, 'horse': 6, 'raced': 7, 'past': 8, 'barn': 9, 'fell': 10, 'completely': 13, 'new': 14, 'words': 15, '<unk>': 16, '<eos>': 17, '<pad>': 18})
    ```
  
    Both methods and index() also work with an already tokenized corpus in the form of 
    
        [["colorless", "green", "ideas", "dream", "furiously"], ["the", "horse", "raced", "past", "the", "barn", "fell"]]    

* **Easy indexing (of batches)**
    
    Index multiple sentences at once in a single function call!

    ```python
    >>> t2i.index(["the green horse raced <eos>", "ideas are a dream <eos>"])
    [[5, 1, 6, 7, 12], [2, 11, 11, 3, 12]]
    ```
    
    where unknown tokens are always mapped to `unk_token`.
    
* **Easy conversion back to strings**
    
    Reverting indices back to strings is equally as easy:
    
    ```python
    >>> t2i.unindex([5, 14, 16, 3, 6])
    'the new <unk> dream horse'
    ```

* **Automatic padding**

    You are indexing multiple sentences of different length and want to add padding? No problem! `index()` has two
    options available via the `pad_to` argument. The first is padding to the maximum length of all the sentences:
    
    ```python
    >>> padded_sents = t2i.index(["the green horse raced <eos>", "ideas <eos>"], pad_to="max")
    >>> padded_sents
    [[5, 1, 6, 7, 12], [2, 12, 13, 13, 13]]
    >>> t2i.unindex(padded_sents)
    [['the green horse raced <eos>', 'ideas <eos> <pad> <pad> <pad>']]
    ```
  
    Alternatively, you can also pad to a pre-defined length:
    
    ```python
    >>> padded_sents = t2i.index(["the green horse <eos>", "past ideas <eos>"], pad_to=5)
    >>> padded_sents
    [[5, 1, 6, 12, 13], [8, 2, 12, 13, 13]]
    >>> t2i.unindex(padded_sents)
    [['the green horse <eos> <pad>', 'past ideas <eos> <pad> <pad>']]
    ```
    
* **Vocab from file**

    Using `T2I.from_file()`, the index can be created directly by reading from an existing vocab file. 
    Refer to its documentation [here](https://token2index.readthedocs.io/en/latest/#t2i.T2I.from_file) for more info.
    
* **Fixed memory size**

    Although the `defaultdict` class from Python's `collections` package also posses the functionality to map unknown 
    keys to a certain value, it grows in size for every new key. `T2I` memory size stays fixed after the index is built.
    
* **Support for special tokens**
    
    To enable flexibility in modern NLP applications, `T2I` allows for an arbitrary number of special tokens (like a 
    masking or a padding token) during init! 
    
    ```python
    >>> t2i = T2I(special_tokens=["<mask>"])
    >>> t2i
    T2I(Size: 3, unk_token: <unk>, eos_token: <eos>, pad_token: <pad>, {'<unk>': 0, '<eos>': 1, '<mask>': 2, '<pad>': 3})
    ```

* **Explicitly supported programmer laziness**

    Too lazy to type? The library saves you a few keystrokes here and there. instead of calling `t2i.index(...)` you can
    directly call `t2i(...)` to index one or multiple sequences. Furthermore, key functions like `index()`, `unindex()`,
    `build()` and `extend()` support strings or iterables of strings as arguments alike.

### :electric_plug: Compatibility with other frameworks (Numpy, PyTorch, Tensorflow)

It is also ensured that `T2I` is easily compatible with frameworks like Numpy, PyTorch and 
Tensorflow, without needing them as requirements:

**Numpy**

```python
>>> import numpy as np
>>> t = np.array(t2i.index(["the new words are ideas <eos>", "the green horse <eos> <pad> <pad>"]))
>>> t
array([[ 5, 15, 16, 17,  2, 18],
   [ 5,  1,  6, 18, 19, 19]])
>>> t2i.unindex(t)
['the new words <unk> ideas <eos>', 'the green horse <eos> <pad> <pad>']
```

**PyTorch**

```python
>>> import torch
>>> t = torch.LongTensor(t2i.index(["the new words are ideas <eos>", "the green horse <eos> <pad> <pad>"]))
>>> t
tensor([[ 5, 15, 16, 17,  2, 18],
    [ 5,  1,  6, 18, 19, 19]])
>>> t2i.unindex(t)
['the new words <unk> ideas <eos>', 'the green horse <eos> <pad> <pad>']
```

**Tensorflow**

```python
>>> import tensorflow as tf
>>> t = tf.convert_to_tensor(t2i.index(["the new words are ideas <eos>", "the green horse <eos> <pad> <pad>"]), dtype=tf.int32)
>>> t
tensor([[ 5, 15, 16, 17,  2, 18],
    [ 5,  1,  6, 18, 19, 19]])
>>> t2i.unindex(t)
['the new words <unk> ideas <eos>', 'the green horse <eos> <pad> <pad>']
```

### :inbox_tray: Installation

Installation can simply be done using ``pip``:

    pip3 install token2index

### :mortar_board: Citing

If you use ``token2index`` for research purposes, please cite the library using the following citation info:

    @misc{ulmer2020token2index,
        title={token2index: A lightweight but powerful library for token indexing},
        author={Ulmer, Dennis},
        journal={https://github.com/Kaleidophon/token2index},
        year={2020}
    }
