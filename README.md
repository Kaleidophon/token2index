# :zap: :card_index: token2index: A lightweight but powerful library for token indexing

[![Build](https://travis-ci.org/Kaleidophon/token2index.svg?branch=master)](https://travis-ci.org/github/Kaleidophon/token2index/builds)
[![Documentation Status](https://readthedocs.org/projects/token2index/badge/?version=latest)](https://token2index.readthedocs.io/en/latest/?badge=latest)
[![Coverage Status](https://coveralls.io/repos/github/Kaleidophon/token2index/badge.svg?branch=master)](https://coveralls.io/github/Kaleidophon/token2index?branch=master)
[![Compatibility](https://img.shields.io/badge/Python-3.5%20%7C%203.6%20%7C%203.7%20%7C%203.8-blue)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)

``token2index`` is a small yet powerful library facilitating the fast and easy creation of a data structure mapping 
tokens to indices, primarily aimed at applications for Natural Language Processing. The library is fully tested, and 
does not require any additional requirements. The documentation can be found [here](token2index.readthedocs.io), some feature highlights are 
shown below.

### :sparkles: Feature Highlights

* **Building and extending vocab**

    One way to build the index from a corpus is using the build() function:

    ```python
    >>> from t2i import T2I
    >>> t2i = T2I.build(["colorless green ideas dream furiously", "the horse raced past the barn fell"])
    >>> t2i
    T2I(Size: 13, unk_token: <unk>, eos_token: <eos>, {'colorless': 0, 'green': 1, 'ideas': 2, 'dream': 3, 'furiously': 4, 'the': 5, 'horse': 6, 'raced': 7, 'past': 8, 'parn': 9, 'fell': 10, '<unk>': 11, '<eos>': 12})
    ```
  
    The index can always be extended again later using `extend()`:
    
    ```python
    >>> t2i = t2i.extend("completely new words")
    T2I(Size: 16, unk_token: <unk>, eos_token: <eos>, {'colorless': 0, 'green': 1, 'ideas': 2, 'dream': 3, 'furiously': 4, 'the': 5, 'horse': 6, 'raced': 7, 'past': 8, 'barn': 9, 'fell': 10, 'completely': 13, 'new': 14, 'words': 15, '<unk>': 16, '<eos>': 17})
    ```

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
    T2I(Size: 3, unk_token: <unk>, eos_token: <eos>, {'<unk>': 0, '<eos>': 1, '<mask>': 2})
    ```

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

    @article{token2index,
        title={token2index: A lightweight but powerful library for token indexing},
        author={Dennis Ulmer},
        year={2020},
    }
