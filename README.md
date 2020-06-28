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

* Building and extending vocab

    One way to build the index from a corpus is using the build() function:

    > from t2i import T2I
    > t2i = T2I.build(["colorless green ideas dream furiously", "the horse raced past the barn fell"])
    > t2i
    T2I(Size: 13, unk_token: <unk>, eos_token: <eos>, {'colorless': 0, 'green': 1, 'ideas': 2, 'dream': 3, 'furiously': 4, 'the': 5, 'horse': 6, 'raced': 7, 'past': 8, 'parn': 9, 'fell': 10, '<unk>': 11, '<eos>': 12})

* Easy indexing (of batches)

    @TODO Example
    
* Easy conversion back to strings
    
    @TODO
    
* Vocab from file

    @TODO Example 
    
* Compatibility with major frameworks! 

    @TODO Example
    
* Fixed memory size

    @TODO Example
    
* Support for special tokens
    
    @TODO Exmaple

### :electric_plug: Compatibility with other frameworks (NumPy, PyTorch, Tensorflow)

@TODO NumPy example

@TODO PyTorch 

@TODO Tensorfow

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
