"""
Unit tests for T2I class.
"""

# STD
import os
import random
import string
import unittest

# PROJECT
from t2i import T2I, Index, Corpus

# TODO: Missing tests
#   - Test behavior for unexpected input types
#   - Test usage of arbitrary special tokens
#   - Test assignment of <unk> and <eos> when building vocab from file
#   - Test updating of i2t after extending
#   - Test init of T2I class with index, in particular
#       - Using an empty index
#       - Using an index missing the unk and / or eos token
#       - Using the index with existing unk and / or eos token
#       - Using the index with a custom unk and / or eos token


class IndexingTest(unittest.TestCase):
    """
    Test whether the building of the index and the indexing and un-indexing of an input work.
    """

    def setUp(self):
        self.test_corpus1 = "A B C D B C A E"
        self.indexed_test_corpus1 = [0, 1, 2, 3, 1, 2, 0, 4]

        self.test_corpus2 = "AA-CB-DE-BB-BB-DE-EF"
        self.indexed_test_corpus2 = [0, 1, 2, 3, 3, 2, 4]

        self.test_corpus3 = "This is a test sentence"
        self.test_corpus3b = "This is a test sentence <eos>"
        self.indexed_test_corpus3 = [0, 1, 2, 3, 4, 6]

        self.test_corpus4 = "This is a <unk> sentence <eos>"
        self.test_corpus4b = "This is a goggledigook sentence <eos>"
        self.indexed_test_corpus45 = [0, 1, 2, 5, 4, 6]

        self.test_corpus5 = "This is a #UNK# sentence #EOS#"
        self.test_corpus5b = "This is a goggledigook sentence #EOS#"

    def _assert_indexing_consistency(
        self, corpus: Corpus, t2i: T2I, joiner: str = " ", delimiter: str = " "
    ):
        """
        Test whether first indexing and then un-indexing yields the original sequence.
        """
        self.assertEqual(
            t2i.unindex(t2i.index(corpus, delimiter=delimiter), joiner=joiner), corpus
        )

    def test_default_indexing(self):
        """
        Test normal indexing case.
        """
        t2i = T2I.build(self.test_corpus1)

        self.assertEqual(t2i.index(self.test_corpus1), self.indexed_test_corpus1)
        self._assert_indexing_consistency(self.test_corpus1, t2i)

    def test_extend(self):
        """
        Test extending an existing index with an additional corpus.
        """
        t2i = T2I.build(self.test_corpus3)
        additional_corpus = "These are new words"

        t2i = t2i.extend(additional_corpus)

        for token in additional_corpus.split(" "):
            self.assertIn(token, t2i)
            self.assertEqual(token, t2i.i2t[t2i[token]])

        test_sentence = "This is a new sentence"
        indexed_test_sentence = [0, 1, 2, 9, 4]
        self.assertEqual(t2i.index(test_sentence), indexed_test_sentence)
        self._assert_indexing_consistency(test_sentence, t2i)

        # TODO: Test that i2t was updated

    def test_delimiter_indexing(self):
        """
        Test indexing with different delimiter.
        """
        t2i = T2I.build(self.test_corpus2, delimiter="-")

        self.assertEqual(
            t2i.index(self.test_corpus2, delimiter="-"), self.indexed_test_corpus2
        )
        self._assert_indexing_consistency(
            self.test_corpus2, t2i, joiner="-", delimiter="-"
        )

    def test_eos_indexing(self):
        """
        Test indexing with (default) end-of-sequence token.
        """
        t2i = T2I.build(self.test_corpus3)

        self.assertEqual(t2i.index(self.test_corpus3b), self.indexed_test_corpus3)
        self._assert_indexing_consistency(self.test_corpus3b, t2i)

    def test_unk_indexing(self):
        """
        Test indexing with unknown words.
        """
        t2i = T2I.build(self.test_corpus3)

        self.assertEqual(t2i.index(self.test_corpus4), self.indexed_test_corpus45)
        self.assertEqual(t2i.index(self.test_corpus4b), self.indexed_test_corpus45)
        self._assert_indexing_consistency(self.test_corpus4, t2i)

    def test_custom_special_tokens_indexing(self):
        """
        Test indexing with custom eos / unk token.
        """
        t2i = T2I.build(self.test_corpus3, unk_token="#UNK#", eos_token="#EOS#")

        self.assertEqual(t2i.index(self.test_corpus5), self.indexed_test_corpus45)
        self.assertEqual(t2i.index(self.test_corpus5b), self.indexed_test_corpus45)
        self._assert_indexing_consistency(self.test_corpus5, t2i)

    def test_immutability(self):
        """
        Test whether the T2I stays immutable after object init.
        """
        t2i = T2I.build(self.test_corpus1)
        with self.assertRaises(NotImplementedError):
            t2i["banana"] = 66


class TypeConsistencyTest(unittest.TestCase):
    """
    Test whether T2I correctly infers the data structure of the input. This is important because some methods are
    expected to work with both single sentence or a list of sentences (or the indexed equivalents of that).
    """

    def setUp(self):
        test_corpus = "This is a long test sentence . It contains many words."
        self.t2i = T2I.build(test_corpus)

    def test_build_and_extend_consistency(self):
        """
        Make sure that index is built correctly no matter whether the input to build() is a single sentence or a list of
        sentences.
        """
        # Test build()
        test_sentence = "This is a test sentence"
        test_corpus = ["This is a", "test sentence"]

        t2i1 = T2I.build(test_sentence)
        t2i2 = T2I.build(test_corpus)
        self.assertEqual(t2i1, t2i2)

        # Test extend()
        test_sentence2 = "These are new words"
        test_corpus2 = ["These are", "new words"]

        self.assertEqual(t2i1.extend(test_sentence2), t2i2.extend(test_corpus2))

        # Test extend with a mix of types
        self.assertEqual(t2i1.extend(test_corpus2), t2i2.extend(test_sentence2))

    def test_indexing_consistency(self):
        """
        Test whether indexing is consistent with respect to the input type. Therefore, indexing a sentence should yield
        a list of indices, and indexing a list of sentences should yield a list of lists of indices, i.e.

        str -> List[int]
        List[str] -> List[List[str]]

        The reverse should hold for un-indexing, i.e.

        List[int] -> str or List[str]
        List[List[int]] -> List[str]
        """
        # Check indexing consistency for single sentence
        test_sentence = "This is a test sentence"
        indexed_test_sentence = self.t2i(test_sentence)

        self.assertEqual(type(indexed_test_sentence), list)
        self.assertTrue(all([type(idx) == int for idx in indexed_test_sentence]))

        # Check un-indexing consistency for single sentence
        unindexed_test_sentence = self.t2i.unindex(indexed_test_sentence)
        self.assertEqual(type(unindexed_test_sentence), str)
        self.assertEqual(test_sentence, unindexed_test_sentence)
        self.assertEqual(
            test_sentence.replace(" ", "###"),
            self.t2i.unindex(indexed_test_sentence, joiner="###"),
        )

        # Check un-indexing consistency for single sentence without a joiner
        unjoined_test_sentence = self.t2i.unindex(indexed_test_sentence, joiner=None)
        self.assertEqual(test_sentence.split(" "), unjoined_test_sentence)
        self.assertEqual(type(unjoined_test_sentence), list)
        self.assertTrue(all([type(token) == str for token in unjoined_test_sentence]))

        # Check indexing consistency for a list of sentences
        test_corpus = ["This is a", "test sentence"]
        indexed_test_corpus = self.t2i(test_corpus)

        self.assertEqual(type(indexed_test_corpus), list)
        self.assertTrue(all([type(sent) == list for sent in indexed_test_corpus]))
        self.assertTrue(
            all([type(idx) == int for sent in indexed_test_corpus for idx in sent])
        )

        # Check un-indexing consistency for a list of sentences
        unindexed_test_corpus = self.t2i.unindex(indexed_test_corpus)
        self.assertEqual(type(unindexed_test_corpus), list)
        self.assertTrue([type(sent) == str for sent in unindexed_test_corpus])
        self.assertEqual(unindexed_test_corpus, test_corpus)
        self.assertEqual(
            [sent.replace(" ", "###") for sent in test_corpus],
            self.t2i.unindex(indexed_test_corpus, joiner="###"),
        )

        # Check un-indexing consistency for a list of sentence  without a joiner
        unjoined_test_corpus = self.t2i.unindex(indexed_test_corpus, joiner=None)
        self.assertEqual(type(unindexed_test_corpus), list)
        self.assertTrue(all([type(sent) == list for sent in unjoined_test_corpus]))
        self.assertTrue(
            all([type(token) == str for sent in unjoined_test_corpus for token in sent])
        )


class VocabFileTest(unittest.TestCase):
    """
    Test building a T2I object from a vocab file.
    """

    def setUp(self):
        num_tokens = 30
        self.tokens = [random_str(random.randint(3, 8)) for _ in range(num_tokens)]

        # First vocab file format: One token per line
        self.vocab_path1 = "vocab1.txt"
        with open(self.vocab_path1, "w") as vocab_file1:
            vocab_file1.write("\n".join(self.tokens))

        # Second vocab file format: Token and index, separated by tab
        self.vocab_path2 = "vocab2.csv"
        self.indices2 = list(range(num_tokens))
        random.shuffle(self.indices2)
        with open(self.vocab_path2, "w") as vocab_file2:
            vocab_file2.write(
                "\n".join(
                    [
                        f"{token}\t{index}"
                        for token, index in zip(self.tokens, self.indices2)
                    ]
                )
            )

        # Second vocab file format, this time with higher indices
        self.vocab_path3 = "vocab3.csv"
        self.indices3 = list(range(0, num_tokens, 2))
        random.shuffle(self.indices3)
        with open(self.vocab_path3, "w") as vocab_file3:
            vocab_file3.write(
                "\n".join(
                    [
                        f"{token}\t{index}"
                        for token, index in zip(self.tokens, self.indices3)
                    ]
                )
            )

        # Second vocab file format, but with different delimiter
        self.vocab_path4 = "vocab4.csv"
        with open(self.vocab_path4, "w") as vocab_file4:
            vocab_file4.write(
                "\n".join(
                    [
                        f"{token}###{index}"
                        for token, index in zip(self.tokens, self.indices2)
                    ]
                )
            )

    def tearDown(self):
        os.remove(self.vocab_path1)
        os.remove(self.vocab_path2)
        os.remove(self.vocab_path3)
        os.remove(self.vocab_path4)

    def test_building_from_file(self):
        """
        Test building a T2I object from a vocab file.
        """
        # First vocab file format: One token per line
        t2i1 = T2I.from_file(self.vocab_path1)
        self.assertTrue(
            [
                t2i1[token] == idx
                for token, idx in zip(self.tokens, range(len(self.tokens)))
            ]
        )

        # Second vocab file format: Token and index, separated by tab
        t2i2 = T2I.from_file(self.vocab_path2)
        self.assertTrue(
            [t2i2[token] == idx for token, idx in zip(self.tokens, self.indices2)]
        )

        # Second vocab file format, this time with higher indices
        t2i3 = T2I.from_file(self.vocab_path3)
        self.assertTrue(
            [t2i3[token] == idx for token, idx in zip(self.tokens, self.indices3)]
        )

        # Second vocab file format, but with different delimiter
        t2i4 = T2I.from_file(self.vocab_path4, delimiter="###")
        self.assertTrue(
            [t2i4[token] == idx for token, idx in zip(self.tokens, self.indices2)]
        )

    def test_correct_indexing(self):
        """
        Test if indexing of new tokens is done correctly if the indices in the T2I class so far are arbitrary. In that
        case, indexing should be continued from the highest index.
        """
        t2i = T2I.from_file(self.vocab_path3)
        highest_index = max(t2i.values())
        test_sent = "These are definitely new non-random tokens ."

        t2i = t2i.extend(test_sent)

        self.assertTrue(
            all([t2i[token] > highest_index for token in test_sent.split(" ")])
        )


class SerializationTest(unittest.TestCase):
    """
    Test saving and loading of a T2I object.
    """

    def setUp(self):
        self.path = "test_t2i.pkl"

    def tearDown(self):
        os.remove(self.path)

    def test_serialization(self):
        """ The above. """
        t2i = T2I.build(
            " ".join(
                [
                    random_str(random.randint(3, 10))
                    for _ in range(random.randint(20, 40))
                ]
            )
        )

        t2i.save(self.path)

        self.assertEqual(T2I.load(self.path), t2i)


class IndexTest(unittest.TestCase):
    """
    Test the Index class (and initializing a T2I with an index).
    """

    def test_index(self):
        """
        Test whether functionalities of the Index class work correctly.
        """
        # Empty index
        index = Index()
        self.assertEqual(index.highest_idx, -1)

        # Look up some random words
        num_tokens = random.randint(5, 15)
        for token in [random_str(5) for _ in range(num_tokens)]:
            index[token]

        # Check if indexing was done correctly
        self.assertEqual(set(range(num_tokens)), set(index.values()))

        # Now add some more distant indices and check if highest_idx changes accordingly
        high_indices = [random.randint(50, 100) for _ in range(30)]

        for high_index in high_indices:
            index[random_str(5)] = high_index

        self.assertEqual(len(index), num_tokens + len(high_indices))
        self.assertEqual(index.highest_idx, max(high_indices))


class ModuleImportTest(unittest.TestCase):
    """
    Test the behavior of certain module imports.
    """

    def test_import_restrictions(self):
        """
        Decorators from t2i.decorators shouldn't be exposed and available for importing for user of the package, so make
        sure that trying to import those directly results in exceptions. Unfortunately, they are still available
        by importing the import from t2i directly (dumb).
        """
        with self.assertRaises(ModuleNotFoundError):
            from t2i.decorators import indexing_consistency

        with self.assertRaises(ModuleNotFoundError):
            from t2i.decorators import unindexing_consistency


def random_str(length: int) -> str:
    """ Return a random, lowercase string of a certain length. """
    return "".join([random.choice(string.ascii_lowercase) for _ in range(length)])


if __name__ == "__main__":
    unittest.main()
