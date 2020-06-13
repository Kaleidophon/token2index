"""
Unit tests for T2I class.
"""

# STD
import unittest

# PROJECT
from t2i import T2I, Corpus


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

    def _assert_indexing_consistency(self, corpus: Corpus, t2i: T2I, joiner: str = " ", delimiter: str = " "):
        """
        Test whether first indexing and then un-indexing yields the original sequence.
        """
        self.assertEqual(t2i.unindex(t2i.index(corpus, delimiter=delimiter), joiner=joiner), corpus)

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

    def test_delimiter_indexing(self):
        """
        Test indexing with different delimiter.
        """
        t2i = T2I.build(self.test_corpus2,  delimiter="-")

        self.assertEqual(t2i.index(self.test_corpus2, delimiter="-"), self.indexed_test_corpus2)
        self._assert_indexing_consistency(self.test_corpus2, t2i, joiner="-", delimiter="-")

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
        self.assertEqual(test_sentence.replace(" ", "###"), self.t2i.unindex(indexed_test_sentence, joiner="###"))
        
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
        self.assertTrue(all([type(idx) == int for sent in indexed_test_corpus for idx in sent]))

        # Check un-indexing consistency for a list of sentences
        unindexed_test_corpus = self.t2i.unindex(indexed_test_corpus)
        self.assertEqual(type(unindexed_test_corpus), list)
        self.assertTrue([type(sent) == str for sent in unindexed_test_corpus])
        self.assertEqual(unindexed_test_corpus, test_corpus)
        self.assertEqual(
            [sent.replace(" ", "###") for sent in test_corpus],
            self.t2i.unindex(indexed_test_corpus, joiner="###")
        )

        # Check un-indexing consistency for a list of sentence  without a joiner
        unjoined_test_corpus = self.t2i.unindex(indexed_test_corpus, joiner=None)
        self.assertEqual(type(unindexed_test_corpus), list)
        self.assertTrue(all([type(sent) == list for sent in unjoined_test_corpus]))
        self.assertTrue(all([type(token) == str for sent in unjoined_test_corpus for token in sent]))


class VocabFileTest(unittest.TestCase):
    # TODO: Test seed index with arbitrary indices
    ...  # TODO


class NumpyTest(unittest.TestCase):
    ...  # TODO


class PyTorchTest(unittest.TestCase):
    ...  # TODO


class TensorflowTest(unittest.TestCase):
    ...  # TODO


if __name__ == "__main__":
    unittest.main()
