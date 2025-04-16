import unittest

from nltk.corpus import reuters

from src.datasets import TokenizedCorpus


class TestTokenizedCorpus(unittest.TestCase):
    def setUp(self):
        self.corpus = [reuters.raw(fieldid) for fieldid in reuters.fileids()]
        self.dataset = TokenizedCorpus(self.corpus, 1000, 32)

    def test_encode(self):
        x, y = self.dataset[0]
        self.assertEqual(x[0], 76)  # make sure first token is start sentinel

        x, y = self.dataset[-1]
        self.assertEqual(y[-1], 77)  # make sure last token is end sentinel

    def test_decode(self):
        x, y = self.dataset[0]
        decoded_x = self.dataset.decode_document(x.unsqueeze(-1))
        decoded_y = self.dataset.decode_document(y.unsqueeze(-1))
        self.assertEqual(decoded_x, self.corpus[0][:31])
        self.assertEqual(decoded_y, self.corpus[0][:32])

        x, y = self.dataset[-1]
        decoded_x = self.dataset.decode_document(x.unsqueeze(-1))
        decoded_y = self.dataset.decode_document(y.unsqueeze(-1))
        self.assertEqual(decoded_x, self.corpus[-1][-32:])
        self.assertEqual(decoded_y, self.corpus[-1][-31:])
