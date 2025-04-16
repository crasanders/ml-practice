from collections import Counter
from typing import Iterable

import torch
from torch.utils.data import Dataset


class TokenizedCorpus(Dataset):
    def __init__(
        self, documents: Iterable[Iterable[str]], char_thresh: int, context_size: int
    ):
        self.context_size = context_size
        raw = "".join(documents)
        counts = Counter(raw)
        self.char_list = sorted(
            [char for char in counts if counts[char] >= char_thresh]
        )
        self.char_to_token = {char: i for i, char in enumerate(self.char_list)}

        self.start_doc = len(self.char_list)
        self.end_doc = len(self.char_list) + 1

        self.doc_tokens = [self.encode_document(doc) for doc in documents]

        self.batches_per_doc = [
            len(range(len(doc) - context_size)) for doc in self.doc_tokens
        ]
        self.length = sum(self.batches_per_doc)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if index < 0:
            index = self.length + index
        tally = 0
        for i, doc_batch in enumerate(self.batches_per_doc):
            tally += doc_batch
            if index < tally:
                index -= sum(self.batches_per_doc[:i])
                doc = self.doc_tokens[i]
                x = torch.LongTensor(doc[index : index + self.context_size])
                y = torch.LongTensor(doc[index + 1 : index + self.context_size + 1])

                return x, y

    def encode_document(self, doc: Iterable[str]):
        tokens = [self.char_to_token[char] for char in doc if char in self.char_list]
        # add sentinel tokens to indicate start and end of docs
        tokens.insert(0, self.start_doc)
        tokens.append(self.end_doc)

        return tokens

    def decode_document(self, doc: Iterable[str]):
        chars = [self.char_list[idx] for idx in doc if idx < self.start_doc]
        return "".join(chars)
