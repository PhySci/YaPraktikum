from pickle import load
from torch.utils.data import Dataset, DataLoader
from torch import nn
from typing import List
import random


class TextDataset(Dataset):

    def __init__(self, file_pth, window_size=2):
        super(TextDataset, self).__init__()
        self._corpus = None
        self._window_size = window_size
        self._load_file(file_pth)
        self._indx_list = list(range(len(self._corpus)))

    def _load_file(self, file_pth):
        with open(file_pth, "rb") as fid:
            self._corpus = load(fid)

    def __len__(self):
        return len(self._corpus) - 10

    def __getitem__(self, item):
        item2 = item + 5
        target = self._corpus[item2]
        left_context = self._corpus[item2-self._window_size:item2]
        right_context = self._corpus[item2+1:item2+self._window_size+1]
        context = left_context + right_context

        fake_words = random.choices(self._indx_list, 9)

        return target, context


class Word2Vec(nn.Module):

    def __init__(self, vocab_size=17005300, emb_size=128):
        super(Word2Vec, self).__init__()
        self._emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_size)


    def forward(self, target: int, context: List[int]):
        target_emb = self._emb(target)
        context_emb = self._emb(context).sum(dim=1)


def test_dataset():
    ds = TextDataset("./corpus.pkl")
    for el in ds:
        print(el)
        break

def main():
    test_dataset()

if __name__ == "__main__":
    main()