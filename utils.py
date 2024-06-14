import math

import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple

from torch import Tensor
from torch.utils.data import Dataset

PADDING_WORDS = ["word_minus_2", "word_minus_1", "word_plus_1", "word_plus_2"]
UNK = "UNK"


def create_word_indexer(vocabulary: List[str]) -> Dict[str, int]:
    return {word: i for i, word in enumerate(vocabulary)}


def extract_vocabulary_and_labels(parsed_sentences):
    words = set()
    labels = set()

    for parsed_sentence in parsed_sentences:
        for a, b in parsed_sentence:
            words.add(a)
            labels.add(b)

    return list(words), list(labels)


def parse_labeled_data(file_path: Path):
    lines = file_path.read_text().splitlines()

    sentences = []
    current_sentence = []

    for line in lines:
        line = line.strip()
        if line:
            word, label = line.split()
            current_sentence.append((word, label))
        else:
            if current_sentence:
                sentences.append(current_sentence)
                current_sentence = []

    if current_sentence:
        sentences.append(current_sentence)

    return sentences


def parsed_sentences_from_files(train_files: List[Path]):
    parsed_sentences = []
    for file in train_files:
        parsed_sentences.extend(parse_labeled_data(file))
    return parsed_sentences


def create_vocab_chars_and_labels_from_files(train_files: List[Path]):
    parsed_sentences = parsed_sentences_from_files(train_files)
    vocab, labels = extract_vocabulary_and_labels(parsed_sentences)
    vocab = list(PADDING_WORDS) + [UNK] + vocab
    characters = "".join(set("".join(vocab)))
    return vocab, characters, labels


class SentenceCharacterEmbeddingDataset(Dataset):
    def __init__(
        self,
        sentences: List[List[Tuple[str, str]]],
        characters: str,
        labels: List[str],
        max_word_size: int,
        device: torch.device,
    ):
        self.sentences = sentences
        self.characters = characters
        self.labels = labels
        self.max_word_size = max_word_size
        self.device = device

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        words = [word for word, _ in sentence]
        word_tensor = torch.tensor(
            [
                [0] * (padding_size // 2)
                + [self.characters.index(character) for character in word]
                + [0] * math.ceil(padding_size / 2)
                for word in words
                for padding_size in [self.max_word_size - len(word)]
            ],
            device=self.device,
            dtype=torch.float,
        )
        word_embedding = torch.nn.functional.one_hot(word_tensor.long(), len(self.characters))
        word_embedding = word_embedding.swapaxes(-1, -2)
        labels = torch.tensor(
            [self.labels.index(label) for word, label in sentence],
            device=self.device,
            dtype=torch.float,
        )
        labels = torch.nn.functional.one_hot(labels.long(), len(self.labels))
        return word_embedding.float(), words, labels.float()


class WordEmbedding:
    UNK = "UUUNKKK"

    def __init__(self, vecs: Tensor, vocab: List[str]):
        self.vecs = vecs
        self.vocab = vocab

    def __getitem__(self, word: str):
        if word in self.vocab:
            return self.vecs[self.vocab.index(word)]
        return self.vecs[self.vocab.index(self.UNK)]


def create_word_embedding_from_files(vec_file_name: str, words_file_name: str):
    vocab = Path(words_file_name).read_text().split()
    vecs = torch.tensor(np.loadtxt(vec_file_name))
    return WordEmbedding(vecs, vocab)


def check_accuracy_on_dataset(model, dataset):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for sentences, words, labels in dataset:
            output = model(sentences, words)
            _, predicted = torch.max(output, 1)
            _, true = torch.max(labels, 1)
            total += true.size(0)
            correct += (predicted == true).sum().item()
    return correct / total
