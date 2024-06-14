from pathlib import Path

import torch
import tqdm
from torch.nn import Module, Conv1d, MaxPool1d, Linear
from torch.utils.data import DataLoader, Dataset

from utils import (
    create_vocab_chars_and_labels_from_files,
    parsed_sentences_from_files,
    SentenceCharacterEmbeddingDataset,
    create_word_embedding_from_files,
    check_accuracy_on_dataset,
)


class ConvBaseSubWordModel(Module):
    def __init__(self, num_of_characters: int, word_size: int, num_of_labels: int, embeddings):
        super(ConvBaseSubWordModel, self).__init__()
        self.pool = MaxPool1d(word_size)
        self.conv = Conv1d(num_of_characters, 30, 3)
        self.lin = Linear(30 + len(embeddings[embeddings.UNK]), num_of_labels)
        self.embeddings = embeddings

    def forward(self, embedded_words, words):
        x = self.conv(embedded_words)
        x = torch.max(x, dim=2).values
        existing_embedding = torch.stack([self.embeddings[word[0]] for word in words]).to(
            device=x.device, dtype=torch.float
        )
        x = torch.cat((x, existing_embedding), dim=-1)
        x = self.lin(x)
        x = torch.softmax(x, dim=1)
        return x


def train(model: Module, training_data: Dataset, dev_data: Dataset, batch_size: int, epochs: int):
    optimizer = torch.optim.Adam(model.parameters())
    for i in tqdm.trange(epochs):
        print(check_accuracy_on_dataset(model, dev_data))
        loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
        model.train()
        for sentences, words, label in tqdm.tqdm(loader, leave=False):
            optimizer.zero_grad()
            output = model(sentences[0], words)
            loss = torch.nn.functional.cross_entropy(output, label[0])
            loss.backward()
            optimizer.step()



def main():
    ner_path = Path("ner")
    pos_path = Path("pos")
    files = [
        (ner_path / "train", ner_path / "dev", ner_path / "test"),
        (pos_path / "train", pos_path / "dev", pos_path / "test"),
    ]
    dev_files = [file[1] for file in files]
    batch_size = 1
    device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")
    training_files = [file[0] for file in files]
    vocabulary, characters, labels = create_vocab_chars_and_labels_from_files(training_files)
    sentences = parsed_sentences_from_files(training_files)
    labeled_words = [labeled_word for sentence in sentences for labeled_word in sentence]
    max_word_len = max([len(word) for word, _ in labeled_words])
    dataset = SentenceCharacterEmbeddingDataset(sentences, characters, labels, max_word_len, device)
    dev_dataset = SentenceCharacterEmbeddingDataset(
        parsed_sentences_from_files(dev_files), characters, labels, max_word_len, device
    )

    # embedding = create_word_indexer(vocabulary)
    vec_file_name = r"wordVectors.txt"
    words_file_name = r"vocab.txt"
    word_embeddings = create_word_embedding_from_files(vec_file_name, words_file_name)

    model = ConvBaseSubWordModel(
        len(characters), max_word_len, len(labels), word_embeddings
    ).to(device=device)

    train(model, dataset, dev_dataset, batch_size, 100)


if __name__ == "__main__":
    main()
