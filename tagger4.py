from pathlib import Path

import click
import matplotlib.pyplot as plt
import torch
import tqdm
from torch.nn import Module, Conv1d, Linear
from torch.utils.data import DataLoader, Dataset

from utils import (
    create_vocab_chars_and_labels_from_files,
    parsed_sentences_from_files,
    SentenceCharacterEmbeddingDataset,
    create_word_embedding_from_files,
    check_accuracy_on_dataset,
    DatasetTypes, correct_predictions,
)


class ConvBaseSubWordModel(Module):
    def __init__(
        self,
        character_embedding_size: int,
        channel: int,
        window_size: int,
        num_of_labels: int,
        embeddings,
    ):
        super(ConvBaseSubWordModel, self).__init__()
        self.conv = Conv1d(character_embedding_size, channel, window_size)
        self.lin = Linear(channel + len(embeddings[embeddings.UNK]), num_of_labels)
        self.embeddings = embeddings

    def forward(self, embedded_words, words):
        x = self.conv(embedded_words)
        x = torch.max(x, dim=2).values
        x = torch.relu(x)
        existing_embedding = torch.stack([self.embeddings[word[0]] for word in words]).to(
            device=x.device, dtype=torch.float
        )
        x = torch.cat((x, existing_embedding), dim=-1)
        x = self.lin(x)
        x = torch.softmax(x, dim=1)
        return x


def train(
    model: Module, training_data: Dataset, dev_data: Dataset, batch_size: int, epochs: int
):
    losses = []
    acc = []
    train_acc = []
    optimizer = torch.optim.Adam(model.parameters())
    for i in range(epochs):
        loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
        model.train()
        total_loss = 0
        total_items = 0
        correct = 0
        for sentences, words, label in tqdm.tqdm(loader, leave=False):
            sentence = sentences[0]
            optimizer.zero_grad()
            output = model(sentence, words)
            total_items += len(sentence)
            correct += correct_predictions(output, label)
            loss = torch.nn.functional.cross_entropy(output, label[0])
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
        train_acc.append(correct / total_items)
        losses.append(total_loss)
        accuracy_dev = check_accuracy_on_dataset(model, dev_data)
        acc.append(accuracy_dev)
        print(accuracy_dev)
    return losses, acc, train_acc


@click.command()
@click.option("--dataset", type=DatasetTypes, default=DatasetTypes.NER)
@click.option("--epochs", type=int, default=100)
@click.option("--channels", type=int, default=30)
@click.option("--window_size", type=int, default=3)
@click.option(
    "--device",
    type=int,
    default=1 if not torch.cuda.is_available() else None,
)
@click.option("--vec_file_name", type=str, default="wordVectors.txt")
@click.option("--words_file_name", type=str, default="vocab.txt")
def main(dataset, epochs, channels, window_size, device, vec_file_name, words_file_name):
    dataset_path = Path(dataset.value)
    files = [(dataset_path / "train", dataset_path / "dev", dataset_path / "test")]
    dev_files = [file[1] for file in files]
    batch_size = 1
    training_files = [file[0] for file in files]
    vocabulary, characters, labels = create_vocab_chars_and_labels_from_files(training_files)
    sentences = parsed_sentences_from_files(training_files, ignore_o=True)
    labeled_words = [labeled_word for sentence in sentences for labeled_word in sentence]
    max_word_len = max([len(word) for word, _ in labeled_words])
    database = SentenceCharacterEmbeddingDataset(
        sentences, characters, labels, max_word_len, device
    )
    dev_database = SentenceCharacterEmbeddingDataset(
        parsed_sentences_from_files(dev_files, ignore_o=True),
        characters,
        labels,
        max_word_len,
        device,
    )

    word_embeddings = create_word_embedding_from_files(vec_file_name, words_file_name)

    model = ConvBaseSubWordModel(
        len(characters), channels, window_size, len(labels), word_embeddings
    ).to(device=device)

    losses, acc, train_acc = train(model, database, dev_database, batch_size, epochs)
    plt.plot(losses)
    plt.title(f"{dataset} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(f"{dataset}-{channels}-{epochs}-{window_size}loss.png")
    plt.clf()
    plt.plot(acc)
    plt.title(f"{dataset} Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig(f"{dataset}-{channels}-{epochs}-{window_size}accuracy.png")
    plt.clf()
    plt.plot(train_acc)
    plt.title(f"{dataset} Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig(f"{dataset}-{channels}-{epochs}-{window_size}-accuracy.png")


if __name__ == "__main__":
    main()
