from pathlib import Path

import click
import torch

from tagger4 import ConvBaseSubWordModel
from utils import (
    create_vocab_chars_and_labels_from_files,
    create_word_embedding_from_files,
    encode_words, parse_test_file,
)


@click.command()
@click.option(
    "--device",
    type=int,
    default=1 if not torch.cuda.is_available() else None,
)
@click.option("--part", type=int, default=5)
@click.option("--vec_file_name", type=str, default="wordVectors.txt")
@click.option("--words_file_name", type=str, default="vocab.txt")
@click.option("--model_path", type=Path)
def main(device, part, vec_file_name, words_file_name, model_path):
    dataset, window_size, channels, _ = model_path.stem.split("-")
    window_size, channels = int(window_size), int(channels)
    dataset_path = Path(dataset.split(".")[-1].lower())
    train_file = dataset_path / "train"
    test_file = dataset_path / "test"
    training_files = [train_file]
    vocabulary, characters, labels = create_vocab_chars_and_labels_from_files(training_files, test_file)
    max_word_len = max([len(word) for word in vocabulary])

    word_embeddings = create_word_embedding_from_files(vec_file_name, words_file_name)

    model = ConvBaseSubWordModel(
        len(characters) + 1, channels, window_size, len(labels), word_embeddings
    ).to(device=device)

    data = torch.load(model_path)
    model.load_state_dict(data)

    results = []
    for word in parse_test_file(test_file):
        if not word:
            results.append("")

        word_embeddings = encode_words([word], characters, max_word_len, device)
        output = model(word_embeddings, [[word]])
        value_index = torch.argmax(output, dim=1)[0]
        results.append(labels[value_index])

    Path(f"test{part}.{dataset.lower()}").write_text("\n".join(results))


if __name__ == "__main__":
    main()
