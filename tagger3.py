from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from datetime import datetime
import tqdm
from torch.nn import Module
from torch.utils.data import TensorDataset, DataLoader

from data_parser import (
    parse_labeled_data,
    extract_vocabulary_and_labels,
    parse_unlabeled_data,
)

PADDING_WORDS = ("word_minus_2", "word_minus_1", "word_plus_1", "word_plus_2")
UNK = "UUUNKKK"


def glorot_init(first_dim, second_dim):
    epsilon = np.sqrt(6 / (first_dim + second_dim))
    return np.random.uniform(-epsilon, epsilon, (first_dim, second_dim))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


class WindowTaggerWithSuffixPrefix(nn.Module):
    def __init__(
            self,
            vocabulary,
            labels,
            hidden_dim,
            learning_rate,
            task,
            print_file,
            test_data,
            words_embeddings,
            prefixes,
            suffixes,
            embedding_dim=50,
            window_shape=(2, 2),
            padding_words=PADDING_WORDS,

    ):
        super(WindowTaggerWithSuffixPrefix, self).__init__()
        self.padding_words = padding_words
        self.unknown_word = UNK
        self.vocabulary_list = vocabulary
        self.vocabulary_dict = {word: index for index, word in enumerate(vocabulary)}
        self.outside_vocab_words_indices = [self.vocabulary_dict[word] for word in self.padding_words] + [
            self.vocabulary_dict[self.unknown_word]]
        self.labels = labels
        surrounding_window_length = window_shape[0] + window_shape[1]
        self.input_size = surrounding_window_length + 1
        assert surrounding_window_length == len(self.padding_words)
        if words_embeddings is None:
            self.words_embedding = nn.Embedding(len(self.vocabulary_dict), embedding_dim)
        else:
            self.words_embedding = nn.Embedding.from_pretrained(words_embeddings, freeze=False)
        self.prefixes = {prefix: index for index, prefix in
                         enumerate(prefixes + list(self.padding_words) + [self.unknown_word])}
        self.suffixes = {suffix: index for index, suffix in
                         enumerate(suffixes + list(self.padding_words) + [self.unknown_word])}
        self.prefix_embedding = nn.Embedding(len(self.prefixes), embedding_dim)
        self.suffix_embedding = nn.Embedding(len(self.suffixes), embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * self.input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, len(labels))
        self.criterion = torch.nn.CrossEntropyLoss().to(device)
        self.learning_rate = learning_rate
        self.accuracy_list = []
        self.task = task
        self.test_data = test_data
        self.print_file = print_file

    def get_word_index(self, word):
        index = self.vocabulary_dict.get(word)
        if index is None:
            index = self.vocabulary_dict.get(self.unknown_word)
        return index

    def get_indices_of_list_of_words(self, words):
        return [self.get_word_index(word) for word in words]

    def get_windows_word_indices_for_sentence(self, sentence):
        res = []
        for word_idx_in_sentence in range(len(sentence)):
            word_indices = []
            self.generate_start_of_sentence_indices(
                sentence, word_idx_in_sentence, word_indices
            )
            word_indices.append(self.get_word_index(sentence[word_idx_in_sentence]))
            self.generate_end_of_sentence_indices(
                sentence, word_idx_in_sentence, word_indices
            )
            res.append(word_indices)
        return res

    def generate_end_of_sentence_indices(
            self, sentence, word_idx_in_sentence, word_indices
    ):
        if word_idx_in_sentence == len(sentence) - 1:
            word_indices.extend(
                self.get_indices_of_list_of_words(self.padding_words[2:4])
            )
        elif word_idx_in_sentence == len(sentence) - 2:
            word_indices.append(self.get_word_index(sentence[word_idx_in_sentence + 1]))
            word_indices.extend(
                self.get_indices_of_list_of_words(self.padding_words[2:3])
            )
        else:
            word_indices.extend(
                self.get_indices_of_list_of_words(
                    sentence[word_idx_in_sentence + 1: word_idx_in_sentence + 3]
                )
            )

    def generate_start_of_sentence_indices(
            self, sentence, word_idx_in_sentence, word_indices
    ):
        if word_idx_in_sentence == 0:
            word_indices.extend(
                self.get_indices_of_list_of_words(self.padding_words[0:2])
            )
        elif word_idx_in_sentence == 1:
            word_indices.extend(
                self.get_indices_of_list_of_words(self.padding_words[1:2])
            )
            word_indices.append(self.get_word_index(sentence[0]))
        else:
            word_indices.extend(
                self.get_indices_of_list_of_words(
                    sentence[word_idx_in_sentence - 2: word_idx_in_sentence]
                )
            )

    def get_gold(self, labeled_sentence, word_index_in_sentence):
        y = (torch.zeros(1, len(self.labels), dtype=torch.float64))
        y[0][self.labels.index(labeled_sentence[word_index_in_sentence][1])] = 1
        return y

    def get_ner_filtered_preds_and_labels(self, predictions, true_labels):
        filtered_predictions = []
        filtered_true_labels = []
        for prediction, true_label in zip(predictions, true_labels):
            if not (prediction == self.labels.index("O") and prediction == true_label):
                filtered_predictions.append(prediction)
                filtered_true_labels.append(true_label)
        true_labels = filtered_true_labels
        predictions = filtered_predictions
        return predictions, true_labels

    def forward(self, prefixes_indices, suffixes_indices, window_indices):
        window_indices = window_indices.to(device)
        prefixes_indices = prefixes_indices.to(device)
        suffixes_indices = suffixes_indices.to(device)
        word_embeds = self.words_embedding(window_indices)  # Shape: (batch_size, 5, embedding_dim)
        suffix_embeds = self.suffix_embedding(suffixes_indices)
        prefix_embeds = self.prefix_embedding(prefixes_indices)
        # Flatten the embeddings
        word_embeds = word_embeds.view(word_embeds.size(0), -1)  # Shape: (batch_size, 5 * embedding_dim)
        suffix_embeds = suffix_embeds.view(suffix_embeds.size(0), -1)
        prefix_embeds = prefix_embeds.view(prefix_embeds.size(0), -1)
        # summing the embeds
        full_embeds = word_embeds + suffix_embeds + prefix_embeds
        out = self.fc1(full_embeds)
        out = torch.tanh(out)
        out = self.fc2(out)
        out = torch.softmax(out, dim=1)
        return out

    def get_data_in_x_y_format(self, train_labeled_sentences):
        window_indices = torch.empty((0, self.input_size), dtype=torch.int32)
        prefixes_indices = torch.empty((0, self.input_size), dtype=torch.int32)
        suffixes_indices = torch.empty((0, self.input_size), dtype=torch.int32)
        y = torch.empty((0, len(self.labels)))
        j = 1
        for labeled_sentence in train_labeled_sentences:
            j += 1
            # if j > 3:
            #     break
            sentence = [labeled_word[0] for labeled_word in labeled_sentence]
            sentence_windows_word_indices = torch.tensor(self.get_windows_word_indices_for_sentence(sentence),
                                                         dtype=torch.int32)
            sentence_prefixes_indices, sentence_suffix_indices = self.get_prefix_and_suffix_indices_for_sentence(
                sentence_windows_word_indices)
            prefixes_indices = torch.cat((prefixes_indices, torch.tensor(sentence_prefixes_indices, dtype=torch.int32)),
                                         dim=0)
            suffixes_indices = torch.cat((suffixes_indices, torch.tensor(sentence_suffix_indices, dtype=torch.int32)),
                                         dim=0)
            window_indices = torch.cat((window_indices, sentence_windows_word_indices), dim=0)
            for word_index_in_sentence, window_word_indices in enumerate(sentence_windows_word_indices):
                y = torch.cat((y, self.get_gold(labeled_sentence, word_index_in_sentence)), dim=0)
        return prefixes_indices, suffixes_indices, window_indices, y

    def get_prefix_and_suffix_indices_for_sentence(self, sentence_windows_word_indices):
        prefix_indices = []
        suffix_indices = []
        for word_indices in sentence_windows_word_indices:
            prefixes = []
            suffixes = []
            for index in word_indices:
                prefix, suffix = self.get_prefix_and_suffix_of_word_index(index)
                prefixes.append(prefix)
                suffixes.append(suffix)
            prefix_indices.append(prefixes)
            suffix_indices.append(suffixes)
        return prefix_indices, suffix_indices

    def get_prefix_and_suffix_of_word_index(self, word_index):
        word = self.vocabulary_list[word_index]
        if word_index in self.outside_vocab_words_indices:
            prefix = suffix = word
        else:
            prefix = word[0:3]
            suffix = word[-3:]
        return self.prefixes[prefix], self.suffixes[suffix]


def train(model: Module, training_data: DataLoader, dev_data: DataLoader, test_data: DataLoader, epochs: int, lr=0.001):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_list = []
    accuracy_list = []
    for i in tqdm.trange(epochs):
        running_loss = 0.0
        print("iteration {}\n".format(i), file=model.print_file)
        j = 0
        for prefixes_indices, suffixes_indices, window_indices, label_vec in tqdm.tqdm(training_data, leave=False, disable=True):
            j += 1
            # if j > 3:
            #     break
            optimizer.zero_grad()
            output = model(prefixes_indices, suffixes_indices, window_indices)
            label_vec = label_vec.to(device)
            loss = model.criterion(output, label_vec)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            del window_indices
            del prefixes_indices
            del suffixes_indices
            del label_vec
        torch.cuda.empty_cache()
        epoch_loss = running_loss / len(training_data)
        print(f"avarage loss in epoch: {epoch_loss} ", file=model.print_file)
        loss_list.append(epoch_loss)
        accuracy = calculate_accuracy_on_dev(dev_data, model)
        accuracy_list.append(accuracy)
        print_predictions_on_test(model, test_data, i)
    return loss_list, accuracy_list


def print_predictions_on_test(model, test_data, i):
    predictions = []
    j = 1
    for prefixes_indices, suffixes_indices, window_indices in tqdm.tqdm(test_data, leave=False, disable=True):
        j += 1
        # if j > 3:
        #     break
        output = model(prefixes_indices, suffixes_indices, window_indices)
        predictions.extend((torch.argmax(output, dim=1)).tolist())
    print_file = str(i) + "_test1_" + model.task + "_" + model.print_file.name
    with open(print_file, "a") as output:
        print(predictions, file=output)


def calculate_accuracy_on_dev(dev_data, model):
    predictions = []
    true_labels = []
    j = 0
    for prefixes_indices, suffixes_indices, window_indices, label_vec in tqdm.tqdm(dev_data, leave=False, disable=True):
        j += 1
        # if j > 3:
        #     break
        output = model(prefixes_indices, suffixes_indices, window_indices)
        true_labels.extend((torch.argmax(label_vec, dim=1)).tolist())
        predictions.extend((torch.argmax(output, dim=1)).tolist())
    if model.task == "ner":
        predictions, true_labels = model.get_ner_filtered_preds_and_labels(predictions, true_labels)
    accuracy = accuracy_score(true_labels, predictions)
    print(f"Accuracy: {accuracy}", file=model.print_file)
    conf_matrix = confusion_matrix(true_labels, predictions)
    print("Confusion Matrix:", file=model.print_file)
    print(conf_matrix, file=model.print_file)
    return accuracy


def without_pre_trained_vecs():
    files = [("pos/train", "pos/dev", "pos/test"), ("ner/train", "ner/dev", "ner/test")]
    now = datetime.now()
    hidden_dim = 20
    lr = 0.001
    epochs = 100
    output_file = f"tagger1_hidim_{hidden_dim}_lr_{lr}_epochs_{epochs}_{now}.txt"

    with open(output_file, "a") as print_file:
        for train_file, dev_file, test_file in files:
            train_labeled_sentences = parse_labeled_data(train_file)
            vocabulary, labels = extract_vocabulary_and_labels(train_labeled_sentences)
            prefixes, suffixes = get_unique_prefixes_and_suffixes(vocabulary)
            dev_labeled_sentences = parse_labeled_data(dev_file)
            test_unlabeled_sentences = parse_unlabeled_data(test_file)
            vocab = list(PADDING_WORDS) + [UNK] + vocabulary
            window_tagger = WindowTaggerWithSuffixPrefix(
                vocab,
                labels,
                hidden_dim,
                lr,
                train_file[0:3],
                print_file,
                test_unlabeled_sentences,
                None,
                prefixes,
                suffixes)
            run_train_and_eval(dev_labeled_sentences, epochs, lr, print_file, test_unlabeled_sentences,
                               train_labeled_sentences, window_tagger, 32)


def run_train_and_eval(dev_labeled_sentences, epochs, lr, print_file, test_unlabeled_sentences, train_labeled_sentences,
                       window_tagger, batch_size):
    train_dataloader = get_labeled_data_loader(train_labeled_sentences, window_tagger, batch_size)
    dev_dataloader = get_labeled_data_loader(dev_labeled_sentences, window_tagger, 8)
    test_dataloader = get_unlabeled_data_loader(test_unlabeled_sentences, window_tagger, 8)
    window_tagger.to(device)
    loss_list, accuracy_list = train(window_tagger, train_dataloader, dev_dataloader, test_dataloader, epochs,
                                     lr)
    print(f"loss list: {loss_list}", file=print_file)
    print(f"accuracy list: {accuracy_list}", file=print_file)
    show_graph(loss_list, 'Loss')
    show_graph(accuracy_list, 'Accuracy')


def with_pre_trained_vecs():
    files = [("pos/train", "pos/dev", "pos/test"), ("ner/train", "ner/dev", "ner/test")]
    now = datetime.now()
    hidden_dim = 20
    lr = 0.001
    epochs = 100
    embedding_dim = 50
    words_file_name = r"vocab.txt"
    vec_file_name = r"wordVectors.txt"
    vocab_pre_trained = Path(words_file_name).read_text().split()
    vecs_pre_trained = np.loadtxt(vec_file_name)
    output_file = f"tagger2_output_hid_dim_{hidden_dim}_learning_rate_{lr}_epochs_{epochs}_{now}.txt"

    with open(output_file, "a") as print_file:
        for train_file, dev_file, test_file in files:
            train_labeled_sentences = parse_labeled_data(train_file)
            vocabulary, labels = extract_vocabulary_and_labels(train_labeled_sentences)
            prefixes, suffixes = get_unique_prefixes_and_suffixes(vocabulary)
            dev_labeled_sentences = parse_labeled_data(dev_file)
            test_unlabeled_sentences = parse_unlabeled_data(test_file)
            window_tagger = get_window_tagger_with_pre_trained_embeddings(
                embedding_dim,
                hidden_dim,
                labels,
                lr,
                print_file,
                test_unlabeled_sentences,
                train_file,
                vecs_pre_trained,
                vocab_pre_trained,
                vocabulary, suffixes,
                prefixes
            )
            run_train_and_eval(dev_labeled_sentences, epochs, lr, print_file, test_unlabeled_sentences,
                               train_labeled_sentences, window_tagger, 32)


def get_unique_prefixes_and_suffixes(vocabulary):
    suffixes = []
    prefixes = []
    for word in vocabulary:
        if len(word) < 3:
            suffixes.append(word)
            prefixes.append(word)
        else:
            prefixes.append(word[0:3])
            suffixes.append(word[-3:])
    return list(set(prefixes)), list(set(suffixes))


def main():
    without_pre_trained_vecs()
    with_pre_trained_vecs()


def show_graph(val_list, metric):
    plt.plot(val_list, label=('Training %s' % metric))
    plt.xlabel('Epoch')
    plt.ylabel('%s' % metric)
    plt.title('%s over Epochs' % metric)
    plt.legend()
    plt.show()


def get_labeled_data_loader(train_labeled_sentences, window_tagger, batch_size=1):
    prefixes_indices, suffixes_indices, window_indices, y = window_tagger.get_data_in_x_y_format(
        train_labeled_sentences)
    labeled_dataset = TensorDataset(prefixes_indices, suffixes_indices, window_indices, y)
    dataloader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def get_unlabeled_data_loader(unlabeled_sentences, window_tagger, batch_size=1):
    window_indices = torch.empty((0, window_tagger.input_size), dtype=torch.int32)
    prefixes_indices = torch.empty((0, window_tagger.input_size), dtype=torch.int32)
    suffixes_indices = torch.empty((0, window_tagger.input_size), dtype=torch.int32)
    j = 0
    for sentence in unlabeled_sentences:
        j += 1
        # if j > 3:
        #     break
        sentence_windows_word_indices = torch.tensor(window_tagger.get_windows_word_indices_for_sentence(sentence),
                                                     dtype=torch.int32)
        sentence_prefixes_indices, sentence_suffix_indices = window_tagger.get_prefix_and_suffix_indices_for_sentence(
            sentence_windows_word_indices)
        prefixes_indices = torch.cat((prefixes_indices, torch.tensor(sentence_prefixes_indices, dtype=torch.int32)),
                                     dim=0)
        suffixes_indices = torch.cat((suffixes_indices, torch.tensor(sentence_suffix_indices, dtype=torch.int32)),
                                     dim=0)
        window_indices = torch.cat((window_indices, sentence_windows_word_indices), dim=0)
    dataset = TensorDataset(prefixes_indices, suffixes_indices, window_indices)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader


def get_window_tagger_with_pre_trained_embeddings(
        embedding_dim,
        hidden_dim,
        labels,
        lr,
        print_file,
        test_unlabeled_sentences,
        train_file,
        vecs_pre_trained,
        vocab_pre_trained,
        vocabulary, suffixes,
        prefixes
):
    vocab_to_add_embedding_vectors_lower_cased = []
    vocab_to_add_new_vectors = []
    for word in vocabulary:
        if word not in vocab_pre_trained:
            if str.lower(word) in vocab_pre_trained:
                vocab_to_add_embedding_vectors_lower_cased.append(word)
            else:
                vocab_to_add_new_vectors.append(word)
    vocab_to_add_new_vectors = vocab_to_add_new_vectors + list(PADDING_WORDS) + [UNK]
    new_words_vectors = torch.tensor(
        glorot_init(len(vocab_to_add_new_vectors), embedding_dim), requires_grad=True
    )
    embedding_vectors_for_upper_cased = torch.tensor([
        vecs_pre_trained[vocab_pre_trained.index(str.lower(word))]
        for word in vocab_to_add_embedding_vectors_lower_cased])
    E = torch.cat([new_words_vectors,
                   embedding_vectors_for_upper_cased,
                   torch.tensor(vecs_pre_trained)]).float()
    full_vocab = (
            vocab_to_add_new_vectors
            + vocab_to_add_embedding_vectors_lower_cased
            + vocab_pre_trained
    )
    window_tagger = WindowTaggerWithSuffixPrefix(
        full_vocab,
        labels,
        hidden_dim,
        lr,
        train_file[0:3],
        print_file,
        test_unlabeled_sentences,
        E,
        prefixes,
        suffixes
    )
    return window_tagger


if __name__ == "__main__":
    main()
