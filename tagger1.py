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
DEBUG = False

def glorot_init(first_dim, second_dim):
    epsilon = torch.sqrt(torch.tensor(6.0) / (first_dim + second_dim))
    return torch.empty(first_dim, second_dim).uniform_(-epsilon.item(), epsilon.item())


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


class BaseWindowTagger(nn.Module):
    def __init__(self, vocabulary, labels, hidden_dim, learning_rate, task, print_file, test_data,
                 indices_of_embeddings_not_to_train_range, embedding_dim=50, window_shape=(2, 2),
                 padding_words=PADDING_WORDS):
        super(BaseWindowTagger, self).__init__()
        self.padding_words = padding_words
        self.unknown_word = UNK
        self.vocabulary_dict = {word: index for index, word in enumerate(vocabulary)}
        self.labels = labels
        surrounding_window_length = window_shape[0] + window_shape[1]
        self.input_size = surrounding_window_length + 1
        assert surrounding_window_length == len(self.padding_words)
        self.fc1 = nn.Linear(embedding_dim * self.input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, len(labels))
        self.criterion = torch.nn.CrossEntropyLoss().to(device)
        self.learning_rate = learning_rate
        self.accuracy_list = []
        self.task = task
        self.print_file = print_file
        self.test_data = test_data
        indices_to_freeze = torch.arange(indices_of_embeddings_not_to_train_range[0], indices_of_embeddings_not_to_train_range[1])
        self.indices_to_freeze = indices_to_freeze

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


class WindowTagger(BaseWindowTagger):
    def __init__(self, vocabulary, labels, hidden_dim, learning_rate, task, print_file, test_data, embeddings,
                 indices_of_embeddings_not_to_train_range, embedding_dim=50):
        super(WindowTagger, self).__init__(vocabulary, labels, hidden_dim, learning_rate, task, print_file, test_data,
                                           indices_of_embeddings_not_to_train_range)
        if embeddings is None:
            self.embedding = nn.Embedding(len(self.vocabulary_dict), embedding_dim)
        else:
            self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=False)

    def forward(self, window_word_indices):
        x = window_word_indices.to(device)
        embeds = self.embedding(x)  # Shape: (batch_size, 5, embedding_dim)
        # Flatten the embeddings
        embeds = embeds.view(
            embeds.size(0), -1
        )  # Shape: (batch_size, 5 * embedding_dim)
        out = self.fc1(embeds)
        out = torch.tanh(out)
        out = self.fc2(out)
        out = torch.softmax(out, dim=1)
        return out

    def get_data_in_x_y_format(self, train_labeled_sentences):
        x = torch.empty((0, self.input_size), dtype=torch.int32)
        y = torch.empty((0, len(self.labels)))
        j = 1
        for labeled_sentence in train_labeled_sentences:
            j += 1
            if DEBUG:
                if j > 3:
                    break
            sentence = [labeled_word[0] for labeled_word in labeled_sentence]
            sentence_windows_word_indices = torch.tensor(self.get_windows_word_indices_for_sentence(sentence),
                                                         dtype=torch.int32)
            x = torch.cat((x, sentence_windows_word_indices), dim=0)
            for word_index_in_sentence, window_word_indices in enumerate(sentence_windows_word_indices):
                y = torch.cat((y, self.get_gold(labeled_sentence, word_index_in_sentence)), dim=0)
        return x, y


def train(model: Module, training_data: DataLoader, dev_data: DataLoader, test_data: DataLoader, epochs: int, lr=0.001):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_list = []
    accuracy_list = []
    for i in tqdm.trange(epochs):
        running_loss = 0.0
        print("iteration {}\n".format(i), file=model.print_file)
        j = 0
        for window_indices, label_vec in tqdm.tqdm(training_data, leave=False, disable=True):
            j += 1
            if DEBUG:
                if j > 3:
                    break
            optimizer.zero_grad()
            output = model(window_indices)
            label_vec = label_vec.to(device)
            loss = model.criterion(output, label_vec)
            running_loss += loss.item()
            loss.backward()
            model.embedding.weight.grad[model.indices_to_freeze] = 0
            optimizer.step()
            del label_vec
            del output
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
    for window_indices in tqdm.tqdm(test_data, leave=False, disable=True):
        j += 1
        if DEBUG:
            if j > 3:
                break
        output = model(window_indices[0])
        predictions.extend((torch.argmax(output, dim=1)).tolist())
    print_file = str(i) + "_test1_" + model.task + "_" + model.print_file.name
    with open(print_file, "a") as output:
        print(predictions, file=output)


def calculate_accuracy_on_dev(dev_data, model):
    predictions = []
    true_labels = []
    j = 0
    for window_indices, label_vec in tqdm.tqdm(dev_data, leave=False, disable=True):
        j += 1
        if DEBUG:
            if j > 3:
                break
        output = model(window_indices)
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


def task_1():
    files = [("pos/train", "pos/dev", "pos/test"), ("ner/train", "ner/dev", "ner/test")]
    now = datetime.now()
    hidden_dim = 20
    lr = 0.001
    epochs = 3
    output_file = f"tagger1_hidim_{hidden_dim}_lr_{lr}_epochs_{epochs}_{now}.txt"

    with open(output_file, "a") as print_file:
        for train_file, dev_file, test_file in files:
            train_labeled_sentences = parse_labeled_data(train_file)
            vocabulary, labels = extract_vocabulary_and_labels(train_labeled_sentences)
            dev_labeled_sentences = parse_labeled_data(dev_file)
            test_unlabeled_sentences = parse_unlabeled_data(test_file)
            vocab = list(PADDING_WORDS) + [UNK] + vocabulary
            window_tagger = WindowTagger(vocab, labels, hidden_dim, lr, train_file[0:3], print_file,
                                         test_unlabeled_sentences, None, (0, 0))
            run_train_and_eval(dev_labeled_sentences, epochs, lr, print_file, test_unlabeled_sentences,
                               train_labeled_sentences, window_tagger, 8)


def run_train_and_eval(dev_labeled_sentences, epochs, lr, print_file, test_unlabeled_sentences, train_labeled_sentences,
                       window_tagger, batch_size):
    train_dataloader = get_labeled_data_loader(train_labeled_sentences, window_tagger, 8)
    dev_dataloader = get_labeled_data_loader(dev_labeled_sentences, window_tagger, 8)
    test_dataloader = get_unlabeled_data_loader(test_unlabeled_sentences, window_tagger, 8)
    window_tagger.to(device)
    loss_list, accuracy_list = train(window_tagger, train_dataloader, dev_dataloader, test_dataloader, epochs,
                                     lr)
    print(f"loss list: {loss_list}", file=print_file)
    print(f"accuracy list: {accuracy_list}", file=print_file)
    show_graph(loss_list, 'Loss')
    show_graph(accuracy_list, 'Accuracy')


def task_2():
    files = [("pos/train", "pos/dev", "pos/test"), ("ner/train", "ner/dev", "ner/test")]
    now = datetime.now()
    hidden_dim = 20
    lr = 0.001
    epochs = 3
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
            dev_labeled_sentences = parse_labeled_data(dev_file)
            test_unlabeled_sentences = parse_unlabeled_data(test_file)
            full_vocab, E, indices_not_to_train = get_full_vocabulary_and_embeddings(embedding_dim, vecs_pre_trained,
                                                                                     vocab_pre_trained, vocabulary)
            window_tagger = WindowTagger(full_vocab, labels, hidden_dim, lr, train_file[0:3], print_file,
                                         test_unlabeled_sentences, E, indices_not_to_train)
            run_train_and_eval(dev_labeled_sentences, epochs, lr, print_file, test_unlabeled_sentences,
                               train_labeled_sentences, window_tagger, 8)


def main():
    task_1()
    task_2()


def show_graph(val_list, metric):
    plt.plot(val_list, label=('Training %s' % metric))
    plt.xlabel('Epoch')
    plt.ylabel('%s' % metric)
    plt.title('%s over Epochs' % metric)
    plt.legend()
    plt.show()


def get_labeled_data_loader(train_labeled_sentences, window_tagger, batch_size=1):
    x, y = window_tagger.get_data_in_x_y_format(train_labeled_sentences)
    labeled_dataset = TensorDataset(x, y)
    dataloader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def get_unlabeled_data_loader(unlabeled_sentences, window_tagger, batch_size=1):
    x = torch.empty((0, window_tagger.input_size), dtype=torch.int32)
    for sentence in unlabeled_sentences:
        sentence_windows_word_indices = torch.tensor(window_tagger.get_windows_word_indices_for_sentence(sentence),
                                                     dtype=torch.int32)
        x = torch.cat((x, sentence_windows_word_indices), dim=0)
    dataset = TensorDataset(x)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader


def get_full_vocabulary_and_embeddings(embedding_dim, vecs_pre_trained, vocab_pre_trained, vocabulary):
    vocab_to_add_embedding_vectors_lower_cased = []
    vocab_to_add_new_vectors = []
    for word in vocabulary:
        if word not in vocab_pre_trained:
            if str.lower(word) in vocab_pre_trained:
                vocab_to_add_embedding_vectors_lower_cased.append(word)
            else:
                vocab_to_add_new_vectors.append(word)
    vocab_to_add_new_vectors = vocab_to_add_new_vectors + list(PADDING_WORDS) + [UNK]
    new_words_vectors = glorot_init(len(vocab_to_add_new_vectors), embedding_dim).clone().detach().requires_grad_(True)
    embedding_vectors_for_upper_cased = torch.tensor(np.array(
        [vecs_pre_trained[vocab_pre_trained.index(str.lower(word))] for word in
         vocab_to_add_embedding_vectors_lower_cased]))
    E = torch.cat([new_words_vectors,
                   embedding_vectors_for_upper_cased,
                   torch.tensor(vecs_pre_trained)]).float()
    full_vocab = (vocab_to_add_new_vectors
                  + vocab_to_add_embedding_vectors_lower_cased
                  + vocab_pre_trained)
    indices_not_to_train = (
        len(vocab_to_add_new_vectors) + len(vocab_to_add_embedding_vectors_lower_cased), len(full_vocab))
    return full_vocab, E, indices_not_to_train


if __name__ == "__main__":
    main()
