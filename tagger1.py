import os
import random
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from datetime import datetime

from data_parser import parse_labeled_data, extract_vocabulary_and_labels, parse_unlabeled_data


def set_seed(seed):
    print("before fixing seed")
    print(np.random.rand(3))
    print(np.random.rand(3))
    print(torch.randn(3))
    print(torch.randn(3))
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    np.random.RandomState(seed)
    np.random.default_rng(seed)
    torch.manual_seed(seed)
    print("after fixing seed")
    print(np.random.rand(1))
    print(np.random.rand(1))
    print(torch.randn(1))
    print(torch.randn(1))
    rng = np.random.default_rng(seed)
    random_numbers = rng.random(5)
    print(random_numbers)
    random_numbers = rng.random(5)
    print(random_numbers)


def glorot_init(first_dim, second_dim):
    epsilon = np.sqrt(6 / (first_dim + second_dim))
    return np.random.uniform(-epsilon, epsilon, (first_dim, second_dim))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

class WindowTagger:
    def __init__(self, vocabulary, labels, hidden_dim, learning_rate, task, print_file, test_data, embedding_dim=50,
                 window_shape=(2, 2),
                 padding_words=('word_minus_2', 'word_minus_1', 'word_plus_1', 'word_plus_2')):
        self.padding_words = padding_words
        self.unknown_word = 'UNK'
        vocabulary_list = list(self.padding_words) + [self.unknown_word] + vocabulary
        self.vocabulary = {word: index for index, word in enumerate(vocabulary_list)}
        self.labels = labels
        self.window_shape = window_shape
        surrounding_window_length = window_shape[0] + window_shape[1]
        assert surrounding_window_length == len(padding_words)
        self.E = torch.tensor(glorot_init(len(self.vocabulary), embedding_dim), requires_grad=True).to(device)
        self.W1 = torch.tensor(glorot_init((surrounding_window_length + 1) * embedding_dim, hidden_dim),
                               requires_grad=True).to(device)
        self.B1 = torch.tensor(glorot_init(1, hidden_dim), requires_grad=True).to(device)
        self.W2 = torch.tensor(glorot_init(hidden_dim, len(labels)), requires_grad=True).to(device)
        self.B2 = torch.tensor(glorot_init(1, len(labels)), requires_grad=True).to(device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.accuracy_list = []
        self.task = task
        self.print_file = print_file
        self.test_data = test_data

    def get_word_index(self, word):
        index = self.vocabulary.get(word)
        if index is None:
            index = self.vocabulary.get(self.unknown_word)
        return index

    def get_indices_of_list_of_words(self, words):
        return [self.get_word_index(word) for word in words]

    def get_windows_word_indices_for_sentence(self, sentence):
        res = []
        for word_idx_in_sentence in range(len(sentence)):
            word_indices = []
            self.generate_start_of_sentence_indices(sentence, word_idx_in_sentence, word_indices)
            word_indices.append(self.get_word_index(sentence[word_idx_in_sentence]))
            self.generate_end_of_sentence_indices(sentence, word_idx_in_sentence, word_indices)
            res.append(word_indices)
        return res

    def generate_end_of_sentence_indices(self, sentence, word_idx_in_sentence, word_indices):
        if word_idx_in_sentence == len(sentence) - 1:
            word_indices.extend(self.get_indices_of_list_of_words(self.padding_words[2:4]))
        elif word_idx_in_sentence == len(sentence) - 2:
            word_indices.append(self.get_word_index(sentence[word_idx_in_sentence + 1]))
            word_indices.extend(self.get_indices_of_list_of_words(self.padding_words[2:3]))
        else:
            word_indices.extend(
                self.get_indices_of_list_of_words(sentence[word_idx_in_sentence + 1:word_idx_in_sentence + 3]))

    def generate_start_of_sentence_indices(self, sentence, word_idx_in_sentence, word_indices):
        if word_idx_in_sentence == 0:
            word_indices.extend(self.get_indices_of_list_of_words(self.padding_words[0:2]))
        elif word_idx_in_sentence == 1:
            word_indices.extend(self.get_indices_of_list_of_words(self.padding_words[1:2]))
            word_indices.append(self.get_word_index(sentence[0]))
        else:
            word_indices.extend(
                self.get_indices_of_list_of_words(sentence[word_idx_in_sentence - 2:word_idx_in_sentence]))

    def train(self, iterations_num, train_labeled_sentences, dev_labeled_sentences):
        loss_list = []
        i = 0
        for iteration in range(iterations_num):
            loss_in_epoch = []
            print("iteration {}\n".format(iteration), file=self.print_file)
            for labeled_sentence in train_labeled_sentences:
                # if i > 2:
                #     break
                sentence = [labeled_word[0] for labeled_word in labeled_sentence]
                sentence_windows_word_indices = self.get_windows_word_indices_for_sentence(sentence)
                for word_index_in_sentence, window_word_indices in enumerate(sentence_windows_word_indices):
                    y = self.get_gold(labeled_sentence, word_index_in_sentence)
                    layer_2_softmax = self.forwards(window_word_indices)
                    loss = self.criterion(layer_2_softmax, y)
                    i = i + 1
                    # if i > 2:
                    #     break
                    if i % 900 == 0:
                        print(f"avarage loss in epoch: {sum(loss_in_epoch) / len(loss_in_epoch)} after {i} samples",
                              file=self.print_file)
                    loss_list.append(loss)
                    loss_in_epoch.append(loss)
                    loss.backward()
                    self.back_prop()
            self.print_accuracy_on_dev(dev_labeled_sentences)
            self.print_prediction_on_test(iteration)
        losses = [l.detach().numpy() for l in loss_list]
        plt.plot(range(len(losses)), losses, "g")
        plt.xlabel("forward pass")
        plt.ylabel("cross entropy loss")
        plt.grid(True)
        plt.show()

    def get_gold(self, labeled_sentence, word_index_in_sentence):
        y = torch.zeros(1, len(self.labels), dtype=torch.float64).to(device)
        y[0][self.labels.index(labeled_sentence[word_index_in_sentence][1])] = 1
        return y

    def print_accuracy_on_dev(self, dev_labeled_sentences):
        predictions = []
        true_labels = []
        i = 0
        for labeled_sentence in dev_labeled_sentences:
            i += 1
            # if i > 6:
            #     break
            sentence = [labeled_word[0] for labeled_word in labeled_sentence]
            sentence_windows_word_indices = self.get_windows_word_indices_for_sentence(sentence)
            for word_index_in_sentence, window_word_indices in enumerate(sentence_windows_word_indices):
                true_labels.append(self.labels.index(labeled_sentence[word_index_in_sentence][1]))
                layer_2_softmax = self.forwards(window_word_indices)
                predictions.append(int(torch.argmax(layer_2_softmax, dim=1)))
        if self.task == 'ner':
            filtered_predictions = []
            filtered_true_labels = []
            for prediction, true_label in zip(predictions, true_labels):
                if not (prediction == self.labels.index('O') and prediction == true_label):
                    filtered_predictions.append(prediction)
                    filtered_true_labels.append(true_label)
            true_labels = filtered_true_labels
            predictions = filtered_predictions
        accuracy = accuracy_score(true_labels, predictions)
        self.accuracy_list.append(accuracy)
        print(f"Accuracy: {accuracy}", file=self.print_file)
        # Generate confusion matrix
        conf_matrix = confusion_matrix(true_labels, predictions)
        print("Confusion Matrix:", file=self.print_file)
        print(conf_matrix, file=self.print_file)

    def print_prediction_on_test(self, iteration):
        predictions = []
        i = 0
        for sentence_array in self.test_data:
            i += 1
            # if i>2:
            #     break
            sentence_windows_word_indices = self.get_windows_word_indices_for_sentence(sentence_array)
            for word_index_in_sentence, window_word_indices in enumerate(sentence_windows_word_indices):
                layer_2_softmax = self.forwards(window_word_indices)
                predictions.append(int(torch.argmax(layer_2_softmax, dim=1)))
        print_file = str(iteration) + "test1_" + self.task + "_" + self.print_file.name
        with open(print_file, "a") as output:
            print(predictions, file=output)

    def back_prop(self):
        W2 = self.W2
        self.W2 = W2 - self.learning_rate * W2.grad
        self.W2.retain_grad()
        B2 = self.B2
        self.B2 = B2 - self.learning_rate * B2.grad
        self.B2.retain_grad()
        W1 = self.W1
        self.W1 = W1 - self.learning_rate * W1.grad
        self.W1.retain_grad()
        B1 = self.B1
        self.B1 = B1 - self.learning_rate * B1.grad
        self.B1.retain_grad()
        E = self.E
        self.E = E - self.learning_rate * E.grad
        self.E.retain_grad()

    def forwards(self, window_word_indices):
        window_embeddings = []
        for window_word_index in window_word_indices:
            word_one_hot_vec = torch.zeros(1, len(self.vocabulary), dtype=torch.float64).to(device)
            word_one_hot_vec[0][window_word_index] = 1
            word_embedding = word_one_hot_vec @ self.E
            window_embeddings.append(word_embedding)
        concatenated_window_embeddings = torch.cat(window_embeddings, dim=1).to(device)
        layer_1_output = (concatenated_window_embeddings @ self.W1) + self.B1
        layer_1_tanh = (torch.tanh(layer_1_output))
        layer_2_output = ((layer_1_tanh @ self.W2) + self.B2)
        layer_2_softmax = torch.softmax(layer_2_output, dim=1)
        return layer_2_softmax


def main():
    set_seed(100)
    files = [('ner/train', 'ner/dev', 'ner/test'), ('pos/train', 'pos/dev', 'pos/test')]
    now = datetime.now()
    hidden_dim = 20
    lr = 0.01
    epochs = 5
    output_file = f"tagger1_hidim_{hidden_dim}_lr_{lr}_epochs_{epochs}_{now}.txt"

    with open(output_file, "a") as print_file:
        for train_file, dev_file, test_file in files:
            train_labeled_sentences = parse_labeled_data(train_file)
            vocabulary, labels = extract_vocabulary_and_labels(train_labeled_sentences)
            dev_labeled_sentences = parse_labeled_data(dev_file)
            test_unlabeled_sentences = parse_unlabeled_data(test_file)
            window_tagger = WindowTagger(vocabulary, labels, hidden_dim, lr, train_file[0:3], print_file,
                                         test_unlabeled_sentences)
            window_tagger.train(epochs, train_labeled_sentences, dev_labeled_sentences)
            print(window_tagger.accuracy_list, file=print_file)


if __name__ == "__main__":
    main()
