import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from datetime import datetime

from torch import optim

from data_parser import parse_labeled_data, extract_vocabulary_and_labels, parse_unlabeled_data

PADDING_WORDS = ('word_minus_2', 'word_minus_1', 'word_plus_1', 'word_plus_2')
UNK = "UNK"


def glorot_init(first_dim, second_dim):
    epsilon = np.sqrt(6 / (first_dim + second_dim))
    return np.random.uniform(-epsilon, epsilon, (first_dim, second_dim))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


class WindowTagger(nn.Module):
    def __init__(self, vocabulary, labels, hidden_dim, learning_rate, task, print_file, test_data, embeddings,
                 embedding_dim=50, window_shape=(2, 2),
                 padding_words=PADDING_WORDS):
        super(WindowTagger, self).__init__()
        self.padding_words = padding_words
        self.unknown_word = UNK
        self.vocabulary = {word: index for index, word in enumerate(vocabulary)}
        self.labels = labels
        self.window_shape = window_shape
        surrounding_window_length = window_shape[0] + window_shape[1]
        assert surrounding_window_length == len(self.padding_words)
        if embeddings is None:
            self.embedding = nn.Embedding(len(self.vocabulary), embedding_dim)
        else:
            self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=False)
        self.fc1 = nn.Linear(embedding_dim * 5, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, len(labels))
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

    def train_it(self, iterations_num, train_labeled_sentences, dev_labeled_sentences, optimizer):
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
                    optimizer.step()
                    optimizer.zero_grad()
            self.print_accuracy_on_dev(dev_labeled_sentences)
            self.print_prediction_on_test(iteration)
        losses = [l.cpu().detach().numpy() for l in loss_list]
        plt.plot(range(len(losses)), losses, "g")
        plt.xlabel("forward pass")
        plt.ylabel("cross entropy loss")
        plt.grid(True)
        plt.show()

    def get_gold(self, labeled_sentence, word_index_in_sentence):
        y = (torch.zeros(1, len(self.labels), dtype=torch.float64)).to(device)
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
            predictions, true_labels = self.get_ner_filtered_preds_and_labels(predictions, true_labels)
        accuracy = accuracy_score(true_labels, predictions)
        self.accuracy_list.append(accuracy)
        print(f"Accuracy: {accuracy}", file=self.print_file)
        conf_matrix = confusion_matrix(true_labels, predictions)
        print("Confusion Matrix:", file=self.print_file)
        print(conf_matrix, file=self.print_file)

    def get_ner_filtered_preds_and_labels(self, predictions, true_labels):
        filtered_predictions = []
        filtered_true_labels = []
        for prediction, true_label in zip(predictions, true_labels):
            if not (prediction == self.labels.index('O') and prediction == true_label):
                filtered_predictions.append(prediction)
                filtered_true_labels.append(true_label)
        true_labels = filtered_true_labels
        predictions = filtered_predictions
        return predictions, true_labels

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

    def forwards(self, window_word_indices):
        input = torch.tensor(window_word_indices).reshape(1, 5).to(device)
        embeds = self.embedding(input)  # Shape: (batch_size, 5, embedding_dim)
        # Flatten the embeddings
        embeds = embeds.view(embeds.size(0), -1)  # Shape: (batch_size, 5 * embedding_dim)
        out = self.fc1(embeds)
        out = torch.tanh(out)
        out = self.fc2(out)
        out = torch.softmax(out, dim=1)
        return out


def main():
    use_pre_trained_embeddings = True
    files = [('ner/train', 'ner/dev', 'ner/test'), ('pos/train', 'pos/dev', 'pos/test')]
    now = datetime.now()
    hidden_dim = 20
    lr = 0.01
    epochs = 5
    embedding_dim = 50
    words_file_name = r"vocab.txt"
    vec_file_name = r"wordVectors.txt"
    vocab_pre_trained = Path(words_file_name).read_text().split()
    vecs_pre_trained = np.loadtxt(vec_file_name)
    if use_pre_trained_embeddings:
        output_file = f"tagger2_output_hid_dim_{hidden_dim}_learning_rate_{lr}_epochs_{epochs}_{now}.txt"
    else:
        output_file = f"tagger1_hidim_{hidden_dim}_lr_{lr}_epochs_{epochs}_{now}.txt"

    with open(output_file, "a") as print_file:
        for train_file, dev_file, test_file in files:
            train_labeled_sentences = parse_labeled_data(train_file)
            vocabulary, labels = extract_vocabulary_and_labels(train_labeled_sentences)
            dev_labeled_sentences = parse_labeled_data(dev_file)
            test_unlabeled_sentences = parse_unlabeled_data(test_file)
            if use_pre_trained_embeddings:
                window_tagger = get_window_tagger_with_pre_trained_embeddings(embedding_dim, hidden_dim, labels, lr, print_file,
                                                                              test_unlabeled_sentences, train_file, vecs_pre_trained, vocab_pre_trained,
                                                                              vocabulary)
            else:
                vocab = list(PADDING_WORDS) + [UNK] + vocabulary
                window_tagger = WindowTagger(vocab, labels, hidden_dim, lr, train_file[0:3], print_file,
                                             test_unlabeled_sentences, None)
            window_tagger.to(device)
            optimizer = optim.Adam(window_tagger.parameters(), lr=0.001)
            window_tagger.train_it(epochs, train_labeled_sentences, dev_labeled_sentences, optimizer)
            print(window_tagger.accuracy_list, file=print_file)


def get_window_tagger_with_pre_trained_embeddings(embedding_dim, hidden_dim, labels, lr, print_file, test_unlabeled_sentences, train_file,
                                                  vecs_pre_trained, vocab_pre_trained, vocabulary):
    vocab_to_add_embedding_vectors_lower_cased = []
    vocab_to_add_new_vectors = []
    for word in vocabulary:
        if word not in vocab_pre_trained:
            if str.lower(word) in vocab_pre_trained:
                vocab_to_add_embedding_vectors_lower_cased.append(word)
            else:
                vocab_to_add_new_vectors.append(word)
    vocab_to_add_new_vectors = vocab_to_add_new_vectors + list(
        PADDING_WORDS) + [UNK]
    new_words_vectors = torch.tensor(glorot_init(len(vocab_to_add_new_vectors), embedding_dim), requires_grad=True)
    embedding_vectors_for_upper_cased = torch.tensor(np.array(
        [vecs_pre_trained[vocab_pre_trained.index(str.lower(word))] for word in
         vocab_to_add_embedding_vectors_lower_cased]))
    E = torch.cat([new_words_vectors, embedding_vectors_for_upper_cased, torch.tensor(vecs_pre_trained)]).float()
    full_vocab = vocab_to_add_new_vectors + vocab_to_add_embedding_vectors_lower_cased + vocab_pre_trained
    window_tagger = WindowTagger(full_vocab, labels, hidden_dim, lr, train_file[0:3], print_file,
                                 test_unlabeled_sentences, E)
    return window_tagger


if __name__ == "__main__":
    main()
