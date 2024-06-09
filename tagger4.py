import torch
from torch.nn import Module, Conv1d, MaxPool1d, Linear
from data_parser import parse_labeled_data, parse_unlabeled_data, extract_vocabulary_and_labels


class ConvBaseSubWordModel(Module):
    def __init__(self, num_of_characters: int, embeddings):
        super(ConvBaseSubWordModel, self).__init__()
        self.pool = MaxPool1d(3)
        self.conv = Conv1d(10, 30, 3)
        self.lin = Linear(30, num_of_characters)
        self.embeddings = embeddings

    def forward(self, word):
        x = self.conv(word)
        x = self.pool(x)
        x = torch.stack([x, self.embeddings(word)])
        x = self.lin(x)
        return x


def train():
    model = ConvBaseSubWordModel(10, torch.rand(10))
    model.train()
    optimizer = torch.optim.Adam(model.parameters())
    for i in range(100):
        word = torch.rand(10)
        target = torch.rand(10)
        optimizer.zero_grad()
        output = model(word)
        loss = torch.nn.functional.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        print(loss)


def main():
    files = [("ner/train", "ner/dev", "ner/test"), ("pos/train", "pos/dev", "pos/test")]
    for train_file, dev_file, test_file in files:
        train_labeled_sentences = parse_labeled_data(train_file)
        vocabulary, labels = extract_vocabulary_and_labels(train_labeled_sentences)
        dev_labeled_sentences = parse_labeled_data(dev_file)
        test_unlabeled_sentences = parse_unlabeled_data(test_file)
