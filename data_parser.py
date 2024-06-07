def parse_labeled_data(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

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


def parse_unlabeled_data(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

    sentences = []
    current_sentence = []

    for line in lines:
        line = line.strip()
        if line:
            word = line
            current_sentence.append(word)
        else:
            if current_sentence:
                sentences.append(current_sentence)
                current_sentence = []

    if current_sentence:
        sentences.append(current_sentence)

    return sentences


def extract_vocabulary_and_labels(parsed_sentences):
    words = set()
    labels = set()

    for parsed_sentence in parsed_sentences:
        for a, b in parsed_sentence:
            words.add(a)
            labels.add(b)

    return list(words), list(labels)


def main():
    # Example usage
    for labeled_file, unlabeled_file in [("ner/dev", "ner/test"), ("pos/dev", "pos/test")]:
        labeled_sentences = parse_labeled_data(labeled_file)
        unlabeled_sentences = parse_unlabeled_data(unlabeled_file)
        vocabulary, labels = extract_vocabulary_and_labels(labeled_sentences)

        print("Labeled Sentences:")
        for sentence in labeled_sentences:
            print(sentence)

        print("\nVocabulary:")
        print(vocabulary)

        print("\nLabels:")
        print(labels)

        print("\nUnlabeled Sentences:")
        for sentence in unlabeled_sentences:
            print(sentence)


if __name__ == "__main__":
    main()
