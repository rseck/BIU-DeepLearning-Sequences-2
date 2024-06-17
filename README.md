# BIU-DeepLearning-Sequences-2
Task No. 2 in deep learning for sequences course by Yoav Goldberg. semester b year 23-24

all executions run ner and pos learning process. 
the code also evaluates dev set accuracy in every epoch and prints test evaluation every epoch.

part 1
to run tagger1, simply run 'python tagger1.py part_1'

part 3
to run tagger2, simply run 'python tagger1.py part_3'

part 4
to run tagger3 with pre-trained embeddings, simply run python tagger3.py with_pre_trained_vecs
to run tagger3 without pre-trained embeddings, simply run python tagger3.py without_pre_trained_vecs

part 5
For this part you'll need the tagger4.py and the utils.py files.
Additionally you'll need to install the click package to run it (a simple and common extension to argparse).
all The parameters have default values so you can run it without any arguments.
However, you need to have a ner directory and a pos directory with the relevant files in the main directory you run the code from.
Here is the command line to run the code:

```python tagger4.py --dataset <(ner/pos)> --epochs <number of epochs for training> --batch_size <batch size for training> --channels <channels for conv layer> --window_size <...> --vec_file_name <The words vector> --words_file_name <the file with the words for the vec fiel>```

As I said, All with default values