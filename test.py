import json
from pprint import pprint

import nltk
import pandas as pd
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
nltk.download('omw-1.4')

import utils
import baseline_2


def test_baseline(json_path, filename):

    '''Use baseline system to produce label of each token
       of the input sentences. The output tokens and labels
       are saved in a csv file'''

    token_label_count = baseline_2.create_temp_lookupdict_fr_trainset(
        "Data/tokens_labels_train.json")
    lookup = baseline_2.create_lookupdict_fr_trainset(token_label_count)

    sentences_test_file = utils.read_file(json_path)
    # sentences_test_file = utils.read_file('Data/sentences_test.json')

    tokens = []
    labels = []
    for sid, sent in sentences_test_file.items():
        result_dict = baseline_2.baseline_2(sent, lookup)
        for token, label in result_dict.items():
            tokens.append(token)
            labels.append(label)

        # pprint(result_dict)

    print(len(tokens))
    print(len(labels))

    utils.write_to_csv(tokens, labels, filename)


def main():
    # test on test set
    json_path = 'Data/sentences_test.json'
    filename = "output_baseline_on_testset"
    test_baseline(json_path, filename)

    # test on dev set
    json_path = 'Data/sentences_dev.json'
    filename = "output_baseline_on_devset"
    test_baseline(json_path, filename)

    # test on eval set
    json_path = 'Data/sentences_eval.json'
    filename = "output_baseline_on_evalset"
    test_baseline(json_path, filename)



if __name__ == "__main__":
    main()
