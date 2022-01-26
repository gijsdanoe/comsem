import json
from pprint import pprint
import argparse

import nltk
import pandas as pd
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
nltk.download('omw-1.4')

import utils


# def create_arg_parser():
#     """
#     Description:

#     This method is an arg parser

#     Return

#     This method returns a map with commandline parameters taken from the user
#     """
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-ts", "--testset",type=str, default='rbf',
#                         help="Input kernel")
#     parser.add_argument("-n1", "--n1", default=1, type=int,
#                         help="Ngram Start point")
#     parser.add_argument("-n2", "--n2", default=1, type=int,
#                         help="Ngram End point")
#     parser.add_argument("-t", "--tfidf", action="store_true",
#                         help="Use the TF-IDF vectorizer instead of CountVectorizer")
#     args = parser.parse_args()
#     return args


def create_tokens_labels_filtered(json_path, filename):


    sid_tokens_labels = utils.read_file(json_path)
    # print(len(sid_tokens_labels.items()))
    tokens_before_filtered = []
    labels_before_filtered = []
    tokens_after_filtered = []
    labels_after_filtered = []

    for tokens_labels in sid_tokens_labels.values():


        tagged_tokens = nltk.pos_tag(tokens_labels.keys())

        for token, pos in tagged_tokens:
            tokens_before_filtered.append(token)

            label = tokens_labels[token][0]
            labels_before_filtered.append(label)

            if utils.skip_token(token, pos):
                continue

            tokens_after_filtered.append(token)
            labels_after_filtered.append(label)

    print("before filtered", len(tokens_before_filtered))
    print("before filtered", len(labels_before_filtered))

    print("after filtered", len(tokens_after_filtered))
    print("after filtered", len(labels_after_filtered))
    utils.write_to_csv(tokens_after_filtered, labels_after_filtered, filename)


def main():
    json_path = 'Data/tokens_labels_test.json'
    filename = "tokens_labels_test_filtered_for_eval"
    create_tokens_labels_filtered(json_path, filename)




if __name__ == "__main__":
    main()
