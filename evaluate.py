import json
from pprint import pprint
import csv

import nltk
import pandas as pd
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
nltk.download('omw-1.4')

import utils
import baseline_2


OUTPUT_DIR = 'Output/'


def write_result_to_csv(c1, c2, c3, c4, filename):
    df = pd.DataFrame()
    df['Token'] = c1
    df['Labels gold'] = c2
    df['Labels baseline'] = c3
    df['Matches'] = c4
    df.to_csv(OUTPUT_DIR + filename + ".csv", index=False)


def evaluate(test_gold_filepath, result_filepath):

    tokens = []
    labels_gold = []
    labels_baseline = []
    matches = []
    with open(test_gold_filepath, 'r') as f1:
        reader = csv.reader(f1, delimiter=",")
        for line in reader:
            # print(line[1])
            tokens.append(line[0])
            labels_gold.append(line[1])

    with open(result_filepath, 'r') as f2:
        reader = csv.reader(f2, delimiter=",")
        for line in reader:
            # print(line[1])
            # FIXME: In the above loop you also appended a token?
            # tokens.append(line[0])
            labels_baseline.append(line[1])

    count_match = 0
    for i, j in zip(labels_gold, labels_baseline):
        if i == j:
            matches.append("1")
            count_match = count_match + 1
        else:
            matches.append("0")


    print(len(tokens), len(labels_gold), len(labels_baseline), len(matches))
    write_result_to_csv(tokens, labels_gold, labels_baseline, matches, "results_baseline")

    print("Accuracy baseline", count_match / len(labels_gold) * 100)
    print(count_match)


def main():
    test_gold_filepath = "Output/tokens_labels_test_filtered_for_eval.csv"
    result_filepath = "Output/result_baseline_on_testset.csv"
    print(evaluate(test_gold_filepath, result_filepath))



if __name__ == "__main__":
    main()
