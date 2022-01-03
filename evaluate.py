import pandas as pd
import json
import utils


OUTPUT_DIR = "Output/"
DATA_DIR = "Data/"


def get_labels(file, n):
    labels = []
    doc_dict = utils.read_file(file)
    for i in doc_dict:
        for x in doc_dict[i]:
            labels.append(doc_dict[i][x][n])

    return labels


def get_predicted_label(file):
    pass


def evaluate(Y_gold, Y_pred):

    total_count = len(Y_gold)
    print(total_count)
    correct_pred = 0

    for g, p in zip(Y_gold, Y_pred):
        if g == p:
            correct_pred = correct_pred + 1

    acc = round((correct_pred / total_count * 100), 2)

    return acc


def main():
    file = DATA_DIR + "tokens_test.json"
    Y_gold = get_labels(file, 0)
    Y_pred_baseline = get_labels(file, 1)
    print(f'baseline accuracy: {evaluate(Y_gold, Y_pred_baseline)}')


if __name__ == '__main__':
    main()
