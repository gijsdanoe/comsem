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


def get_predicted_label(file, n):
    labels = []
    with open(file, 'r') as infile:
        for line in infile:
            line = line.split(",")
            labels.append(line[n])

    return labels


def evaluate(Y_gold, Y_pred):

    total_count = len(Y_gold)
    # print(total_count)
    correct_pred = 0

    for g, p in zip(Y_gold, Y_pred):
        if g == p:
            correct_pred = correct_pred + 1

    acc = round((correct_pred / total_count * 100), 2)

    return acc


def main():
    test_file = DATA_DIR + "tokens_test.json"
    predict_file = OUTPUT_DIR + "output.csv"

    Y_gold = get_labels(test_file, 0)
    # print(Y_gold)
    Y_pred_baseline = get_labels(test_file, 1)
    # Y_pred_baseline = get_predicted_label(predict_file, 2)
    # print(Y_pred_baseline)
    # Y_pred_system = get_predicted_label(predict_file, 3)
    # print(Y_pred_system)

    print(f'baseline accuracy: {evaluate(Y_gold, Y_pred_baseline)}')
    # print(f'system accuracy: {evaluate(Y_gold, Y_pred_system)}')



if __name__ == '__main__':
    main()
