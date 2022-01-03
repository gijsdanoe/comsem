import pandas as pd
import json
import utils

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

OUTPUT_DIR = "Output/"
DATA_DIR = "Data/"


def get_labels(file):
    doc_dict = utils.read_file(file)
    for i in doc_dict:
        for x in doc_dict[i]:
            for y in doc_dict[i][x]:
                print(y)

    # return file


def get_predicted_label(file):
    pass


def evaluate(Y_test, Y_pred):
    # Save the confusion matrix of the model in png file
    cm = confusion_matrix(Y_test, Y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    plt.savefig(OUTPUT_DIR + "{}.png".format(experiment_name))

    # Obtain the accuracy score of the model
    acc = accuracy_score(Y_test, Y_pred)
    print("Final accuracy: {}".format(acc))


def main():
    file = DATA_DIR + "tokens.json"
    get_labels(file)



if __name__ == '__main__':
    main()
