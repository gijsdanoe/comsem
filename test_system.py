"""
This script will test the accuracy of our results by comparing it to baselien
"""
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import wordnet
import spacy
import utils
sp = spacy.load('en_core_web_sm')
from nltk.corpus.reader.wordnet import WordNetError
import csv
import re
import argparse



def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", default='test', type=str, help="Dataset for testing system")
    parser.add_argument("-g", "--gloss", action="store_true", help="Use gloss only rules")
    args = parser.parse_args()
    
    return args

def match(gold, system):
    """
    This function will check the accuracy of our system by comparing the word
    sense labels with the gold labels
    
    Returns a list o 0 and 1 with 0 = mismatch and 1= match
    """
    labels = []
    for i,j in zip(gold, system):
        if i == "O" or j == "O":
            if i== j:
                labels.append(1)
            else:
                labels.append(0)
        else:
            try:
                sense1 = wordnet.synset(i).name()
                sense2 = wordnet.synset( re.sub(r"_[0-9]*", "", j)).name() #regex here is to remove the version number is gloss/example was used
            except WordNetError:
                sense1 = i
                sense2 = re.sub(r"_[0-9]*", "", j)
            if sense1 == sense2:
                labels.append(1)
            else:
                labels.append(0)
    return labels

def get_accuracy(labels):
    """
    Calculate percentages of 1 in labels
    """
    n_correct = labels.count(1)
    n_incorrect = labels.count(0)
    n_all = n_correct +n_incorrect
    accuracy = n_correct/n_all
    return accuracy*100

def filter_tokens(tokens, gold, results):
    """
    Filterout tokens that are not present in baseline
    """
    new_tokens = []
    new_sense = []
    cont = True
    while cont:
        for i in results:
            for x in results[i]:
                if len(tokens) == 0:
                    cont = False
                elif x == tokens[0]:
                    new_tokens.append(x)
                    new_sense.append(results[i][x][3])
                    tokens = tokens[1:]
    return new_tokens, new_sense

def main():
    args = create_arg_parser()
    tokens,gold, base, label = utils.read_data2(f"Output/result_baseline_on_{args.dataset}set.csv")
    if args.gloss:
        results = utils.read_file(f"Output/results_{args.dataset}.json")
    else:
        results = utils.read_file(f"Output/results_{args.dataset}_mix.json")
    new_tokens, new_sense = filter_tokens(tokens,gold,results)
    new_labels = match(gold, new_sense)
    print(len(new_tokens))
    print(get_accuracy(new_labels))
    with open('Output/system_{args.dataset}.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(zip(new_tokens, gold,base,new_sense,new_labels))


if __name__ == "__main__":
    main()
