import utils
import nltk
import json
from nltk.corpus import wordnet as wn
nltk.download('omw-1.4')


def getSenseBasline(doc_dict):
    for i in doc_dict:
        # print(i)
        for x in doc_dict[i]:
            try:
                doc_dict[i][x].append(''+str(wn.synsets(x)[0]).replace("Synset(","").replace(")","").replace("'",""))
            except IndexError:
                doc_dict[i][x].append("O")
    return doc_dict


def write2csv(Ids, sentence):
    output = '\n'.join('\t'.join(map(str,row)) for row in zip(Ids, sentence))
    with open('Data/sentences.txt', 'w') as f:
        f.write(output)


def write2Json(doc_dict, path):
    with open(path, 'w') as fp:
        json.dump(doc_dict, fp)


def main():
    Ids, sentence, doc_dict = utils.read_data("Data/train.txt")
    # get tokens and their labels from trainset file and
    # save them to a json file
    write2Json(doc_dict, 'Data/tokens_labels_train.json')
    write2Json(getSenseBasline(doc_dict),'Data/tokens.json')
    write2Json({Ids[i]: sentence[i] for i in range(len(Ids))}, 'Data/sentences.json')


    Ids_test, sentence_test, doc_dict_test = utils.read_data("Data/test.txt")
    # get tokens and their labels from testset file and
    # save them to a json file
    write2Json(doc_dict_test, 'Data/tokens_labels_test.json')
    write2Json(getSenseBasline(doc_dict_test),'Data/tokens_test.json')
    write2Json({Ids_test[i]: sentence_test[i] for i in range(len(Ids_test))}, 'Data/sentences_test.json')
    # print("length of train set:", len(Ids)) # 7668
    # print("length of test set:", len(Ids_test)) # 1048


    Ids_dev, sentence_dev, doc_dict_dev = utils.read_data("Data/dev.txt")
    # get tokens and their labels from devset file and
    # save them to a json file
    write2Json(doc_dict_dev, 'Data/tokens_labels_dev.json')
    write2Json({Ids_dev[i]: sentence_dev[i] for i in range(len(Ids_dev))}, 'Data/sentences_dev.json')


    Ids_eval, sentence_eval, doc_dict_eval = utils.read_data("Data/eval.txt")
    # get tokens and their labels from evalset file and
    # save them to a json file
    write2Json(doc_dict_eval, 'Data/tokens_labels_eval.json')
    write2Json({Ids_eval[i]: sentence_eval[i] for i in range(len(Ids_eval))}, 'Data/sentences_eval.json')

if __name__ == "__main__":
    main()

