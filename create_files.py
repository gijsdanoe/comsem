"""
The goal of this script is to create 2 json files

The first file has the following template \{id:\{token:[gold wordsense, POS, Baseline Wordsens (we ignore this baseline for the final evaluation) ]\}\}
The second file has the following template \{id:sentence\}
"""
import utils
import nltk
import json
import spacy
import argparse
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import WordNetError
nltk.download('omw-1.4')
nlp = spacy.load("en_core_web_sm")

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", default='test', type=str, help="Dataset for testing system")
    args = parser.parse_args()
    
    return args


def getSenseBasline(doc_dict, sentences):
    """
    This function obtain the POS and baseline sense (first sense) of each token
    
    """
    for i,j in zip(doc_dict, sentences):
        j = j.replace("~","")
        j = j.replace("-","")
        j = j.replace("'m","am")
        
        pos_list = nlp(j)
   
       
        for x, y in zip(doc_dict[i],pos_list):
            
            try:
                x2 = x.replace("~","_")
                print(f"{y.pos_} --{y} -- {x} -- {x2}")
  
                doc_dict[i][x].append(y.pos_)
                sense = ''+str(wn.synsets(x2)[0]).replace("Synset(","").replace(")","").replace("'","")
                print(f"Sense --{sense} = {doc_dict[i][x][0]}")

                if doc_dict[i][x][0] != "O":
                    if wn.synset(sense).lch_similarity(wn.synset(doc_dict[i][x][0])) == wn.synset(sense).lch_similarity(wn.synset(sense)):# this is to check if senses have the same definitions
                        doc_dict[i][x].append(doc_dict[i][x][0])
                    else:
                        doc_dict[i][x].append(''+str(wn.synsets(x2)[0]).replace("Synset(","").replace(")","").replace("'",""))
                else:
                    doc_dict[i][x].append(''+str(wn.synsets(x2)[0]).replace("Synset(","").replace(")","").replace("'",""))
            except IndexError:
                doc_dict[i][x].append("O")
            except WordNetError:
                # If lch_similarity throws an error
                doc_dict[i][x].append(''+str(wn.synsets(x2)[0]).replace("Synset(","").replace(")","").replace("'",""))
                
    return doc_dict


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
    # print("train set (sentences):", len(Ids)) # 7668


    Ids_test, sentence_test, doc_dict_test = utils.read_data("Data/test.txt")
    # get tokens and their labels from testset file and
    # save them to a json file
    write2Json(doc_dict_test, 'Data/tokens_labels_test.json')
    write2Json(getSenseBasline(doc_dict_test),'Data/tokens_test.json')
    write2Json({Ids_test[i]: sentence_test[i] for i in range(len(Ids_test))}, 'Data/sentences_test.json')

    # print("test set (sentences):", len(Ids_test)) # 1048


    Ids_dev, sentence_dev, doc_dict_dev = utils.read_data("Data/dev.txt")
    # get tokens and their labels from devset file and
    # save them to a json file
    write2Json(doc_dict_dev, 'Data/tokens_labels_dev.json')
    write2Json({Ids_dev[i]: sentence_dev[i] for i in range(len(Ids_dev))}, 'Data/sentences_dev.json')
    # print("dev set (sentences):", len(Ids_dev)) # 1169

    Ids_eval, sentence_eval, doc_dict_eval = utils.read_data("Data/eval.txt")
    # get tokens and their labels from evalset file and
    # save them to a json file
    write2Json(doc_dict_eval, 'Data/tokens_labels_eval.json')
    write2Json({Ids_eval[i]: sentence_eval[i] for i in range(len(Ids_eval))}, 'Data/sentences_eval.json')
    # print("dev set (sentences):", len(Ids_eval)) # 830

if __name__ == "__main__":
    main()

