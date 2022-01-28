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
    args = create_arg_parser()
    Ids_test, sentence_test, doc_dict_test = utils.read_data(f"Data/{args.dataset}.txt")
    write2Json(getSenseBasline(doc_dict_test, sentence_test),f'Data/tokens_{args.dataset}.json')
    write2Json({Ids_test[i]: sentence_test[i] for i in range(len(Ids_test))}, f'Data/sentences_{args.dataset}.json')
    print("length of set:", len(Ids_test)) 

if __name__ == "__main__":
    main()

