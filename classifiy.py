import utils
import nltk
import json
from nltk.corpus import wordnet as wn
nltk.download('omw-1.4')

def getSenseBasline(doc_dict):
    for i in doc_dict:
        print(i)
        for x in doc_dict[i]:
            try:
                doc_dict[i][x].append(''+str(wn.synsets(x)[0]).replace("Synset(","").replace(")","").replace("'",""))
            except IndexError:
                doc_dict[i][x].append("0")
    return doc_dict

def write2csv(Ids, sentence):
    output = '\n'.join('\t'.join(map(str,row)) for row in zip(Ids, sentence))
    with open('Data/sentences.txt', 'w') as f:
        f.write(output)

def write2Json(doc_dict):
    with open('Data/result.json', 'w') as fp:
        json.dump(doc_dict, fp)

def main():
    Ids, sentence, doc_dict = utils.read_data("Data/train.txt")
    write2Json(getSenseBasline(doc_dict))
    write2csv(Ids, sentence)
if __name__ == "__main__":
    main()
    
    
