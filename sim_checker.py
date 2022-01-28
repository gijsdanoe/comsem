import nltk
import spacy
import re
import argparse
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from utils import read_file
from utils import write2Json
from utils import write2CsvList
from clauses import get_wn_definition
from clauses import get_wn_definition_examples
from sim_checker import check_similarity
from transformers import AutoTokenizer, AutoModel
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nlp = spacy.load("en_core_web_sm")


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", default='test', type=str, help="Dataset for testing system")
    parser.add_argument("-g", "--gloss", action="store_true", help="Use gloss only rules")
    args = parser.parse_args()
    
    return args

def base_similarity_check(sentences, model):
    """
     Calculate the similarity between the first item in 'sentences' and the rest using embeddings from 'model'
     
     Returns a list of cosine_similarities
    """
    #Encoding:
    sentence_embeddings = model.encode(sentences)

    #Calculate cosine similarity for sentence 0:
    similarities = cosine_similarity(
    [sentence_embeddings[0]],
    sentence_embeddings[1:]
    )

    return similarities.tolist()[0]


def get_wordsense_evaluation(sentence_json, token_json, gloss_only, dataset):
    """
     This function takes a dictionary of ids:sentences and a nested dictionary of tokens and their label and POS information (ids:tokens:[Gold,POS,Baseline]) as input
     Using the sentences it identify the most similair word sense for each tokens and add it to 'token_json'
     
     This function returns a modified token_json (ids:tokens:[Gold,POS,Baseline, System])
    """
    sentences = []
    count = 0
    
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
    model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
    
    base_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    
    for doc in sentence_json:
        print(sentence_json[doc])#sentence
        for token in token_json[doc]:
            sentences = []
            sentences.append(sentence_json[doc])
            try:
                if gloss_only:
                    definitions = get_wn_definition(token,token_json[doc][token][1])
                else:
                    definitions = get_wn_definition_examples(token,token_json[doc][token][1])
                sentences.extend(list(definitions.values()))
                sentence_match, score = get_match(sentences, base_model)
                if score < 0.3:
                    #get 1st sense of the word
                    sense = get_defaut(definitions)
                else:
                    sense = get_key(sentence_match,definitions)
                token_json[doc][token].append(sense)
            except KeyError:
                # If the POS of the word is not supported by wordnet (for example:AUX)
                token_json[doc][token].append('O')
            except ValueError:
                # If wordsense for word + POS doesn't exit
                token_json[doc][token].append('O')
    if gloss_only:
        write2Json(token_json, f"Output/results_{dataset}.json")
    else:
        write2Json(token_json, f"Output/results_{dataset}_mix.json")

def get_wordsense(ids, sentence_list, gloss_only):
    """
     This function takes a list of ids and a list of sentences as input
     Firstly this function tokinised the sentences and identify the most similair word sense for each tokens and add it to a custom nested dictionary
     
     This function returns a nested dictionary(ids:tokens:wordsense(system))
    """
    
    sentences = []
    output = {}
    
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
    model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
    
    base_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    for Id, sentence in zip(ids, sentence_list):
        Id_dict = {}
        tokens = word_tokenize(sentence)
        sentence = sentence.replace("~","")
        sentence = sentence.replace("-","")
        sentence = sentence.replace("'m","am")
       
        tokens_pos = nlp(sentence)
        
        for token, pos in zip(tokens,tokens_pos):
            print(pos.pos_)
            sentences = []
            try:
                if gloss_only:
                    definitions = get_wn_definition(token,pos.pos_)
                else:
                    definitions = get_wn_definition_examples(token,pos.pos_)
                sentences.extend(list(definitions.values()))
                sentence_match, score = get_match(sentences, base_model)
                if score < 0.3:
                    #get 1st sense of the word
                    sense = get_defaut(definitions)
                else:
                    sense = get_key(sentence_match,definitions)
                Id_dict[token] = re.sub(r"_[0-9]*", "", sense)
            except KeyError:
                # If the POS of the word is not supported by wordnet (for example:AUX)
                Id_dict[token] ='O'
            except ValueError:
                # If the POS of the word is not supported by wordnet (for example:AUX)
                Id_dict[token] ='O'
            print(Id_dict)
        output[Id] = Id_dict
    write2Json(output, "Output/output.json")

            

def get_match(sentences, model):
    """
     This function get the gloss/example that is the most similair to our data and their similarity
     
     Return String containing gloss/example and score as float
    """
    best_sent = ""
    score = 0.0
    difference = 0.0
    for sent, sim in zip(sentences[1:], base_similarity_check(sentences, model)):
        if round(sim, 4) > score:
            difference = round(sim, 4) -  score
            best_sent = sent
            score = round(sim, 4) 
    return best_sent, score

def get_key(val, my_dict):
    """
     This function get the sense that has the gloss/example that is in the val input
    """
    
    for key, value in my_dict.items():
         if val == value:
             return key
 
    return "key doesn't exist"
    
def get_defaut(my_dict):
    """
     This function get the 1st sense in my_dict
    """
    for key, value in my_dict.items():
        return key
 
    return "key doesn't exist"

def main():
    args = create_arg_parser()
    sentence_json = read_file(f"Data/sentences_{args.dataset}.json")
    token_json = read_file(f"Data/tokens_{args.dataset}.json")
    get_wordsense_evaluation(sentence_json,token_json, args.gloss, args.dataset)
    


if __name__ == '__main__':
     main()
