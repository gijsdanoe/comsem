from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from utils import read_file
from utils import write2Json
from clauses import get_clauses
import nltk


def base_similarity_check(sentences):

    '''Calculate the consine-similarity of Bert embeddings
       of sentences using BERT sentencetransforms base.
       Return a list of scores of sentences'''

    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    #Encoding:
    sentence_embeddings = model.encode(sentences)

    #Calculate cosine similarity for sentence 0:
    similarities = cosine_similarity(
    [sentence_embeddings[0]],
    sentence_embeddings[1:]
    )

    return similarities.tolist()[0]


def mini_similarity_check(sentences):
    '''Calculate the consine-similarity of Bert embeddings
       of sentences using BERT sentencetransforms mini.
       Return a list of scores of sentences'''

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    #Encoding:
    sentence_embeddings = model.encode(sentences)

    #Calculate cosine similarity for sentence 0:
    similarities = cosine_similarity(
    [sentence_embeddings[0]],
    sentence_embeddings[1:]
    )

    return similarities.tolist()[0]

def classify(sentence_json, token_json):
    sentences = []
    for doc in sentence_json:
        
        for token in token_json[doc]:
            sentences = []
            sentences.append(sentence_json[doc])
            print(sentence_json[doc])
           
            if token_json[doc][token][1] == "O":
                token_json[doc][token].append("O")
            else:
                definitions = get_clauses(token)
                print(token)
                sentences.extend(list(definitions.values()))
                sentence_match = get_match(sentences)
                print(sentence_match)
                sense = get_key(sentence_match,definitions)
                print(sense)
                token_json[doc][token].append(sense)
    write2Json(token_json, "Data/results.json")
                

def get_match(sentences):
    best_sent = ""
    score = 0.0
    for sent, sim in zip(sentences[1:], base_similarity_check(sentences).tolist()[0]):
        if round(sim, 4) > score:
 
            best_sent = sent
            score = round(sim, 4) 
    return best_sent

def get_key(val, my_dict):
    for key, value in my_dict.items():
         if val == value:
             return key
 
    return "key doesn't exist"

def main():
    sentence_json = read_file("Data/sentences.json")
    token_json = read_file("Data/tokens.json")
    classify(sentence_json,token_json)
    
    sentences = [
          "I'm a tennis player.",
         "the arm of the record player", 
         "a person who participates in or is skilled at some game", 
         "someone who plays a musical instrument (as a profession)"
        "any instrument or instrumentality used in fighting or hunting",
        "someone who takes part in an activity",
        "a theatrical performer"
    ]

    # print(similarity_check(sentences))
    for sent, sim in zip(sentences[1:], base_similarity_check(sentences)):
        print(sent, round(sim, 4))


if __name__ == '__main__':
     main()
