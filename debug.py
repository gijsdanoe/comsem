from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from utils import read_file
from utils import write2Json
# from clauses import get_clauses
import nltk


def base_similarity_check(sentences):

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

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    #Encoding:
    sentence_embeddings = model.encode(sentences)

    #Calculate cosine similarity for sentence 0:
    similarities = cosine_similarity(
    [sentence_embeddings[0]],
    sentence_embeddings[1:]
    )

    return similarities.tolist()[0]


def main():
    sentences = [
    "A girl is styling her hair",

    ]

    print(base_similarity_check(sentences))


if __name__ == '__main__':
     main()
