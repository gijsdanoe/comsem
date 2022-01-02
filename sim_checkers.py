from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def base_similarity_check(sentences):

    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    #Encoding:
    sentence_embeddings = model.encode(sentences)

    #Calculate cosine similarity for sentence 0:
    similarities = cosine_similarity(
    [sentence_embeddings[0]],
    sentence_embeddings[1:]
    )

    return similarities


def mini_similarity_check(sentences):

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    #Encoding:
    sentence_embeddings = model.encode(sentences)

    #Calculate cosine similarity for sentence 0:
    similarities = cosine_similarity(
    [sentence_embeddings[0]],
    sentence_embeddings[1:]
    )

    return similarities


def main():
    sentences = [
        "Three years later, the coffin was still full of Jello.",
        "The fish dreamed of escaping the fishbowl and into the toilet where he saw his friend go.",
        "The person box was packed with jelly many dozens of months later.",
        "He found a leprechaun in his walnut shell."
    ]

    # print(similarity_check(sentences))
    for sent, sim in zip(sentences[1:], base_similarity_check(sentences).tolist()[0]):
        print(sent, round(sim, 4))


if __name__ == '__main__':
     main()
