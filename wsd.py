#!/usr/bin/env python

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity


def check_similarity(sentences):

    #nitialize our model and tokenizer:
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
    model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')

    # initialize dictionary: stores tokenized sentences
    token = {'input_ids': [], 'attention_mask': []}
    for sentence in sentences:
        # encode each sentence, append to dictionary
        new_token = tokenizer.encode_plus(sentence, max_length=128,
                                       truncation=True, padding='max_length',
                                       return_tensors='pt')
        token['input_ids'].append(new_token['input_ids'][0])
        token['attention_mask'].append(new_token['attention_mask'][0])

    #Reformat list of tensors to single tensor
    token['input_ids'] = torch.stack(token['input_ids'])
    token['attention_mask'] = torch.stack(token['attention_mask'])

    #Process tokens through model:
    output = model(**token)

    #The dense vector representations of text are contained within the outputs 'last_hidden_state' tensor
    embeddings = output.last_hidden_state

    #To perform this operation, we first resize our attention_mask tensor:
    att_mask = token['attention_mask']
    mask = att_mask.unsqueeze(-1).expand(embeddings.size()).float()
    mask_embeddings = embeddings * mask

    #Then we sum the remained of the embeddings along axis 1:
    summed = torch.sum(mask_embeddings, 1)

    #Then sum the number of values that must be given attention in each position of the tensor:
    summed_mask = torch.clamp(mask.sum(1), min=1e-9)
    mean_pooled = summed / summed_mask

    #Convert from PyTorch tensor to numpy array
    mean_pooled = mean_pooled.detach().numpy()

    #Calculate cosine similarity for sentence 0:
    similarities = cosine_similarity([mean_pooled[0]], mean_pooled[1:])

    return similarities


def main():
    sentences = [
            "Three years later, the coffin was still full of Jello.",
            "The fish dreamed of escaping the fishbowl and into the toilet where he saw his friend go.",
            "The person box was packed with jelly many dozens of months later.",
            "He found a leprechaun in his walnut shell."
    ]

    print(sentences[0])
    for sent, sim in zip(sentences[1:], check_similarity(sentences).tolist()[0]):
        print(sent, round(sim, 4))


if __name__ == '__main__':
    main()
