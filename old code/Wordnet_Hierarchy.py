
import open_clip
import torch
import numpy as np
# find me the NLTK wordnet hierarchy
from nltk.corpus import wordnet as wn
import nltk
nltk.download('wordnet')


# clip model
def load_clip_model():
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    return model, preprocess
  
# import wordnet hierarchy from imagent datasets

# Get a synset for a word (e.g., "cat")
synset = wn.synsets('cat')[0]

# Get hypernyms (parent synsets)
hypernyms = synset.hypernyms()

# Get hyponyms (child synsets)
hyponyms = synset.hyponyms()

def print_hypernyms(synset, level=0):
    indentation = '  ' * level
    print(indentation + synset.name().split('.')[0])
    for hypernym in synset.hypernyms():
        print_hypernyms(hypernym, level + 1)

def print_hyponyms(synset, level=0):
    indentation = '  ' * level
    print(indentation + synset.name().split('.')[0])
    for hyponym in synset.hyponyms():
        print_hyponyms(hyponym, level + 1)

# Print hypernym hierarchy for "cat"
print("Hypernym Hierarchy:")
print_hypernyms(synset)

# Print hyponym hierarchy for "cat"
print("Hyponym Hierarchy:")
print_hyponyms(synset)


