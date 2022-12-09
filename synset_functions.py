import re
import numpy as np
import nltk
#import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from autocorrect import Speller
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize


#Synset Functions
def convert_tag(tag):
    """Convert the tag given by nltk.pos_tag to the tag used by wordnet.synsets"""
    
    tag_dict = {'N': 'n', 'J': 'a', 'R': 'r', 'V': 'v'}
    #simplifying this to work with bron_ic
    # tag_dict = {'N': 'n', 'J': 'n', 'R': 'v', 'V': 'v'}
    try:
        return tag_dict[tag[0]]
    except KeyError:
        return None




def doc_to_synsets(doc):
    """
    Returns a list of synsets in document.

    Tokenizes and tags the words in the document doc.
    Then finds the first synset for each word/tag combination.
    If a synset is not found for that combination it is skipped.

    Args:
        doc: string to be converted

    Returns:
        list of synsets

    Example:
        doc_to_synsets('Fish are nvqjp friends.')
        Out: [Synset('fish.n.01'), Synset('be.v.01'), Synset('friend.n.01')]
    """
    
    words_nltk =  nltk.pos_tag(word_tokenize(doc))
    words = [(w[0],convert_tag(w[1])) for w in words_nltk]
    words_synset_list = [wn.synsets(w[0],pos=w[1]) for w in words]
    #taking the first synset here won't always be the most accurate.
    result = [w[0] for w in words_synset_list if bool(w)]
    result = [word for word in result if bool(re.search(str(word)[-6],'nvar'))]
    #a lot of extra work here to make sure you end up with the tags that genesis_ic can handle
    
    return result
