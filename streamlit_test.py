import streamlit as st
import numpy as np
import re
import nltk
from nltk.corpus import genesis
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from autocorrect import Speller
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('genesis')

st.markdown('## Find-a-Verse')
st.markdown('This site runs a program that looks through the King James Version of the Bible (KJVB) to find verses. \nIt has some work to do generating synonyms now - it may take a while. I promise there will be a search bar, it just might take a minute to load.')
st.markdown('The program uses Cosine distance and Res similarity to search for bible verses from the KJVB. Res simliarity uses synonyms and word meanings from a contemporary context to find results. It may give strange results, especially for uncommon words that are common in the KJVB. It also may miss things because the KJVB has older phrasing, e.g. bring forth a son. I chose KJVB because it is in the public domain. Cosine distance should give a more word for word result but takes longer to load.\n\n')

def res_similarity_score(s1, s2):
    """
    Calculate the normalized similarity score of s1 onto s2

    For each synset in s1, finds the synset in s2 with the largest similarity value.
    Sum of all of the largest similarity values and normalize this value by dividing it by the
    number of largest similarity values found.

    Args:
        s1, s2: list of synsets from doc_to_synsets

    Returns:
        normalized res_similarity score for document
    """
    

    similarity_values = []
    for d in s1:
        #get this to stop if it reaches some value?
        #so remove the list comprehension, put in a for loop and if the path_similarity is bigger
        #than 0.3?? just take that?? ie stop searching if you get close enough in a verse.
        lst = [d.res_similarity(s,genesis_ic) for s in s2 if d._pos == s._pos]
        if lst:
            similarity_values.append(max(lst))
    length = len(similarity_values)
    if length>0:
        return sum(similarity_values)/length
    else:
        return 0

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

with open('kjvb.txt') as f:
    bible = f.read()

books = bible[0:2900].split('\n')[24:92]

#deleting a blank line and the new testament head (once one is deleted, index 39 changes item)
del books[39]
del books[39]

bible = bible.replace('  ',' ')

info = re.split('\*\*\*',bible)

NT = info[3][51:]
OT = info[2][2112:]
bible_text = OT+NT

books = [books[i]+'\n\n' for i in range(len(books))]

bible_books={}
rest_of_bible = bible_text

#spliting (max once) based on the names of books - ditched the linespaces for keys.
#this puts the text of the books into a dictionary with the official title as the key
#strange iteration is to put Revelation last (otherwise it gets defined first)
for i in range(1,len(books)):
    bible_books[books[i-1].replace('\n','')] = rest_of_bible.split(books[i],1)[0]
    rest_of_bible = rest_of_bible.split(books[i],1)[1]
bible_books[books[65].replace('\n','')] = rest_of_bible

#splitting dictinary book values into lists. 
for book in bible_books:
    #this split keeps all the verese references (at start of verse and in seperate index)
    bible_books[book] = (re.split('[\s\S][^\d](?=(\d{1,3}:\d{1,3}))',bible_books[book]))
    bible_books[book] = bible_books[book][::2]
    #deleting pesky \n at start of verses. Had to use int iterator not verses.
    for i in range(len(bible_books[book])):
        if bible_books[book][i][0] == '\n':
            bible_books[book][i] = bible_books[book][i][1:]
    #deleting first element if it doesn't start with 1:1
    if bible_books[book][0][0:3] != '1:1':
        del bible_books[book][0]

#makeing the verse_synsets - this takes a while
verse_synsets = []
for book in bible_books:
    verse_synsets.extend((book,verse,doc_to_synsets(verse)) for verse in bible_books[book])


genesis_ic = wn.ic(genesis, False, 0.0)


st.markdown("Write what you remember of the verse you are looking for: ")
user_input = st.text_input(label="Type:\n",value="Christ",label_visibility="hidden")

#spell checker

spell=Speller(lang="en")
WORD = re.compile(r'\w+')
def reTokenize(doc):
    tokens = WORD.findall(doc)
    return tokens

text = user_input
def spell_correct(text):
    sptext = (' '.join([spell(w).lower() for w in reTokenize(text)]))      
    return sptext    

user_input  = spell_correct(user_input)

#this is splitting into letters rtaher than words. ?
st.markdown('\nSearching for "'+user_input+'"...\n')


user_synsets = doc_to_synsets(user_input)

top_ten = [(res_similarity_score(user_synsets,verse[2]),verse[0],verse[1]) for verse in verse_synsets]

st.markdown('\n ### Res_similiarity results: ')

top_ten.sort(reverse=True)
top_ten = top_ten[0:5]
for verse in top_ten:
    book = verse[1]
    v = verse[2]
    v_num = re.findall(r'[\d]{,3}:[\d]{,3}',v)
    st.text(book + ' ' + v_num[0])
    v_words = re.split('[\d]{,3}:[\d]{,3}', v, 1)[1]
    st.text(v_words)
    st.text('')
    st.text('')


st.markdown('\n ### Cosine distance results: \n')

bible_verses = [user_input]
for book in bible_books:
    bible_verses.extend(bible_books[book])
#change max_df to get faster results? 
count_vect = CountVectorizer()

count_matrix = count_vect.fit_transform(bible_verses)

num_rows, num_cols = count_matrix.shape

similarity_scores = [(cosine_similarity(count_matrix[0], count_matrix[i]).item(),bible_verses[i]) for i in range(1,(num_rows))]    

similarity_scores.sort(reverse=True)
similarity_scores = similarity_scores[0:5]

for verse in similarity_scores:
    for book in bible_books:
        for v in bible_books[book]:
            if v == verse[1]:
                st.text('\n')
                v_num = re.findall(r'[\d]{,3}:[\d]{,3}',v)
                st.text(book + ' ' + v_num[0])
                v_words = re.split('[\d]{,3}:[\d]{,3}', v, 1)[1]
                st.text(v_words)
                st.text('')


