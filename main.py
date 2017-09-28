# -*- coding: utf-8 -*-
from __future__ import print_function, division
import os, codecs
import io
import numpy as np
import pygtrie
import sys
import argparse
import string
import re
import pickle
from collections import Counter
from nltk.stem import WordNetLemmatizer, PorterStemmer

# Using Trie Data Structure for storing the Dictionary of the corpus
reload(sys)
sys.setdefaultencoding('utf8')
ngrams = [pygtrie.StringTrie() for i in range(5)]
total_ngrams = {}

data = {}
all_data = []
def truncate_punctuations(text):
    # Replace all the punctuation marks from the text
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    out = regex.sub(' ', text)

    return out

# Reference code from norvig.org for spelling correction
def words(text): return re.findall(r'\w+', text.lower())

WORDS = {}

def set_WORDS():
    wnl = WordNetLemmatizer()
    port = PorterStemmer()

    #tmp = [str(wnl.lemmatize(i)) for i in all_data]
    #tmp2 = [str(port.stem(i)) for i in tmp]
    WORDS = Counter(all_data)
    #print(WORDS)


def P(word, N=sum(WORDS.values())):
    "Probability of `word`."
    val = 0
    if word in WORDS.keys():
        val = WORDS[word] / N
    return val

def correction(word):
    "Most probable spelling correction for word."
    return max(candidates(word), key=P)

def candidates(word):
    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

def known(words):
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)

def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word):
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))


def parse_input_query(query):
    q = query.split(',')
    q = [truncate_punctuations(x) for x in q]
    #TODO: Take care of the noun/verbs via stemming and lemmatization

    q = [correction(x) for x in q]
    wnl = WordNetLemmatizer()
    q = [str(wnl.lemmatize(i)) for i in q]

    port = PorterStemmer()
    q = [str(port.stem(i)) for i in q]

    return q




def get_ngrams(text_list, grams):
    return zip(*[text_list[i:] for i in range(grams)])

def generate_ngram(grams, text, years):

    out = truncate_punctuations(text)
    # Split the the text into lists. ngram is by word in this context
    text_list = out.split()

    # Generate the ngrams in the form of list of tuples
    ngrams_lists = get_ngrams(text_list, grams)
    '''
    for i in ngrams_lists:
        ngrams_text = ' '.join(map(''.join, i))
        ngrams[grams-1][ngrams_text] = set()
        try:
            ngrams[grams-1][ngrams_text].add(years)# += years
            #ngrams[grams-1][ngrams_text] = set(ngrams[grams-1][ngrams_text])
        except KeyError:
            ngrams[grams-1][ngrams_text].add(years)# set(years)
    '''

    wnl = WordNetLemmatizer()
    port = PorterStemmer()

    # Instead lets make a inverse DF
    for i in ngrams_lists:
        ngrams_text = ' '.join(map(''.join, i.lower()))
        if grams == 1:
            all_data.append(str(ngrams_text).lower())
        '''
        try:
            pass
            #ngrams_text = wnl.lemmatize(ngrams_text)#.decode('utf-8'))
            #ngrams_text = port.stem(ngrams_text)#.decode('utf-8'))
        except UnicodeDecodeError:
            pass
        '''
        try:
            ngrams[grams-1][years].append(ngrams_text)
        except KeyError:
            ngrams[grams-1][years] = ngrams_text

def value_by_key_prefix(d, partial):
    matches = [val for key, val in d.iteritems() if key.startswith(partial)]
    if not matches:
        raise KeyError(partial)
        if len(matches) > 1:
            raise ValueError('{} matches more than one key'.format(partial))
    return matches[0]

if __name__ == '__main__':

    # Add support for more intuitive options for arguments in command line

    parser = argparse.ArgumentParser(description='Simulation of Google nGram Viewer')
    #parser.add_argument('--ngram', type=int, help='ngram support 1 to 5 supported')
    parser.add_argument('--data_dir', type=str, help='path to dataset rootdir of books')
    parser.add_argument('--input_query', type=str, help='input query with , delimited and supported 2-bigrams and 1-unigrams')
    args = parser.parse_args()

    # Support 1 to 5 ngrams as per specifications

    # Need the input query to perform the search
    if(args.input_query == ""):
        parser.error('input_query required')

    years_ = set()

    data_dir = args.data_dir
    lines = ""
    for subdir, dirs, files in os.walk(data_dir):
        for file in files:
            dat = ""
            with open(os.path.join(subdir, file), 'rb') as f:
                    dat += f.read().replace('\n', ' ')
            lines += dat
        # Distincting the files by years
        year = subdir.split('/')
        if len(year) == 1:
            continue
        yr = year[1]
        years_.add(yr)
        try:
            data[yr]+= lines
        except KeyError:
            data[yr] = lines

    postings = range(1, 6)
    for i in postings:
        ngrams[i-1] = {}
        for years in years_:
            ngrams[i-1][years] = [] #{years: []}

    index_files = [str(i)+'gram_indexing.txt' for i in postings]
    for i in postings:
        if os.path.exists(index_files[i-1]):
            print(index_files[i-1], " file present already")
            with open(index_files[i-1], 'rb') as fr:
                ngrams[i-1] = pickle.loads(fr.read())
        else:

            # Generate the ngrams from 1 to 5 all at once while indexing for faster operations on query
            print('Generating the index files '+index_files[i-1])
            for years in data.keys():
                generate_ngram(i, data[years], years)

            # Dump the indexing into a text file for faster processing of queries
            with open(index_files[i-1], 'wb') as fw:
                pickle.dump(ngrams[i-1], fw)

    for i in range(1, 6):
        total_ngrams[i-1] = {}
        for y in years_:
            if y in ngrams[i-1].keys():
                total_ngrams[i-1][y] = len(ngrams[i-1][y])
            else:
                total_ngrams[i-1][y] = 0

    set_WORDS()
    # Pre-process the input_query
    query = parse_input_query(args.input_query)
    # Search for input query ngrams in the ngrams generated
    stats = {}
    yrs = ngrams[1].keys()
    query = [item.encode('utf-8') for item in query]
    print(query)
    for i in query:
        stats[i] = {}
        temp = i.split()
        for y in yrs:
            hits = 0
            percent = 0
            if y in ngrams[len(temp)-1].keys():
                res = Counter(ngrams[len(temp)-1][y])
                if i in res.keys():
                    hits = Counter(ngrams[len(temp)-1][y])[i]
                else:
                    hits = len(list(v for k,v in res.iteritems() if i in k.lower()))
                    #print(hits)
            if total_ngrams[len(temp)-1][y] != 0:
                percent = hits*100 / total_ngrams[len(temp)-1][y]
            stats[i][y] = percent

    print(stats)

    import matplotlib.pylab as plt

    for i in query:
        lists = sorted(stats[i].items()) # sorted by key, return a list of tuples
        x, y = zip(*lists) # unpack a list of pairs into two tuples
        #print(x, y)
        plt.plot(x, y, label=i)
        plt.legend()
        plt.xlabel('Years')
        plt.ylabel('Percentage')
    plt.show()




