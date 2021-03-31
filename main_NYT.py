# downloader script for nyt metadata archive
# see information here: https://archive.nytimes.com/www.nytimes.com/ref/membercenter/nytarchive.html
import NewYorkTime.api_NYT
import os
import glob
import importlib
import NewYorkTime.utils
import pandas as pd
import nltk
nltk.download('stopwords')
importlib.reload(NewYorkTime.api_NYT)

"""  Step 1: API requests and data retrieval    """
api_key='8X5rRqGAYeRHvcNWsihrGVGm6z7CIa0O'
print(api_key)

wk_dir='/Applications/anaconda/anaconda3/NYT_archive'
NewYorkTime.api_NYT.ApiRetrival(api_key, startYear=2019, endYear=2020, working_dir=wk_dir)

"""  Step 2: Parse JSON files to dataframe   """
import NewYorkTime.ReadJSON 

fileNameSearch = wk_dir + '/nyt_*'
fileNames = glob.glob(fileNameSearch)

result = []
for fileName in fileNames:
    print(fileName)
    result.append(NewYorkTime.utils.parse_monthly_json(fileName))

print(result)
df = pd.concat(result)
df.sort_values(by="pub_date", inplace=True)
df.reset_index(drop=True, inplace=True)
df.keys()

"""  Step 3: Slice dataframe   """
sections = ["Amazon", "Finance", "Big data", "Technology","Machine learning"]
desks = ["Business/Financial Desk", "Business", "Financial Desk"]
print(df.columns)

df = df[df.section_name.isin(sections) | df.news_desk.isin(desks)]

df = df[(df.abstract.str.len() >=20) & (df.abstract.str.len() <2000)]

"""  Step 4: Basic NLP   """
df.abstract.replace({r'[^\x00-\x7F]+':''}, regex=True, inplace=True)

df.abstract.replace({r'[^a-zA-z\s]':''}, regex=True, inplace=True)

df.abstract.replace({r'_':''}, regex=True, inplace=True)

df.abstract = df.abstract.str.strip().str.lower()

stop = nltk.corpus.stopwords.words("english")
print(stop)
df['abstract'] = df['abstract'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
df.to_csv('NYT_DF.csv')

""" Demonstrate
    1. NLP (Natural Language Processing)
    2. MarryNYT_DF with the lexicon
    3. Generate a wordcloud from Shakespear text

."""

import re
import nltk
nltk.download('punkt')
from collections import Counter
from wordcloud import WordCloud # using python 3.7
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import pandas as pd


NEGWORDS = ["not", "no", "none", "neither", "never", "nobody", "n't", 'nor',"however","except"]
STOPWORDS = ["an", "a", "the", "or", "and", "thou", "must", "that", "this", "self", "unless", "behind", "for", "which",
             "whose", "can", "else", "some", "will", "so", "from", "to", "by", "within", "of", "upon", "th", "with",
             "it"]


def _remove_stopwords(txt):
    """Delete from txt all words contained in STOPWORDS."""
    words = txt.split()
    # words = txt.split(" ")
    for i, word in enumerate(words):
        if word in STOPWORDS:
            words[i] = " "
    return (" ".join(words))

data = pd.read_csv('/Users/mac/Desktop/python/financial information/DataRetrieval-master/NYT_DF.csv', sep='，')
data.to_csv("/Users/mac/Desktop/python/financial information/DataRetrieval-master/tesla.txt",sep='\t',index=None,header=None)
  
tesla='Amazon.txt'
with open(tesla, 'r', encoding='utf-8') as tesla_read:
    tesla_string = tesla_read.read()

tesla_split = str.split(tesla_string, sep=',')
print(tesla_split)
len(tesla_split)

doc_out = []
for k in tesla_split:
    cleantextprep = str(k)
    expression = "[^a-zA-Z ]"  # keep only letters, numbers and whitespace
    cleantextCAP = re.sub(expression, '', cleantextprep)  # apply regex
    cleantext = cleantextCAP.lower()  # lower case
    cleantext = _remove_stopwords(cleantext)
    bound = ''.join(cleantext)
    doc_out.append(bound)       # a list of sentences

print(doc_out)
print(tesla_split)

# print clean text
for line in doc_out:
    print(line)