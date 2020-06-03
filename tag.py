# -*- coding: utf-8 -*-
"""
Created on Mon May 18 01:53:09 2020

@author: jines
"""

import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns

import warnings

import pickle
import time

import re
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import ToktokTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from string import punctuation

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import hamming_loss
from sklearn.cluster import KMeans


import logging

from scipy.sparse import hstack

warnings.filterwarnings("ignore")
plt.style.use('bmh')

from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score
import re

np.random.seed(seed=42)



from joblib import dump,load
import numpy as np
model = load('LinearSVC_body_code.joblib')





title="Should I use nested classes in this case?"

body="<p>I am working on a collection of classes used for video playback and recording. I have one main class which acts like the public interface, with methods like <code>play()</code>, <code>stop()</code>, <code>pause()</code>, <code>record()</code> etc... Then I have workhorse classes which do the video decoding and video encoding. </p>\n\n<p>I just learned about the existence of nested classes in C++, and I'm curious to know what programmers think about using them. I am a little wary and not really sure what the benefits/drawbacks are, but they seem (according to the book I'm reading) to be used in cases such as mine.</p>\n\n<p>The book suggests that in a scenario like mine, a good solution would be to nest the workhorse classes inside the interface class, so there are no separate files for classes the client is not meant to use, and to avoid any possible naming conflicts? I don't know about these justifications. Nested classes are a new concept to me. Just want to see what programmers think about the issue.</p>\n"

#title="MySQL/Apache Error in PHP MySQL query"

#body='<p>I am getting the following error:</p>\n\n<blockquote>\n  <p>Access denied for user \'apache\'@\'localhost\' (using password: NO)</p>\n</blockquote>\n\n<p>When using the following code:</p>\n\n<pre><code>&lt;?php\n\ninclude("../includes/connect.php");\n\n$query = "SELECT * from story";\n\n$result = mysql_query($query) or die(mysql_error());\n\necho "&lt;h1&gt;Delete Story&lt;/h1&gt;";\n\nif (mysql_num_rows($result) &gt; 0) {\n    while($row = mysql_fetch_row($result)){\n          echo \'&lt;b&gt;\'.$row[1].\'&lt;/b&gt;&lt;span align="right"&gt;&lt;a href="../process/delete_story.php?id=\'.$row[0].\'"&gt;Delete&lt;/a&gt;&lt;/span&gt;\';\n      echo \'&lt;br /&gt;&lt;i&gt;\'.$row[2].\'&lt;/i&gt;\';\n    }\n}\nelse {\n   echo "No stories available.";\n}\n?&gt;\n</code></pre>\n\n<p>The connect.php file contains my MySQL connect calls that are working fine with my INSERT queries in another portion of the software.  If I comment out the $result = mysql_query line, then it goes through to the else statement.  So, it is that line or the content in the if.</p>\n\n<p>I have been searching the net for any solutions, and most seem to be related to too many MySQL connections or that the user I am logging into MySQL as does not have permission.  I have checked both.  I can still perform my other queries elsewhere in the software, and I have verified that the account has the correct permissions.</p>\n'


tags_features=['javascript','sql','java','c#','python','php','c++','c','typescript','ruby','swift','objective-c','vb.net','assembly','r','perl','vba','matlab','go','scala','groovy','coffeescript','lua','haskell']
def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub(r"\'\n", " ", text)  # r for line feed
    text = re.sub(r"\'\xa0", " ", text) #for space
    text = re.sub('\s+', ' ', text) # mathes one or more whitespace in row
    text = text.strip(' ')
    return text
token=ToktokTokenizer()
punctuation
punct = '!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~'
def strip_list_noempty(mylist):
    newlist = (item.strip() if hasattr(item, 'strip') else item for item in mylist)
    return [item for item in newlist if item != '']

#Removing whitespace and space in list that contains different types of elements
def clean_punct(text): 
    words=token.tokenize(text)
    punctuation_filtered = []
    regex = re.compile('[%s]' % re.escape(punct))
    remove_punctuation = str.maketrans(' ', ' ', punct) #Best way to strip punctuation from a string
    for w in words:
        if w in tags_features:
            punctuation_filtered.append(w)
        else:
            punctuation_filtered.append(regex.sub('', w))
  
    filtered_list = strip_list_noempty(punctuation_filtered)
        
    return ' '.join(map(str, filtered_list))
lemma=WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
def lemitizeWords(text):
    words=token.tokenize(text)
    listLemma=[]
    for w in words:
        x=lemma.lemmatize(w, pos="v")
        listLemma.append(x)
    return ' '.join(map(str, listLemma))
def stopWordsRemove(text):
    
    stop_words = set(stopwords.words("english"))
    
    words=token.tokenize(text)
    
    filtered = [w for w in words if not w in stop_words]
    
    return ' '.join(map(str, filtered))
def body_split(body):
    # print(body)
    soup = BeautifulSoup(body, "html5lib")
    t2_soup=soup.find_all('code')
    code_text = ""
    text_without_code = re.sub(r'<code>.*?</code>', ' ', body)
    soup2 = BeautifulSoup(text_without_code, "html5lib")
    text_without_code_tags = soup2.get_text()

    for item in t2_soup:
        code_text = code_text + str(item.text) + "\n"

    return text_without_code_tags

def code_split(body):
    # print(body)
    soup = BeautifulSoup(body, "html5lib")
    t2_soup=soup.find_all('code')
    code_text = ""
    text_without_code = re.sub(r'<code>.*?</code>', ' ', body)
    soup2 = BeautifulSoup(text_without_code, "html5lib")
    text_without_code_tags = soup2.get_text()

    for item in t2_soup:
        code_text = code_text + str(item.text) + "\n"

    return code_text



t1=title 
t1=str(t1)
t1=clean_text(t1)
t1=clean_punct(t1)
t1=lemitizeWords(t1)
t1=stopWordsRemove(t1)

t1

t2=body

t2_text=body_split(t2)

t2_code=code_split(t2)





t2_text=BeautifulSoup(t2_text).get_text()
t2_text=clean_text(t2_text)
t2_text=clean_punct(t2_text)
t2_text=lemitizeWords(t2_text)
t2_text=stopWordsRemove(t2_text)



t2_code=re.sub(r'\A\s+|\s+\Z','',t2_code)
t2_code=re.sub('\n',' ',t2_code)



t2_code=BeautifulSoup(t2_code).get_text()
t2_code=clean_text(t2_code)
t2_code=clean_punct(t2_code)
t2_code=lemitizeWords(t2_code)
t2_code=stopWordsRemove(t2_code)



def preprocess(title,body):
    t1=title 
    t1=str(t1)
    t1=clean_text(t1)
    t1=clean_punct(t1)
    t1=lemitizeWords(t1)
    t1=stopWordsRemove(t1)
    t2=body
    t2_text=body_split(t2)
    t2_code=code_split(t2)    
    t2_text=BeautifulSoup(t2_text).get_text()
    t2_text=clean_text(t2_text)
    t2_text=clean_punct(t2_text)
    t2_text=lemitizeWords(t2_text)
    t2_text=stopWordsRemove(t2_text)
    t2_code=re.sub(r'\A\s+|\s+\Z','',t2_code)
    t2_code=re.sub('\n',' ',t2_code)
    t2_code=BeautifulSoup(t2_code).get_text()
    t2_code=clean_text(t2_code)
    t2_code=clean_punct(t2_code)
    t2_code=lemitizeWords(t2_code)
    t2_code=stopWordsRemove(t2_code)
    vectorizer_X1 = TfidfVectorizer(analyzer = 'word',
                                       min_df=0.0,
                                       max_df = 1.0,
                                       strip_accents = None,
                                       encoding = 'utf-8', 
                                       preprocessor=None,
                                       token_pattern=r"(?u)\S\S+",
                                       max_features=1000)

    vectorizer_X2 = TfidfVectorizer(analyzer = 'word',
                                           min_df=0.0,
                                           max_df = 1.0,
                                           strip_accents = None,
                                           encoding = 'utf-8', 
                                           preprocessor=None,
                                           token_pattern=r"(?u)\S\S+",
                                           max_features=1000)

    vectorizer_X3 = TfidfVectorizer(analyzer = 'word',
                                           min_df=0.0,
                                           max_df = 1.0,
                                           strip_accents = None,
                                           encoding = 'utf-8', 
                                           preprocessor=None,
                                           token_pattern=r"(?u)\S\S+",
                                           max_features=1000)
    
    
    
    df = pd.DataFrame()
    row_df=pd.DataFrame([pd.Series([t1,t2_text,t2_code])])
    df=pd.concat([row_df, df], ignore_index=True)
    X1_tfidf = vectorizer_X1.fit_transform(df[0])
    X2_tfidf = vectorizer_X2.fit_transform(df[1])
    X3_tfidf = vectorizer_X3.fit_transform(df[2])
    
    temp1=3000-(X1_tfidf.size+X2_tfidf.size+X3_tfidf.size)
    tp=np.zeros((1,temp1))
    
    X_tfidf = hstack([X1_tfidf,X2_tfidf,X3_tfidf,tp])
    

    return model.predict(X_tfidf)





ans= preprocess(title,body)


tag_idx=['php','c','c#','javascript','coffeescript','go','groovy','haskell','java','c++','lua','matlab','objective-c','perl','assembly','python','r','ruby','scala','sql','swift','typescript','vb.net','vba']
print(tag_idx[np.argmax(ans)])