from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from classes.helpers.File import File
from classes.nlp.WordTokenaizer import RussianWordTokenaizer
import os
import collections






files = os.listdir('data/russian/posts/')

fileNames = [x.split('.')[0] for x in files ]

text1 = File("data/russian/posts/oop.txt").getContent()
text2 = File("data/russian/posts/ci_di.txt").getContent()
text3 = File("data/russian/posts/yandex.txt").getContent()


testText = File("data/russian/posts/oop_test.txt").getContent()

russianTextTokenaizer = RussianWordTokenaizer(testText).make()

texts = [text1, text2]
texts_labels = ['oop', 'ci/di']
 
text_clf = Pipeline([
                     ('tfidf', TfidfVectorizer()),
                     ('clf', SGDClassifier())
                     ])
 
text_clf.fit(texts, texts_labels)
 

res = (text_clf.predict(russianTextTokenaizer))
print(collections.Counter(res))
#print(res)