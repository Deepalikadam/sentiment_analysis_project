import pandas as pd
import numpy as np

df1=pd.read_csv('amazon_cells_labelled.txt', names=['sentence','labels'],sep='\t')
names=['col1','col2']
print(names)

from sklearn.feature_extraction.text import CountVectorizer
sentences = ['John likes ice cream', 'John hates chocolate.']
vectorizer = CountVectorizer(min_df=0, lowercase=False)
vectorizer.fit(sentences)
vectorizer.vocabulary_
vectorizer.transform(sentences).toarray()

from sklearn.model_selection import train_test_split
sentences = df1['sentence'].values
# print(sentences)
y = df1['labels'].values
# print(y)

sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=1000)

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
vectorizer.fit(sentences_train)

X_train = vectorizer.transform(sentences_train)
X_test  = vectorizer.transform(sentences_test)
print(X_train)

from sklearn.linear_model import LogisticRegression
lg = LogisticRegression()
lg.fit(X_train, y_train)
model_predict =lg.predict(X_test)
print(model_predict)
print(y_test)
from sklearn.metrics import accuracy_score
logistics_accuracy = accuracy_score(y_test,model_predict)
print(logistics_accuracy)

from sklearn.neighbors import KNeighborsClassifier
k=KNeighborsClassifier()
k.fit(X_train,y_train)
model2_predict=k.predict(X_train)
print(model2_predict)
print(y_train)
from sklearn.metrics import accuracy_score
knn_accuracy = accuracy_score(y_train,model2_predict)
print(knn_accuracy)

from sklearn.tree import DecisionTreeClassifier
d=DecisionTreeClassifier()
d.fit(X_train,y_train)
model3_predict=d.predict(X_train)
print(model3_predict)
print(y_train)
from sklearn.metrics import accuracy_score
dectree_accuracy = accuracy_score(y_train,model3_predict)
print(dectree_accuracy)

from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
a=rf.fit(X_train,y_train)
model4_predict=rf.predict(X_train)
print(model4_predict)
print(y_train)
from sklearn.metrics import accuracy_score
rf_accuracy = accuracy_score(y_train,model4_predict)
print(rf_accuracy)

from sklearn.naive_bayes import MultinomialNB
g=MultinomialNB()
g.fit(X_train,y_train)
model5_predict=g.predict(X_train)
print(model5_predict)
print(y_train)
from sklearn.metrics import accuracy_score
nb_accuracy = accuracy_score(y_train,model5_predict)
print(nb_accuracy)

