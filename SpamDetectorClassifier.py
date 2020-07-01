import pandas as pd
import numpy as np
import nltk
import re
import pickle
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

df = pd.read_csv('Spamdata', sep='\t', names=['labels','message'])
corpus = []
ps = PorterStemmer()
for i in range(0,df.shape[0]):
    message = re.sub(pattern='[^a-zA-Z]',repl=' ', string=df.message[i])
    message = message.lower()
    words = message.split()
    words = [word for word in words if word not in set(stopwords.words('english'))]
    words = [ps.stem(word) for word in words]
    message = ' '.join(words)
    corpus.append(message)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()
y = pd.get_dummies(df['labels'])
y = y.iloc[:,1].values
pickle.dump(cv,open('cv-transform.pkl','wb'))

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0, test_size=0.2)

from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB(alpha=0.3)
classifier.fit(X_train,y_train)
filename = 'Spam_Model.pkl'
pickle.dump(classifier,open(filename,'wb'))