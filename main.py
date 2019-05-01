#Multi-Class Text Classification Model Comparison and Selection Natural Language Processing, word2vec, Support Vector Machine, bag-of-words, deep learning. [Online]
#Author(s): Susan Li, (2018)
#Available at: https://towardsdatascience.com/multi-class-text-classification-model-comparison-and-selection-5eb066197568
#[Accessed 14 02 2019].

import pandas as pd
import numpy as np
from numpy import random
import gensim
import nltk
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup


df = pd.read_csv('data/HTMLdataset.csv')
df = df[pd.notnull(df['label'])]
print(df.head(10))
print(df['text'].apply(lambda x: len(x.split(' '))).sum())

labels = ['Form','Text','Link','Table','JavaScript','Image']
plt.figure(figsize=(10,4))
df.label.value_counts().plot(kind='bar');

def print_plot(index):
    example = df[df.index == index][['text', 'label']].values[0]
    if len(example) > 0:
        print(example[0])
        print('label:', example[1])
print_plot(10)

print_plot(30)


#Text Preprocessing

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    """
        text: a string
        
        return: modified initial string
    """
    text = BeautifulSoup(text, "lxml").text # HTML decoding
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwors from text
    return text
    
df['text'] = df['text'].apply(clean_text)
print_plot(10)

df['text'].apply(lambda x: len(x.split(' '))).sum()

X = df.text
y = df.label
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state = 42)



# *** Naive Bayes Classification *** #
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer

nb = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', MultinomialNB()),
              ])
nb.fit(X_train, y_train)

#%%time
from sklearn.metrics import classification_report
y_pred = nb.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred,target_names=labels))



# *** Linear Support Vector Machine *** #
from sklearn.linear_model import SGDClassifier

sgd = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)),
               ])
sgd.fit(X_train, y_train)

#%%time

y_pred = sgd.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred,target_names=labels))


# *** Logistic Regression *** #
from sklearn.linear_model import LogisticRegression

logreg = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression(n_jobs=1, C=1e5)),
               ])
logreg.fit(X_train, y_train)

#%%time

y_pred = logreg.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred,target_names=labels))



#Multi-Class Text Classification Model Comparison and Selection. [Online]
#Author(s): Susan Li, (2018).
#Available at: https://towardsdatascience.com/multi-class-text-classification-model-comparison-and-selection-5eb066197568
#[Accessed 17 02 2019].

print(" ######### ---------- Text Classification with Keras ---------- ######### ")


import itertools
import os

#%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.preprocessing import text, sequence
from keras import utils


train_size = int(len(df) * .9)
train_text = df['text'][:train_size]
train_label = df['label'][:train_size]

test_text = df['text'][train_size:]
test_label = df['label'][train_size:]

max_words = 1000
tokenize = text.Tokenizer(num_words=max_words, char_level=False)
tokenize.fit_on_texts(train_text) # only fit on train

x_train = tokenize.texts_to_matrix(train_text)
x_test = tokenize.texts_to_matrix(test_text)

encoder = LabelEncoder()
encoder.fit(train_label)
y_train = encoder.transform(train_label)
y_test = encoder.transform(test_label)

num_classes = np.max(y_train) + 1
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)

batch_size = 100
epochs = 20

# Build the model
model = Sequential()
model.add(Dense(512, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
              
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.2)
										

score = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)
print('Test accuracy:', score[1])


#Implementing predictions

#Intro to text classification with Keras: automatically tagging Stack Overflow posts. [Online]
#Author(s): Sara Robinson, Josh Gordon, Marianne Linhares Monteiro (2017).
#Available at: https://cloud.google.com/blog/products/gcp/intro-to-text-classification-with-keras-automatically-tagging-stack-overflow-posts
#[Accessed 19 02 2019].

train_size = int(len(df) * .8)
train_text = df['text'][:train_size]
train_label = df['label'][:train_size]
test_text = df['text'][train_size:]
test_label = df['label'][train_size:]

for i in range(25):    
    prediction = model.predict(np.array([x_test[i]]))
	

text_labels = encoder.classes_ 
predicted_label = text_labels[np.argmax(prediction[0])]
print(test_text.iloc[i][:25], "...")

actual_label = test_label.iloc[i]
print('\nActual label:' + actual_label)

print("\nPredicted label: " + predicted_label + "\n\n")	



#Test Cases

Form = "Enter a name in the text box.\n"
Text = "Verify text is displayed.\n"
Image = "Image is loaded.\n"
Link = "Hyperlink is selected and new page opens.\n"
JS = "JavaScript loads.\n"
Table = "Table is loaded.\n"

#Python GUI Tkinter Tutorial [#1] - Creating Your First GUI; Python GUI Tkinter Tutorial [#3] - Widgets. [Online]
#Author(s): Kody Simpson (2018).
#Available at: https://www.youtube.com/watch?v=3li1SdZ1Ru8; https://www.youtube.com/watch?v=HzY2ck-uTCY
#[Accessed 01 05 2019].

import tkinter as tk

#Crete instance
win = tk.Tk()

#Window title
win.title("Test Case Application")

frame = tk.Frame(win)
frame.pack()

#Label 1 
tk.Label(frame, text="Displaying test case\n\n").pack()

#Label 2
tk.Label(frame, text="The test case to be performed is as follows:\n\n").pack()

if predicted_label == "Form":
	tk.Label(frame, text= Form).pack()
	
elif predicted_label == "Text":
	tk.Label(frame, text= Text).pack()

elif predicted_label == "Image":
	tk.Label(frame, text= Image).pack()

elif predicted_label == "Link":
	tk.Label(frame, text= Link).pack()
	
elif predicted_label == "JavaScript":
	tk.Label(frame, text= JS).pack()
	
elif predicted_label == "Table":
	tk.Label(frame, text= Table).pack()
	
else:
	tk.Label(frame, text="No test case found matching label: " + predicted_label).pack()

#How do I close a tkinter window? [Online]
#Author(s): Steven M. Vascellaro (2018).
#Available at: https://stackoverflow.com/questions/110923/how-do-i-close-a-tkinter-window
#[Accessed 01 05 2019].

#Quit button
tk.Button(frame, text="Exit", command=quit).pack()

#Start GUI
win.mainloop()