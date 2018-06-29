# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 14:08:24 2018

@author: Wanyu Du
"""

import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Embedding
from keras.layers.core import Dense, Reshape, Flatten, Dropout
from keras.layers.convolutional import Conv1D, Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

# load dataset function
def get_texts(file_path, mode=None):
    file = pd.read_csv(file_path)
    q1, q2 = file['q1'], file['q2']
    texts1 = []
    texts2 = []
    for sent in q1:
        texts1.append(sent)
    for sent in q2:
        texts2.append(sent)
    if mode=='train':
        labels=list(file['label'])
        return np.array(labels), texts1, texts2
    else:
        return texts1, texts2
  
    
    
embed_size=300
text1_maxlen=50
text2_maxlen=50
filters_1d=text2_maxlen
kernel_size_1d=3
num_conv2d_layers=2
filters_2d=[128,64]
kernel_size_2d=[[3,3], [3,3]]
mpool_size_2d=[[2,2], [2,2]]
dropout_rate=0.5
batch_size=128

TRAIN_PATH = 'data/train.csv'
TEST_PATH = 'data/test.csv'
WORD_EMBED='data/char_embed.txt'
model_path='checkpoints/arcii.h5'
log_path='checkpoints/log_arcii.txt'

print('Load files...')
word_embedding = pd.read_csv(WORD_EMBED, header=None, sep=' ')
word_embed={}
for i in range(len(word_embedding)):
    word_embed[word_embedding.iloc[i,0]]=list(word_embedding.iloc[i,1:])

all_labels, all_texts1, all_texts2 = get_texts(TRAIN_PATH, 'train')
test_texts1, test_texts2 = get_texts(TEST_PATH, 'test')

print('Prepare word embedding...')
all_texts=all_texts1+all_texts2+test_texts1+test_texts2
# prepare tokenizer
t=Tokenizer(lower=False)
t.fit_on_texts(all_texts)
vocab_size=len(t.word_index)+1
# integer encode the docs
encoded_texts1=t.texts_to_sequences(all_texts1)
encoded_texts2=t.texts_to_sequences(all_texts2)
encoded_test_texts1=t.texts_to_sequences(test_texts1)
encoded_test_texts2=t.texts_to_sequences(test_texts2)
# pad docs
padded_texts1=pad_sequences(encoded_texts1, maxlen=text1_maxlen, padding='post')
padded_texts2=pad_sequences(encoded_texts2, maxlen=text2_maxlen, padding='post')
padded_test_texts1=pad_sequences(encoded_test_texts1, maxlen=text1_maxlen, padding='post')
padded_test_texts2=pad_sequences(encoded_test_texts2, maxlen=text1_maxlen, padding='post')
# create a weight matrix for words in training docs
embedding_matrix=np.zeros((vocab_size, embed_size))
for word, i in t.word_index.items():
    embedding_vector=word_embed[word]
    if embedding_vector is not None:
        embedding_matrix[i]=embedding_vector

print('Split train and valid set...')
sample_size=len(all_labels)*4//5
train_labels, train_texts1, train_texts2 = all_labels[:sample_size], padded_texts1[:sample_size], padded_texts2[:sample_size]
valid_labels, valid_texts1, valid_texts2 = all_labels[sample_size:], padded_texts1[sample_size:], padded_texts2[sample_size:]


print('Build model...')
query=Input(shape=(text1_maxlen,), name='query')
doc=Input(shape=(text2_maxlen,), name='doc')

embedding = Embedding(vocab_size, embed_size, weights=[embedding_matrix], trainable=False)
q_embed = embedding(query)
d_embed = embedding(doc)

layer1_input=concatenate([q_embed, d_embed])

layer1_conv1=Conv1D(filters=filters_1d, kernel_size=kernel_size_1d, padding='same', activation='relu')(layer1_input)
layer1_reshaped=Reshape((text1_maxlen, text2_maxlen, -1))(layer1_conv1)
z=MaxPooling2D(pool_size=(2,2))(layer1_reshaped)

for i in range(num_conv2d_layers):
    z=Conv2D(filters=filters_2d[i], kernel_size=kernel_size_2d[i], padding='same', activation='relu')(z)
    z=MaxPooling2D(pool_size=(mpool_size_2d[i][0], mpool_size_2d[i][1]))(z)
    
pool1_flat=Flatten()(z)
pool1_flat_drop=Dropout(rate=dropout_rate)(pool1_flat)
mlp1=Dense(64, activation='relu')(pool1_flat_drop)
mlp2=Dense(32, activation='relu')(mlp1)
out=Dense(1, activation='sigmoid')(mlp2)

model=Model(inputs=[query, doc], outputs=out)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['loss', 'acc'])
model.summary()

# build dataset generator
def generator(texts1, texts2, labels, batch_size, min_index, max_index):
    i=min_index
    
    while True:
        if i+batch_size>=max_index:
            i=min_index
        rows=np.arange(i, min(i+batch_size, max_index))
        i+=batch_size
        
        samples1=texts1[rows]
        samples2=texts2[rows]
        targets=labels[rows]
        yield {'query':samples1, 'doc':samples2}, targets
        
def test_generator(texts1, texts2, batch_size, min_index, max_index):
    i=min_index
    
    while True:
        if i+batch_size>=max_index:
            i=min_index
        rows=np.arange(i, min(i+batch_size, max_index))
        i+=batch_size
        
        samples1=texts1[rows]
        samples2=texts2[rows]
        yield {'query':samples1, 'doc':samples2}
            
train_gen=generator(train_texts1, train_texts2, train_labels, batch_size=batch_size, min_index=0, max_index=len(train_texts1))
valid_gen=generator(valid_texts1, valid_texts2, valid_labels, batch_size=batch_size, min_index=0, max_index=len(valid_texts1))
test_gen=test_generator(padded_test_texts1, padded_test_texts2, batch_size=1, min_index=0, max_index=len(test_texts1))

print('Train classifier...')
history=model.fit_generator(train_gen, epochs=10, steps_per_epoch=len(train_texts1)//batch_size,
                  validation_data=valid_gen, validation_steps=len(valid_texts1)//batch_size, verbose=1,
                  callbacks=[ModelCheckpoint(model_path, monitor='val_loss', mode='min', save_best_only=True), 
                             EarlyStopping(monitor='val_loss', patience=3), CSVLogger(log_path)])

print('Predict...')
model=load_model(model_path)
preds=model.predict_generator(test_gen, steps=len(test_texts1))
