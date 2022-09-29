#!/usr/bin/env python
# coding: utf-8

# In[99]:


import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split

import unicodedata
import re
import numpy as np
import os
import io
import time
import pandas as pd

from flask import Flask,render_template,request
from nltk import tokenize
from nltk.util import ngrams
import nltk
from collections import Counter
import random
import csv
import sqlite3


# In[100]:


path_to_file = "All_data_NL_SQL_Answer.xlsx"


# In[101]:


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform'))

    def call(self, x, hidden):
        x = self.embedding(x)
        output, forward_h, backward_h = self.gru(x)
        state_h = tf.keras.layers.Concatenate()([forward_h, backward_h])
       
        encoder_states = state_h
        #output, state = self.gru(x, initial_state=hidden)
        return output, encoder_states

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))
    
    def get_config(self):
        return {"vocab_size": self.vocab_size,"embedding_dim": self.embedding_dim,"enc_units": self.enc_units,
                "batch_sz": self.batch_sz}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# In[102]:


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
    # query hidden state shape == (batch_size, hidden size)
    # query_with_time_axis shape == (batch_size, 1, hidden size)
    # values shape == (batch_size, max_len, hidden size)
    # we are doing this to broadcast addition along the time axis to calculate the score
        query_with_time_axis = tf.expand_dims(query, 1)

    # score shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(
        self.W1(query_with_time_axis) + self.W2(values)))

    # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


# In[103]:


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units*2,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)

        # used for attention
        self.attention = BahdanauAttention(self.dec_units*2)

    def call(self, x, hidden, enc_output):
    # enc_output shape == (batch_size, max_length, hidden_size)
        context_vector, attention_weights = self.attention(hidden, enc_output)

    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

    # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # passing the concatenated vector to the GRU
        output, state = self.gru(x)

    # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

    # output shape == (batch_size, vocab)
        x = self.fc(output)

        return x, state, attention_weights
    
    def get_config(self):
        return {"vocab_size": self.vocab_size,"embedding_dim": self.embedding_dim,"dec_units": self.dec_units,
                "batch_sz": self.batch_sz}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# In[104]:


# Converts the unicode file to ascii
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                 if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
    w = unicode_to_ascii(w.strip())

  # creating a space between a word and the punctuation following it
  # eg: "he is a boy." => "he is a boy ."
  # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    #w = re.sub(r"([?.!,¿])", r" \1 ", w)
    #w = re.sub(r'[" "]+', " ", w)
    w = re.sub('(?<! )(?=[.,!?()])|(?<=[.,!?()])(?! )', r' ', w)
    w = re.sub(r'([0-9]) . ([0-9])', r"\1.\2", w)
    w = re.sub(r'[" "]+', " ", w)
  # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
  #w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

    w = w.strip()

  # adding a start and an end token to the sentence
  # so that the model know when to start and stop predicting.
    w = '<start> ' + w + ' <end>'
    return w


# In[105]:


# 1. Remove the accents
# 2. Clean the sentences
# 3. Return word pairs in the format: [ENGLISH, SPANISH]
def create_dataset(path, num_examples):
    
    df = pd.read_excel(path)
    NL = df['NL']
    SQL = df['SQL']
    Processed_NL = []
    Processed_SQL = []
    for line in NL:
        Processed_NL.append(preprocess_sentence(line).lower())
    for line in SQL:
        Processed_SQL.append(preprocess_sentence(line))    
        
        
        
    

    return Processed_NL,Processed_SQL


# In[106]:


def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='',lower=False)
    lang_tokenizer.fit_on_texts(lang)

    tensor = lang_tokenizer.texts_to_sequences(lang)

    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                         padding='post')

    return tensor, lang_tokenizer


# In[107]:


def load_dataset(path, num_examples=None):
  # creating cleaned input, output pairs
    inp_lang,targ_lang = create_dataset(path, num_examples)

    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)

    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer


# In[108]:


input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(path_to_file)

print(targ_lang.word_index)

max_length_targ, max_length_inp = target_tensor.shape[1], input_tensor.shape[1]


# In[109]:


# Creating training and validation sets using an 80-20 split
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)

# Show length
print(len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val))


# In[110]:


BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 64
steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
embedding_dim = 256
units = 512
vocab_inp_size = len(inp_lang.word_index)+1
vocab_tar_size = len(targ_lang.word_index)+1

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)


# In[111]:


encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)
optimizer = tf.keras.optimizers.Adam()


# In[112]:


checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)


# In[113]:


# restoring the latest checkpoint in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
print(targ_lang.index_word)


# In[114]:


def evaluate(sentence):
    

    sentence = preprocess_sentence(sentence)

    inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                         maxlen=max_length_inp,
                                                         padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)

    for t in range(max_length_targ):
        predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                         dec_hidden,
                                                         enc_out)

    

        predicted_id = tf.argmax(predictions[0]).numpy()

        if(targ_lang.index_word[predicted_id] != '<end>'):
            result += targ_lang.index_word[predicted_id] + ' '

        if targ_lang.index_word[predicted_id] == '<end>':
            return result, sentence

    # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence


# In[115]:


def translate(sentence):
    result, sentence = evaluate(sentence)

    print('Input:', sentence)
    print('Predicted translation:', result)
    return result

  







app = Flask(__name__)



@app.route('/')
def hello():
    return render_template('index.html')


def load_data(file_name):
    df = pd.read_excel(file_name)
    corpus = df['NL']
    corp_sentences = [x.lower() for x in corpus]
    return corp_sentences


#Module for generating ngrams dictionary from the text
def get_ngrams_from_text(n,text):
    frequencies = Counter([])
    for s in text:
        token = nltk.word_tokenize(s)
        n_grams = ngrams(token,n)
        frequencies += Counter(n_grams)

    return  frequencies

#Module for generating predictions for the entered text
def get_predictions(n,text,ngram_model):
    vals = text.split(" ")
    predicted_vals = []
    for elem in ngram_model.elements():
        if n==2:
            if(elem[0] == vals[0]):
                next_vals = []
                next_vals.append(elem[0])
                next_vals.append(elem[1])

                if next_vals not in predicted_vals:
                    predicted_vals.append(next_vals)
        elif n==3:
            if(elem[0] == vals[0] and elem[1] == vals[1] ):
                next_vals = []
                next_vals.append(elem[0])
                next_vals.append(elem[1])
                next_vals.append(elem[2])
                if next_vals not in predicted_vals:
                    predicted_vals.append(next_vals)
        elif n==4:
            if(elem[0] == vals[0] and elem[1] == vals[1] and elem[2] == vals[2] ):
                next_vals = []
                next_vals.append(elem[0])
                next_vals.append(elem[1])
                next_vals.append(elem[2])
                next_vals.append(elem[3])
                if next_vals not in predicted_vals:
                    predicted_vals.append(next_vals)            
    predict = {}
    for values in predicted_vals:
        if n==2:
            predict[values[1]] = ngram_model[tuple(values)]
        elif n==3:
            predict[values[2]] = ngram_model[tuple(values)]
        elif n==4:
            predict[values[3]] = ngram_model[tuple(values)]

    return sorted(predict, key=predict.get, reverse=True)[:3]        



corpus = load_data('All_data_NL_SQL_Answer.xlsx')
two_gram_model = get_ngrams_from_text(2,corpus)
three_gram_model = get_ngrams_from_text(3,corpus)
four_gram_model = get_ngrams_from_text(4,corpus)

def get_next_word(text):
     
     txt_len = len(text.split(" "))
     print(txt_len)
     if(txt_len>4):
         text = " ".join(text.split(" ")[-4:])
         txt_len=4
     print(text)
   
    
     if txt_len ==2:
         vals = get_predictions(txt_len,text,two_gram_model)
     elif txt_len ==3:
         vals = get_predictions(txt_len,text,three_gram_model)
     else:
         vals = get_predictions(txt_len,text,four_gram_model)
     
     
         
     return vals


def fetch_results(query):
    inpsql3 = sqlite3.connect('college_2/college_2.sqlite')
    sql3_cursor = inpsql3.cursor()
    sql3_cursor.execute(query)
    columnNames = [d[0] for d in sql3_cursor.description]
    result = "<table border=1> <tr>"
    for col in columnNames:
        result+= "<th> "+ col +"</th>"
    result+= "</tr>"
    for res in sql3_cursor:
        result+= "<tr>"
        for i in range(len(columnNames)):
            result+= "<td>"+ str(res[i]) + "</td>"
            
        result+= "</tr>"
        
    inpsql3.close()     
    result += "</table>"
    return result
    



@app.route('/autocomplete',methods = ['POST'])
def autocomplete():
   if request.method == 'POST':
     
     
        qual = request.form['search']

        new_vals = []

        oov = ""

        for val in qual.lower().split():
         
          if((val in inp_lang.word_index.keys())):
              new_vals.append(val)
          else:
              oov = val + " "
              
        qual1 = str(' ').join(new_vals)

        qual1 += " "
        print("Called")
        print(qual)
        vals = get_next_word(qual1.lower())
        print(vals)
        options = ""
        for val in vals:
          options += '<option value="' + qual + val + '"></option>'
        print(options)    
        return options


  
     
@app.route('/generatesql',methods = ['POST'])
def generatesql():
   if request.method == 'POST':
     
     
      text = request.form['nl']

      vals = text.lower().split()
      
      new_vals = []

      for val in vals:
         
          if(val in inp_lang.word_index.keys()):
              new_vals.append(val)
              
      vals = str(' ').join(new_vals)
      
      sql = translate(vals)
      print(sql)
      return sql

    


@app.route('/generateresults',methods = ['POST'])
def generateresults():
   if request.method == 'POST':
     
     
      text = request.form['query']
      
      
      
      results = fetch_results(text)
      print(results)
      return results    



if __name__ == '__main__':
    
    app.run()   
   
  





