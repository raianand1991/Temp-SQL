#!/usr/bin/env python
# coding: utf-8

# In[130]:


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


# In[131]:


path_to_file = "All_data_NL_SQL_Answer.xlsx"


# In[132]:


# Converts the unicode file to ascii
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                 if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
    w = unicode_to_ascii(w.strip())
    
  # creating a space between a word and the punctuation following it
  # eg: "he is a boy." => "he is a boy ."
  # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
   # w = re.sub(r"([?.!,¿])", r" \1 ", w)
   # w = re.sub(r'[" "]+', " ", w)
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


# In[133]:


# 1. Remove the accents
# 2. Clean the sentences
# 3. Return word pairs in the format: [NL, SQL]
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


# In[134]:


NL, SQL = create_dataset(path_to_file, None)
print(NL[0])
print(SQL[0])


# In[135]:
#Tokeninzing the Input NL & SQL and making them of uniqform 

def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='',lower=False)
    lang_tokenizer.fit_on_texts(lang)

    tensor = lang_tokenizer.texts_to_sequences(lang)

    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                         padding='post')

    return tensor, lang_tokenizer


# In[136]:


def load_dataset(path, num_examples=None):
  # creating cleaned input, output pairs
    inp_lang,targ_lang = create_dataset(path, num_examples)

    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)

    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer


# In[137]:


# Try experimenting with the size of that dataset
num_examples = 10000
input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(path_to_file,
                                                                num_examples)

# Calculate max_length of the target tensors
max_length_targ, max_length_inp = target_tensor.shape[1], input_tensor.shape[1]


# In[138]:


# Creating training and validation sets using an 80-20 split
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2,random_state=42)

# Show length
print(len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val))


# In[140]:


BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 64
steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
embedding_dim = 256
units = 512
vocab_inp_size = len(inp_lang.word_index)+1
vocab_tar_size = len(targ_lang.word_index)+1

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)


# In[141]:


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


# In[142]:


example_input_batch, example_target_batch = next(iter(dataset))
example_input_batch.shape, example_target_batch.shape


# In[143]:


encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)

# sample input
sample_hidden = encoder.initialize_hidden_state()
sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)
print('Encoder output shape: (batch size, sequence length, units)', sample_output.shape)
print('Encoder Hidden state shape: (batch size, units)', sample_hidden.shape)


# In[144]:


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


# In[145]:


attention_layer = BahdanauAttention(7)
attention_result, attention_weights = attention_layer(sample_hidden, sample_output)

print("Attention result shape: (batch size, units)", attention_result.shape)
print("Attention weights shape: (batch_size, sequence_length, 1)", attention_weights.shape)


# In[146]:


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


# In[147]:


decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

sample_decoder_output, _, _ = decoder(tf.random.uniform((BATCH_SIZE, 1)),
                                      sample_hidden, sample_output)

print('Decoder output shape: (batch_size, vocab size)', sample_decoder_output.shape)


# In[148]:


optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                            reduction='none')


def loss_function(real, pred):
    
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


# In[149]:


checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)


# In[150]:


@tf.function
def train_step(inp, targ, enc_hidden):
    loss = 0

    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)

        dec_hidden = enc_hidden

        dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * BATCH_SIZE, 1)

    # Teacher forcing - feeding the target as the next input
        for t in range(1, targ.shape[1]):
      # passing enc_output to the decoder
          predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

          loss += loss_function(targ[:, t], predictions)

      # using teacher forcing
          dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = (loss / int(targ.shape[1]))

    variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, variables)

    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss


# In[ ]:
# restoring the latest checkpoint in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

EPOCHS = 10

for epoch in range(EPOCHS):
    start = time.time()

    enc_hidden = encoder.initialize_hidden_state()
    total_loss = 0

    for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
        batch_loss = train_step(inp, targ, enc_hidden)
        total_loss += batch_loss

        if batch % 100 == 0:
            print(f'Epoch {epoch+1} Batch {batch} Loss {batch_loss.numpy():.4f}')
  # saving (checkpoint) the model every 2 epochs
    if (epoch + 1) % 1 == 0:
        checkpoint.save(file_prefix=checkpoint_prefix)

    print(f'Epoch {epoch+1} Loss {total_loss/steps_per_epoch:.4f}')
    print(f'Time taken for 1 epoch {time.time()-start:.2f} sec\n')


# In[ ]:


# restoring the latest checkpoint in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


# In[ ]:


def evaluate(sentence):
    

    

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

        if targ_lang.index_word[predicted_id] != '<end>':
            result += targ_lang.index_word[predicted_id] + ' '

        if targ_lang.index_word[predicted_id] == '<end>':
            return result

    # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    return result


# In[ ]:


def decode_target(vals):
    result=""
    for val in vals:
        
        if  val!=0 and  val!=inp_lang.word_index['<start>'] and val!=inp_lang.word_index['<end>']:
            result+= inp_lang.index_word[val] + " "
            
    return result


# In[ ]:


def decode_prediction(vals):
    res=""
    for val in vals:
        if  val!=0 and  val!=targ_lang.word_index['<start>'] and val!=targ_lang.word_index['<end>']:
            res+= targ_lang.index_word[val] + " "
            
    return res


# In[ ]:


print(inp_lang.word_index)


# In[ ]:


from nltk.translate.bleu_score import sentence_bleu


# In[ ]:


def extract_nums(text):
    text = re.sub(r"([?.!,])", r" \1 ", text)
    text = re.sub(r'([0-9]) . ([0-9])', r"\1.\2", text)
    text = re.sub(r'[" "]+', " ", text)
    text = " " + text.strip() + " "
    cleantext = text
    
    floats = re.findall(r' \d+\.\d+ ', text)
    for f in floats:
        text = re.sub(f,"",text)
    ints = re.findall(r' \d+ ', text)
    nums_in_text = floats + ints
    return nums_in_text, cleantext

def fix_numerical_mismatch(nl, pred_sql):
    nums_in_nl, nl = extract_nums(nl)
    nums_in_sql, sql = extract_nums(pred_sql)
    sr = list(set(nums_in_nl).symmetric_difference(set(nums_in_sql)))

    if len(sr)==2:
        #print(sql)
        if str(sr[0]) in sql :
            #print(sr[0],">>",sr[1])
            corr_sql = sql.replace(str(sr[0]),str(sr[1]))
            return corr_sql.strip()
        
        if str(sr[1]) in sql :
            #print(sr[1],">>",sr[0])
            corr_sql = sql.replace(str(sr[1]),str(sr[0]))
            return corr_sql.strip()

    return pred_sql.strip()


# In[ ]:


#Train Accuracy
avg_train_accuracy = 0.0
avg_tain_bleu_score = 0.0
training_data = []
for i in range(len(input_tensor_train)):
    inp = decode_target(input_tensor_train[i])
    pred = evaluate(inp.strip())
    pred = fix_numerical_mismatch(inp.strip(),pred)
    targ = decode_prediction(target_tensor_train[i])
    training_data.append([inp,targ.strip(),pred])
    print(pred)
    print(targ)
    print()
    if (pred.strip() == targ.strip()):
        avg_train_accuracy+=1
         
    avg_tain_bleu_score+= sentence_bleu([targ.strip().split(' ')],pred.strip().split(' '))
    print(f'Sample {i} Train Accuracy {avg_train_accuracy/(i+1):.4f}  Training BLEU {avg_tain_bleu_score/(i+1):.4f}')    


    
    


print(f'Train Accuracy {avg_train_accuracy/len(input_tensor_train):.4f}  Training BLEU {avg_tain_bleu_score/len(input_tensor_train):.4f}')    


# In[ ]:


#Test Accuracy
avg_test_accuracy = 0.0
avg_test_bleu_score = 0.0
test_data = []
for i in range(len(input_tensor_val)):
    inp = decode_target(input_tensor_val[i])
   
    pred = evaluate(inp.strip())
    pred = fix_numerical_mismatch(inp.strip(),pred)
    targ = decode_prediction(target_tensor_val[i])
    
    test_data.append([inp,targ.strip(),pred])
    if (pred.strip() == targ.strip()):
        avg_test_accuracy+=1
         
    avg_test_bleu_score+= sentence_bleu([targ.strip().split(' ')],pred.strip().split(' '))
    print(f'Sample {i} Test Accuracy {avg_test_accuracy/(i+1):.4f}  Test BLEU {avg_test_bleu_score/(i+1):.4f}')    


    
    


print(f'Test Accuracy {avg_test_accuracy/len(input_tensor_val):.4f}  Test BLEU {avg_test_bleu_score/len(input_tensor_val):.4f}')    


# In[ ]:


import xlsxwriter


# In[ ]:


with xlsxwriter.Workbook('training_data_all.xlsx') as workbook:
    worksheet = workbook.add_worksheet()

    for row_num, data in enumerate(training_data):
        worksheet.write_row(row_num, 0, data)


# In[ ]:


with xlsxwriter.Workbook('test_data_all.xlsx') as workbook:
    worksheet = workbook.add_worksheet()

    for row_num, data in enumerate(test_data):
        worksheet.write_row(row_num, 0, data)


# In[ ]:




