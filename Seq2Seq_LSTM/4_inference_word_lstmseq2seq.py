"""
## Setup
"""
print("loading libs...")
import numpy as np
import tensorflow as tf
from tensorflow import keras
import re
import random
import pandas as pd
import csv
import nltk

vecs = []
print("loading Glove model...")
glove_data_file = 'glove.6B.50d.txt'
glove_words = pd.read_table(glove_data_file, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
        
def vec(w):
    try:
        return glove_words.loc[w].to_numpy()
    except:
        return vec(".")

def find_nearest(word,input_characters):
    if len(vecs) == 0:
        for ww in input_characters:
            try:
                vecs.append(vec(ww))
            except:
                vecs.append(vec("."))
    diff = vecs - vec(word)
    delta = np.sum(diff * diff, axis=1)
    i = np.argsort(delta)
    print(word + " >>> " + input_characters[i[0]])
    return input_characters[i[0]]

"""
## Configuration
"""
num_samples = 0  # Number of samples to train on.
batch_size = 0#64  # Batch size for training.
epochs = 10  # Number of epochs to train for.
latent_dim = 128  # Latent dimensionality of the encoding space.
all_data_path = 'All_data_NL_SQL_Answer.xlsx'
train_data_path = 'Training_data_NL_SQL_Answer.xlsx'
#test_data_path = 'Training_data_NL_SQL_Answer.xlsx'
test_data_path = 'Test_data_NL_SQL_Answer.xlsx'
start = "<<start>>"
end = "<<end>>"
batches = [1,2,4,8,16,32,64,128,256,512]
c_sel = 1
c_where = 1
c_join = 1
c_and = 1
c_or = 1
c_order = 1
c_dist = 1
c_grp = 1

"""
## Prepare the data
"""
def clean(line):
    line = line.replace("(" , " ( ")
    line = line.replace(")" , " ) ")
    line = line.replace(",", " , ")
    line = line.replace("=", " = ")
    line = line.replace("! =", "!=")
    line = line.replace("!=", " != ")
    line = line.replace(">", " > ")
    line = line.replace("<", " < ")
    line = line.replace("?", " ? ")
    line = line.replace("  ", " ")
    line = line.replace("  ", " ")
    line = line.replace("  ", " ")
    line = line.replace("  ", " ")
    line = line.strip()
    return line

# Vectorize the data.
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()

print("reading the data set...")
df = pd.read_excel(all_data_path)
nl = df['NL'].values
sql = df['SQL'].values
cnt = len(nl)

print("cleaning all data...")
#extract unique words from NL and SQL
for i in range(cnt):
    #clean 
    nlq = clean(nl[i])
    #clean and then add start and end tokens
    sqlq = start + " " + clean(sql[i]) + " " + end
    for word in nlq.split(" "):
        if word not in input_characters and word != "":
            input_characters.add(word)

    for word in sqlq.split(" "):
        if word not in target_characters and word != "":
            target_characters.add(word)

'''
    # if the counts of the words match the requirement then add the NL and SQL to the train data set
    if c_sel >= sqlq.lower().count("select"):
        if  c_where >= sqlq.lower().count("where"):
            if c_join >= sqlq.lower().count(" join "):
                if c_and >= sqlq.lower().count(" and "):
                    if c_or >= sqlq.lower().count(" or "):
                        if c_order >= sqlq.lower().count("order by"):
                            if c_dist >= sqlq.lower().count("distinct"):
                                if c_grp >= sqlq.lower().count("group by"):
                                    input_texts.append(nlq)
                                    target_texts.append(sqlq)
                                    '''

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
input_characters_lower = [w.lower() for w in input_characters]

num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt.split(" ")) for txt in nl])
max_decoder_seq_length = max([len(txt.split(" ")) for txt in sql])

print("Number of unique input tokens:", num_encoder_tokens)
print("Number of unique output tokens:", num_decoder_tokens)
print("Max sequence length for inputs:", max_encoder_seq_length)
print("Max sequence length for outputs:", max_decoder_seq_length)

modelname = "s2s" + "_" +  str(latent_dim) + "_" + str(c_sel) + "_" + str(c_where) + "_" + str(c_join) + "_" + str(c_and) + "_" + str(c_or) + "_" + str(c_order) + "_" + str(c_dist) + "_" + str(c_grp) + "_" + str(num_encoder_tokens) + "_" + str(num_decoder_tokens) 

input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])


'''
Load Test Data
'''

print("reading the test data set...")
test_df = pd.read_excel(test_data_path)
test_nl = test_df['NL'].values
test_sql = test_df['SQL'].values
test_cnt = len(test_nl)
test_input_texts = []
test_target_texts = []
                                    
print("cleaning test data...")
#extract unique words from NL and SQL
for i in range(test_cnt):
    #clean 
    nlq = clean(test_nl[i])
    #clean and then add start and end tokens
    sqlq = start + " " + clean(test_sql[i]) + " " + end
    # check if out of vocab
    for word in nlq.split(" "):
        if word not in input_characters:
            print("Out of Input Vocab : " + word)

    for word in sqlq.split(" "):
        if word not in target_characters:
            print("Out of Target Vocab : " + word)

    # if the counts of the words match the requirement then add the NL and SQL to the train data set
    if c_sel >= sqlq.lower().count("select"):
        if  c_where >= sqlq.lower().count("where"):
            if c_join >= sqlq.lower().count(" join "):
                if c_and >= sqlq.lower().count(" and "):
                    if c_or >= sqlq.lower().count(" or "):
                        if c_order >= sqlq.lower().count("order by"):
                            if c_dist >= sqlq.lower().count("distinct"):
                                if c_grp >= sqlq.lower().count("group by"):
                                    test_input_texts.append(nlq)
                                    test_target_texts.append(sqlq)

test_encoder_input_data = np.zeros((len(test_input_texts), max_encoder_seq_length, num_encoder_tokens), dtype="float32")
for i, (input_text, target_text) in enumerate(zip(test_input_texts, test_target_texts)):
    for t, char in enumerate(input_text.split(" ")):
        test_encoder_input_data[i, t, input_token_index[char]] = 1.0

"""
## Build the model
"""


try:
    print("")
    print("loading model from the disk ...")
    model = keras.models.load_model(modelname)
except:
    print("#######could not load model from file#######")
    print("")

while (True):

    """
    ## Run inference (sampling)

    1. encode input and retrieve initial decoder state
    2. run one step of decoder with this initial state
    and a "start of sequence" token as target.
    Output will be the next target token.
    3. Repeat with the current target token and current states
    """

    # Define sampling models
    # Restore the model and construct the encoder and decoder.
    #model = keras.models.load_model(modelname)

    encoder_inputs = model.input[0]  # input_1
    encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output  # lstm_1
    encoder_states = [state_h_enc, state_c_enc]
    encoder_model = keras.Model(encoder_inputs, encoder_states)

    decoder_inputs = model.input[1]  # input_2
    decoder_state_input_h = keras.Input(shape=(latent_dim,), name="input_3")
    decoder_state_input_c = keras.Input(shape=(latent_dim,), name="input_4")
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_lstm = model.layers[3]
    decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs
    )
    decoder_states = [state_h_dec, state_c_dec]
    decoder_dense = model.layers[4]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = keras.Model(
        [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
    )

    # Reverse-lookup token index to decode sequences back to
    # something readable.
    reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
    reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())


    def decode_sequence(input_seq):
        # Encode the input as state vectors.
        states_value = encoder_model.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        # Populate the first character of target sequence with the start character.
        #print(target_token_index)
        target_seq[0, 0, target_token_index[start]] = 1.0

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ""
        while not stop_condition:
            output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = reverse_target_char_index[sampled_token_index]
            if sampled_char != end:
                decoded_sentence = decoded_sentence + " " + sampled_char

            # Exit condition: either hit max length
            # or find stop character.
            if sampled_char == end or len(decoded_sentence.split(" ")) > max_decoder_seq_length:
                #print("last char = " + sampled_char + " length = " + str(len(decoded_sentence)))
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.0

            # Update states
            states_value = [h, c]
        return decoded_sentence.replace(start,"").replace(end,"").strip()


    """
    You can now generate decoded sentences as such:
    """
    tot_bleu_score = 0
    tot_acc_score = 0
    print("Running Inference on Test data...")
    print("")
    cnt_test_data = len(test_input_texts)
    for ii in range(cnt_test_data):
        #ii = random.randint(0,cnt_test_data-1)
        line = clean(test_input_texts[ii])
        l=""
        for word in line.split(" "):
            if word in input_characters:
                l = l+ " " + word
            elif word in input_characters_lower:
                # if there is Case mismatch then follow the Case in the Training data
                l = l + " " + input_characters[input_characters_lower.index(word)]                
            else:
                # replace with the nearest word in the Training data
                l = l + " " + find_nearest(word.lower(),input_characters)
        line = clean(l)
        encoder_val_data = np.zeros((1, max_encoder_seq_length, num_encoder_tokens), dtype="float32")
        for t, char in enumerate(clean(line).split(" ")):
            encoder_val_data[0, t, input_token_index[char]] = 1.0

        decoded_sentence = decode_sequence(encoder_val_data)
        print(line)
        
        ori_sql = test_target_texts[ii].replace(start,"").replace(end,"").strip()
        pred_sql = decoded_sentence
        print(ori_sql)
        print(pred_sql)
        hypothesis = ori_sql.split(" ")
        reference = pred_sql.split(" ")
        BLEUscore = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis)
        tot_bleu_score = tot_bleu_score + BLEUscore
        print("BLEUscore = " + str(BLEUscore))
        print("")
        if BLEUscore == 1:
            tot_acc_score = tot_acc_score + 1
    avg_bleu_score = tot_bleu_score / cnt_test_data
    print("Avg BLEUscore = " + str(avg_bleu_score))
    print("Accuracy = " + str(tot_acc_score / cnt_test_data))
    break

