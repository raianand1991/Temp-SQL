import nltk
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
import os
import pandas
from PyDictionary import PyDictionary
import re
from pandas import ExcelWriter

# Load the Template file
df = pandas.read_excel('template.xls')
print (df.columns)
NL = df['NL'].values

# Retrieve the unique words from the NL queries
uniquewords = []

for nlq in NL:
    nlq = re.sub("{\w*.\w*}","", nlq)
    
    words = word_tokenize(nlq)
    pos = (nltk.pos_tag(words))
    for i in range (len(pos)):
        word = pos[i][0]
        
        if str(pos[i][1]).count("NN") > 0:
            if word not in uniquewords:
                print(pos[i])
                uniquewords.append(word)
uniquewords.sort()
print(uniquewords)

# Find Synonyms of the unique words
outdf = pandas.DataFrame(columns=["word","Dict","WNet"])
row = 0
for w in uniquewords:
    print(w)
    dictionary=PyDictionary(w)
        
    synonyms = []
    for syn in wordnet.synsets(w):
        for l in syn.lemmas():
            synonyms.append(l.name())
    
    outdf.loc[row] = [w , str(dictionary.getSynonyms()) , str(set(synonyms))]
    row=row+1

    
# Store the unique words and respective synonyms in the output file
writer = ExcelWriter('Synonyms.xlsx')
outdf.to_excel(writer,'Sheet1')
writer.save()

import winsound
frequency = 2500  # Set Frequency To 2500 Hertz
duration = 1000  # Set Duration To 1000 ms == 1 second
winsound.Beep(frequency, duration)
