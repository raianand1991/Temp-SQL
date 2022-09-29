import sqlite3
from pathlib import Path
import os.path
#from textaugment import Wordnet
import nltk
import pandas as pd
import re
from pandas import ExcelWriter
from collections import defaultdict

# Reading the template file
print("reading the template data...")
tdf = pd.read_excel('template.xls')
#tdf = tdf.sort_values(by = ['NeedsTemplate'], ascending=False)
NL = tdf['NL'].values
SQL = tdf['SQL'].values
NT = tdf['NeedsTemplate'].values

# Reading the synonyms file
print("reading the synonyms data...")
df = pd.read_excel('synonyms.xlsx')
df = df.sort_values(by = ['Synonym'], ascending=False)
words = df['word'].values
syns = df['Synonym'].values
wss = []
for i in range(len(words)):
    syn = str(syns[i])
    if syn != "" and syn != 'nan':
        word = words[i]
        ss = syn.split(',')
        for s in ss:
            wss.append([word,s.strip()])

            
# Augmenting the NL queries by replacing the unique words present in Synonym data file with their synonyms
outdf = pd.DataFrame(columns=["NL","SQL","NeedsTemplate"])
row = 0

#for each templates in the excel file
for i in range (len(NL)):
    nl = NL[i]
    #print(nl, SQL[i],NT[i])
    outdf.loc[row] = [nl, SQL[i],NT[i]]
    row = row + 1
    for ws in wss:
        if ws[0].lower() in nl.lower():
            #print(ws)
            pattern = re.compile(ws[0], re.IGNORECASE)
            nnl = pattern.sub(ws[1], nl)
            outdf.loc[row] = [nnl, SQL[i],NT[i]]
            row = row + 1

# Storing the augmented template in to a separate file
writer = ExcelWriter('AugmentedTemplate.xlsx')
outdf.to_excel(writer,'Sheet1')
writer.save()

