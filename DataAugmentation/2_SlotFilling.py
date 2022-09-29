import sqlite3
from pathlib import Path
import os.path
#from textaugment import Wordnet
import nltk
import pandas as pd
import re
from pandas import ExcelWriter

# connect to the college_2 database
db = sqlite3.connect("c:\P\database\college_2\college_2.sqlite")
print("database opened")
cursor = db.cursor()

# load the template data
print("reading the template data...")
df = pd.read_excel('Augmentedtemplate.xlsx')
df = df.sort_values(by = ['NeedsTemplate'], ascending=False)
NL = df['NL'].values
SQL = df['SQL'].values


outdf = pd.DataFrame(columns=["NL","SQL","Answer","select","where","join","and","or","order","distinct","group"])
row = 0
                           
# for each templates in the excel file
for i in range (len(NL)):
    name_values = []
    #print("")
    #print("Template " + str(i+1))
    nl = NL[i]
    #print(nl)
    sql = SQL[i]
    #print(sql)
    #print("")
    
    # find all the Fields/Attributes used in the template
    fieldnames = re.findall("{\w*.\w*}", nl)
    #print(fieldnames)
    num_field_names = len(fieldnames)
    
    # create a list of attributes and respective unique values from the database    
    if num_field_names > 0:
        name_values1 = []
        field_name_1 = fieldnames[0]
        fn1 = field_name_1.replace("{","")
        fn1 = fn1.replace("}","")
        tn1 = fn1.split(".")[0]
        fn1 = fn1.split(".")[1]
        #Get unique values of Field Name 1
        qry1 = "SELECT DISTINCT " + fn1 + " from " + tn1
        #print(qry1)
        cursor.execute(qry1)
        distinct_values_1 = cursor.fetchall()
        #print(distinct_values_1)
        for distinct_value_1 in distinct_values_1:
            dv1 = ""
            if type(distinct_value_1[0]) == type(""):
                dv1 = "'" + distinct_value_1[0] + "'"
            else:
                dv1 = str(distinct_value_1[0])
            name_values1.append([tn1+"."+fn1,dv1])
        name_values = name_values1
    if num_field_names > 1:
        name_values2 = []
        field_name_2 = fieldnames[1]
        fn2 = field_name_2.replace("{","")
        fn2 = fn2.replace("}","")
        tn2 = fn2.split(".")[0]
        fn2 = fn2.split(".")[1]
        #Get unique values of Field Name 2 
        for distinct_value_1 in distinct_values_1:
            dv1 = ""
            if type(distinct_value_1[0]) == type(""):
                dv1 = "'" + distinct_value_1[0] + "'"
            else:
                dv1 = str(distinct_value_1[0])
            #print(fn1,dv1)
            qry2 = "SELECT DISTINCT " + fn2 + " from " + tn2
            #print(qry2)
            cursor.execute(qry2)
            distinct_values_2 = cursor.fetchall()
            for distinct_value_2 in distinct_values_2:
                dv2 = ""
                if type(distinct_value_2[0]) == type(""):
                    dv2 = "'" + distinct_value_2[0] + "'"
                else:
                    dv2 = str(distinct_value_2[0])
                #print(dv2)
                name_values2.append([tn1+"."+fn1,dv1,tn2+"."+fn2,dv2])
        name_values = name_values2
    if num_field_names > 2:
        name_values3 = []
        field_name_3 = fieldnames[2]
        fn3 = field_name_3.replace("{","")
        fn3 = fn3.replace("}","")
        tn3 = fn3.split(".")[0]
        fn3 = fn3.split(".")[1]
        #Get unique values of Field Name 3 
        for name_value in name_values:
            fn1 = name_value[0]
            dv1 = str(name_value[1])
            fn2 = name_value[2]
            dv2 = str(name_value[3])
            qry3 = "SELECT DISTINCT " + fn3 + " from " + tn3 
            cursor.execute(qry3)
            distinct_values_3 = cursor.fetchall()
            for distinct_value_3 in distinct_values_3:
                dv3 = ""
                if type(distinct_value_3[0]) == type(""):
                    dv3 = "'" + distinct_value_3[0] + "'"
                else:
                    dv3 = str(distinct_value_3[0])
                name_values3.append([fn1,dv1,fn2,dv2,tn3+"."+fn3,dv3])
        name_values = name_values3
    if num_field_names > 3:
        print("*************ERROR ***********")

    # Create NL and SQL pairs by searching and replacing the attributes with the unique values
    for name_value in name_values:
        NLQ = nl
        for ii in range (num_field_names):
            NLQ = NLQ.replace("{"+name_value[ii*2]+"}",name_value[(ii*2)+1].replace("'",""))
        #print (NLQ)
        SQLQ = sql
        for ii in range (num_field_names):
            SQLQ = SQLQ.replace("{"+name_value[ii*2]+"}",name_value[(ii*2)+1])
        #print (SQLQ)
        cursor.execute(SQLQ)
        Answer = cursor.fetchall()
        #print("Answer : " + str(Answer))
        #print("")
        c_sel = SQLQ.lower().count("select")
        c_where = SQLQ.lower().count("where")
        c_join = SQLQ.lower().count(" join ")
        c_and = SQLQ.lower().count(" and ")
        c_or = SQLQ.lower().count(" or ")
        c_order = SQLQ.lower().count("order by")
        c_dist = SQLQ.lower().count("distinct")
        c_grp = SQLQ.lower().count("group by")
        
        outdf.loc[row] = [NLQ , SQLQ , str(Answer),c_sel,c_where,c_join,c_and,c_or,c_order,c_dist,c_grp]
        row=row+1
        
    if num_field_names == 0:
        cursor.execute(sql)
        Answer = cursor.fetchall()
        
        c_sel = SQLQ.lower().count("select")
        c_where = SQLQ.lower().count("where")
        c_join = SQLQ.lower().count(" join ")
        c_and = SQLQ.lower().count(" and ")
        c_or = SQLQ.lower().count(" or ")
        c_order = SQLQ.lower().count("order by")
        c_dist = SQLQ.lower().count("distinct")
        c_grp = SQLQ.lower().count("group by")
        
        outdf.loc[row] = [nl , sql , str(Answer),c_sel,c_where,c_join,c_and,c_or,c_order,c_dist,c_grp]
        row=row+1

# store the augmented data in the output file
writer = ExcelWriter('All_data_NL_SQL_Answer.xlsx')
outdf = outdf.sort_values(['select', 'where','join','and','or',"order","distinct","group"], ascending=[True, True,True,True,True,True,True,True])
outdf.to_excel(writer,'Sheet1')
writer.save()

# wake me up when done, good night !!
import winsound
frequency = 2500  # Set Frequency To 2500 Hertz
duration = 500  # Set Duration To 1000 ms == 1 second
winsound.Beep(frequency, duration)



            
    

    










