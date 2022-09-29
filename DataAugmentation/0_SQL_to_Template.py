from moz_sql_parser import parse
import json
import pandas
import pandas as pd
from pandas import ExcelWriter

dummy = []

def isnum(s):
    try:
        f = float(s)
        return True
    except:
        return False

def isliteral(l):
    if len(l) == 1:
        try:
            value = l["literal"]
            return True , value
        except:
            return False, ""

def unfold(col_value):
    ret = []
    for cv in col_value:
        if type(cv) is list:
            uf = unfold(cv)
            for u in uf:
                if type(u) is int:
                    ret.append(uf)
                    break
                else:
                    ret.append(u)
        else:
            return(col_value)
    return ret

def get_colname_value(op, arr):
    ret = []
    col_value = arr
    #print(op)
    if op == "in":
        return [4 , "ex" , "ex"] 
    if (op == "and" or op == "or"):
        for wherelist in arr:
            ret.append(parsewherelist(wherelist))
        return ret
    #print(col_value)

    try:
        if isnum(col_value[1]) :
            col = col_value[0]
            value = col_value[1]
            if type(col) is str:
                ret.append([1,col,value])
            else:
                ret.append([3,col,value])
            return ret
    except:
        pass
    try:
        islit, value = isliteral(col_value[1])
        if islit:
            col = col_value[0]
            if type(col) is str:
                ret.append([1,col,value])
            else:
                ret.append([3,col,value])
            return ret
    except:
        pass
    
    try:
        #print(arr[1]["where"])
        return parsewherelist(arr[1]["where"]) 
    except:
        pass

    #print(op, arr)
    return [4 , "ex" , "ex"] 

def parsewherelist(wherelist):
    ret = []
    for op in wherelist:
        ret.append(get_colname_value(op, wherelist[op]))
    return ret

def findfrom(resp):
    #print(resp)
    #print(type(resp))
    
    if type(resp) is str:
        return None

    if type(resp) is dict:
        try:
            if tables["name"] == tn:
                return tables["value"]
        except:
            for t in tables:
                v = findTvalue(tn, tables[t])
                if v != None:
                    return v
    return None

def gettabcolname(sql):
    if("INTERSECT" in sql):
        sql1 = sql.split("INTERSECT")[0]
        sql2 = sql.split("INTERSECT")[1]
        r1, t1 = gettabcolname(sql1)
        r2 , t2 = gettabcolname(sql2)        
        return [r1,r2],[t1,t2] 
    if("EXCEPT" in sql):
        sql1 = sql.split("EXCEPT")[0]
        sql2 = sql.split("EXCEPT")[1]
        r1, t1 = gettabcolname(sql1)
        r2 , t2 = gettabcolname(sql2)        
        return [r1,r2],[t1,t2] 
    if("UNION" in sql):
        sql1 = sql.split("UNION")[0]
        sql2 = sql.split("UNION")[1]
        r1, t1 = gettabcolname(sql1)
        r2 , t2 = gettabcolname(sql2)        
        return [r1,r2],[t1,t2]                         

    json_str = json.dumps(parse(sql))
    resp = json.loads(json_str)
    #print(resp)
    table =""
    where = ""

    if " IN " in sql:
        return [[4 , "ex" , "ex"], None] 

    try:
        table = (resp['from'])
        #print(table)
    except:
        #print(resp)
        return [2, "par","par","par"] , None
    try:     
        where = (resp['where'])
        #print(where)
    except:
        try:
            where = (resp['having'])
            #print(where)
        except:
            #print(resp)
            #print(tables)
            #print(where)
            return [2, "par","par","par"] , None
            
    ret = parsewherelist(where)
    return ret , table

def findTvalue(tn, tables):
    #print(tables)
    #print(type(tables))
    if type(tables) is str:
        return None

    if type(tables) is dict:
        try:
            if tables["name"] == tn:
                return tables["value"]
        except:
            for t in tables:
                v = findTvalue(tn, tables[t])
                if v != None:
                    return v

    if type(tables) is list:
        for t in tables:
            v = findTvalue(tn,t)
            if v != None:
                return v
    return None

def createsearhreparray(col_values, tables):
    sr = []
    tn = tables

    if type(tables) is list:
        if type(tables[0]) is str:
            tn = tables[0]
        
    for cv in col_values:
        cn = cv[1]
        val = cv[2]
        # if there is "." in the column name that means it is a Join
        if "." in cn:
            tn = cn.split(".")[0]
            print("finding " + tn)
            tn = findTvalue(tn, tables)
            cn = cn.split(".")[1]
            # find table name
        if type(tn) is list and "." not in cn:
            continue
        if tn != None:
            sr.append([tn + "." + cn, val])
            
    return sr



df = pandas.read_excel('college_2.xlsx')
NL = df['NL'].values
SQL = df['SQL'].values


cnt = 0
par = 0
ex = 0
tbd = 0
skip = 0
outdf = pd.DataFrame(columns=["NL","SQL","NeedsTemplate"])
row=0

for i in range (169):
    nt = 0
    nl = NL[i]
    sql = SQL[i]
    #print()
    #print(sql)
    ret, table = (gettabcolname(sql))
    #print (ret)

    if ret == [4, 'ex', 'ex']:
        ex = ex + 1
    elif ret == (3, 'skip', 'skip', 'skip'):
        #print(nl)
        #print(sql)
        #print(ret)
        #print()
        skip = skip +1
    elif ret == [2, 'par', 'par', 'par']:
        #print(sql)
        par = par +1
    elif ret == [(0, '', '')]:
        tbd = tbd +1
    else:
        uf = unfold(ret)        
        if(uf[0][0] == 3):
            #print(nl)
            #print(sql)
            #print(ret)
            #print()
            skip += 1
        else:
            print()
            print(nl)
            print(sql)
            #print(table)
            #print(uf)
            SR = (createsearhreparray(uf,table))
            c = 0
            SRR = []
            for sr in SR:
                add = True
                for srr in SRR:
                    if srr[0] == sr[0]:
                        add = False
                if add:
                    SRR.append(sr)
            print(SRR)
            for sr in SRR:                
                if c >= 2:
                    break
                if type(sr[1]) is str:
                    if str(sr[1]) in nl:
                        nl = nl.replace(str(sr[1]), "{" + str(sr[0])+"}")
                        sql = sql.replace("'" + str(sr[1]) + "'" , "{" + str(sr[0])+"}")
                        nt = 1
                else:
                    if str(sr[1]) in nl:
                        nl = nl.replace(str(sr[1]),"{" + str(sr[0])+"}")
                        sql = sql.replace(str(sr[1]),"{" + str(sr[0])+"}")
                        nt = 1
                c += 1
            print(nl)
            print(sql)
            if nt == 1:
                cnt += 1
            else:
                skip += 1
            
    outdf.loc[row] = [nl, sql, nt]
    row += 1

print()
print("unable to identify  = " + str(tbd))
print("count = " + str(cnt))
print("can not be converted to template = " + str(par + skip + ex))

writer = ExcelWriter('Template.xlsx')
outdf.to_excel(writer,'Sheet1')
writer.save()
