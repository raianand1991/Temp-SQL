import pandas as pd
#converts NL-SQL pairs in the SPIDER JSON file to CSV file
df = pd.read_json('train_spider.json')
df.to_csv("SpiderJSON_to_NL_SQL_Pairs.csv")
