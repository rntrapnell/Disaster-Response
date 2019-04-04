import pandas as pd
from sqlalchemy import create_engine
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("messages_database" )
parser.add_argument("categories_database" )
parser.add_argument("sql_path", default = 'sqlite:///Database.db')
parser.add_argument("--table_name", default='table1' )
args = parser.parse_args()

# load messages dataset
messages = pd.read_csv(args.messages_database)

# load categories dataset
categories = pd.read_csv(args.categories_database)

# merge datasets
df = messages.merge(categories, on = 'id')


# create a dataframe of the 36 individual category columns
categories = df['categories'].str.split(';', expand=True)


# select the first row of the categories dataframe
row = categories.iloc[0]

# use this row to extract a list of new column names for categories.
# one way is to apply a lambda function that takes everything
# up to the second to last character of each string with slicing
category_colnames = [x[0] for x in row.str.split('-',n = 1)]

# rename the columns of `categories`
categories.columns = category_colnames

for column in categories:
    # set each value to be the last character of the string
    categories[column] = [x[1] for x in categories[column].astype(str).str.split('-',n = 1)]

    # convert column from string to numeric
    categories[column] = [int(x) for x in categories[column]]

# drop the original categories column from `df`
df = df.drop(['categories'], axis = 1)

# concatenate the original dataframe with the new `categories` dataframe
df = pd.concat([df, categories], axis=1)

# drop duplicates
df = df.drop_duplicates()

# save to SQL
engine = create_engine('sqlite:///{}'.format(args.sql_path))
df.to_sql(args.table_name, engine, index=False)
