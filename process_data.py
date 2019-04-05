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

def categories_to_columns(df):
    '''Converts column with list of categories to columns of categoriesself.

    The original csv contains a list of categories with each category name followed
    by -0 or -1 indicated whether the column is marked for the message in the row.
    This function converts those lists into seperate columns (one for each category)
    with 0's and 1's in order to prepare the data for machine learning techniques.

    Parameters:
    df - An unconverted dataframe

    Returns:
    df - The converted dataframe

    '''
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0]
    category_colnames = [x[0] for x in row.str.split('-',n = 1)]
    categories.columns = category_colnames

    for column in categories:
        categories[column] = [x[1] for x in categories[column].astype(str).str.split('-',n = 1)]
        categories[column] = [int(x) for x in categories[column]]


    df = df.drop(['categories'], axis = 1)

    df = pd.concat([df, categories], axis=1)
    return(df)
    
df = categories_to_columns(df)

# drop duplicates
df = df.drop_duplicates()

# save to SQL
engine = create_engine('sqlite:///{}'.format(args.sql_path))
df.to_sql(args.table_name, engine, index=False)
