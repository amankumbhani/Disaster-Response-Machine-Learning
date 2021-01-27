# import libraries
import sys
import re
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Loads the data from two CSV files - messages.csv, categories.csv & merges 
    them using an inner merge on "ID's" which are the same
    
    Args:
    messages_filepath: Path to the messages.csv file
    categories_filepath: Path to the categories.csv file
    
    Returns:
    A merged dataframe containing data from both, messages.csv & categories.csv
    '''
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how='inner', on='id')
    
    return df

def clean_data(df):
    '''
    Cleans the merged dataframe to store it in an SQLite Database
    This cleaned data can directly be used for training any classifier
    
    Args:
    df: Merged dataframe to be cleaned
    
    Returns:
    df: A clean master dataframe
    '''
    
    # Splits the categories column using a semicolon into a Series object containing all split elements
    categories = df['categories'].str.split(";", expand=True)
    
    # Removes the last two characters at the end of the row variable to obtain a clean column name
    categories.columns = categories.iloc[0].apply(lambda x: x[:-2])
    
    # Set values under each column to the values as the last character of the string
    for column in categories:
        # set each value to be the last character of the string & set them as an integer
        categories[column] = (categories[column].astype(str).str[-1:]).astype(int)
    
    # Drop the original categories column
    df.drop('categories', inplace=True, axis=1)
    
    # Concatenate the master dataframe with the new categories columns ( 36 of them ) 
    df = pd.concat([df, categories], axis=1)
    
    # Convert values of the categories columns into binary form for MultiClassClassification
    df.drop_duplicates(df.drop(df[df['related'] == 2].index, inplace=True), inplace=True)
    
    return df
    
def save_data(df, database_filename):
    '''
    Saves the master dataframe into a SQLite database

    Args: 
    df: Master cleaned dataframe
    database_filename: The database filename to be saved as
    
    Returns:
    None
    '''
    
    # Creates an SQLite Engine using the database filename passed by user as an argument
    engine = create_engine('sqlite:///'+ str(database_filename))
    
    # Saves the dataframe to the SQLite database with the given name
    # If the database already exists, it gets replaced
    df.to_sql(database_filename, engine, index=False, if_exists='replace')

def main():
    '''
    Main function
    
    Args: 
    None
    
    Returns:
    None
    '''
    
    # Requires three arguments
    if len(sys.argv) == 4:

        # Requires path to messages.csv, categories.csv & the name of the database to save DF as
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        
        # Calls the load_data() function that returns the merged dataframe
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        
        # Calls the clean_data() function that returns the cleaned dataframe
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        
        # Calls the save_data() function that saves the dataframe to an SQLite database
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')

if __name__ == '__main__':
    main()