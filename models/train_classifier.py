# import libraries
import sys
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
import re
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(database_filepath):
    '''
    Loads the clean dataset from the database whose filepath is user mentioned
    
    Args: 
    database_filepath: The input filepath to the database

    Returns:
    X, y: The messages & output labels (35 classes)
    category_names: Category names of the output (35 classes)
    '''

    # Load data from database
    engine = create_engine('sqlite:///' + str(database_filepath))
    df = pd.read_sql_table("data/DisasterResponse.db", engine)
    
    # Input & Output to the Machine Learning Algorithm
    X, y = df['message'], df.iloc[:, 4:]

    return X, y, list(df.iloc[:, 4:].columns)

def tokenize(text):
    '''
    Tokenizes the input text received by the CountVectorizer()
    
    Args: 
    text: Input text sentence

    Returns:
    clean_tokens: A list of clean returned lemmatized words of the input sentence
    '''

    # Remove punctuation marks
    text = re.sub("[^a-zA-Z0-9]", " ", text)
    
    # Tokenize the text into words
    words = word_tokenize(text)
    
    # Remove stop words
    words = [w for w in words if w not in stopwords.words('english')]
    
    # Initialize the Lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    
    for word in words:
        # Lemmatize each word and convert it into a lower case without whitespaces
        clean_tok = lemmatizer.lemmatize(word).lower().strip()
        clean_tokens.append(clean_tok)
    
    return clean_tokens

def build_model():
    '''
    Builds the machine learning pipeline, defines parameters for hyper tuning & initializes
    GridSearchCV

    Args:
    None

    Returns:
    model: Initialized GridSearchCV object
    '''

    # Defining the pipline using CountVectorizer(Bag of words), TfIdfTransformer(Document Term
    # Frequency) & a classifier (DecisionTreeClassifier)
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(estimator=DecisionTreeClassifier()))
    ])
    
    # Parameters that could be used for hypertuning the model
    parameters_new = {
        'vect__binary': [True, False],
        'clf__estimator__min_samples_leaf': [10, 20, 30],
        'clf__estimator__min_samples_split': [5, 7, 9]
    }

    # Initialized gridsearchcv object using pipeline and parameters
    model = GridSearchCV(pipeline, param_grid=parameters_new, n_jobs=-1, cv=2, verbose=7)
    
    return model

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluates the model performance on the test set using F1 Scores

    Args:
    model: The model to be fit
    X_test: The test set inputs
    Y_test: The test set output labels
    category_names: 35 Category Names (OUTPUT)

    Returns:
    None
    '''

    y_pred_new = model.predict(X_test)
    print(classification_report(Y_test.values, y_pred_new, target_names=category_names))
    
def save_model(model, model_filepath):
    '''
    Saves the model as a pickle file for further usage

    Args:
    model: The trained model 
    model_filepath: The name of the model to be saved as

    Returns:
    None
    '''
    pickle.dump(model, open(model_filepath, 'wb'))

def main():

    # Requires 3 arguments while calling the script
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))

        # Calls the load_data function which returns X, Y & category_names
        X, Y, category_names = load_data(database_filepath)

        # Splits the dataset into training and test sets (80% training, 20% test)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        # Calls the build_model function which returns the instantiated GridSearchCV model
        print('Building model...')
        model = build_model()
        
        # Trains the model with the training set
        print('Training model...')
        model.fit(X_train, Y_train)
        
        # Evaluates the model using evalute_model function on the test sets
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        # Saves the trained model using the save_model function
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()