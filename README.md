# Disaster Response Classifier

# Table Of Contents
1. [Installation](#installation)
2. [Project Motivation](#projectmotivation)
3. [File Descriptions](#filedescriptions)
4. [How to run](#results)
5. [Liscensing, Authors and Acknowledgements](#liscense)

## Installation
<a id='installation'></a>
The following packages need to be installed to run the classifier dashboard;
1. Plotly
2. Flask
3. NLTK
4. Sqlalchemy
```
pip install plotly
pip install Flask
pip install nltk
pip install SQLAlchemy
```
The code should run with no issues using Python versions 3.*.

## Project Motivation
<a id='projectmotivation'></a>
At the time of a natural disaster, thousands of people are in need of essential services like fresh drinking water, medical aid, food etc. The disaster response team receives thousands of requests for the same. However, there is a need to sort the messages in terms of priority so that help can be provided at the earliest to the ones who need it the most at that time. For these reasons, I developed a Machine Learning classifier which classifies distress messages into 36 categories. Using these categories, disaster response teams can categorize messages & provide help to those in dire need at the right time.

## File Descriptions
<a id='filedescriptions'></a>
There are three file directories;
1. models
    * Contains a python script named train_classifier.py to train the cleaned dataset
    * Contains a saved model as a pickle object named classifier.pkl
2. data
    * Contains a python script named process_data.py to clean & store the dataset into an SQLite Database
    * disaster_messages.csv - Contains distress messages
    * disaster_categories.csv - Contains 36 categories of types of distress messages
3. app - contains code to run the data dashboard using flask
    * run.py - A flask file that runs the dashboard
    * template
        * master.html - Main page of the dashboard
        * go.html - Classification results page of the dashboard

## How to run
<a id='results'></a>
In order to run the project, the following steps need to be followed;
1. Cleaning & Processing Data 
    * Go to the data directory from your command prompt & enter the following command;
    ```
    python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
    ```
    This script cleans the input data & stores it into a SQLite Database with the name DiasterResponse.db

2. Training the classifier
    To train your model on the clean data, run the following command from the models directory;
    ```
    python train_classifier.py ../data/DisasterResponse.db classifier.pkl
    ```

3. Running the data dashboard
    * Go to the app folder & run python run.py. This will deploy the dashboard locally & will print out the IP address for the same

## Limitations
Due to the skewed nature of the classes in the dataset, the F1 score that is obtained is very less (around 0.63). This can be improved if the imbalance in the dataset is eradicated.

## Licensing, Authors, Acknowledgements
<a id='liscense'></a>
This dataset belongs to the Figure Eight & is a part of the Udacity Data Science Nano Degree Program.
