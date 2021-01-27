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
1. models - contains train_classifier.py to train the classifier
2. data - contains process_data.py to preprocess & clean the data
3. app - contains code to run the data dashboard using flask

## How to run
<a id='results'></a>
Go to app folder & run python run.py. This will deploy the dashboard locally. Go to the link mentioned & give the classifier a try!

## Limitations
Due to the skewed nature of the classes in the dataset, the F1 score that is obtained is very less (around 0.63). This can be improved if the imbalance in the dataset is eradicated.

## Licensing, Authors, Acknowledgements
<a id='liscense'></a>
This dataset belongs to the Figure Eight & is a part of the Udacity Data Science Nano Degree Program.
