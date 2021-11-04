# Fake-News-Detection

## Intro

The contents of this repository were developed in the Baloise Codecamp 2021. 
We wanted to get started with Machine Learning and tried to find a fun use case.
We decided to go for a fake news detection and started with a tutorial with training data in English.
When we finished we tried to apply that to German datasets we found online.

## Structure

### Tutorial
Based on [https://towardsdatascience.com/fake-news-detection-with-machine-learning-using-python-3347d9899ad1](https://towardsdatascience.com/fake-news-detection-with-machine-learning-using-python-3347d9899ad1).
We have two CSV files (Fake.csv and True.csv) with our training data (available on the tutorial page). We trained our model with the tutorial.py file and did predictions in the inference.py file. The model was saved as "saved_weights.pt".

### Fake-News
Here we tried to apply the same approach to the German data that we found independently from the tutorial. Our data was in "fake.csv" and "true.csv" and we trained using "train.py" and predicted using "inference.py". The model was stored in "awesome-model.pt".

### Slides
Contains the slides for the presentation of our Codecamp results.

