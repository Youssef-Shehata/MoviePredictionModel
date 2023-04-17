import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics 
movies =pd.read_csv("movies-regression-dataset.csv")





#this block is for cleaning missing data 
######################################################
# Check for missing values

#fix inconsistencies 
# Fix missing values by filling with the mean or mode
movies['homepage'].fillna('Uknown', inplace=True)
movies['overview'].fillna('No overview available', inplace=True)
movies['tagline'].fillna('No tagline available', inplace=True)

missing_values = movies['budget'].isnull().sum()

# Calculate the percentage of missing values 
percent_missing = (missing_values / len(movies['budget'])) * 100

# Check the number of missing values
total_missing = missing_values.sum()

# Calculate the percentage of missing values 
percent_total_missing = (total_missing / movies['budget'].shape[0] ) * 100

# Decide whether to drop the missing value records or impute them with mean
if percent_total_missing < 5:
    # Drop rows with missing values
    movies = movies.dropna(subset=['budget'])
else:
    # Impute missing values with mean
    movies = movies['budget'].fillna(movies.mean())



#drop dupes 
movies = movies.drop_duplicates()






# Fix inconsistent values in categorical columns by replacing with a standardized value
movies['status'].replace(['Rumored', 'Post Production'], 'Unknown', inplace=True)

# Fix inconsistent values in numerical columns by clipping or replacing with a standardized value
movies['runtime'] = movies['runtime'].clip(lower=0, upper=400)
movies.loc[movies['vote_count'] < 10, 'vote_average'] = 0

# Fix inconsistent date format in release_date column
movies['release_date'] = pd.to_datetime(movies['release_date'], format='%m/%d/%Y')

######################################################



