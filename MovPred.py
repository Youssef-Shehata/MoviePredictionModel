import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics 
from scipy.stats import zscore

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
num_cols = ['budget', 'viewercount', 'revenue', 'runtime', 'vote_count', 'vote_average']
removed = 0 
movies[num_cols] = movies[num_cols].apply(zscore)
for col in num_cols:
    z_scores = zscore(movies[col])
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 4)
    print((abs_z_scores < 4))
    movies = movies[filtered_entries]
# create boxplot for each numerical column
plt.figure(figsize=(10, 6))
movies[num_cols].boxplot()
plt.xticks(rotation=45)
plt.show()
# print(movies[1])


# Remove outliers using Z-score method
cols = ['budget', 'revenue', 'vote_average', 'vote_count']

# print(movies.columns)

######################################################
