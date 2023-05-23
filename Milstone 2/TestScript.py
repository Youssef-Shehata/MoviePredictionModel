import pandas as pd 
import numpy as np
import sys
import matplotlib.pyplot as plt
import pickle
from scipy import stats
from sklearn.calibration import LabelEncoder
from sklearn import metrics 
from sklearn.preprocessing import StandardScaler

import pickle
from nltk.stem import WordNetLemmatizer

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import json
import re
from urllib.parse import urlparse
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

import numpy as np
import time
import pickle

from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import string
import matplotlib

def main():
    # Check that the script is called with the correct number of arguments
    if len(sys.argv) != 2:
        print('Usage: python predict.py <data_file>')
        return


    # Load the trained models
    svm_model = pickle.load(open("svm_model.pkl", "rb"))
    random_forest_model = pickle.load(open("random_forest_model.pkl", "rb"))
    decision_tree_model = pickle.load(open("decision_tree_model.pkl", "rb"))
    




    # Load the new dataset from the file specified as an argument
    data_file = sys.argv[1]
    
    movies =pd.read_csv(data_file)
    with open("classificationFeatures.json", "r") as f:
        selectedFeatures = json.load(f)
    preprocess_titles(movies, 'original_title')
    preprocess_titles(movies, 'title')


    # PREPROCCESSING
    # Fix missing values by filling with the mean or mode
    ratemappins ={
        'High' : 3,
        'Intermediate' : 2,
        'Low' : 1,

    }
    movies['Rate']=movies['Rate'].map(ratemappins)
    # movies = clean_and_tokenize_tagline(movies,"tagline")
    movies = clean_and_encode_homepage(movies,'homepage')
    clean_and_tokenize(movies , 'overview')
    clean_and_tokenize(movies , 'tagline')
    movies = CatEncoding(movies,'genres')
    movies = CatEncoding(movies,'keywords')

    movies =LabelEncoding(movies)
    movies['homepage_freq_enc'].fillna(0, inplace=True)
    # movies = Keyencoding(movies)
    movies['overview'].fillna('', inplace=True)
    movies['tagline'].fillna('Nan', inplace=True)
    # Fix inconsistent values in numerical columns by clipping or replacing with a standardized value
    movies['runtime'] = movies['runtime'].clip(lower=0, upper=400)



    movies['release_date'] =  pd.to_datetime(movies.index)


    # Cast the Unix timestamp to a float
    movies['release_date'] = movies['release_date'].astype('int')

    movies = handleMissingNumValues(movies, 'budget')
    movies = handleMissingNumValues(movies, 'viewercount')
    movies = handleMissingNumValues(movies, 'revenue')
    movies = handleMissingNumValues(movies, 'runtime')
    movies = handleMissingNumValues(movies, 'vote_count')

    # drop dupes
    movies.drop(['id', 'original_title', 'title' ,'tagline', "overview"], axis=1, inplace=True)
     

    movies = movies.drop_duplicates()



    # Extract the columns that need to be scaled
    cols_to_scale = ['budget', 'viewercount', 'revenue', 'runtime', 'vote_count']


    # Create a StandardScaler object
    scaler = StandardScaler()

    # Apply the scaler to the selected columns
    movies[cols_to_scale] = scaler.fit_transform(movies[cols_to_scale])



    # Select numerical columns to normalize
    numerical_cols = ['budget', 'viewercount', 'revenue', 'runtime', 'vote_count']

    # Initialize scaler
    scaler = MinMaxScaler()

    # Normalize numerical columns
    movies[numerical_cols] = scaler.fit_transform(movies[numerical_cols])

    # Predict the labels of the new data
    x = movies.drop('Rate', axis=1)

        # Set the hyperparameters on the model
    svmhp = {'C': 1, 'kernel': 'rbf',}
    rfhp = {'max_features': 4, 'n_estimators': 10,}
    dthp = {'criterion': 'gini', 'max_depth': 5,}


    svm_model.set_params(**svmhp)
    random_forest_model.set_params(**rfhp)
    decision_tree_model.set_params(**dthp )

    # Calculate the accuracy score
    svm_accuracy = svm_model.score(x, movies['Rate'])
    tree_accuracy = decision_tree_model.score(x, movies['Rate'])
    rf_accuracy = random_forest_model.score(x, movies['Rate'])
    # Create a dictionary of hyperparameters



    print('SVM accuracy:', svm_accuracy)
    print('Decision tree accuracy:', tree_accuracy)
    print('Random forest accuracy:', rf_accuracy)










    
def preprocess_titles(df, column_name):
    # remove leading and trailing whitespaces
    df[column_name] = df[column_name].str.strip()

    # convert all characters to lowercase
    df[column_name] = df[column_name].str.lower()

    # remove punctuation marks
    df[column_name] = df[column_name].str.translate(str.maketrans('', '', string.punctuation))

    # tokenize the titles
    df[column_name] = df[column_name].str.split()




def handleMissingNumValues(movies, col):
    missing_values = movies[col].isnull().sum()

    # Calculate the percentage of missing values
    percent_missing = (missing_values / len(movies[col])) * 100

    # Check the number of missing values
    total_missing = missing_values.sum()

    # Calculate the percentage of missing values
    percent_total_missing = (total_missing / movies[col].shape[0]) * 100

    # Decide whether to drop the missing value records or impute them with mean
    if percent_total_missing < 5:
        # Drop rows with missing values
        movies = movies.dropna(subset=[col])
    else:
        # Impute missing values with mean
        movies = movies[col].fillna(movies.mean())
    return movies


def findBestFeatures(movies):
    # Calculate the correlation matrix
    corr_matrix = movies.corr()

    # Get the correlation coefficients between the dependent variable and the independent variables
    correlation_coefficients = corr_matrix['vote_average'].drop('vote_average')

    # Sort the coefficients by their absolute values in descending order
    sorted_coefficients = correlation_coefficients.abs().sort_values(ascending=False)

    # Select the top n features with the highest correlation coefficients
    n = 2
    selected_features = sorted_coefficients[:n].index.tolist()
    # print(selected_features)
    return selected_features




def CatEncoding(df, column_name):
    def update_column(x):
        import json
        json_string = x
        data = json.loads(json_string)
        names = [d["name"] for d in data]
        return names

    df[column_name] = df[column_name].apply(lambda x: update_column(x))

    column_counts = df[column_name].explode().value_counts()
    threshold = int(5/ 100 * len(df))  # Calculate the threshold based on the dataset size
    popular_columns = column_counts[column_counts > threshold].index.tolist()

    for column in popular_columns:
        df[column] = df[column_name].apply(lambda x: 1 if column in x else 0)

    df.drop(column_name, axis=1, inplace=True)



    return df


from sklearn.preprocessing import LabelEncoder


def LabelEncoding(df):
    features = ['original_language', 'production_countries', 'spoken_languages', 'production_companies']
    for feature in features:
        le = LabelEncoder()
        df[feature] = le.fit_transform(df[feature].astype(str))

    status_values = ['Released', 'Post Production', 'Unknown']
    le.fit(status_values)
    df['status'] = df['status'].apply(lambda x: 'Unknown' if x not in le.classes_ else x)
    df['status'] = le.transform(df['status'])

    return df




def clean_and_encode_homepage(df, column_name):
    # Extract domain name from URL
    df[column_name] = df[column_name].apply(lambda x: urlparse(str(x)).hostname if pd.notnull(x) else x)
    # Create a dictionary mapping each unique value in the column to its frequency
    value_counts = df[column_name].value_counts().to_dict()
    
    # Replace each value in the column with its frequency
    df[column_name] = df[column_name].map(value_counts)
    
    # Rename the column to indicate that it has been frequency encoded
    df = df.rename(columns={column_name: column_name + '_freq_enc'})
    
    return df
    


import string
import nltk
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
def clean_and_tokenize(df, column_name):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    def remove_non_ascii(text):
        return "".join([char for char in text if ord(char) < 128 or char == "'"])

    df[column_name].fillna("", inplace=True)

    df[column_name] = df[column_name].apply(lambda x: [
        lemmatizer.lemmatize(word) for word in nltk.word_tokenize(remove_non_ascii(str(x).lower()))
        if word not in stop_words and all(char not in string.punctuation for char in word) and word != ""
    ])

    return df

def Keyencoding(df):
    def updateWords(x):

        import json

        json_string = x

        # parse the JSON string into a Python object
        movies = json.loads(json_string)

        # iterate through the list of dictionaries and extract the values of the "name" key
        names = [d["name"] for d in movies]
        return names
        # apply function to genres column

    df['keywords'] = df['keywords'].apply(lambda x: updateWords(x))
    words = set(genre for movie_genres in df['keywords'] for genre in movie_genres)
    keywords = list(words)
    
    # Output: {'Drama', 'War', 'Action', 'Documentary', 'Comedy', 'Horror', 'Music', 'Crime', 'Thriller', 'Romance'}
    # Create a new moviesframe with one column for each unique genre
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_selection import SelectKBest, chi2
    import numpy as np

    # Define a list of keywords

    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Apply TF-IDF vectorization to the list of keywords
    tfidf_matrix = vectorizer.fit_transform(keywords)

    # Convert the TF-IDF matrix to a dense NumPy array
    tfidf_array = tfidf_matrix.toarray()

    # Perform feature selection using chi-squared
    k = 3  # Number of top features to select
    chi2_selector = SelectKBest(chi2, k=k)
    selected_features = chi2_selector.fit_transform(tfidf_array, np.array([1]*len(keywords)))

    # Get the indices of the selected features
    feature_indices = chi2_selector.get_support(indices=True)

    # Print the selected features
    selected_keywords = [keywords[i] for i in feature_indices]
    # for word in selected_keywords:
    #     df[word] = 
    # Drop the original 'genres' column
    df.drop('keywords', axis=1, inplace=True)


main()

#  rewrite the tagline function to fix the current error 
# # train and save the models and learn the metrics of classification 
# # apply feature selection and hyperparameter tunning 

if __name__ == '__main__':
    main()
