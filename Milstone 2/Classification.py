import re
from urllib.parse import urlparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle
from sklearn.model_selection import GridSearchCV

from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import string
import matplotlib
from nltk.stem import WordNetLemmatizer


matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
def main():
    # Read the movies from the CSV file
    movies = pd.read_csv('movies-classification-dataset.csv' , parse_dates=['release_date'])
    # iterating the columns
    # for col in movies.columns:
    #     print(col)
    # print(movies['keywords'].head(5))
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
    movies = clean_and_tokenize_tagline(movies,"tagline")
    movies = clean_and_encode_homepage(movies,'homepage')

    movies['homepage_freq_enc'].fillna(0, inplace=True)
    # movies = Keyencoding(movies)
    movies['overview'].fillna('', inplace=True)
    movies['tagline'].fillna('Nan', inplace=True)
    # Fix inconsistent values in numerical columns by clipping or replacing with a standardized value
    movies['runtime'] = movies['runtime'].clip(lower=0, upper=400)



    movies['release_date'] =  pd.to_datetime(movies.index)

    # Convert the datetime object to a Unix timestamp
    # movies['release_date'] = movies['release_date'].to_timestamp()

    # Cast the Unix timestamp to a float
    movies['release_date'] = movies['release_date'].astype('int')

    movies = handleMissingNumValues(movies, 'budget')
    movies = handleMissingNumValues(movies, 'viewercount')
    movies = handleMissingNumValues(movies, 'revenue')
    movies = handleMissingNumValues(movies, 'runtime')
    movies = handleMissingNumValues(movies, 'vote_count')

    # drop dupes
    movies.drop(['id', 'original_title', 'title' ,'tagline', 'keywords'], axis=1, inplace=True)
     

    movies = movies.drop_duplicates()
    movies = preOverview(movies , 'overview')

    total = movies.shape[0]
    threshold = total * .0005
    cols = ['overview'] 
    movies = movies.apply(lambda x:x.mask(x.map(x.value_counts()) < threshold, 'RARE') if x.name in cols else x)
    movies = pd.get_dummies(data = movies , columns = cols)
    movies.drop(['overview_RARE'  ], axis=1, inplace=True)


    # Extract the columns that need to be scaled
    cols_to_scale = ['budget', 'viewercount', 'revenue', 'runtime', 'vote_count']


    # Create a StandardScaler object
    scaler = StandardScaler()

    # Apply the scaler to the selected columns
    movies[cols_to_scale] = scaler.fit_transform(movies[cols_to_scale])

    movies = CatEncoding(movies)

    movies = cleanOutliers(movies)

    # Select numerical columns to normalize
    numerical_cols = ['budget', 'viewercount', 'revenue', 'runtime', 'vote_count']

    # Initialize scaler
    scaler = MinMaxScaler()

    # Normalize numerical columns
    movies[numerical_cols] = scaler.fit_transform(movies[numerical_cols])

    movies.to_csv("modified_movies.csv", index=False)
        #  Split the data into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(movies.drop('Rate', axis=1), movies['Rate'], test_size=0.25)

    # Scale the features to a common scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train the models
    svm_model = SVC()
    svm_model.fit(X_train, y_train)

    decision_tree_model = DecisionTreeClassifier()
    decision_tree_model.fit(X_train, y_train)

    random_forest_model = RandomForestClassifier()
    random_forest_model.fit(X_train, y_train)

    # Evaluate the models
    svm_accuracy = svm_model.score(X_test, y_test)
    decision_tree_accuracy = decision_tree_model.score(X_test, y_test)
    random_forest_accuracy = random_forest_model.score(X_test, y_test)

    print('SVM accuracy:', svm_accuracy)
    print('Decision tree accuracy:', decision_tree_accuracy)
    print('Random forest accuracy:', random_forest_accuracy)

    max_features_range = np.arange(1,6,1)
    n_estimators_range = np.arange(10,11,10)
    rf_param_grid = dict(max_features=max_features_range, n_estimators=n_estimators_range)
    svm_param_grid = {'kernel': ['linear', 'poly', 'rbf'], 'C': [1, 10, 100]}
    decTree_param_grid = {'criterion': ['gini', 'entropy'], 'max_depth': [3, 5, 7]}

    # hyperparameter tunningfor random forest 
    grid = GridSearchCV(estimator=random_forest_model, param_grid=rf_param_grid, cv=5)
    grid.fit(X_train, y_train)
    print("The best random forest parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))
    
    # hyperparameter tunningfor svm
    grid = GridSearchCV(svm_model, param_grid=svm_param_grid, cv=5)
    grid.fit(X_train, y_train)
    print("The best svm parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))
    
    # hyperparameter tunningfor random forest 
    grid = GridSearchCV(estimator=decision_tree_model, param_grid=decTree_param_grid, cv=5)
    grid.fit(X_train, y_train)
    print("The best decision tree forest parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))
    

    # Select the best model
    best_model = max(svm_model, decision_tree_model, random_forest_model, key=lambda model: model.score(X_test, y_test))


   

def preOverview(movies,coulmn_name):
    # Initialize the lemmatizer and stop words
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    # Define the pattern to remove non-ASCII characters
    pattern = r'[^\x00-\x7F]+'

    # Iterate over the rows in the specified column
    for i, row in movies.iterrows():
        content = row[coulmn_name]

        # Remove non-ASCII characters
        content_clean = re.sub(pattern, '', content)

        # Tokenize the content into sentences and words
        tokens = nltk.word_tokenize(content_clean)

        # Remove stop words and punctuation
        tokens_filtered = [lemmatizer.lemmatize(word) for word in tokens if
                           word.casefold() not in stop_words and word.casefold() not in string.punctuation]

        # Join the filtered tokens back into a string
        clean_text = ' '.join(tokens_filtered)

        # Update the value in the specified column with the cleaned text
        movies.at[i, coulmn_name] = clean_text
        return movies


def preprocess_titles(df, column_name):
    # remove leading and trailing whitespaces
    df[column_name] = df[column_name].str.strip()

    # convert all characters to lowercase
    df[column_name] = df[column_name].str.lower()

    # remove punctuation marks
    df[column_name] = df[column_name].str.translate(str.maketrans('', '', string.punctuation))

    # tokenize the titles
    df[column_name] = df[column_name].str.split()

    # # print the preprocessed movies
    # fc = df[column_name].tolist()
    # print(f'{column_name}:', fc)
    # print("////////////////////////////////////////////////////")



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


def cleanOutliers(movies):
    threshold = 3
    numeric_columns = movies.select_dtypes(include=np.number).columns.tolist()
    for column in numeric_columns:
        mean = np.mean(movies[column])
        std_dev = np.std(movies[column])
        outliers = []
        if (std_dev == 0):
            continue
        for index, row in movies.iterrows():

            Z_score = (row[column] - mean) / std_dev
            if np.abs(Z_score) > threshold:
                outliers.append(index)
        movies = movies.drop(outliers)
    return movies


def CatEncoding(df):
    def update_genres(x):

        import json

        json_string = x

        # parse the JSON string into a Python object
        movies = json.loads(json_string)

        # iterate through the list of dictionaries and extract the values of the "name" key
        names = [d["name"] for d in movies]
        return names
        # apply function to genres column

    df['genres'] = df['genres'].apply(lambda x: update_genres(x))
    unique_genres = set(genre for movie_genres in df['genres'] for genre in movie_genres)
    # print(unique_genres)
    # Output: {'Drama', 'War', 'Action', 'Documentary', 'Comedy', 'Horror', 'Music', 'Crime', 'Thriller', 'Romance'}
    # Create a new moviesframe with one column for each unique genre
    for genre in unique_genres:
        df[genre] = df['genres'].apply(lambda x: 1 if genre in x else 0)

    # Drop the original 'genres' column
    df.drop('genres', axis=1, inplace=True)

    features = ['original_language', 'production_countries', 'spoken_languages', 'production_companies']

    # Loop over each feature and label encode it
    for feature in features:
        le = LabelEncoder()
        df[feature] = le.fit_transform(df[feature].astype(str))
    # Define the two possible values of the 'status' column
    status_values = ['Released', 'Post Production', 'Unknown']

    # Fit the encoder to the status values
    le.fit(status_values)
    df['status'] = df['status'].apply(lambda x: 'Unknown' if x not in le.classes_ else x)

    # Apply label encoding to the 'status' column
    df['status'] = le.transform(df['status'])

    return df

    # # Extract the genres column as a list
    # genres_list = movies['genres'].str.split('|')

    # # Get a list of all unique genres
    # unique_genres = set([genre for genres in genres_list for genre in genres])

    # # Create a dictionary mapping each unique genre to a binary column
    # genre_dict = {}
    # for genre in unique_genres:
    #     genre_dict[genre] = [1 if genre in genres else 0 for genres in genres_list]
    # print(genre_dict)
    # # Create a moviesframe from the dictionary and concatenate it to the original moviesset
    # genre_movies = pd.moviesFrame(genre_dict)
    # movies = pd.concat([movies, genre_movies], axis=1)

    # # Drop the original 'Genres' column
    # movies.drop('genres', axis=1, inplace=True)



import category_encoders as ce

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

def clean_and_tokenize_tagline(df, column_name):
    # Remove punctuation, convert to lowercase, and tokenize text while removing stop words
    stop_words = set(stopwords.words('english'))
    df[column_name] = df[column_name].apply(lambda x: tuple(word for word in nltk.word_tokenize(str(x).lower().replace('.', '').replace(',', '').replace('?', '')) if word not in stop_words))

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
