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
from sklearn.feature_selection import SelectKBest, mutual_info_regression

import time
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

    preprocess_titles(movies, 'original_title')
    preprocess_titles(movies, 'title')
    with open("mean.csv", 'w') as file:
        file.write(str(movies.mean(numeric_only=True).round(1)))

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
    # movies['overview'] = ListEncoding(movies['overview'], 14)
    # movies['tagline'] = ListEncoding(movies['tagline'], 4)


    movies =LabelEncoding(movies)
    movies['homepage_freq_enc'].fillna(0, inplace=True)
    # movies = Keyencoding(movies)
    movies['overview'].fillna('', inplace=True)
    movies['tagline'].fillna('Nan', inplace=True)
    # Fix inconsistent values in numerical columns by clipping or replacing with a standardized value
    movies['runtime'] = movies['runtime'].clip(lower=0, upper=400)



    movies['release_date'] =  pd.to_datetime(movies.index)


    # Cast the Unix timestamp to a float
    movies['release_date'] = movies['release_date'].astype('int64').astype('int32')

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


    movies = cleanOutliers(movies)

    # Select numerical columns to normalize
    numerical_cols = ['budget', 'viewercount', 'revenue', 'runtime', 'vote_count']

    # Initialize scaler
    scaler = MinMaxScaler()

    # Normalize numerical columns
    movies[numerical_cols] = scaler.fit_transform(movies[numerical_cols])

    movies.to_csv("modified_movies.csv", index=False)
        #  Split the data into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(movies.drop('Rate', axis=1), movies['Rate'], test_size=0.2)

    # Scale the features to a common scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # Apply feature selection

    # def my_score(X, y):
    #     return mutual_info_regression(X, y, random_state=0)

    # selector = SelectKBest(score_func=my_score, k=5)
    
    # X_train = selector.fit_transform(X_train, y_train)
    # X_test = selector.transform(X_test)

    # Save the dictionary to a file
    # import json 
    # with open("classificationFeatures.json", "w") as f:
    #     json.dump(X_test.tolist(), f, indent=4)


    # Train the models
    svm_model = SVC()
    start_time = time.time()
    svm_model.fit(X_train, y_train)
    training_time_svm = time.time() - start_time

    decision_tree_model = DecisionTreeClassifier()
    start_time = time.time()
    decision_tree_model.fit(X_train, y_train)
    training_time_decision = time.time() - start_time

    random_forest_model = RandomForestClassifier()
    start_time = time.time()
    random_forest_model.fit(X_train, y_train)
    training_time_random_forest = time.time() - start_time

    # Evaluate the models
    start_time = time.time()
    svm_accuracy = svm_model.score(X_test, y_test)
    testing_time_svm = time.time() - start_time

    start_time = time.time()
    decision_tree_accuracy = decision_tree_model.score(X_test, y_test)
    testing_time_decision = time.time() - start_time

    start_time = time.time()
    random_forest_accuracy = random_forest_model.score(X_test, y_test)
    testing_time_random_forest = time.time() - start_time

    print('SVM accuracy:', svm_accuracy)
    print('Decision tree accuracy:', decision_tree_accuracy)
    print('Random forest accuracy:', random_forest_accuracy)

    accuracy_values_plot = [svm_accuracy, decision_tree_accuracy,
                            random_forest_accuracy]  # Example accuracy values
    model_labels = ['SVM', 'Decision_Tree', 'Random_forest']  # Example model labels

    plt.bar(model_labels, accuracy_values_plot, width=0.7)

    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison before hyperparameter tunning')
    plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
               ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'])
    # plt.xticks(rotation=45)
    plt.ylim(0, 1)  # Set the y-axis limits (0 to 1 in this case)

    plt.show()

    training_time_plot = [training_time_svm, training_time_decision,
                          training_time_random_forest]  # Example accuracy values

    plt.bar(model_labels, training_time_plot, width=0.7)

    plt.xlabel('Models')
    plt.ylabel('Time by seconds')
    plt.title('Model Training Time Comparison before hyperparameter tunning')
    # plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    #            ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'])
    # plt.xticks(rotation=45)
    # plt.ylim(0, 1)  # Set the y-axis limits (0 to 1 in this case)

    plt.show()


    max_features_range = np.arange(1, 6, 1)
    n_estimators_range = np.arange(10, 11, 10)
    rf_param_grid = dict(max_features=max_features_range, n_estimators=n_estimators_range)
    svm_param_grid = {'kernel': ['linear', 'poly', 'rbf'], 'C': [1, 10, 100]}
    decTree_param_grid = {'criterion': ['gini', 'entropy'], 'max_depth': [3, 5, 7]}

    # hyperparameter tunningfor random forest
    grid = GridSearchCV(estimator=random_forest_model, param_grid=rf_param_grid, cv=5)
    start_time = time.time()
    grid.fit(X_train, y_train)
    training_time_random_after = time.time() - start_time

    print("The best random forest parameters are %s with a score of %0.2f"
          % (grid.best_params_, grid.best_score_))
    random_forest_accuracy_plot = grid.best_score_

    # hyperparameter tunning for svm
    grid = GridSearchCV(svm_model, param_grid=svm_param_grid, cv=5)
    start_time = time.time()
    grid.fit(X_train, y_train)
    training_time_svm_after = time.time() - start_time
    print("The best svm parameters are %s with a score of %0.2f"
          % (grid.best_params_, grid.best_score_))
    svm_accuracy_plot = grid.best_score_

    # hyperparameter tunning for decision tree
    grid = GridSearchCV(estimator=decision_tree_model, param_grid=decTree_param_grid, cv=5)
    start_time = time.time()
    grid.fit(X_train, y_train)
    training_time_decision_after = time.time() - start_time

    print("The best decision tree forest parameters are %s with a score of %0.2f"
          % (grid.best_params_, grid.best_score_))
    decision_tree_accuracy_plot = grid.best_score_


    # Select the best model
    best_model = max(svm_model, decision_tree_model, random_forest_model, key=lambda model: model.score(X_test, y_test))

    accuracy_values_plot = [svm_accuracy_plot, decision_tree_accuracy_plot, random_forest_accuracy_plot]  # Example accuracy values
    model_labels = ['SVM', 'Decision_Tree', 'Random_forest']  # Example model labels

    plt.bar(model_labels, accuracy_values_plot,width=0.7)

    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison after hyperparameter tunning')
    plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5,0.6,0.7,0.8,0.9,1.0], ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'])
    # plt.xticks(rotation=45)
    plt.ylim(0, 1)  # Set the y-axis limits (0 to 1 in this case)

    plt.show()

    training_time_plot_after = [training_time_svm_after, training_time_decision_after, training_time_random_after]  # Example accuracy values

    plt.bar(model_labels, training_time_plot_after, width=0.7)

    plt.xlabel('Models')
    plt.ylabel('Time by seconds')
    plt.title('Model Training Time Comparison after hyperparameter tunning')
    # plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    #            ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'])
    # plt.xticks(rotation=45)
    # plt.ylim(0, 1)  # Set the y-axis limits (0 to 1 in this case)

    plt.show()

    testing_time_plot = [testing_time_svm, testing_time_decision,testing_time_random_forest]  # Example accuracy values
    plt.bar(model_labels, testing_time_plot, width=0.7)

    plt.xlabel('Models')
    plt.ylabel('Time by seconds')
    plt.title('Model Testing Time Comparison')
    # plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    #            ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'])
    # plt.xticks(rotation=45)
    # plt.ylim(0, 1)  # Set the y-axis limits (0 to 1 in this case)

    plt.show()

    import json


    # Save the models
    pickle.dump(svm_model, open("svm_model.pkl", "wb"))
    pickle.dump(random_forest_model, open("random_forest_model.pkl", "wb"))
    pickle.dump(decision_tree_model, open("decision_tree_model.pkl", "wb"))


    # Select the best model
    best_model = max(svm_model, decision_tree_model, random_forest_model, key=lambda model: model.score(X_test, y_test))


   

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



def ListEncoding(column, threshold_percentage):
    def update_column(x):
        names = x
        return names

    column = column.apply(lambda x: update_column(x))

    # Filter out values that are not lists or have different lengths
    column = column.apply(lambda x: x if isinstance(x, list) and len(x) == len(column) else [])

    column_counts = column.explode().value_counts()
    threshold = int(threshold_percentage / 100 * len(column))  # Calculate the threshold based on the dataset size
    popular_columns = column_counts[column_counts > threshold].index.tolist()

    print("Popular Columns:", popular_columns)

    df = pd.DataFrame()
    for col in popular_columns:
        df[col] = column.apply(lambda x: 1 if col in x else 0)

    print("Encoded DataFrame:", df)

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
