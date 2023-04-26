import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import pickle
# from scipy import stats
from sklearn.calibration import LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn import metrics
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import string
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def main():
    movies = pd.read_csv("movies-regression-dataset.csv")
    preprocess_titles(movies, 'original_title')
    preprocess_titles(movies, 'title')


    movies.drop(['id', 'original_title', 'title'], axis=1, inplace=True)
    # PREPROCCESSING
    # Fix missing values by filling with the mean or mode
    movies['homepage'].fillna('Uknown', inplace=True)
    movies['overview'].fillna('No overview available', inplace=True)
    movies['tagline'].fillna('No tagline available', inplace=True)

    # Fix inconsistent values in numerical columns by clipping or replacing with a standardized value
    movies['runtime'] = movies['runtime'].clip(lower=0, upper=400)
    movies.loc[movies['vote_count'] < 10, 'vote_average'] = 0

    # Fix inconsistent date format in release_date column
    movies['release_date'] = pd.to_datetime(movies['release_date'], format='%m/%d/%Y')

    movies = handleMissingNumValues(movies, 'budget')
    movies = handleMissingNumValues(movies, 'viewercount')
    movies = handleMissingNumValues(movies, 'revenue')
    movies = handleMissingNumValues(movies, 'runtime')
    movies = handleMissingNumValues(movies, 'vote_count')
    movies = handleMissingNumValues(movies, 'vote_average')

    # drop dupes
    movies = movies.drop_duplicates()

    # Extract the columns that need to be scaled
    cols_to_scale = ['budget', 'viewercount', 'revenue', 'runtime', 'vote_count', 'vote_average']

    # Create a StandardScaler object
    scaler = StandardScaler()

    # Apply the scaler to the selected columns
    movies[cols_to_scale] = scaler.fit_transform(movies[cols_to_scale])

    movies = CatEncoding(movies)

    movies = cleanOutliers(movies)

    # Select numerical columns to normalize
    numerical_cols = ['budget', 'viewercount', 'revenue', 'runtime', 'vote_count', 'vote_average']

    # Initialize scaler
    scaler = MinMaxScaler()

    # Normalize numerical columns
    movies[numerical_cols] = scaler.fit_transform(movies[numerical_cols])

    movies.to_csv("modified_movies.csv", index=False)
    selected_features = findBestFeatures(movies)
    polyResults = PolyReg(movies, selected_features)
    ridgeResults = ridge(movies, selected_features)



def preprocess_titles(df, column_name):
    # remove leading and trailing whitespaces
    df[column_name] = df[column_name].str.strip()

    # convert all characters to lowercase
    df[column_name] = df[column_name].str.lower()

    # remove punctuation marks
    df[column_name] = df[column_name].str.translate(str.maketrans('', '', string.punctuation))

    # tokenize the titles
    df[column_name] = df[column_name].str.split()

    # # print the preprocessed data
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

def ridge(movies,selected_features):
    X = movies[selected_features]
    y = movies['vote_average']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)

    # Evaluate the model
    y_pred = ridge.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred , squared=False)
    mse = mean_squared_error(y_test, y_pred )
    r2 = r2_score(y_test, y_pred)
    overfitting = abs(rmse - mean_squared_error(y_train, ridge.predict(X_train), squared=False))

    print(f"ridge RMSE: {rmse}")
    print(f"ridge MSE: {mse}")

    print(f"ridge R-squared score: {r2}")

    print(f"ridge overfitting: {overfitting}")


    filename = 'ridge.pkl'

    with open(filename, 'wb') as file:
        pickle.dump(ridge, file)


    # Plot the actual vs predicted values
    




    fig, ax = plt.subplots()
    X_test=np.arange(0,len(X_test),1)
    ax.scatter(X_test, y_test , color='blue')
    ax.plot(X_test, y_pred , color='red')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.show()
    return { "rigde_error": rmse, "ridge_r2": r2,
            "ridge_overfitting": overfitting}

def PolyReg(movies, selected_features):
    X = movies[selected_features]
    y = movies['vote_average']

    # Split the dataset into training and testing sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the best parameters
    best_degree = 1
    best_error = float('inf')
    best_r2 = float('-inf')
    best_overfitting = float('inf')
    best_mse = float('inf')

    # Loop over different degrees and select the best one
    for degree in range(1, 5):
        # Apply polynomial regression
        poly = PolynomialFeatures(degree=degree)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)
        poly_reg = LinearRegression()
        poly_reg.fit(X_train_poly, y_train)
        # Save the model to a file using pickle

        # Make predictions on the testing set
        y_pred = poly_reg.predict(X_test_poly)

        # Evaluate the model
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        overfitting = abs(rmse - mean_squared_error(y_train, poly_reg.predict(X_train_poly), squared=False))

        # Update the best parameters
        if r2 > best_r2 :
            best_degree = degree
            best_error = rmse
            best_r2 = r2
            best_overfitting = overfitting
            best_model = poly_reg
            best_polModel = poly
            best_mse = mse
    print(mean_squared_error(y_test, y_pred))

    filename = 'Poly_Reg.pkl'
    filename2 = 'poly.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(best_model, file)
    with open(filename2, 'wb') as file:
        pickle.dump(best_polModel, file)
    # Print the best parameters
    print(f"poly degree: {best_degree}")
    print(f"poly MSE: {best_mse}")

    print(f"poly RMSE: {best_error}")
    print(f"poly R-squared score: {best_r2}")
    print(f"poly overfitting: {best_overfitting}")






    fig, ax = plt.subplots()
    X_test=np.arange(0,len(X_test),1)
    X_test_poly=np.arange(0,len(X_test),1)


    ax.scatter(X_test, y_test , color='blue')
    ax.plot(X_test, y_pred , color='red')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.show()

    return {"poly_degree": best_degree, "poly_error": best_error, "poly_r2": best_r2,
            "poly_overfitting": best_overfitting}


def cleanOutliers(data):
    threshold = 3
    numeric_columns = data.select_dtypes(include=np.number).columns.tolist()
    for column in numeric_columns:
        mean = np.mean(data[column])
        std_dev = np.std(data[column])
        outliers = []
        if (std_dev == 0):
            continue
        for index, row in data.iterrows():

            Z_score = (row[column] - mean) / std_dev
            if np.abs(Z_score) > threshold:
                outliers.append(index)
        data = data.drop(outliers)
    return data


def CatEncoding(df):
    def update_genres(x):

        import json

        json_string = x

        # parse the JSON string into a Python object
        data = json.loads(json_string)

        # iterate through the list of dictionaries and extract the values of the "name" key
        names = [d["name"] for d in data]
        return names
        # apply function to genres column

    df['genres'] = df['genres'].apply(lambda x: update_genres(x))
    unique_genres = set(genre for movie_genres in df['genres'] for genre in movie_genres)
    # print(unique_genres)
    # Output: {'Drama', 'War', 'Action', 'Documentary', 'Comedy', 'Horror', 'Music', 'Crime', 'Thriller', 'Romance'}
    # Create a new dataframe with one column for each unique genre
    for genre in unique_genres:
        df[genre] = df['genres'].apply(lambda x: 1 if genre in x else 0)

    # Drop the original 'genres' column
    df.drop('genres', axis=1, inplace=True)

    features = ['original_language', 'production_countries', 'spoken_languages']

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
    # # Create a dataframe from the dictionary and concatenate it to the original dataset
    # genre_movies = pd.DataFrame(genre_dict)
    # movies = pd.concat([movies, genre_movies], axis=1)

    # # Drop the original 'Genres' column
    # movies.drop('genres', axis=1, inplace=True)


main()
