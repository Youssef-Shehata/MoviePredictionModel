import string
import pandas as pd 
import numpy as np
import sys
import matplotlib.pyplot as plt
import pickle
from scipy import stats
from sklearn.calibration import LabelEncoder
from sklearn import metrics 
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error


def main():
    # Check that the script is called with the correct number of arguments
    if len(sys.argv) != 2:
        print('Usage: python predict.py <data_file>')
        return

    # Load the saved Pol_model from the file using pickle
    with open('Poly_Reg.pkl', 'rb') as file:
        reg_model = pickle.load(file)
    with open('poly.pkl', 'rb') as file:
        Pol_model = pickle.load(file)
    with open('ridge.pkl', 'rb') as file:
        Ridge_model = pickle.load(file)

    # Load the new dataset from the file specified as an argument
    data_file = sys.argv[1]
 
    movies =pd.read_csv(data_file)
    # print(movies.head(6))

    # Preprocess the data if necessary
    preprocess_titles(movies, 'original_title')
    preprocess_titles(movies, 'title')

        #PREPROCCESSING
    # Fix missing values by filling with the mean or mode
    movies['homepage'].fillna('Uknown', inplace=True)
    movies['overview'].fillna('No overview available', inplace=True)
    movies['tagline'].fillna('No tagline available', inplace=True)

 
    # Fix inconsistent values in numerical columns by clipping or replacing with a standardized value
    movies['runtime'] = movies['runtime'].clip(lower=0, upper=400)
    movies.loc[movies['vote_count'] < 10, 'vote_average'] = 0

    # Fix inconsistent date format in release_date column
    movies['release_date'] = pd.to_datetime(movies['release_date'], format='%m/%d/%Y')

    movies = handleMissingNumValues(movies,'budget')
    movies = handleMissingNumValues(movies,'viewercount')
    movies = handleMissingNumValues(movies,'revenue')
    movies = handleMissingNumValues(movies,'runtime')
    movies = handleMissingNumValues(movies,'vote_count')
    movies = handleMissingNumValues(movies,'vote_average')

    #drop dupes 

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

    movies.to_csv("TEST_modified_movies.csv", index=False)
    selected_features = ['runtime','vote_count']
    # selected_features = findBestFeatures(movies)











    # print(selected_features)
    # Use the loaded Pol_model to make predictions on the new dataset 

    X_test = movies[selected_features]
    X_test_poly = Pol_model.transform(X_test)
    predictions = reg_model.predict(X_test_poly)

    # Evaluate the Pol_model
    mse = mean_squared_error(movies['vote_average'], predictions)
    r2 = r2_score(movies['vote_average'], predictions)

    print(f"Pol_Regression MSE  : {mse}")
    print(f"Pol_Regression R-squared score: {r2}")




    # Use the loaded Ridge_model to make predictions on the new 
    y_pred =Ridge_model.predict(movies[selected_features]) 

    # Evaluate the Pol_model
    mse = mean_squared_error(movies['vote_average'], y_pred )
    r2 = r2_score(movies['vote_average'], y_pred)
    
    print(f"ridge MSE: {mse}")
    print(f"ridge R-squared score: {r2}")


def handleMissingNumValues(movies,col):

    missing_values = movies[col].isnull().sum()

    # Calculate the percentage of missing values 
    percent_missing = (missing_values / len(movies[col])) * 100

    # Check the number of missing values
    total_missing = missing_values.sum()
    
    # Calculate the percentage of missing values 
    percent_total_missing = (total_missing / movies[col].shape[0] ) * 100

    # Decide whether to drop the missing value records or impute them with mean
    if percent_total_missing > 5:
        # Drop rows with missing values
        # movies = movies.dropna(subset=[col])

    
        # Impute missing values with mean
        movies = movies[col].fillna(movies.mean())
    return movies



def cleanOutliers(data):
    threshold = 3
    numeric_columns = data.select_dtypes(include=np.number).columns.tolist()
    for column in numeric_columns:
        mean = np.mean(data[column])
        std_dev = np.std(data[column])
        outliers = []
        if(std_dev == 0):
            continue
        for index, row in data.iterrows():
            
            Z_score = (row[column] - mean) / std_dev
            if np.abs(Z_score) > threshold:
                outliers.append(index)
        data = data.drop(outliers)
    return data


def CatEncoding(df ):
    


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
    status_values = ['Released', 'Post Production','Unknown']

    # Fit the encoder to the status values
    le.fit(status_values)
    df['status'] = df['status'].apply(lambda x: 'Unknown' if x not in le.classes_ else x)


    # Apply label encoding to the 'status' column
    df['status'] = le.transform(df['status'])




    return df


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
    print(selected_features)
    return selected_features

if __name__ == '__main__':
    main()
