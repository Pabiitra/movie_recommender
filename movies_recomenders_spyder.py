import pandas as pd
import numpy as np

movies = pd.read_csv('tmdb_5000_movies.csv')
credit = pd.read_csv('tmdb_5000_credits.csv')

movies = movies.merge(credit, on='title')

# genres
# id
# keywords
# title
# overview
# cast
# crew

movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]


movies.isnull().sum(axis=0)
movies.dropna(inplace=True)
movies.duplicated().sum()

movies.iloc[0].genres

# '[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]'

# ['Action', 'Adventure', 'Fantasy', 'Science Fiction']


import ast

def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)




def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    
    return L

movies['cast'] = movies['cast'].apply(convert3)



def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L

movies['crew'] = movies['crew'].apply(fetch_director)


# to convert overview string to list

movies['overview'] = movies['overview'].apply(lambda x:x.split())


# remove space from aall features

movies['genres'] = movies['genres'].apply(lambda x:[i.replace(' ', '')for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(' ', '')for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(' ', '')for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(' ', '')for i in x])



movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']



new_df = movies[['movie_id', 'title', 'tags']]

# convert tags to string format
new_df['tags'] = new_df['tags'].apply(lambda x:' '.join(x))

new_df['tags'][0]

# covert in lower case
new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())





### TEXT FEATERING / TEXT REPRESENTATION / TEXT VECTORIZATION


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words='english')

vectors = cv.fit_transform(new_df['tags']).toarray()




# to convert words to it's root word like played to play
import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return ' '.join(y)    

new_df['tags'] = new_df['tags'].apply(stem)

vectors[0]

cv.get_feature_names_out()


# calculate cosin distance in between vactors . the distance is less the similarity will more.

from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vectors)


sorted(list(enumerate(similarity[0])), reverse=True, key=lambda x:x[1])[1:6]



def recommend(movie):
    # Find the index of the movie in the DataFrame
    movie_index = new_df[new_df['title'] == movie].index[0]
    
    # Get the similarity scores for the movie
    distance = similarity[movie_index]
    
    # Sort the movies based on similarity and get the top 5 recommendations
    movies_list = sorted(list(enumerate(distance)), key=lambda x: x[1], reverse=True)[1:6]
    
    # Print the indices of the recommended movies
    for i in movies_list:
        print(new_df.iloc[i[0]].title)
        

# Example usage:
recommend("Avatar")





### Convert the entire code into a website.

















