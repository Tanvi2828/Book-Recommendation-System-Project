# Book Recommendation System - README

## Overview
This project is a **Book Recommendation System** that recommends books to users based on the title of a book they provide. It uses **Natural Language Processing (NLP)** techniques, specifically **TF-IDF (Term Frequency-Inverse Document Frequency)** for text vectorization and **Cosine Similarity** or **Nearest Neighbors** for finding similar books. The system is built using **Python** and various machine learning libraries, such as **scikit-learn** and **pandas**.

## Features
- **Text-based recommendation**: The system recommends books based on the similarity of book titles, authors, and publishers.
- **Efficient Similarity Search**: Uses **Nearest Neighbors** to efficiently find similar books without creating a massive similarity matrix.
- **User-friendly Interface**: Can be deployed as a web app (using **Streamlit**) or run interactively in the terminal.

## Installation

### Prerequisites
Ensure that you have **Python 3.x** installed on your system. You also need to install the following Python packages:

- **pandas**: For data manipulation and analysis.
- **scikit-learn**: For machine learning algorithms.
- **joblib**: For saving and loading the trained models.
- **Streamlit** (optional, for deployment): For building a web-based user interface.

You can install the necessary dependencies by running the following command:

```bash
pip install pandas scikit-learn joblib streamlit
```

### Dataset
The dataset used for this project contains a list of books with the following columns:
- `Book-Title`: The title of the book.
- `Book-Author`: The author of the book.
- `Publisher`: The publisher of the book.
- `Year-Of-Publication`: The year the book was published.
- `ISBN`: International Standard Book Number.

The dataset should be placed in the project folder with the name `Books.csv`.

## Usage

### Step 1: Preprocessing the Dataset
To begin using the recommendation system, load and preprocess the dataset. This step involves combining the relevant features (book title, author, publisher) into a single column that will be used for generating recommendations.

```python
import pandas as pd

# Load the dataset
books = pd.read_csv('Books.csv')

# Combine features (title, author, publisher) for recommendation
books['combined_features'] = (
    books['Book-Title'].fillna('') + " " +
    books['Book-Author'].fillna('') + " " +
    books['Publisher'].fillna('')
)
```

### Step 2: Train the Recommendation Model
After preprocessing the data, we use **TF-IDF** to vectorize the text features and the **Nearest Neighbors** algorithm to find similar books.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Use TF-IDF Vectorizer to convert text to numerical format
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(books['combined_features'])

# Fit NearestNeighbors model
nn_model = NearestNeighbors(metric='cosine', algorithm='brute')
nn_model.fit(tfidf_matrix)
```

### Step 3: Save the Models
After training, the models can be saved using **joblib** for future use without retraining.

```python
import joblib

# Save the TF-IDF vectorizer and NearestNeighbors model
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
joblib.dump(nn_model, 'nearest_neighbors_model.pkl')
```

### Step 4: Make Book Recommendations
The main function for recommending books takes a book title as input and returns a list of recommended books based on similarity.

```python
def recommend_books_nn(title, n=5):
    indices = pd.Series(books.index, index=books['Book-Title']).drop_duplicates()
    if title not in indices:
        return f"'{title}' not found in the dataset."
    
    idx = indices[title]
    
    # Find nearest neighbors
    distances, neighbors = nn_model.kneighbors(tfidf.transform([books['combined_features'][idx]]), n_neighbors=n+1)
    
    recommended = books.iloc[neighbors[0][1:]][['Book-Title', 'Book-Author', 'Publisher']]
    return recommended
```

### Step 5: Interactive User Input
You can now run the recommendation system interactively. Users can input a book title and get recommendations.

```python
book_title = input("Enter a book title to get recommendations: ")
recommendations = recommend_books_nn(book_title, n=5)
print("Top Recommendations:")
print(recommendations)
```

### Step 6: Optional - Web App Deployment (Streamlit)
You can deploy the recommendation system as a web app using **Streamlit**. Create a file called `app.py` with the following content:

```python
import streamlit as st
import joblib
import pandas as pd

# Load the saved models
tfidf = joblib.load('tfidf_vectorizer.pkl')
nn_model = joblib.load('nearest_neighbors_model.pkl')
books = pd.read_csv('Books.csv')

# Combine features for interactivity
books['combined_features'] = (
    books['Book-Title'].fillna('') + " " +
    books['Book-Author'].fillna('') + " " +
    books['Publisher'].fillna('')
)

# Define the recommendation function
def recommend_books_streamlit(title, n=5):
    indices = pd.Series(books.index, index=books['Book-Title']).drop_duplicates()
    if title not in indices:
        return f"'{title}' not found in the dataset."
    
    idx = indices[title]
    distances, neighbors = nn_model.kneighbors(tfidf.transform([books['combined_features'][idx]]), n_neighbors=n+1)
    recommended = books.iloc[neighbors[0][1:]][['Book-Title', 'Book-Author', 'Publisher']]
    return recommended

# Streamlit app
st.title("Book Recommendation System")
book_title = st.text_input("Enter a book title to get recommendations:")
if book_title:
    recommendations = recommend_books_streamlit(book_title)
    if isinstance(recommendations, str):
        st.write(recommendations)
    else:
        st.write("Top Recommendations:")
        st.write(recommendations)
```

To run the app:
```bash
streamlit run app.py
```

---

## Model Saving and Loading
You can save the trained models (TF-IDF vectorizer and NearestNeighbors model) using **joblib** as shown above. To load the models, use:

```python
# Load the saved models
tfidf = joblib.load('tfidf_vectorizer.pkl')
nn_model = joblib.load('nearest_neighbors_model.pkl')
```

---

## Future Improvements
- **Enhanced Preprocessing**: Handle missing values and remove special characters or punctuation.
- **Interactive Features**: Add more features to the recommendations, such as filtering by genre or publication year.
- **Recommendation Evaluation**: Implement a user feedback loop to improve recommendations.

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
