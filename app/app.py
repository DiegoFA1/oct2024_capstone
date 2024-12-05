### KICKOFF - CODING AN APP IN STREAMLIT

### import libraries
import os
import pandas as pd
import streamlit as st
import joblib

import json
from recommendation import recommend
from scipy.sparse import load_npz
from joblib import load

# Get the directory of the current script
script_dir = os.path.dirname(__file__)

st.write('Streamlit is an open-source app framework for Machine Learning and Data Science teams. For the docs, please click [here](https://docs.streamlit.io/en/stable/api.html). \
    This is is just a very small window into its capabilities.')


#######################################################################################################################################
### LAUNCHING THE APP ON THE LOCAL MACHINE
### 1. Save your *.py file (the file and the dataset should be in the same folder)
### 2. Open git bash (Windows) or Terminal (MAC) and navigate (cd) to the folder containing the *.py and *.csv files
### 3. Execute... streamlit run <name_of_file.py>
### 4. The app will launch in your browser. A 'Rerun' button will appear every time you SAVE an update in the *.py file



#######################################################################################################################################
### Create a title

st.title("Kickoff - Live coding an app")

# Press R in the app to refresh after changing the code and saving here

### You can try each method by uncommenting each of the lines of code in this section in turn and rerunning the app

### You can also use markdown syntax.
#st.write('# Our last morning kick off :sob:')

### To position text and color, you can use html syntax
#st.markdown("<h1 style='text-align: center; color: blue;'>Our last morning kick off</h1>", unsafe_allow_html=True)


#######################################################################################################################################
### DATA LOADING

### A. define function to load data
@st.cache_data # <- add decorators after tried running the load multiple times
def load_data(file, num_rows):
    # Get the directory of the current script
    script_dir = os.path.dirname(__file__)
    path = os.path.join(script_dir, file)
    df = pd.read_csv(path, index_col=0)
    df_snippet = pd.read_csv(path, nrows=num_rows, index_col=0)
    df_snippet = df_snippet.drop(['img_s','img_m','img_l'], axis=1)
    
    return df, df_snippet

### B. Load first 100 rows
df, df_snippet = load_data("Books_data.csv", 100)

### C. Display the dataframe in the app
st.dataframe(df_snippet)


#######################################################################################################################################




#######################################################################################################################################
### DATA ANALYSIS & VISUALIZATION

### B. Add filter on side bar after initial bar chart constructed


#counts = df["Start Time"].dt.hour.value_counts()
#st.bar_chart(counts)





### The features we have used here are very basic. Most Python libraries can be imported as in Jupyter Notebook so the possibilities are vast.
#### Visualizations can be rendered using matplotlib, seaborn, plotly etc.
#### Models can be imported using *.pkl files (or similar) so predictions, classifications etc can be done within the app using previously optimized models
#### Automating processes and handling real-time data


#######################################################################################################################################
### MODEL INFERENCE

st.subheader("Using pretrained models with user input")

# A. Load the model using joblib
# Load the model
model_path = os.path.join(script_dir, 'model.joblib')
model = load(model_path)

# Load the csr_matrix from the file
csr_path = os.path.join(script_dir, 'csr_data.npz')
csr_data = load_npz(csr_path)

# Load book isbn IDs
book_path = os.path.join(script_dir, 'book_isbn.json')
with open(book_path, 'r') as f:
    book_isbns = json.load(f)
    
# B. Set up input field

# Get all unique book titles from the dataframe
book_titles = df['book_title'].unique().tolist()

# Use a selectbox with search capabilities
selected_title = st.selectbox(
    'Start typing a book title:', 
    options=book_titles, 
)


# C. Use the model to predict sentiment & write result
if selected_title:
    # Find the corresponding ISBN for the selected title
    selected_isbn = df.loc[df['book_title'] == selected_title, 'isbn'].values

    if len(selected_isbn) > 0:  # Ensure we have a valid ISBN
        selected_isbn = selected_isbn[0]  # Take the first match if duplicates exist
        
        # Use the recommender function
        recommendations = recommend(
            book_isbn=selected_isbn,  
            knn_model=model,
            csr_data=csr_data,
            book_isbns=book_isbns,  # List of ISBNs corresponding to the rows in csr_data
            dataframe=df,  # Original DataFrame with book metadata
            n_neighbors=10
        )


        # D. Display the recommendations

        
        # Ensure the subset columns exist in your DataFrame
        st.dataframe(recommendations)
    else:
        st.error("No ISBN found for the selected book title.")



#######################################################################################################################################
### Streamlit Advantages and Disadvantages
    
st.subheader("Streamlit Advantages and Disadvantages")
st.write('**Advantages**')
st.write(' - Easy, Intuitive, Pythonic')
st.write(' - Free!')
st.write(' - Requires no knowledge of front end languages')
st.write('**Disadvantages**')
st.write(' - Apps all look the same')
st.write(' - Not very customizable')
st.write(' - A little slow. Not good for MLOps, therefore not scalable')
