# Diego's Capstone Project

### Project
A key challenge in book recommendation systems is delivering personalized suggestions that resonate with readers' preferences. With countless books available, finding one that matches individual tastes can be tedious. Existing systems may not fully capture a reader's interests based on factors like author, publication year, or user ratings.

### Goal
Develop a recommendation system using nearest neighbor or collaborative filtering models. These models will analyze user preferences and book attributes, such authors, publication years, categories and ratings. The nearest neighbor approach will identify similar users (age, country) or books (categories, reviews), while collaborative filtering will suggest books based on users with similar reading patterns.

### Data
The Book-Crossing dataset comprises three key files joined into an already preprocessed main file:

    Users: Contains anonymized user information, including demographics like location and age (if available).
    
    Books: Identified by ISBN, this file includes book titles, authors, publication years, summary, publishers, and cover image URLs from Amazon.
    
    Ratings: Includes user ratings for books, both explicit (on a scale of 1-10) and implicit (indicated by a rating of 0).
    This data is essential for analyzing user preferences and behavior, which is critical for developing an effective recommendation model
    using nearest neighbor and collaborative filtering techniques.


####  Demographic & User data:
- User ID
- Location
- Age
- city
- state
- country
- Rating

####  Book data:
- ISBN
- Book title
- Book author
- Year of publication
- Publisher
- img_s
- img_m
- img_l
- Summary
- Language
- Category


### Work 
For the analysis of the data:

    Dropped columns that are not useful for model creation, such as image URLs and book summaries.

    Examined each column to explore distributions and patterns, analyzing frequencies of authors, users, ages, etc.

    Grouped the 6,000 unique category values into 11 broader categories to simplify analysis and facilitate encoding for future modeling.
    
    Grouped age into 5 different groups for the same reason.
    

Findigs

    User Activity: Some users have left up to 6,000 reviews, which may indicate potential bot activity or anomalies within the dataset.

    Book Titles: While book titles are numerous and unique, the title with the highest reviews has only 2.5k out of 1 million total reviews, suggesting that titles alone may not be strong predictors of review trends.

    Authors & Publishers: Authors and publishers both show a wide range of unique values, but many of these have high occurrence counts, potentially making them useful for differentiating reviews.

    Year of Publication: Newer books generally receive more interactions, showing that publication year influences review volume.

    Geographical Data: The majority of reviews come from users in the USA, presenting two possible analytical directions: focusing on US states or analyzing data by continent.

    Language: Most of the reviewed books in the dataset are in English, which may limit insights into preferences for non-English literature.

Correlations

    Age and Year of Publication: Both show a slight correlation with ratings, but there is a statistically significant across different ranges of these independent variables, suggesting meaningful differences based on age and publication year.

    Categories and Age: Visualizations indicate varying category preferences across age groups. Statistical differences will be assessed after encoding the category column, which should help clarify age-based category trends.

Preprocessing Categorical Data

    Categorical Columns: The majority of columns in this dataset are categorical and need encoding for analysis.

    Book Title: Since it offers limited information and has low review variance, it’s likely to be dropped.

    Book Author and Publisher: These can be encoded using frequency encoding or by the mean of reviews, which may help capture their influence more effectively.

    Category: Now grouped into broader categories, either one-hot encoding or binary encoding would be suitable.

    Geographical Data: Depending on whether we focus on continents or states, one-hot encoding is likely appropriate for better geographical insights.
    

### Results
