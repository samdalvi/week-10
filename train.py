import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the coffee analysis data
url = "https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv"
df = pd.read_csv(url)

# Exercise 1: 
X1 = df[['100g_USD']].dropna()
y1 = df.loc[X1.index, 'rating']

# Remove any remaining NaN values in y
mask = ~y1.isna()
X1 = X1[mask]
y1 = y1[mask]

# Train the model
lr_model = LinearRegression()
lr_model.fit(X1, y1)

# Save the model
with open('model_1.pickle', 'wb') as f:
    pickle.dump(lr_model, f)

# Exercise 2: 
def roast_category(roast_value):
    """Map roast values to numerical categories"""
    if pd.isna(roast_value):
        return np.nan
    
    roast_mapping = {
        'Light': 0,
        'Medium-Light': 1,
        'Medium': 2,
        'Medium-Dark': 3,
        'Dark': 4
    }
    
    # Return the mapped value or a default if not found
    return roast_mapping.get(roast_value, -1)

# Create roast_cat column
df['roast_cat'] = df['roast'].apply(roast_category)

# Prepare data for model 2
X2 = df[['100g_USD', 'roast_cat']].copy()
y2 = df['rating'].copy()

# Remove rows where rating is NaN
mask = ~y2.isna()
X2 = X2[mask]
y2 = y2[mask]

# Train Decision Tree Regressor
dtr_model = DecisionTreeRegressor(random_state=42)
dtr_model.fit(X2, y2)

# Save the model
with open('model_2.pickle', 'wb') as f:
    pickle.dump(dtr_model, f)

# Bonus Exercise 4: TF-IDF vectorization for text-based prediction
# Prepare text data
text_data = df['desc_3'].dropna()
y3 = df.loc[text_data.index, 'rating']

# Remove any remaining NaN values in y
mask = ~y3.isna()
text_data = text_data[mask]
y3 = y3[mask]

# Vectorize the text
tfidf_vectorizer = TfidfVectorizer(max_features=100)
X3 = tfidf_vectorizer.fit_transform(text_data)

# Train linear regression model on vectorized text
lr_text_model = LinearRegression()
lr_text_model.fit(X3, y3)

# Save the model and vectorizer
with open('model_3.pickle', 'wb') as f:
    pickle.dump({'model': lr_text_model, 'vectorizer': tfidf_vectorizer}, f)

print("All models trained and saved successfully!")
