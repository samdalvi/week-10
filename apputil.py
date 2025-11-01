# your code here
import pandas as pd
import numpy as np
import pickle

def predict_rating(X, text=False):
    """
    Predict rating based on input features.
    
    Parameters:
    -----------
    X : pd.DataFrame or array-like
        If text=False: DataFrame with columns '100g_USD' and 'roast'
        If text=True: DataFrame with column 'text' containing review text
    text : bool
        If True, use text-based prediction (model_3)
        If False, use feature-based prediction (model_1 or model_2)
    
    Returns:
    --------
    array : Predicted ratings
    """
    
    if text:
        # Load text model and vectorizer
        with open('model_3.pickle', 'rb') as f:
            text_model_data = pickle.load(f)
        
        model = text_model_data['model']
        vectorizer = text_model_data['vectorizer']
        
        # Handle text input
        if isinstance(X, pd.DataFrame):
            texts = X.iloc[:, 0].values
        else:
            texts = X
        
        # Transform text using the fitted vectorizer
        X_vectorized = vectorizer.transform(texts)
        
        # Predict
        predictions = model.predict(X_vectorized)
        
        return predictions
    
    else:
        # Load models for feature-based prediction
        with open('model_1.pickle', 'rb') as f:
            model_1 = pickle.load(f)
        
        with open('model_2.pickle', 'rb') as f:
            model_2 = pickle.load(f)
        
        # Define roast mapping (same as in train.py)
        roast_mapping = {
            'Light': 0,
            'Medium-Light': 1,
            'Medium': 2,
            'Medium-Dark': 3,
            'Dark': 4
        }
        
        # Get valid roast values from training data
        valid_roasts = set(roast_mapping.keys())
        
        predictions = []
        
        for idx, row in X.iterrows():
            price = row['100g_USD']
            roast = row['roast']
            
            # Check if roast is valid and not NaN
            if pd.notna(roast) and roast in valid_roasts:
                # Use model_2 with both features
                roast_cat = roast_mapping[roast]
                X_input = np.array([[price, roast_cat]])
                pred = model_2.predict(X_input)[0]
            else:
                # Use model_1 with only price
                X_input = np.array([[price]])
                pred = model_1.predict(X_input)[0]
            
            predictions.append(pred)
        
        return np.array(predictions)
