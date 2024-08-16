from sklearn.linear_model import LogisticRegression
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
import pandas as pd

def explained_variance(X: pd.DataFrame, y: pd.Series) -> float:
    """
    Calculate the explained variance for a generalized linear model (GLM)
    with a binary response variable.
    Ref: https://www.nature.com/articles/nature25501/figures/11

    Description: Logistic regression pseudo-R2 was extracted as a measure of “explained variance”
    in patient response (i.e., the percent of variation in patient response that can be attributed
    to the contributions of the biological inputs)
    
    
    Args:
    X (pd.DataFrame): The feature values (can have multiple columns).
    y (pd.Series): The response labels, with 'R' for response and 'NR' for no response.
    
    Returns:
    float: The pseudo R-squared for the model, a measure of explained variance.
    """
    # Encode the response variable ('R' and 'NR') to numeric values
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Add a constant to the features for the intercept term
    X_with_intercept = sm.add_constant(X)
    
    # Fit a logistic regression model using statsmodels, which provides pseudo R-squared
    glm_binom = sm.GLM(y_encoded, X_with_intercept, family=sm.families.Binomial())
    glm_result = glm_binom.fit()
    ev = glm_result.pseudo_rsquared()
    
    return ev