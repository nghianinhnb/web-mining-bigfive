#!/usr/bin/env python3
"""
TF-IDF + Ridge Regression baseline model for Big Five personality prediction.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
import joblib
from pathlib import Path
from typing import Union, List, Dict, Optional

from src.config import TRAIT_COLS


class TfidfRidgeModel:
    """
    TF-IDF + Ridge Regression baseline for Big Five personality prediction.
    
    This model uses TF-IDF features extracted from text and trains a Ridge
    regression model for each of the 5 personality traits.
    """
    
    def __init__(self, alpha: float = 1.0, max_features: int = 10000):
        """
        Initialize the model.
        
        Args:
            alpha: Ridge regression regularization strength
            max_features: Maximum number of TF-IDF features
        """
        self.alpha = alpha
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
        )
        self.regressor = MultiOutputRegressor(
            Ridge(alpha=alpha, random_state=42)
        )
        self.is_fitted = False
    
    def fit(self, texts: pd.Series, targets: pd.DataFrame) -> "TfidfRidgeModel":
        """
        Fit the model on training data.
        
        Args:
            texts: Series of text strings (concatenated tweets per user)
            targets: DataFrame with columns for each trait (open, conscientious, etc.)
            
        Returns:
            self
        """
        # Convert to list if needed
        if hasattr(texts, 'tolist'):
            texts = texts.tolist()
        
        # Fit vectorizer and transform texts
        X = self.vectorizer.fit_transform(texts)
        
        # Ensure targets is numpy array
        if hasattr(targets, 'values'):
            y = targets.values
        else:
            y = np.array(targets)
        
        # Fit regressor
        self.regressor.fit(X, y)
        self.is_fitted = True
        
        return self
    
    def predict(self, texts: Union[pd.Series, List[str]]) -> np.ndarray:
        """
        Predict personality traits for given texts.
        
        Args:
            texts: List or Series of text strings
            
        Returns:
            Array of shape (n_samples, 5) with predicted trait scores
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Convert to list if needed
        if hasattr(texts, 'tolist'):
            texts = texts.tolist()
        
        X = self.vectorizer.transform(texts)
        predictions = self.regressor.predict(X)
        
        # Clip to [0, 1] range
        predictions = np.clip(predictions, 0, 1)
        
        return predictions
    
    def evaluate(self, texts: pd.Series, targets: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate model on given data.
        
        Args:
            texts: Series of text strings
            targets: DataFrame with trait columns
            
        Returns:
            Dictionary with RMSE and MAE metrics for each trait and average
        """
        predictions = self.predict(texts)
        
        if hasattr(targets, 'values'):
            y_true = targets.values
        else:
            y_true = np.array(targets)
        
        metrics = {}
        
        # Calculate per-trait metrics
        rmse_values = []
        mae_values = []
        
        for i, trait in enumerate(TRAIT_COLS):
            rmse = np.sqrt(mean_squared_error(y_true[:, i], predictions[:, i]))
            mae = mean_absolute_error(y_true[:, i], predictions[:, i])
            
            metrics[f"rmse_{trait}"] = rmse
            metrics[f"mae_{trait}"] = mae
            
            rmse_values.append(rmse)
            mae_values.append(mae)
        
        # Calculate average metrics
        metrics["rmse_avg"] = np.mean(rmse_values)
        metrics["mae_avg"] = np.mean(mae_values)
        
        return metrics
    
    def save(self, path: Union[str, Path]) -> None:
        """Save model to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump({
            'vectorizer': self.vectorizer,
            'regressor': self.regressor,
            'alpha': self.alpha,
            'max_features': self.max_features,
            'is_fitted': self.is_fitted,
        }, path)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "TfidfRidgeModel":
        """Load model from file."""
        path = Path(path)
        data = joblib.load(path)
        
        model = cls(alpha=data['alpha'], max_features=data['max_features'])
        model.vectorizer = data['vectorizer']
        model.regressor = data['regressor']
        model.is_fitted = data['is_fitted']
        
        return model


class TfidfRidgeWithOpinion(TfidfRidgeModel):
    """
    TF-IDF + Ridge model with additional opinion mining features.
    
    Extends the base model to include sentiment and emotion features
    extracted using CardiffNLP models.
    """
    
    def __init__(self, alpha: float = 1.0, max_features: int = 10000):
        super().__init__(alpha=alpha, max_features=max_features)
        self.opinion_dim = None  # Will be set during fit
    
    def fit(self, texts: pd.Series, targets: pd.DataFrame, 
            opinion_features: Optional[np.ndarray] = None) -> "TfidfRidgeWithOpinion":
        """
        Fit model with optional opinion features.
        
        Args:
            texts: Series of text strings
            targets: DataFrame with trait columns
            opinion_features: Optional numpy array of opinion features
            
        Returns:
            self
        """
        # Convert to list if needed
        if hasattr(texts, 'tolist'):
            texts = texts.tolist()
        
        # Fit vectorizer and transform texts
        X_tfidf = self.vectorizer.fit_transform(texts)
        
        # Combine with opinion features if provided
        if opinion_features is not None:
            self.opinion_dim = opinion_features.shape[1]
            X = np.hstack([X_tfidf.toarray(), opinion_features])
        else:
            self.opinion_dim = 0
            X = X_tfidf
        
        # Ensure targets is numpy array
        if hasattr(targets, 'values'):
            y = targets.values
        else:
            y = np.array(targets)
        
        # Fit regressor
        self.regressor.fit(X, y)
        self.is_fitted = True
        
        return self
    
    def predict(self, texts: Union[pd.Series, List[str]], 
                opinion_features: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict with optional opinion features.
        
        Args:
            texts: List or Series of text strings
            opinion_features: Optional numpy array of opinion features
            
        Returns:
            Array of shape (n_samples, 5) with predicted trait scores
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Convert to list if needed
        if hasattr(texts, 'tolist'):
            texts = texts.tolist()
        
        X_tfidf = self.vectorizer.transform(texts)
        
        # Combine with opinion features if provided
        if opinion_features is not None:
            X = np.hstack([X_tfidf.toarray(), opinion_features])
        else:
            X = X_tfidf
        
        predictions = self.regressor.predict(X)
        
        # Clip to [0, 1] range
        predictions = np.clip(predictions, 0, 1)
        
        return predictions
