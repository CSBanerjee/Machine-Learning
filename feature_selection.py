"""
Module for feature selection and importance analysis.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from typing import List, Tuple


class FeatureSelector:
    """
    Handles feature selection and importance analysis.
    """
    def __init__(self, dataframe: pd.DataFrame, target_col: str):
        """
        Initialize FeatureSelector. 
        Args:
            dataframe (pd.DataFrame): Input dataset
            target_col (str): Target column name
        """
        self.df = dataframe.copy()
        self.target_col = target_col
        self. feature_importance = None
    
    def calculate_feature_importance(self, features: List[str], n_estimators: int = 100) -> pd.DataFrame:
        """
        Calculate feature importance using Random Forest.
        Args:
            features (List[str]): List of feature column names
            n_estimators (int): Number of trees in random forest 
        Returns:
            pd.DataFrame: Feature importance dataframe sorted by importance
        """
        X = self.df[features]
        y = self.df[self.target_col]
        # Handle missing values
        X = X.fillna(X.mean())
        # Train Random Forest
        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        rf.fit(X, y)
        # Get feature importance
        self.feature_importance = pd.DataFrame({
            'feature': features,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        return self. feature_importance
    
    def get_top_features(self, top_n: int = 5) -> List[str]:
        """
        Get top N important features.
        Args:
            top_n (int): Number of top features to return
            
        Returns:
            List[str]: List of top feature names
        """
        if self.feature_importance is None:
            raise ValueError("Feature importance not calculated.  Run calculate_feature_importance first.")
        return self.feature_importance.head(top_n)['feature'].tolist()
    
    def calculate_correlation(self, features: List[str], target: str = None) -> pd.DataFrame:
        """
        Calculate correlation of features with target variable.
        Args:
            features (List[str]): List of feature column names
            target (str): Target column name (default: self.target_col)
            
        Returns:
            pd.DataFrame: Correlation dataframe
        """
        if target is None:
            target = self.target_col
        
        corr_data = self.df[features + [target]].corr()
        return corr_data[target]. sort_values(ascending=False)
    
    def remove_low_variance_features(self, features: List[str], threshold: float = 0.01) -> List[str]:
        """
        Remove features with low variance.
        Args:
            features (List[str]): List of feature column names
            threshold (float): Variance threshold  
        Returns: 
            List[str]: List of features with variance above threshold
        """
        variances = self.df[features]. var()
        selected_features = variances[variances > threshold].index.tolist()
        
        return selected_features
    
    def get_feature_importance_plot(self, top_n: int = 10):
        """
        Display feature importance plot.
        Args:
            top_n (int): Number of top features to plot
        """
        import matplotlib.pyplot as plt
        if self.feature_importance is None:
            raise ValueError("Feature importance not calculated. Run calculate_feature_importance first.")
        
        top_features = self.feature_importance.head(top_n)
        plt.figure(figsize=(10, 6))
        plt.barh(top_features['feature'], top_features['importance'])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Top {} Feature Importances'.format(top_n))
        plt.tight_layout()
        plt.show()