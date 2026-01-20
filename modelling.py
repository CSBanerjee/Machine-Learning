"""
Module for model building and evaluation.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict


class DataPreprocessor:
    """
    Handles data preprocessing for modeling.
    """
    def __init__(self, dataframe: pd.DataFrame, target_col: str):
        """
        Initialize DataPreprocessor.
        Args:
            dataframe (pd. DataFrame): Input dataset
            target_col (str): Target column name
        """
        self.df = dataframe.copy()
        self.target_col = target_col
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def check_gaussian_distribution(self, columns: list) -> None:
        """
        Check and visualize Gaussian distribution of features. 
        Args:
            columns (list): List of column names to check
        """
        fig, axes = plt.subplots(len(columns), 1, figsize=(10, 4 * len(columns)))
        for idx, col in enumerate(columns):
            if len(columns) > 1:
                axes[idx].hist(self.df[col]. dropna(), bins=30, edgecolor='black')
                axes[idx].set_title(f'Distribution of {col}')
            else:
                axes[0].hist(self.df[col].dropna(), bins=30, edgecolor='black')
                axes[0].set_title(f'Distribution of {col}')
        plt.tight_layout()
        plt.show()
    
    def rescale_features(self, feature_columns: list) -> pd.DataFrame:
        """
        Rescale features using StandardScaler. 
        
        Args:
            feature_columns (list): List of feature column names
            
        Returns:
            pd.DataFrame: Dataset with rescaled features
        """
        scaled_data = self.scaler. fit_transform(self.df[feature_columns])
        self.df[feature_columns] = scaled_data
        
        return self.df
    
    def train_test_split_data(self, feature_columns: list, test_size: float = 0.2, 
                             random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np. ndarray]:
        """
        Split data into train and test sets. 
        
        Args:
            feature_columns (list): List of feature column names
            test_size (float): Test set proportion
            random_state (int): Random state for reproducibility
            
        Returns:
            Tuple:  X_train, X_test, y_train, y_test
        """
        X = self.df[feature_columns]
        y = self.df[self.target_col]
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def get_train_test_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np. ndarray]:
        """
        Get current train and test data.
        
        Returns:
            Tuple: X_train, X_test, y_train, y_test
        """
        return self.X_train, self.X_test, self.y_train, self.y_test
class LogisticRegressionModel: 
    """
    Logistic Regression model wrapper.
    """
    def __init__(self, random_state: int = 42):
        """
        Initialize LogisticRegressionModel.
        
        Args:
            random_state (int): Random state for reproducibility
        """
        self. model = LogisticRegression(random_state=random_state, max_iter=1000)
        self.predictions = None
        self.predictions_proba = None
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the logistic regression model.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training target
        """
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Make predictions on test data.
        
        Args:
            X_test (np.ndarray): Test features
            
        Returns:
            np.ndarray: Predicted labels
        """
        self.predictions = self.model.predict(X_test)
        return self.predictions
    
    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X_test (np.ndarray): Test features
            
        Returns:
            np. ndarray: Prediction probabilities
        """
        self.predictions_proba = self. model.predict_proba(X_test)
        return self. predictions_proba
    
    def get_coefficients(self) -> np.ndarray:
        """
        Get model coefficients.
        
        Returns:
            np. ndarray: Model coefficients
        """
        return self.model.coef_


class ModelEvaluator:
    """
    Evaluates model performance. 
    """
    def __init__(self):
        """Initialize ModelEvaluator."""
        self.metrics = {}
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         y_proba: np.ndarray = None) -> Dict[str, float]:
        """
        Calculate evaluation metrics.
        
        Args:
            y_true (np. ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            y_proba (np.ndarray): Prediction probabilities (optional)
            
        Returns: 
            Dict[str, float]: Dictionary of metrics
        """
        self.metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred)
        }
        
        if y_proba is not None: 
            self.metrics['roc_auc'] = roc_auc_score(y_true, y_proba[: , 1])
        
        return self.metrics
    
    def print_metrics(self) -> None:
        """Print evaluation metrics."""
        for metric, value in self.metrics.items():
            print(f"{metric}: {value:.4f}")
    
    def plot_confusion_matrix(self, y_true:  np.ndarray, y_pred: np.ndarray) -> None:
        """
        Plot confusion matrix.
        Args:
            y_true (np. ndarray): True labels
            y_pred (np.ndarray): Predicted labels
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        plt.show()
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get calculated metrics.
        Returns:
            Dict[str, float]:  Dictionary of metrics
        """
        return self.metrics