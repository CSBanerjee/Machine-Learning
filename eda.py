"""
Module for Exploratory Data Analysis (EDA).
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib. pyplot as plt
from typing import Dict, List


class EDA:
    """
    Performs exploratory data analysis on the dataset.
    """
    
    def __init__(self, dataframe: pd.DataFrame):
        """
        Initialize EDA with a dataframe.
        
        Args:
            dataframe (pd. DataFrame): Dataset to analyze
        """
        self.df = dataframe
    
    def get_statistical_summary(self) -> pd.DataFrame:
        """
        Get statistical summary of the dataset.
        
        Returns:
            pd. DataFrame: Statistical summary transposed
        """
        return self.df.describe().T
    
    def check_missing_values(self) -> pd.Series:
        """
        Check for missing values in the dataset.
        
        Returns:
            pd.Series: Count of missing values per column
        """
        return self.df.isnull().sum()
    
    def get_missing_percentage(self) -> pd.Series:
        """
        Get percentage of missing values per column. 
        
        Returns:
            pd.Series: Percentage of missing values
        """
        return self.df.isnull().sum() / self.df.shape[0]
    
    def check_target_imbalance(self, target_col: str) -> pd.Series:
        """
        Check for class imbalance in target variable.
        
        Args:
            target_col (str): Target column name
            
        Returns: 
            pd.Series: Value counts of target variable
        """
        return self.df[target_col]. value_counts()
    
    def check_constant_columns(self) -> List[str]:
        """
        Identify columns with unique values (constant columns).
        
        Returns:
            List[str]: List of constant column names
        """
        constant_cols = []
        for col in self.df.columns:
            if len(self. df[col].value_counts()) == 1:
                constant_cols.append(col)
        return constant_cols
    
    def plot_boxplot(self, columns: List[str]) -> None:
        """
        Plot boxplots for specified columns. 
        
        Args:
            columns (List[str]): List of column names to plot
        """
        self.df[columns].plot. box()
        plt.show()
    
    def plot_distribution(self, columns: List[str]) -> None:
        """
        Plot KDE distribution for specified columns.
        
        Args:
            columns (List[str]): List of column names to plot
        """
        fig, axes = plt.subplots(1, len(columns), figsize=(8 * len(columns), 3))
        for idx, col in enumerate(columns):
            sns.kdeplot(self.df[col], ax=axes[idx] if len(columns) > 1 else axes)
            axes[idx].set_title(f'Distribution of {col}') if len(columns) > 1 else axes. set_title(f'Distribution of {col}')
        plt.tight_layout()
        plt.show()
    
    def plot_correlation_heatmap(self, columns: List[str]) -> None:
        """
        Plot correlation heatmap for specified columns.
        
        Args:
            columns (List[str]): List of column names to analyze
        """
        cor = self.df[columns].corr()
        sns.heatmap(cor, cmap='YlGnBu', annot=True)
        plt.show()
    
    def plot_bivariate_analysis(self, target_col: str, numeric_cols: List[str], 
                               categorical_col: str) -> None:
        """
        Plot bivariate analysis with target variable.
        
        Args:
            target_col (str): Target column name
            numeric_cols (List[str]): List of numeric column names
            categorical_col (str): Categorical column name
        """
        fig, axes = plt.subplots(1, len(numeric_cols) + 1, figsize=(14, 3))
        
        for idx, col in enumerate(numeric_cols):
            sns.boxplot(x=target_col, y=col, data=self.df, ax=axes[idx])
        
        sns.countplot(x=target_col, hue=categorical_col, data=self.df, ax=axes[-1])
        plt.tight_layout()
        plt.show()