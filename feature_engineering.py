"""
Module for feature engineering and data preprocessing.
"""
import pandas as pd
import numpy as np
from typing import Tuple, List

class OutlierHandler:
  def __init__(self, dataframe: pd. DataFrame, q1: float = 0.25, q3: float = 0.75):
    """Initialize OutlierHandler.
    dataframe (pd.DataFrame): Input dataset
    q1 (float): First quartile (default: 0.25)
    q3 (float): Third quartile (default: 0.75)
      """
    self. df = dataframe. copy()
    self.q1 = q1
    self.q3 = q3
  
  def remove_outliers_iqr(self, columns: List[str], multiplier: float = 1.5) -> pd.DataFrame:
      """
      Remove outliers using IQR method.
      columns (List[str]): List of columns to check for outliers
      multiplier (float): IQR multiplier for bounds (default: 1.5)   
      Returns:pd.DataFrame: Dataset with outliers removed
      """
      Q1 = self.df[columns].quantile(self.q1)
      Q3 = self.df[columns].quantile(self.q3)
      IQR = Q3 - Q1
      # Remove rows with outliers
      self.df = self.df[~((self.df[columns] < (Q1 - multiplier * IQR)) | (self.df[columns] > (Q3 + multiplier * IQR))).any(axis=1)]

      return self.df
  def get_dataframe(self) -> pd.DataFrame:
    """Get the processed dataframe.
    Returns:
      pd.DataFrame: Processed dataset
    """
    return self.df

class MissingValueHandler:
   """
    Handles missing value imputation. 
  """
   def __init__(self, dataframe:  pd.DataFrame):
    """
    Initialize MissingValueHandler. 
    Args:
        dataframe (pd.DataFrame): Input dataset
    """
    self. df = dataframe.copy()

    def handle_missing_random(self, column: str, random_state: int = 0) -> pd.DataFrame:
        """
        Handle missing values with random imputation.
        Args:
            column (str): Column name with missing values
            random_state (int): Random state for reproducibility 
        Returns:
            pd.DataFrame: Dataset with random imputed values
        """
        random_sample = self.df[column].dropna().sample(
            self.df[column].isnull().sum(), 
            random_state=random_state
        )
        random_sample.index = self.df[self.df[column].isnull()].index
        self.df[f'{column}_random'] = self.df[column].combine_first(random_sample)
        return self.df
    
    def handle_missing_median(self, column: str) -> pd.DataFrame:
        """
        Handle missing values with median imputation.
        Args:
            column (str): Column name with missing values
        Returns:
            pd.DataFrame: Dataset with median imputed values
        """
        self.df[f'{column}_median'] = self.df[column]. fillna(self.df[column]. median())
        
        return self.df
    
    def handle_missing_grouped_median(self, column: str, group_cols: List[str]) -> pd.DataFrame:
        """
        Handle missing values with grouped median imputation
        Args:
            column (str): Column name with missing values
            group_cols (List[str]): List of columns to group by
            
        Returns:
            pd.DataFrame: Dataset with grouped median imputed values
        """
        grouped = self.df.groupby(group_cols)
        self.df[f'{column}_group_median'] = grouped[column]. apply(
            lambda x: x. fillna(x.median())
        )
        return self.df
    
    def handle_missing_forward_fill(self, column: str) -> pd.DataFrame:
        """
        Handle missing values with forward fill method.
        Args:
            column (str): Column name with missing values
            
        Returns: 
            pd.DataFrame: Dataset with forward filled values
        """
        self.df[f'{column}_ffill'] = self.df[column].fillna(method='ffill')
        
        return self.df
    
    def get_dataframe(self) -> pd.DataFrame:
        """
        Get the processed dataframe.
        
        Returns:
            pd.DataFrame: Processed dataset
        """
        return self.df

class CategoricalHandler:
    """
    Handles categorical variable encoding.
    """
    def __init__(self, dataframe: pd.DataFrame):
        """
        Initialize CategoricalHandler.
        Args:
            dataframe (pd. DataFrame): Input dataset
        """
        self.df = dataframe. copy()
        self.encoders = {}
    
    def label_encode(self, column: str) -> pd.DataFrame:
        """
        Apply label encoding to categorical variable.
        Args:
            column (str): Column name to encode
            
        Returns: 
            pd.DataFrame: Dataset with encoded column
        """
        unique_values = self.df[column].unique()
        encoding_dict = {val: idx for idx, val in enumerate(unique_values)}
        self.encoders[column] = encoding_dict
        
        self.df[f'{column}_encoded'] = self. df[column].map(encoding_dict)
        
        return self.df
    
    def one_hot_encode(self, column: str) -> pd.DataFrame:
        """
        Apply one-hot encoding to categorical variable.
        Args:
            column (str): Column name to encode
            
        Returns:
            pd. DataFrame: Dataset with one-hot encoded columns
        """
        one_hot = pd.get_dummies(self.df[column], prefix=column)
        self.df = pd.concat([self.df, one_hot], axis=1)
        return self.df
    
    def get_dataframe(self) -> pd.DataFrame:
        """
        Get the processed dataframe.
        
        Returns:
            pd.DataFrame: Processed dataset
        """
        return self.df 
