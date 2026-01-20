"""
Data Loading Module
Handles loading and initial exploration of the Titanic dataset. 
"""
import pandas as pd 
from typing import Tuple
from config import DATA_CONFIG

class Dataloader:
  """Handles loading and initial exploration of the dataset."""
  def __init__(self,train_path: str, test_path: str):
    self.train_path = DATA_CONFIG[train_path]
    self.test_path = DATA_CONFIG[test_path]
    self.train_df = None
    self.test_df = None
  
  def load_data(self)->pd.DataFrame:
      """Load train and test datasets from CSV files. 
      Returns:
        tuple: (train_data, test_data) DataFrames
      """
      self.train_df = pd.read_csv(self.train_path)
      self.test_df = pd.read_csv(self.test_path)
      return self.train_df, self.test_df
  
  def select_features(self, feature_list: list) -> pd.DataFrame:
    """Select specified features from the training dataset."""
    self.train_df = self. train_df[feature_list]
    return self.train_df
  
  # def get_train_data(self):
  #       """Get training data."""
  #       return self.train_data
    
  # def get_test_data(self):
  #       """Get test data."""
  #       return self.test_data

# a = Dataloader('train_path', 'test_path')
# a.load_data()
# p = a.explore_data()
# print(p)

