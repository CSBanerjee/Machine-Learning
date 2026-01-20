"""
Main pipeline orchestrator for Titanic survival prediction.
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from config import DATA_CONFIG
from data_loader import Dataloader
from eda import EDA
from feature_engineering import OutlierHandler, MissingValueHandler, CategoricalHandler
from feature_selection import FeatureSelector
from modelling import DataPreprocessor, LogisticRegressionModel, ModelEvaluator


class TitanicPipeline: 
    """
    Complete pipeline for Titanic survival prediction. 
    """
    
    def __init__(self, train_path: str, test_path: str = None):
        """
        Initialize the pipeline.
        
        Args:
            train_path (str): Path to training CSV
            test_path (str): Path to test CSV (optional)
        """
        self.train_path = train_path
        self.test_path = test_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self. y_train = None
        self.y_test = None
        self.model = None
        self.evaluator = None
    
    def step_1_data_gathering(self):
        """Step 1: Load and prepare data."""
        print("\n" + "="*50)
        print("STEP 1: DATA GATHERING")
        print("="*50)
        
        loader = Dataloader(self.train_path, self.test_path)
        self.train_df, self.test_df = loader.load_data()
        self.df = self.train_df

        
        # Select relevant features
        features = ['PassengerId', 'Age', 'Sex', 'Fare', 'Embarked', 'Survived']
        self.df = self.df[features]
        
        print(f"Dataset shape: {self.df.shape}")
        print("\nFirst few rows:")
        print(self.df.head())
        
        return self.df
    
    def step_2_eda(self):
        """Step 2: Exploratory Data Analysis."""
        print("\n" + "="*50)
        print("STEP 2: EXPLORATORY DATA ANALYSIS (EDA)")
        print("="*50)
        
        eda = EDA(self.df)
        
        print("\nDataset Info:")
        self.df.info()
        
        print("\nStatistical Summary:")
        print(eda.get_statistical_summary())
        
        print("\nTarget Variable Distribution:")
        print(eda.check_target_imbalance('Survived'))
        
        print("\nMissing Values:")
        print(eda.check_missing_values())
        
        print("\nMissing Values Percentage:")
        print(eda.get_missing_percentage())
        
        print("\nConstant Columns:", eda.check_constant_columns())
        
        print("\nPlotting visualizations...")
        # eda.plot_boxplot(['Age', 'Fare'])
        # eda.plot_distribution(['Age', 'Fare'])
        # eda.plot_correlation_heatmap(['Age', 'Fare'])
    
    def step_3_feature_engineering(self):
        """Step 3: Feature Engineering."""
        print("\n" + "="*50)
        print("STEP 3: FEATURE ENGINEERING")
        print("="*50)
        
        # 3.1: Outlier Treatment
        print("\n3.1 Outlier Treatment (IQR Method)")
        outlier_handler = OutlierHandler(self.df, q1=0.25, q3=0.70)
        self.df = outlier_handler.remove_outliers_iqr(['Age', 'Fare'])
        print(f"Dataset shape after outlier removal: {self.df.shape}")
        
        # 3.2: Missing Value Handling
        print("\n3.2 Missing Value Handling")
        missing_handler = MissingValueHandler(self.df)
        
        # Random imputation
        missing_handler.handle_missing_random('Age')
        
        # Median imputation
        missing_handler. handle_missing_median('Age')
        
        # Grouped median imputation
        missing_handler. handle_missing_grouped_median('Age', ['Embarked', 'Sex'])
        
        # Fill remaining missing values in Embarked
        missing_handler.df['Embarked'].fillna(missing_handler.df['Embarked'].mode()[0], inplace=True)
        
        self.df = missing_handler.get_dataframe()
        print("Missing values handled")
        
        # 3.3: Categorical Variable Handling
        print("\n3.3 Categorical Variable Handling")
        categorical_handler = CategoricalHandler(self.df)
        categorical_handler.label_encode('Sex')
        categorical_handler.one_hot_encode('Embarked')
        
        self.df = categorical_handler.get_dataframe()
        print("Categorical variables encoded")
        
        print(f"\nDataset shape after feature engineering: {self.df.shape}")
        print("\nFeatures after engineering:")
        print(self.df.columns.tolist())
    
    def step_4_feature_selection(self):
        """Step 4: Feature Selection."""
        print("\n" + "="*50)
        print("STEP 4: FEATURE SELECTION")
        print("="*50)
        
        selector = FeatureSelector(self. df, 'Survived')
        
        # Select features for importance calculation
        numeric_features = ['Age_group_median', 'Fare', 'Sex_encoded']
        
        print("\nCalculating feature importance using Random Forest...")
        importance_df = selector.calculate_feature_importance(numeric_features)
        print(importance_df)
        
        print("\nCalculating correlation with target...")
        correlation = selector.calculate_correlation(numeric_features)
        print(correlation)
        
        top_features = selector.get_top_features(top_n=3)
        print(f"\nTop 3 features:  {top_features}")
    
    def step_5_modelling(self):
        """Step 5: Modelling."""
        print("\n" + "="*50)
        print("STEP 5: MODELLING")
        print("="*50)
        
        # Select features for modeling
        feature_columns = ['Age_group_median', 'Fare', 'Sex_encoded', 'Embarked_C', 'Embarked_Q', 'Embarked_S']
        
        # 5.1: Check Gaussian Distribution
        print("\n5.1 Checking Gaussian Distribution...")
        preprocessor = DataPreprocessor(self.df, 'Survived')
        # preprocessor.check_gaussian_distribution(['Age_group_median', 'Fare'])
        
        # 5.2: Rescaling of Data
        print("\n5.2 Rescaling Features...")
        preprocessor.rescale_features(['Age_group_median', 'Fare'])
        
        # 5.3: Train-Test Split
        print("\n5.3 Train-Test Split...")
        self.X_train, self.X_test, self.y_train, self.y_test = preprocessor.train_test_split_data(
            feature_columns, test_size=0.2, random_state=42
        )
        print(f"Training set size: {self.X_train. shape}")
        print(f"Test set size: {self.X_test.shape}")
        
        # 5.4: Model Creation & Training
        print("\n5.4 Model Creation & Training...")
        self.model = LogisticRegressionModel(random_state=42)
        self.model.train(self.X_train, self.y_train)
        print("Model trained successfully!")
        
        # 5.5: Model Evaluation
        print("\n5.5 Model Evaluation...")
        y_pred = self.model.predict(self.X_test)
        y_proba = self.model. predict_proba(self.X_test)
        
        self.evaluator = ModelEvaluator()
        metrics = self.evaluator.calculate_metrics(self.y_test, y_pred, y_proba)
        
        print("\nModel Performance Metrics:")
        self.evaluator.print_metrics()
        
        print("\nConfusion Matrix:")
        self.evaluator.plot_confusion_matrix(self.y_test, y_pred)
    
    def run_pipeline(self):
        """Run the complete pipeline."""
        print("\n" + "█"*50)
        print("TITANIC SURVIVAL PREDICTION PIPELINE")
        print("█"*50)
        
        try:
            self.step_1_data_gathering()
            self.step_2_eda()
            self.step_3_feature_engineering()
            self.step_4_feature_selection()
            self.step_5_modelling()
            
            print("\n" + "█"*50)
            print("PIPELINE COMPLETED SUCCESSFULLY!")
            print("█"*50 + "\n")
            
        except Exception as e:
            print(f"\nError occurred: {str(e)}")
            raise


def main():
    pipeline = TitanicPipeline(
        train_path="train_path",
        test_path="test_path"
    )
    pipeline.run_pipeline()



if __name__ == "__main__":
    main()