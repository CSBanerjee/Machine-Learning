# Titanic Survival Prediction - Modularized OOP Implementation

<img width="2752" height="1536" alt="unnamed (3)" src="https://github.com/user-attachments/assets/19804276-f65e-4524-9aa2-b6f23ff79f15" />


A complete, production-ready, object-oriented implementation of the Titanic survival prediction machine learning project. 

## Project Structure

```
├── data_loader.py           # Data loading and management
├── eda.py                   # Exploratory Data Analysis
├── feature_engineering.py   # Feature engineering & preprocessing
├── feature_selection.py     # Feature selection & importance
├── modelling.py             # Model building & evaluation
├── main.py                  # Pipeline orchestrator
├── requirements.txt         # Project dependencies
└── README.md               # This file
```

## Modules Overview

### 1. **data_loader.py**
Handles data loading and basic dataset operations. 
- `DataLoader` class for loading CSV files
- Feature selection
- Dataset info and statistics

### 2. **eda. py**
Performs exploratory data analysis.
- Statistical summaries
- Missing value analysis
- Target imbalance detection
- Visualization methods (boxplots, distributions, heatmaps)
- Bivariate analysis

### 3. **feature_engineering.py**
Processes and transforms raw features. 
- `OutlierHandler`: IQR-based outlier removal
- `MissingValueHandler`: Multiple imputation strategies
  - Random imputation
  - Median imputation
  - Grouped median imputation
- `CategoricalHandler`: Categorical variable encoding
  - Label encoding
  - One-hot encoding

### 4. **feature_selection.py**
Selects most important features.
- `FeatureSelector` class for:
  - Random Forest feature importance
  - Correlation analysis
  - Low variance feature removal
  - Feature importance visualization

### 5. **modelling.py**
Builds and evaluates ML models.
- `DataPreprocessor`: Data preprocessing for modeling
  - Gaussian distribution checking
  - Feature rescaling (StandardScaler)
  - Train-test splitting
- `LogisticRegressionModel`: Logistic regression wrapper
- `ModelEvaluator`: Model evaluation metrics
  - Accuracy, Precision, Recall, F1-Score, ROC-AUC
  - Confusion matrix visualization

### 6. **main.py**
Orchestrates the complete pipeline.
- `TitanicPipeline` class combining all steps
- 5-step workflow: 
  1. Data Gathering
  2. Exploratory Data Analysis
  3. Feature Engineering
  4. Feature Selection
  5. Modelling

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage
```python
from main import TitanicPipeline

# Initialize pipeline
pipeline = TitanicPipeline(
    train_path="data/train.csv",
    test_path="data/test.csv"
)

# Run complete pipeline
pipeline.run_pipeline()
```

### Module-by-Module Usage
```python
# Data Loading
from data_loader import DataLoader
loader = DataLoader("train.csv")
df = loader.load_data()

# EDA
from eda import EDA
eda = EDA(df)
eda.get_statistical_summary()

# Feature Engineering
from feature_engineering import OutlierHandler, MissingValueHandler
outlier_handler = OutlierHandler(df)
df = outlier_handler.remove_outliers_iqr(['Age', 'Fare'])

# Feature Selection
from feature_selection import FeatureSelector
selector = FeatureSelector(df, 'Survived')
selector.calculate_feature_importance(['Age', 'Fare'])

# Modelling
from modelling import DataPreprocessor, LogisticRegressionModel, ModelEvaluator
preprocessor = DataPreprocessor(df, 'Survived')
X_train, X_test, y_train, y_test = preprocessor.train_test_split_data(['Age', 'Fare'])

model = LogisticRegressionModel()
model.train(X_train, y_train)
predictions = model.predict(X_test)

evaluator = ModelEvaluator()
metrics = evaluator.calculate_metrics(y_test, predictions)
```

## Project Workflow

### Step 1: Data Gathering
- Load Titanic dataset
- Select relevant features:  PassengerId, Age, Sex, Fare, Embarked, Survived
- Dataset shape:  (891, 6)

### Step 2: Exploratory Data Analysis (EDA)
- Understand data structure and statistics
- Check for missing values (Age:  19. 87%, Embarked: 0.22%)
- Identify outliers in Age and Fare
- Analyze target variable distribution
- Correlation analysis

### Step 3: Feature Engineering

#### 3.1 Outlier Treatment
- IQR method with multiplier 1.5
- Remove extreme values from Age and Fare columns

#### 3.2 Missing Value Handling
- **Random Imputation**: Replace with random samples from non-null values
- **Median Imputation**: Replace with median values
- **Grouped Median Imputation**: Replace with median within groups (Embarked, Sex)

#### 3.3 Categorical Variables
- **Sex**: Label encode (male → 0, female → 1)
- **Embarked**: One-hot encode (C, Q, S)

### Step 4: Feature Selection
- Calculate feature importance using Random Forest
- Analyze correlation with target variable
- Select top features for modeling

### Step 5: Modelling

#### 5.1 Check Gaussian Distribution
- Verify normality of features

#### 5.2 Feature Rescaling
- Standardize features using StandardScaler

#### 5.3 Train-Test Split
- 80-20 split with random_state=42

#### 5.4 Model Training
- Logistic Regression classifier
- Binary classification (Survived:  0/1)

#### 5.5 Model Evaluation
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC Score
- Confusion Matrix visualization

## Key Features

✅ **Object-Oriented Design**: Clean, modular, and reusable classes
✅ **Separation of Concerns**: Each module handles a specific responsibility
✅ **Flexible & Extensible**: Easy to add new models or preprocessing steps
✅ **Production-Ready**: Proper error handling and documentation
✅ **Visualizations**: Built-in plotting capabilities
✅ **Reproducibility**: Fixed random states for consistent results

## Data Dictionary

| Feature | Type | Description |
|---------|------|-------------|
| PassengerId | int | Unique passenger identifier |
| Age | float | Age of passenger (years) |
| Sex | str | Gender of passenger (male/female) |
| Fare | float | Ticket price ($) |
| Embarked | str | Port of embarkation (C/Q/S) |
| Survived | int | Target variable (0=No, 1=Yes) |

## Performance Metrics

After model training, you'll get:
- **Accuracy**: Overall correctness of predictions
- **Precision**: True positives among all positive predictions
- **Recall**: True positives among all actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve

## Future Enhancements

- [ ] Add more ML models (Random Forest, SVM, XGBoost)
- [ ] Implement cross-validation
- [ ] Add hyperparameter tuning
- [ ] Create prediction interface
- [ ] Add model persistence (saving/loading)
- [ ] Implement ensemble methods
