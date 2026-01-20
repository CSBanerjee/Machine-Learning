"""
Configuration Module
Centralized configuration for the Titanic ML project.
"""

# Data paths
DATA_CONFIG = {
    'train_path': 'data/train.csv',
    'test_path':  'data/test.csv',
    #'submission_path': 'submissions/submission.csv',
}

# # Feature engineering configuration
# FEATURE_CONFIG = {
#     'age_bins': [0, 12, 18, 35, 60, 100],
#     'age_labels': ['Child', 'Teen', 'Adult', 'Middle', 'Senior'],
#     'fare_quantiles': 4,
#     'fare_labels': ['Low', 'Medium', 'High', 'VeryHigh'],
#     'categorical_features': ['Sex', 'Title', 'Embarked', 'AgeGroup', 'FareGroup'],
#     'selected_features': [
#         'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
#         'Title', 'FamilySize', 'IsAlone', 'Embarked', 'AgeGroup', 'FareGroup'
#     ]
# }

# # Model training configuration
# MODEL_CONFIG = {
#     'random_state': 42,
#     'test_size': 0.2,
#     'cv_folds':  5,
#     'logistic_regression': {
#         'max_iter': 1000,
#         'random_state': 42
#     },
#     'random_forest': {
#         'n_estimators':  100,
#         'random_state': 42,
#         'n_jobs': -1
#     },
#     'gradient_boosting': {
#         'n_estimators': 100,
#         'random_state': 42
#     }
# }

# # Evaluation configuration
# EVAL_CONFIG = {
#     'metrics': ['accuracy', 'precision', 'recall', 'f1', 'auc'],
#     'plot_confusion_matrix': True,
# }