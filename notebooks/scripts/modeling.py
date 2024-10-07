import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt

# Function to train the model
def train_model(train_data, target, model_type='logistic'):
    """Trains a Logistic Regression or Random Forest model and returns the fitted model."""
    X_train = train_data.drop(columns=[target])
    y_train = train_data[target]
    
    if model_type == 'logistic':
        model = LogisticRegression(penalty='l1', C=0.9, solver='liblinear')
    elif model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Choose 'logistic' or 'random_forest'.")
    
    model.fit(X_train, y_train)
    return model

# Function to evaluate the model
def evaluate_model(model, test_data, target):
    """Evaluates the model using metrics like Accuracy, Precision, Recall, F1 Score, and ROC-AUC."""
    X_test = test_data.drop(columns=[target])
    y_test = test_data[target]
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    # AUC and ROC curve
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"AUC Score: {auc}")
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

# Function to perform hyperparameter tuning using Grid Search
def perform_grid_search(train_data, target, model_type='logistic'):
    """Performs hyperparameter tuning using GridSearchCV for logistic regression or random forest."""
    X_train = train_data.drop(columns=[target])
    y_train = train_data[target]
    
    if model_type == 'logistic':
        param_grid = {
            'C': [0.01, 0.1, 1, 10],
            'penalty': ['l1', 'l2']
        }
        model = LogisticRegression(solver='liblinear')
    elif model_type == 'random_forest':
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
        model = RandomForestClassifier(random_state=42)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Choose 'logistic' or 'random_forest'.")
    
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters for {model_type}: {grid_search.best_params_}")
    return grid_search.best_estimator_
