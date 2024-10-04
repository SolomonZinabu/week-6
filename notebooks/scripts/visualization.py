# notebooks/scripts/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns

def plot_distribution(df, column):
    """Plot distribution of a numerical column."""
    plt.figure(figsize=(10, 5))
    sns.histplot(df[column], bins=50, kde=True)
    plt.title(f'Distribution of {column}')
    plt.show()

def plot_categorical_distribution(df, column):
    """Plot count of a categorical column."""
    plt.figure(figsize=(8, 4))
    sns.countplot(df[column])
    plt.title(f'Count of {column}')
    plt.show()

def detect_outliers(df, column):
    """Plot boxplot to detect outliers in a numerical column."""
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=df[column])
    plt.title(f'Boxplot of {column}')
    plt.show()
