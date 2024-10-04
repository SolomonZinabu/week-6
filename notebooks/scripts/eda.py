# notebooks/scripts/eda.py
def summarize_data(df):
    """Generate summary statistics for numerical columns."""
    return df.describe(include='all')

def check_missing_values(df):
    """Check for missing values in the dataset."""
    missing_values = df.isnull().sum()
    return missing_values[missing_values > 0]  # Return only columns with missing values
