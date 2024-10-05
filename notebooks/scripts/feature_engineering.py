import pandas as pd
import numpy as np
from xverse.transformer import WOE

# Function to create aggregate features
def create_aggregate_features(df):
    """Creates aggregate features such as total, average transaction amounts, and transaction count."""
    if 'CustomerId' not in df.columns or 'Amount' not in df.columns or 'TransactionId' not in df.columns:
        raise KeyError("Required columns 'CustomerId', 'Amount', or 'TransactionId' are missing from the dataframe.")
    
    agg_features = df.groupby('CustomerId').agg(
        total_amount=('Amount', 'sum'),
        avg_amount=('Amount', 'mean'),
        transaction_count=('TransactionId', 'count'),
        std_transaction_amount=('Amount', 'std')
    ).reset_index()

    # Handle NaN values that might arise due to standard deviation calculation
    agg_features['std_transaction_amount'].fillna(0, inplace=True)
    
    return agg_features

# Function to extract temporal features
def extract_temporal_features(df):
    """Extracts hour, day, month, and year from TransactionStartTime."""
    if 'TransactionStartTime' not in df.columns:
        raise KeyError("Required column 'TransactionStartTime' is missing from the dataframe.")
    
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], errors='coerce')

    df['transaction_hour'] = df['TransactionStartTime'].dt.hour
    df['transaction_day'] = df['TransactionStartTime'].dt.day
    df['transaction_month'] = df['TransactionStartTime'].dt.month
    df['transaction_year'] = df['TransactionStartTime'].dt.year

    # Fill NaN values generated from invalid datetime conversion
    df[['transaction_hour', 'transaction_day', 'transaction_month', 'transaction_year']].fillna(-1, inplace=True)

    return df

# Function for encoding categorical variables
def encode_categorical(df, columns, method='one-hot'):
    """Encodes categorical columns using one-hot or label encoding."""
    if not isinstance(columns, list):
        raise TypeError("Columns parameter must be a list.")
    
    for col in columns:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' is missing from the dataframe.")
    
    if method == 'one-hot':
        df_encoded = pd.get_dummies(df, columns=columns, drop_first=True)
    elif method == 'label':
        for col in columns:
            df[col] = df[col].astype('category').cat.codes
        df_encoded = df
    else:
        raise ValueError("Method must be either 'one-hot' or 'label'.")
    
    return df_encoded

# Function to handle missing values
def handle_missing_values(df, strategy='mean'):
    """Handles missing values by filling with mean, median, or mode for numerical columns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns  # Select only numeric columns

    if numeric_cols.empty:
        raise ValueError("No numerical columns found for missing value handling.")
    
    if strategy == 'mean':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif strategy == 'median':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    elif strategy == 'mode':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mode().iloc[0])
    elif strategy == 'remove':
        df.dropna(inplace=True)
    else:
        raise ValueError("Invalid strategy. Use 'mean', 'median', 'mode', or 'remove'.")
    
    return df

# Function to normalize or standardize numerical features
def scale_features(df, columns, method='normalize'):
    """Scales numerical features using normalization or standardization."""
    if not isinstance(columns, list):
        raise TypeError("Columns parameter must be a list.")
    
    for col in columns:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' is missing from the dataframe.")
        if not np.issubdtype(df[col].dtype, np.number):
            raise TypeError(f"Column '{col}' is not numerical.")
    
    if method == 'normalize':
        df[columns] = (df[columns] - df[columns].min()) / (df[columns].max() - df[columns].min())
    elif method == 'standardize':
        df[columns] = (df[columns] - df[columns].mean()) / df[columns].std()
    else:
        raise ValueError("Method must be either 'normalize' or 'standardize'.")
    
    return df


# Function to calculate WoE and IV with enhanced error handling
def calculate_woe_iv(df, target, feature_columns):
    """Performs Weight of Evidence (WoE) and Information Value (IV) binning."""
    
    # Ensure the input is a pandas DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError("The input data must be a pandas DataFrame.")
    
    # Ensure target and feature columns are in the DataFrame
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' is missing from the dataframe.")
    
    for col in feature_columns:
        if col not in df.columns:
            raise KeyError(f"Feature column '{col}' is missing from the dataframe.")
        if df[col].dtype != 'object' and not pd.api.types.is_categorical_dtype(df[col]):
            raise ValueError(f"Feature column '{col}' must be categorical or of type object for WoE calculation.")
    
    # Convert columns to categorical if not already done
    df[feature_columns] = df[feature_columns].astype('category')
    
    # Make an explicit copy to avoid chained assignment issues
    feature_df = df[feature_columns].copy()
    target_series = df[target].copy()

    # Check types before applying WoE
    if not isinstance(feature_df, pd.DataFrame):
        raise TypeError(f"Expected feature_df to be a pandas DataFrame, got {type(feature_df)}.")
    
    if not isinstance(target_series, pd.Series):
        raise TypeError(f"Expected target_series to be a pandas Series, got {type(target_series)}.")
    
    print(f"Feature DataFrame shape: {feature_df.shape}, types:\n{feature_df.dtypes}")
    print(f"Target Series shape: {target_series.shape}, type: {target_series.dtype}")
    
    woe_transformer = WOE()

    try:
        # Apply WoE transformation
        df_woe = woe_transformer.fit_transform(feature_df, target_series)
        iv = woe_transformer.woe_df[['Variable', 'IV']].sort_values(by='IV', ascending=False)
    except Exception as e:
        raise ValueError(f"Error during WoE/IV calculation: {e}")
    
    return df_woe, iv

# # Sample call of the function in your notebook
# categorical_columns = ['ProductCategory', 'ChannelId']
# woe_features, iv_values = calculate_woe_iv(data, 'FraudResult', categorical_columns)
# print("WoE-transformed data:")
# print(woe_features.head())
# print("Information Value of features:")
# print(iv_values)
