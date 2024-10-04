# notebooks/scripts/load_data.py
import pandas as pd

def load_data(file_path):
    """Load CSV data into a pandas DataFrame."""
    try:
        data = pd.read_csv(file_path)
        print(f"Data loaded successfully. Shape: {data.shape}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
