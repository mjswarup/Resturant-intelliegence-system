import pandas as pd
from pathlib import Path
from src.config.config import RAW_DATA_PATH

def load_raw_data(path: str | Path = None) -> pd.DataFrame:
    """
    Load the raw restaurant dataset.
    
    Parameters
    ----------
    path : str or Path, optional
        Custom path to the CSV file. Defaults to RAW_DATA_PATH.
    
    Returns
    -------
    pd.DataFrame
        Raw restaurant data.
    """
    file_path = Path(path) if path else RAW_DATA_PATH
    
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {file_path}\n"
            f"Please place your CSV file in: data/raw/Dataset.csv"
        )
    
    df = pd.read_csv(file_path)
    print(f"✅ Dataset loaded successfully")
    print(f"   → Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"   → File : {file_path.name}")
    
    return df