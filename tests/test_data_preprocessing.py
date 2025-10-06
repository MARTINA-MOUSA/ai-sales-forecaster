import os
import subprocess

def test_data_preprocessing_runs():

    result = subprocess.run(
        ["python", "src/data_preprocessing.py"],
        capture_output=True,
        text=True
    )

    assert result.returncode == 0, f"Script failed:\n{result.stderr}"

    processed_path = "data/processed/walmart_enhanced.csv"
    assert os.path.exists(processed_path), f"{processed_path} not found!"

    import pandas as pd
    df = pd.read_csv(processed_path)
    assert "Weekly_Sales" in df.columns, "Weekly_Sales column missing!"

    print(" Data preprocessing ran successfully!")
