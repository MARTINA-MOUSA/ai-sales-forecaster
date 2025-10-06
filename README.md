# AI Sales Forecaster

AI Sales Forecaster is a machine learning project designed to predict future sales using historical data. The project leverages advanced data preprocessing, feature engineering, and state-of-the-art models to help businesses make data-driven decisions for inventory management and sales planning.

## Features
- Automated data cleaning and preprocessing
- Advanced time-based feature extraction
- Outlier detection and removal
- Powerful sales forecasting model (XGBoost)
- Interactive Streamlit web app for predictions and visualization
- Version control for data and models using DVC

## Project Structure
```
├── api/                 # API for model serving
├── data/                # Raw, processed, and split datasets
├── models/              # Trained models and scalers
├── src/                 # Source code for preprocessing, training, prediction, and app
├── requirements.txt     # Python dependencies
├── dvc.yaml, .dvc       # DVC pipeline and tracking files
├── README.md            # Project documentation
```

## Technologies Used
- Python 3.x
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Streamlit
- DVC

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/MARTINA-MOUSA/ai-sales-forecaster.git
cd ai-sales-forecaster
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Reproduce the pipeline (optional, if using DVC)
```bash
dvc pull
dvc repro
```

### 4. Run data preprocessing
```bash
python src/data_preprocessing.py
```

### 5. Train the model
```bash
python src/train.py
```

### 6. Evaluate the model
```bash
python src/evaluate.py
```

### 7. Run the Streamlit app
```bash
streamlit run src/app_streamlit.py
```

## Usage
- Upload your sales data or use the provided dataset
- Visualize trends and predictions in the web app
- Download processed data and prediction results

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for suggestions or improvements.

## License
This project is licensed under the MIT License.

---
For questions or collaboration, feel free to contact the project owner via LinkedIn or GitHub.
