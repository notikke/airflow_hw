import os
import json
import dill
import pandas as pd
from datetime import datetime
from glob import glob
from typing import List, Dict, Any

# Укажем путь к файлам проекта
path = os.environ.get('PROJECT_PATH', '.')

def load_latest_model() -> Any:
    model_files = glob(f'{path}/data/models/cars_pipe_*.pkl')
    if not model_files:
        raise FileNotFoundError("No trained models found in data/models/")
    
    latest_model = max(model_files, key=os.path.getctime)
    with open(latest_model, 'rb') as f:
        return dill.load(f)

def read_json_files() -> List[Dict]:
    test_files = glob(f'{path}/data/test/*.json')
    if not test_files:
        raise FileNotFoundError("No test files found in data/test/")
    
    data = []
    for test_file in test_files:
        with open(test_file, 'r') as f:
            data.append(json.load(f))
    return data

def prepare_data(raw_data: List[Dict]) -> pd.DataFrame:
    return pd.DataFrame(raw_data)

def make_predictions(model: Any, df: pd.DataFrame) -> pd.DataFrame:
    predictions = model.predict(df)
    return pd.DataFrame({
        'id': df['id'],
        'price': df['price'],
        'predicted_price_category': predictions
    })

def save_predictions(predictions: pd.DataFrame) -> str:
    os.makedirs(f'{path}/data/predictions', exist_ok=True)
    pred_filename = f'{path}/data/predictions/preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv'
    predictions.to_csv(pred_filename, index=False)
    return pred_filename

def predict():
    try:
        model = load_latest_model()
        print(f"Loaded model: {type(model.named_steps['classifier']).__name__}")
        
        raw_data = read_json_files()
        test_df = prepare_data(raw_data)
        print(f"Loaded {len(test_df)} test samples")
        
        predictions = make_predictions(model, test_df)
        
        pred_file = save_predictions(predictions)
        print(f"Predictions saved to {pred_file}")
        
        return pred_file
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise

if __name__ == '__main__':
    predict()