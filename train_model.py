import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

def main():
    print("--- Starting Advanced Model Training ---")
    
    # 1. Load the Advanced Engineered Data
    data_path = '/Users/santomukiza/Desktop/Github/LaligaPrediction/Laliga-game-score-prediction/ml_ready_data.csv'
    df = pd.read_csv(data_path)


    # Sort chronologically to prevent data leakage
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # 2. Select Features (Now including Expected Offense, Match Dominance Metrics, Diffs, and H2H)
    features = [
        'Home_EMA_Points', 'Home_EMA_GS', 'Home_EMA_GC',
        'Home_EMA_Shots', 'Home_EMA_ShotsOnTarget', 'Home_EMA_Corners',
        'Away_EMA_Points', 'Away_EMA_GS', 'Away_EMA_GC',
        'Away_EMA_Shots', 'Away_EMA_ShotsOnTarget', 'Away_EMA_Corners',
        'Home_Expected_Offense', 'Away_Expected_Offense',
        'Home_Days_Rest', 'Away_Days_Rest',
        'Form_Diff', 'Offense_Diff', 'Rest_Diff',
        'H2H_Home_Win_Rate'
    ]
    
    X = df[features]
    y = df['Target'] 

    # 3. Time Series Split
    split_index = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    print(f"Training on {len(X_train)} matches...")
    print(f"Testing on {len(X_test)} matches...")

    # 4. Hyperparameter Tuning (Finding the perfect brain structure)
    # We test different sizes and depths of the forest to find the most accurate one
    print("\nRunning Grid Search to find optimal parameters (This will take 30-60 seconds)...")
    
    # New XGBoost specific parameters
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0] # Helps prevent overfitting
    }
    
    # We use TimeSeriesSplit for cross-validation to respect the chronological order
    tscv = TimeSeriesSplit(n_splits=3)
    
    # Initialize XGBoost instead of Random Forest
    xgb_model = XGBClassifier(random_state=42, eval_metric='mlogloss')
    
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=tscv, scoring='accuracy', n_jobs=-1)
    
    grid_search.fit(X_train, y_train)
    
    # Extract the absolute best model from the grid search
    best_model = grid_search.best_estimator_
    print(f"Optimal Parameters Found: {grid_search.best_params_}")

    # 5. Evaluate the Optimized Model
    predictions = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print("\n--- Optimized Model Evaluation ---")
    print(f"Overall Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report (0=Away Win, 1=Draw, 2=Home Win):")
    print(classification_report(y_test, predictions))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    # 6. Save the Model
    model_save_path = 'laliga_rf_model.pkl'
    joblib.dump(best_model, model_save_path)
    
    print(f"\nAdvanced model saved successfully to: {model_save_path}")
    print("You can now update predict.py to use this new .pkl file!")

if __name__ == "__main__":
    main()