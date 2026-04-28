import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
def main():
    data_path = '/Users/santomukiza/Desktop/Github/LaligaPrediction/Laliga-game-score-prediction/ml_ready_data.csv'
    df = pd.read_csv(data_path)
    # Sort by date to ensure strict chronological order
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    # Drop non-numeric columns that shouldn't be features
    features = [
        'Home_EMA_Points', 'Home_EMA_GS', 'Home_EMA_GC',
        'Away_EMA_Points', 'Away_EMA_GS', 'Away_EMA_GC',
        'Home_Expected_Offense', 'Away_Expected_Offense'
    ]
    
    X = df[features]
    y = df['Target'] # 0 = Away Win, 1 = Draw, 2 = Home Win
    split_index = int(len(X) * 0.8)

    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    model= RandomForestClassifier(
        n_estimators=200,# Number of trees in the forest
        max_depth=10,# Maximum depth of the trees
        class_weight='balanced',# Weight to handle imbalanced data
        random_state=42,# For reproducibility
        min_samples_split=10,# Minimum number of samples required to split a node
    )
    print("Training Random Forest Classifier... this might take a few seconds.")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print("\n--- Model Evaluation ---")
    print(f"Overall Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report (0=Away Win, 1=Draw, 2=Home Win):")
    print(classification_report(y_test, predictions))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    # 6. Save the Model
    # We save the trained model so predict.py can load it instantly without retraining
    model_save_path = '/Users/santomukiza/Desktop/Github/LaligaPrediction/Laliga-game-score-prediction/laliga_rf_model.pkl'
    joblib.dump(model, model_save_path)
    
    print(f"\nModel saved successfully to: {model_save_path}")

if __name__ == "__main__":
    main()



    