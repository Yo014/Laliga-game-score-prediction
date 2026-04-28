# La Liga Game Score Prediction

A machine learning project designed to predict the outcomes of Spanish La Liga football matches. The model predicts whether a match will result in a Home Win, Away Win, or Draw, based on historical team form and player offensive statistics.

## Project Overview

This project uses historical match data, top scorer information, and top assist data to engineer features representing a team's current form and offensive power. An XGBoost Classifier is then trained on these features to predict future match outcomes.

### Key Features Engineered
- **Form & Dominance (Exponential Moving Average):** Calculates an EMA for points, goals, shots, shots on target, and corners over recent matches to capture true match dominance.
- **Mathematical Differentials:** Explicitly calculates the numerical difference in form, offensive expected metrics, and rest days between the Home and Away teams.
- **Head-to-Head Bias:** Calculates the historical win-rate of the Home team against the specific Away team to capture tactical advantages.
- **Expected Offensive Index:** Aggregates Expected Goals (xG) and Expected Assists (xA) from top scorers and assisters to create an "Expected Offensive Index".

## Project Structure

- `LaligaSeasons/`: Directory containing raw match data CSVs.
- `Laligascoring/`: Directory containing top scorers data for various seasons.
- `LaligaAssist/`: Directory containing top assisters data for various seasons.
- `Data_processing.py`: The data cleaning script. It recursively reads and combines raw files from `LaligaSeasons/`, `Laligascoring/`, and `LaligaAssist/`, standardizing team names and formatting for different seasons to generate `Processed_Matches.csv`, `Processed_Scorers.csv`, and `Processed_Assists.csv`.
- `feature_engeneering.py`: The feature engineering script. It loads the processed data, calculates the EMA and Expected Offensive Index, and outputs a single `ml_ready_data.csv` file used for training.
- `train_model.py`: The machine learning training script. It loads `ml_ready_data.csv`, trains an `XGBClassifier` from the `xgboost` library via grid search, evaluates its accuracy, and saves the trained model as `laliga_rf_model.pkl`.
- `predict.py`: The inference script. It dynamically calculates match differentials and Head-to-Head history, using the saved model to predict the win/draw probabilities for a specified matchup.

## Prerequisites
Make sure you have the following Python packages installed:
```bash
pip install pandas numpy scikit-learn joblib xgboost
```

## How to Use

1. **Data Processing**
   Clean and combine the raw CSV files from subdirectories across different seasons (including the latest formats):
   ```bash
   python Data_processing.py
   ```
   *This will output `Processed_Matches.csv`, `Processed_Scorers.csv`, and `Processed_Assists.csv`.*

2. **Feature Engineering**
   Process the cleaned CSV files into a dataset ready for machine learning:
   ```bash
   python feature_engeneering.py
   ```
   *This will generate `ml_ready_data.csv`.*

3. **Train the Model**
   Train the XGBoost classifier:
   ```bash
   python train_model.py
   ```
   *This will output the model's accuracy metrics (~52%) and save the trained model to `laliga_rf_model.pkl`.*

4. **Make Predictions**
   To predict a specific matchup, you can edit the `predict_match()` calls at the bottom of `predict.py`, then run:
   ```bash
   python predict.py
   ```
   *Example Output:*
   ```text
   ==============================================
     MATCH PREDICTION: Espanol vs Real Madrid 
   ==============================================
   Current Home xG+xA Index : 37.70
   Current Away xG+xA Index : 53.70
   ----------------------------------------------
   Win Probabilities:
   [Espanol] Home Win : 32.9%
   Draw              : 29.8%
   [Real Madrid] Away Win : 37.2%
   
    Model Prediction: AWAY WIN 
   
   ==============================================
   ```

## Target Variable Classes
The model's internal target mapping is:
- `0`: Away Win
- `1`: Draw
- `2`: Home Win
