# La Liga Game Score Prediction

A machine learning project designed to predict the outcomes of Spanish La Liga football matches. The model predicts whether a match will result in a Home Win, Away Win, or Draw, based on historical team form and player offensive statistics.

## Project Overview

This project uses historical match data, top scorer information, and top assist data to engineer features representing a team's current form and offensive power. A Random Forest Classifier is then trained on these features to predict future match outcomes.

### Key Features Engineered
- **Form (Exponential Moving Average):** Calculates an EMA for points, goals scored, and goals conceded over recent matches to give more weight to a team's most recent performances.
- **Expected Offensive Index:** Aggregates Expected Goals (xG) and Expected Assists (xA) from top scorers and assisters to create an "Expected Offensive Index" for each team per season, reflecting their underlying ability to create and finish chances.

## Project Structure

- `LaligaSeasons/`: Directory containing raw match data CSVs.
- `Laligascoring/`: Directory containing top scorers data for various seasons.
- `LaligaAssist/`: Directory containing top assisters data for various seasons.
- `Data_processing.py`: The data cleaning script. It recursively reads and combines raw files from `LaligaSeasons/`, `Laligascoring/`, and `LaligaAssist/`, standardizing team names and formatting for different seasons to generate `Processed_Matches.csv`, `Processed_Scorers.csv`, and `Processed_Assists.csv`.
- `feature_engeneering.py`: The feature engineering script. It loads the processed data, calculates the EMA and Expected Offensive Index, and outputs a single `ml_ready_data.csv` file used for training.
- `train_model.py`: The machine learning training script. It loads `ml_ready_data.csv`, trains a `RandomForestClassifier` from `scikit-learn`, evaluates its accuracy, and saves the trained model as `laliga_rf_model.pkl`.
- `predict.py`: The inference script. It uses the saved model and recent team stats to predict the win/draw probabilities for a specified matchup.

## Prerequisites
Make sure you have the following Python packages installed:
```bash
pip install pandas numpy scikit-learn joblib
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
   Train the Random Forest classifier:
   ```bash
   python train_model.py
   ```
   *This will output the model's accuracy metrics and save the trained model to `laliga_rf_model.pkl`.*

4. **Make Predictions**
   To predict a specific matchup, you can edit the `predict_match()` calls at the bottom of `predict.py`, then run:
   ```bash
   python predict.py
   ```
   *Example Output:*
   ```text
   ==============================================
     MATCH PREDICTION: Real Madrid vs Barcelona 
   ==============================================
   Current Home xG+xA Index : 43.80
   Current Away xG+xA Index : 24.30
   ----------------------------------------------
   Win Probabilities:
   [Real Madrid] Home Win : 65.9%
   Draw              : 19.3%
   [Barcelona] Away Win : 14.9%
   
    Model Prediction: HOME WIN 
   
   ==============================================
   ```

## Target Variable Classes
The model's internal target mapping is:
- `0`: Away Win
- `1`: Draw
- `2`: Home Win
