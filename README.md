# La Liga Game Score Prediction

A machine learning project designed to predict the outcomes of Spanish La Liga football matches. The model predicts whether a match will result in a Home Win, Away Win, or Draw, based on historical team form and player offensive statistics.

## Project Overview

This project uses historical match data, top scorer information, and top assist data to engineer features representing a team's current form and offensive power. A Random Forest Classifier is then trained on these features to predict future match outcomes.

### Key Features Engineered
- **Form (Exponential Moving Average):** Calculates an EMA for points, goals scored, and goals conceded over recent matches to give more weight to a team's most recent performances.
- **Offensive Index:** Aggregates coefficients from top scorers and assisters to create an "Offensive Power" rating for each team per season.

## Project Structure

- `LaligaSeasons/`: Directory containing raw match data CSVs.
- `Laligascoring/`: Directory containing top scorers data for various seasons.
- `LaligaAssist/`: Directory containing top assisters data for various seasons.
- `feature_engeneering.py`: The data processing script. It loads the raw data, calculates the EMA and Offensive Index, and outputs a single `ml_ready_data.csv` file used for training.
- `train_model.py`: The machine learning training script. It loads `ml_ready_data.csv`, trains a `RandomForestClassifier` from `scikit-learn`, evaluates its accuracy, and saves the trained model as `laliga_rf_model.pkl`.
- `predict.py`: The inference script. It uses the saved model and recent team stats to predict the win/draw probabilities for a specified matchup.

## Prerequisites

Make sure you have the following Python packages installed:
```bash
pip install pandas numpy scikit-learn joblib
```

## How to Use

1. **Feature Engineering**
   Process the raw CSV files into a dataset ready for machine learning:
   ```bash
   python feature_engeneering.py
   ```
   *This will generate `ml_ready_data.csv`.*

2. **Train the Model**
   Train the Random Forest classifier:
   ```bash
   python train_model.py
   ```
   *This will output the model's accuracy metrics and save the trained model to `laliga_rf_model.pkl`.*

3. **Make Predictions**
   To predict a specific matchup, you can edit the `predict_match()` calls at the bottom of `predict.py`, then run:
   ```bash
   python predict.py
   ```
   *Example Output:*
   ```text
   --- MATCH PREDICTION: Real Madrid (Home) vs Barcelona (Away) ---
   Win Probabilities:
   [Real Madrid] Home Win : 45.0%
   Draw              : 25.0%
   [Barcelona] Away Win : 30.0%
   
   Model Prediction: HOME WIN 
   ```

## Target Variable Classes
The model's internal target mapping is:
- `0`: Away Win
- `1`: Draw
- `2`: Home Win
