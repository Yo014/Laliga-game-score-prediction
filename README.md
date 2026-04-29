# La Liga Game Score Prediction

A machine learning project designed to predict the outcomes of Spanish La Liga football matches. The model predicts whether a match will result in a Home Win, Away Win, or Draw, based on historical team form, player offensive statistics, and real-time squad availability.

## Project Overview

This project uses historical match data, top scorer information, top assist data, and per-team injury reports to engineer features representing a team's current form, offensive power, and squad health. An XGBoost Classifier is then trained on these features to predict future match outcomes.

### Key Features Engineered
- **Form & Dominance (Exponential Moving Average):** Calculates an EMA for points, goals, shots, shots on target, and corners over recent matches to capture true match dominance.
- **Mathematical Differentials:** Explicitly calculates the numerical difference in form, offensive expected metrics, rest days, and squad health between the Home and Away teams.
- **Head-to-Head Bias:** Calculates the historical win-rate of the Home team against the specific Away team to capture tactical advantages.
- **Expected Offensive Index:** Aggregates Expected Goals (xG) and Expected Assists (xA) from top scorers and assisters to create an "Expected Offensive Index".
- **Squad Health & Injury Impact:** Quantifies how many key players (≥15 appearances) each team is missing due to injury, what percentage of the team's total playing time those players represent, and how much of the team's goal-scoring output is lost.

## Project Structure

- `LaligaSeasons/`: Directory containing raw match data CSVs.
- `Laligascoring/`: Directory containing top scorers data for various seasons.
- `LaligaAssist/`: Directory containing top assisters data for various seasons.
- `Laliga Squads/`: Directory containing per-team player data CSVs for the current season (appearances, goals, injuries, expected return dates).
- `Data_processing.py`: The data cleaning script. It recursively reads and combines raw files from `LaligaSeasons/`, `Laligascoring/`, and `LaligaAssist/`, standardizing team names and formatting for different seasons to generate `Processed_Matches.csv`, `Processed_Scorers.csv`, and `Processed_Assists.csv`.
- `build_squad_health.py`: The squad health aggregation script. It reads all player CSVs from `Laliga Squads/`, identifies currently injured players, and computes team-level injury metrics saved to `current_squad_health.csv`.
- `feature_engeneering.py`: The feature engineering script. It loads the processed data and squad health, calculates the EMA, Expected Offensive Index, and injury impact features, and outputs a single `ml_ready_data.csv` file used for training.
- `train_model.py`: The machine learning training script. It loads `ml_ready_data.csv`, trains an `XGBClassifier` from the `xgboost` library via grid search, evaluates its accuracy, and saves the trained model as `laliga_rf_model.pkl`.
- `predict.py`: The inference script. It dynamically calculates match differentials, Head-to-Head history, and squad health context, using the saved model to predict the win/draw probabilities for a specified matchup.

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

2. **Build Squad Health**
   Aggregate the current season's player injury data into team-level metrics:
   ```bash
   python build_squad_health.py
   ```
   *This will output `current_squad_health.csv` with per-team injury statistics.*

3. **Feature Engineering**
   Process the cleaned CSV files and squad health into a dataset ready for machine learning:
   ```bash
   python feature_engeneering.py
   ```
   *This will generate `ml_ready_data.csv` with 28 model features plus metadata.*

4. **Train the Model**
   Train the XGBoost classifier:
   ```bash
   python train_model.py
   ```
   *This will output the model's accuracy metrics (~52%) and save the trained model to `laliga_rf_model.pkl`.*

5. **Make Predictions**
   To predict a specific matchup, you can edit the `predict_match()` calls at the bottom of `predict.py`, then run:
   ```bash
   python predict.py
   ```
   *Example Output:*
   ```text
   ==============================================
     MATCH PREDICTION: Espanol vs Real Madrid 
   ==============================================
   Current Home xG+xA Index : 140.40
   Current Away xG+xA Index : 43.80
   ----------------------------------------------
   Squad Health:
     [Espanol] Missing 0 key players (0.0% playing time, 0.0% goals)
     [Real Madrid] Missing 5 key players (22.9% playing time, 33.7% goals)
   ----------------------------------------------
   Win Probabilities:
   [Espanol] Home Win : 48.8%
   Draw              : 26.5%
   [Real Madrid] Away Win : 24.6%

    Model Prediction: HOME WIN 

   ==============================================
   ```

## Model Features (28 total)

| Category | Features |
|---|---|
| Home Form (EMA) | Points, Goals Scored, Goals Conceded, Shots, Shots on Target, Corners |
| Away Form (EMA) | Points, Goals Scored, Goals Conceded, Shots, Shots on Target, Corners |
| Offensive Strength | Home Expected Offense, Away Expected Offense |
| Rest & Fatigue | Home Days Rest, Away Days Rest |
| Squad Health | Home/Away Missing Key Players, Missing Impact %, Missing Goals % |
| Differentials | Form Diff, Offense Diff, Rest Diff, Missing Key Diff, Missing Impact Diff |
| Historical | H2H Home Win Rate |

## Target Variable Classes
The model's internal target mapping is:
- `0`: Away Win
- `1`: Draw
- `2`: Home Win
