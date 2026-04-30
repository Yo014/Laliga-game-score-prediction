# La Liga Game Score Prediction

A machine learning project designed to predict the outcomes of Spanish La Liga football matches. The model predicts whether a match will result in a Home Win, Away Win, or Draw, based on historical team form, player offensive statistics, real-time squad availability, betting market data, and referee characteristics.

## Project Overview

This project uses historical match data, top scorer information, top assist data, and per-team injury reports to engineer features representing a team's current form, offensive power, and squad health. An XGBoost Classifier is then trained on these features to predict future match outcomes.

### Key Features Engineered
- **Market Data (Betting Odds):** Incorporates raw betting odds and calculates **Normalized Implied Probabilities** (Market_Prob_H/D/A). This captures the "wisdom of the crowd" and is currently the model's most influential feature set.
- **Referee Statistics:** Tracks historical referee "personalities" by calculating rolling averages for cards shown and fouls called per game.
- **Form & Dominance (Exponential Moving Average):** Calculates an EMA for points, goals, shots, shots on target, and corners over recent matches to capture true match dominance.
- **Mathematical Differentials:** Explicitly calculates the numerical difference in form, offensive expected metrics, rest days, and squad health between the Home and Away teams.
- **Head-to-Head Bias:** Calculates the historical win-rate of the Home team against the specific Away team to capture tactical advantages.
- **Expected Offensive Index:** Aggregates Expected Goals (xG) and Expected Assists (xA) from top scorers and assisters to create an "Expected Offensive Index".
- **Squad Health & Injury Impact:** Quantifies how many key players (≥15 appearances) each team is missing due to injury, what percentage of the team's total playing time those players represent, and how much of the team's goal-scoring output is lost.

## Project Structure

- `LaligaSeasons/`: Directory containing raw match data CSVs (2015-2025).
- `Laligascoring/`: Directory containing top scorers data for various seasons.
- `LaligaAssist/`: Directory containing top assisters data for various seasons.
- `Laliga Squads/`: Directory containing per-team player data CSVs for the current season (appearances, goals, injuries, expected return dates).
- `Data_processing.py`: The data cleaning script. It recursively reads and combines raw files, standardizing team names and formatting for different seasons. It now captures fouls and cards for referee analysis.
- `build_squad_health.py`: The squad health aggregation script. It reads all player CSVs, identifies currently injured players, and computes team-level injury metrics.
- `feature_engeneering.py`: The feature engineering script. It loads processed data, calculates EMA, Market Probabilities, and Referee Stats. Outputs `ml_ready_data.csv`.
- `train_model.py`: The machine learning training script. Trains an `XGBClassifier` with grid search, evaluates accuracy, and saves the model as `laliga_rf_model.pkl`. Includes feature importance visualization.
- `predict.py`: The inference script. Dynamically calculates match differentials, handles betting odds inputs, and uses the saved model to predict outcomes.

## Prerequisites
Make sure you have the following Python packages installed:
```bash
pip install pandas numpy scikit-learn joblib xgboost
```

## How to Use

1. **Data Processing**
   ```bash
   python Data_processing.py
   ```
   *Combines raw CSVs into `Processed_Matches.csv`, capturing fouls, cards, and betting odds.*

2. **Build Squad Health**
   ```bash
   python build_squad_health.py
   ```
   *Aggregates current season's player injury data into `current_squad_health.csv`.*

3. **Feature Engineering**
   ```bash
   python feature_engeneering.py
   ```
   *Generates `ml_ready_data.csv` with 50+ model features including Market Data and Referee Stats.*

4. **Train the Model**
   ```bash
   python train_model.py
   ```
   *Trains the optimized XGBoost classifier. Current accuracy: **~54.5%**.*

5. **Make Predictions**
   ```bash
   python predict.py
   ```
   *Example call:* `predict_match("Barcelona", "Real Madrid", 7, 7, 2.10, 3.50, 3.30, "Jose Maria Sánchez")`

## Model Features (50 total)

| Category | Features |
|---|---|
| **Market Data** | B365H/D/A Odds, Normalized Implied Probabilities (Prob_H/D/A) |
| **Referee Stats** | Historical Average Cards per Game, Average Fouls per Game |
| **Home Form (EMA)** | Points, GS, GC, GoalDiff, Shots, ShotsOnTarget, Corners, ShotsConceded, SOTConceded, CornersConceded |
| **Away Form (EMA)** | Points, GS, GC, GoalDiff, Shots, ShotsOnTarget, Corners, ShotsConceded, SOTConceded, CornersConceded |
| **Offensive Strength**| Home Expected Offense (xG+xA), Away Expected Offense |
| **Rest & Fatigue** | Home Days Rest, Away Days Rest |
| **Squad Health** | Home/Away Missing Key Players, Missing Impact %, Missing Goals % |
| **Differentials** | Form Diff, Offense Diff, Rest Diff, Missing Key Diff, Missing Impact Diff |
| **Historical** | H2H Home Win Rate, Team Codes, Referee Codes |

## Target Variable Classes
The model's internal target mapping is:
- `0`: Away Win
- `1`: Draw
- `2`: Home Win
