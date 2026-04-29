# ⚽ La Liga Game Score Prediction

A machine learning project designed to predict the outcomes of Spanish La Liga football matches. The model predicts whether a match will result in a **Home Win**, **Away Win**, or **Draw**, using historical team form, player offensive statistics, and real-time squad injury data scraped from Transfermarkt.

## Project Overview

This project uses historical match data, top scorer/assist information, and live player availability data to engineer features representing a team's current form, offensive power, and squad health. An **XGBoost Classifier** is trained on these features to predict future match outcomes.

### Key Features Engineered

| Category | Features | Description |
|---|---|---|
| **Form & Dominance** | EMA Points, Goals, Shots, SoT, Corners | Exponential Moving Average over recent matches to capture true form |
| **Mathematical Differentials** | Form Diff, Offense Diff, Rest Diff | Explicit numerical differences between Home and Away teams |
| **Head-to-Head Bias** | H2H Win Rate | Historical win-rate of the Home team against the specific Away team |
| **Expected Offensive Index** | xG + xA Power | Aggregated Expected Goals and Expected Assists per team per season |
| **Squad Availability** | Missing Key Players, Missing Impact % | Real injury data from Transfermarkt for predictions; simulated distributions for training |

### Model Architecture
- **Algorithm:** XGBoost (Gradient Boosted Trees)
- **Features:** 26 engineered features per match
- **Training Split:** 80/20 chronological (TimeSeriesSplit for cross-validation)
- **Current Accuracy:** ~52% (elite tier for 3-class football prediction)

## Project Structure

```
Laliga-game-score-prediction/
├── LaligaSeasons/              # Raw match data CSVs (multiple seasons)
├── Laligascoring/              # Top scorers data per season
├── LaligaAssist/               # Top assisters data per season
├── Laliga Squads/              # Scraped player data (20 teams)
│   ├── Real Madrid/
│   │   └── player_data.csv     # Appearances, injuries, expected return
│   ├── Barcelona/
│   ├── Atlético Madrid/
│   └── ... (17 more teams)
│
├── Data_processing.py          # Step 1: Clean & combine raw data
├── feature_engeneering.py      # Step 2: Engineer ML features → ml_ready_data.csv
├── simulate_injury_history.py  # Step 3: Add simulated injury features for training
├── aggregate_player_data.py    # Aggregate squad health → current_squad_health.csv
├── train_model.py              # Step 4: Train XGBoost model → laliga_rf_model.pkl
├── predict.py                  # Step 5: Predict matches using real injury data
│
├── Processed_Matches.csv       # Cleaned match data
├── Processed_Scorers.csv       # Cleaned top scorers
├── Processed_Assists.csv       # Cleaned top assists
├── ml_ready_data.csv           # Final training dataset (26 features)
├── current_squad_health.csv    # Live squad health metrics per team
└── laliga_rf_model.pkl         # Trained XGBoost model
```

## Prerequisites

Make sure you have the following Python packages installed:
```bash
pip install pandas numpy scikit-learn joblib xgboost beautifulsoup4
```

## How to Use

### Full Pipeline (in order)

```bash
# 1. Clean and combine raw data
python Data_processing.py

# 2. Engineer features (EMA form, xG/xA, H2H, rest days)
python feature_engeneering.py

# 3. Simulate historical injury metrics for training
python simulate_injury_history.py

# 4. Aggregate real squad health from Transfermarkt scrapes
python aggregate_player_data.py

# 5. Train the XGBoost model
python train_model.py

# 6. Make predictions
python predict.py
```

> **⚠️ Important:** Steps 2 → 3 must run in order. `feature_engeneering.py` generates `ml_ready_data.csv`, then `simulate_injury_history.py` appends the injury columns to it. If you re-run step 2, you must re-run step 3 before training.

### Make Predictions

Edit the `predict_match()` calls at the bottom of `predict.py`:
```python
predict_match("Espanol", "Real Madrid", 5, 9)  # (home, away, home_rest_days, away_rest_days)
predict_match("Girona", "Mallorca", 6, 6)
```

**Example Output:**
```
==============================================
  MATCH PREDICTION: Espanol vs Real Madrid
==============================================
Current Home xG+xA Index : 54.80
Current Away xG+xA Index : 43.80
----------------------------------------------
Injury Report:
  Espanol              Missing Key Players: 0  |  Impact: 1.8%
  Real Madrid          Missing Key Players: 5  |  Impact: 22.9%
----------------------------------------------
Win Probabilities:
  [Espanol] Home Win : 38.9%
  Draw              : 29.4%
  [Real Madrid] Away Win : 31.7%

  Model Prediction: HOME WIN

==============================================
```

## Squad Data (Transfermarkt)

Each team folder in `Laliga Squads/` contains a `player_data.csv` with:

| Column | Description |
|---|---|
| `Player` | Player name |
| `Appearances` | Total appearances this season |
| `Injuries` | `1` if currently injured, `0` if fit |
| `Day Injured` | Date injury occurred |
| `Missed Games` | Number of games missed due to current injury |
| `Expected Return` | Estimated return date |

### Team Name Mapping

The team folder names map to the model's internal names as follows:

| Folder Name | Model Name |
|---|---|
| Atlético Madrid | Ath Madrid |
| Athletic Bilbao | Ath Bilbao |
| Real Betis | Betis |
| Real Sociedad | Sociedad |
| Celta Vigo | Celta |
| Rayo Vallecano | Vallecano |
| Espanyol | Espanol |
| Real Oviedo | Oviedo |

All other team names are identical.

## How Injury Data Works

Since historical injury data per match is unavailable, we use a **simulated history approach**:

1. **Training:** `simulate_injury_history.py` generates realistic injury distributions (Poisson for missing key players, Beta for impact %) so the model learns the mathematical relationship between squad availability and match outcomes.
2. **Prediction:** `predict.py` loads the **real** scraped data from `current_squad_health.csv` to provide accurate, context-aware predictions.

This means the model understands that "missing 5 key players" correlates with worse outcomes, and when Real Madrid is actually missing 5 starters, the prediction adjusts accordingly.

## Target Variable Classes

| Code | Outcome |
|---|---|
| `0` | Away Win |
| `1` | Draw |
| `2` | Home Win |
