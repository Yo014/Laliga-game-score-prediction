import pandas as pd
import joblib


SQUAD_HEALTH_PATH = '/Users/santomukiza/Desktop/Github/LaligaPrediction/Laliga-game-score-prediction/current_squad_health.csv'


def get_latest_team_stats(team_name, is_home, df):
    """
    Scans the dataset to find the most recent form and Expected Offensive Index 
    for the requested team.
    """
    team_history = df[(df['HomeTeam'] == team_name) | (df['AwayTeam'] == team_name)].copy()
    
    if team_history.empty:
        print(f"Error: Could not find team '{team_name}'. Check spelling.")
        return None
        
    latest_match = team_history.iloc[-1]
    
    if latest_match['HomeTeam'] == team_name:
        ema_pts = latest_match['Home_EMA_Points']
        ema_gs  = latest_match['Home_EMA_GS']
        ema_gc  = latest_match['Home_EMA_GC']
        ema_sh  = latest_match['Home_EMA_Shots']
        ema_sot = latest_match['Home_EMA_ShotsOnTarget']
        ema_co  = latest_match['Home_EMA_Corners']
        off_idx = latest_match['Home_Expected_Offense']
    else:
        ema_pts = latest_match['Away_EMA_Points']
        ema_gs  = latest_match['Away_EMA_GS']
        ema_gc  = latest_match['Away_EMA_GC']
        ema_sh  = latest_match['Away_EMA_Shots']
        ema_sot = latest_match['Away_EMA_ShotsOnTarget']
        ema_co  = latest_match['Away_EMA_Corners']
        off_idx = latest_match['Away_Expected_Offense']

    return [ema_pts, ema_gs, ema_gc, ema_sh, ema_sot, ema_co, off_idx]


def get_real_injury_data(team_name, squad_health_df):
    """
    Fetches the REAL current injury metrics from the scraped Transfermarkt data.
    Falls back to league-average values if the team is not found.
    """
    row = squad_health_df[squad_health_df['Team'] == team_name]
    if row.empty:
        print(f"  [WARNING] '{team_name}' not found in squad health data — using league averages.")
        return {'Missing_Key_Players': 1.5, 'Missing_Impact_Pct': 8.0}
    return {
        'Missing_Key_Players': float(row.iloc[0]['Missing_Key_Players']),
        'Missing_Impact_Pct':  float(row.iloc[0]['Missing_Impact_Pct']),
    }


def predict_match(home_team, away_team, home_rest_days, away_rest_days):
    print(f"\nAnalyzing Matchup: {home_team} (Home) vs {away_team} (Away)...")
    
    data_path  = '/Users/santomukiza/Desktop/Github/LaligaPrediction/Laliga-game-score-prediction/ml_ready_data.csv'
    model_path = '/Users/santomukiza/Desktop/Github/LaligaPrediction/Laliga-game-score-prediction/laliga_rf_model.pkl'
    raw_path   = '/Users/santomukiza/Desktop/Github/LaligaPrediction/Laliga-game-score-prediction/Processed_Matches.csv'
    
    try:
        raw_matches   = pd.read_csv(raw_path)
        df            = pd.read_csv(data_path)
        model         = joblib.load(model_path)
        squad_health  = pd.read_csv(SQUAD_HEALTH_PATH)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    # Attach team names so we can search
    df['HomeTeam'] = raw_matches.iloc[df.index]['HomeTeam']
    df['AwayTeam'] = raw_matches.iloc[df.index]['AwayTeam']

    # 1. Fetch current EMA form
    home_stats = get_latest_team_stats(home_team, True,  df)
    away_stats = get_latest_team_stats(away_team, False, df)
    
    if not home_stats or not away_stats:
        return

    # 2. Fetch REAL injury data from Transfermarkt scrape
    home_inj = get_real_injury_data(home_team, squad_health)
    away_inj = get_real_injury_data(away_team, squad_health)

    # 3. Calculate differentials
    form_diff    = home_stats[0] - away_stats[0]
    offense_diff = home_stats[6] - away_stats[6]
    rest_diff    = home_rest_days - away_rest_days

    # Injury differentials: positive = home team is LESS hurt
    missing_key_diff    = away_inj['Missing_Key_Players'] - home_inj['Missing_Key_Players']
    missing_impact_diff = away_inj['Missing_Impact_Pct']  - home_inj['Missing_Impact_Pct']
    
    matchups = raw_matches[
        ((raw_matches['HomeTeam'] == home_team) & (raw_matches['AwayTeam'] == away_team)) |
        ((raw_matches['HomeTeam'] == away_team) & (raw_matches['AwayTeam'] == home_team))
    ]
    
    if len(matchups) == 0:
        h2h_win_rate = 0.5
    else:
        wins = len(matchups[(matchups['HomeTeam'] == home_team) & (matchups['FTR'] == 'H')]) + \
               len(matchups[(matchups['AwayTeam'] == home_team) & (matchups['FTR'] == 'A')])
        h2h_win_rate = wins / len(matchups)

    # 4. Construct the feature array exactly matching training schema
    match_features = pd.DataFrame([[
        home_stats[0], home_stats[1], home_stats[2],   # Home EMA Form
        home_stats[3], home_stats[4], home_stats[5],   # Home Dominance
        away_stats[0], away_stats[1], away_stats[2],   # Away EMA Form
        away_stats[3], away_stats[4], away_stats[5],   # Away Dominance
        home_stats[6], away_stats[6],                  # Expected Offense
        home_rest_days, away_rest_days,
        form_diff, offense_diff, rest_diff,
        h2h_win_rate,
        # ---- REAL injury data (from Transfermarkt scrape) ----
        home_inj['Missing_Key_Players'], away_inj['Missing_Key_Players'],
        home_inj['Missing_Impact_Pct'],  away_inj['Missing_Impact_Pct'],
        missing_key_diff,                missing_impact_diff,
    ]], columns=[
        'Home_EMA_Points', 'Home_EMA_GS', 'Home_EMA_GC',
        'Home_EMA_Shots', 'Home_EMA_ShotsOnTarget', 'Home_EMA_Corners',
        'Away_EMA_Points', 'Away_EMA_GS', 'Away_EMA_GC',
        'Away_EMA_Shots', 'Away_EMA_ShotsOnTarget', 'Away_EMA_Corners',
        'Home_Expected_Offense', 'Away_Expected_Offense',
        'Home_Days_Rest', 'Away_Days_Rest',
        'Form_Diff', 'Offense_Diff', 'Rest_Diff',
        'H2H_Home_Win_Rate',
        'Home_Missing_Key_Players', 'Away_Missing_Key_Players',
        'Home_Missing_Impact_Pct',  'Away_Missing_Impact_Pct',
        'Missing_Key_Diff',          'Missing_Impact_Diff',
    ])

    # 5. Make Prediction
    probabilities = model.predict_proba(match_features)[0]
    prediction    = model.predict(match_features)[0]
    outcomes      = {0: "Away Win", 1: "Draw", 2: "Home Win"}

    print(f"\n==============================================")
    print(f"  MATCH PREDICTION: {home_team} vs {away_team} ")
    print(f"==============================================")
    print(f"Current Home xG+xA Index : {home_stats[6]:.2f}")
    print(f"Current Away xG+xA Index : {away_stats[6]:.2f}")
    print(f"----------------------------------------------")
    print(f"Injury Report:")
    print(f"  {home_team:<20} Missing Key Players: {int(home_inj['Missing_Key_Players'])}  |  Impact: {home_inj['Missing_Impact_Pct']:.1f}%")
    print(f"  {away_team:<20} Missing Key Players: {int(away_inj['Missing_Key_Players'])}  |  Impact: {away_inj['Missing_Impact_Pct']:.1f}%")
    print(f"----------------------------------------------")
    print(f"Win Probabilities:")
    print(f"  [{home_team}] Home Win : {probabilities[2] * 100:.1f}%")
    print(f"  Draw              : {probabilities[1] * 100:.1f}%")
    print(f"  [{away_team}] Away Win : {probabilities[0] * 100:.1f}%")
    print(f"\n  Model Prediction: {outcomes[prediction].upper()} \n")
    print(f"==============================================\n")

if __name__ == "__main__":
    predict_match("Espanol",    "Real Madrid", 5, 9)
    predict_match("Girona",     "Mallorca",    6, 6)
