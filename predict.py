import pandas as pd
import joblib


def get_latest_team_stats(team_name, is_home, df):
    """
    Scans the dataset to find the most recent form and Expected Offensive Index 
    for the requested team.
    """
    # Filter matches where the team played
    team_history = df[(df['HomeTeam'] == team_name) | (df['AwayTeam'] == team_name)].copy()
    
    if team_history.empty:
        print(f"Error: Could not find team '{team_name}'. Check spelling.")
        return None
        
    # Get their absolute most recent game
    latest_match = team_history.iloc[-1]
    
    # Extract the stats depending on if they were home or away in that last match
    if latest_match['HomeTeam'] == team_name:
        stats = [
            latest_match['Home_EMA_Points'], latest_match['Home_EMA_GS'], latest_match['Home_EMA_GC'], latest_match['Home_EMA_GoalDiff'],
            latest_match['Home_EMA_Shots'], latest_match['Home_EMA_ShotsOnTarget'], latest_match['Home_EMA_Corners'],
            latest_match['Home_EMA_ShotsConceded'], latest_match['Home_EMA_SOTConceded'], latest_match['Home_EMA_CornersConceded'],
            latest_match['Home_Expected_Offense'],
            latest_match['Home_EMA_xG_Created'], latest_match['Home_EMA_xG_Conceded'],
            latest_match['Home_EMA_Field_Tilt'], latest_match['Home_PPDA']
        ]
    else:
        stats = [
            latest_match['Away_EMA_Points'], latest_match['Away_EMA_GS'], latest_match['Away_EMA_GC'], latest_match['Away_EMA_GoalDiff'],
            latest_match['Away_EMA_Shots'], latest_match['Away_EMA_ShotsOnTarget'], latest_match['Away_EMA_Corners'],
            latest_match['Away_EMA_ShotsConceded'], latest_match['Away_EMA_SOTConceded'], latest_match['Away_EMA_CornersConceded'],
            latest_match['Away_Expected_Offense'],
            latest_match['Away_EMA_xG_Created'], latest_match['Away_EMA_xG_Conceded'],
            latest_match['Away_EMA_Field_Tilt'], latest_match['Away_PPDA']
        ]

    return stats

def predict_match(home_team, away_team, home_rest_days, away_rest_days, b365h=2.0, b365d=3.0, b365a=3.0, referee_name="Unknown"):
    print(f"\nAnalyzing Matchup: {home_team} (Home) vs {away_team} (Away)...")
    
    # 1. Load Data & Advanced Model
    data_path = '/Users/santomukiza/Desktop/Github/LaligaPrediction/Laliga-game-score-prediction/ml_ready_data.csv'
    model_path = '/Users/santomukiza/Desktop/Github/LaligaPrediction/Laliga-game-score-prediction/laliga_rf_model.pkl'
    squad_health_path = '/Users/santomukiza/Desktop/Github/LaligaPrediction/Laliga-game-score-prediction/current_squad_health.csv'
    
    try:
        raw_matches = pd.read_csv('/Users/santomukiza/Desktop/Github/LaligaPrediction/Laliga-game-score-prediction/Processed_Matches.csv')
        df = pd.read_csv(data_path)
        model = joblib.load(model_path)
        squad_health = pd.read_csv(squad_health_path)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    # Attach team names to the engineered dataset to search for them
    df['HomeTeam'] = raw_matches.iloc[df.index]['HomeTeam']
    df['AwayTeam'] = raw_matches.iloc[df.index]['AwayTeam']

    # 2. Fetch current form
    home_stats = get_latest_team_stats(home_team, True, df)
    away_stats = get_latest_team_stats(away_team, False, df)
    
    if not home_stats or not away_stats:
        return

    # 3. Look up squad health for both teams
    home_health = squad_health[squad_health['Team'] == home_team]
    away_health = squad_health[squad_health['Team'] == away_team]

    home_missing_key = home_health['Missing_Key_Players'].values[0] if len(home_health) > 0 else 0
    home_missing_impact = home_health['Missing_Impact_Pct'].values[0] if len(home_health) > 0 else 0
    home_missing_goals = home_health['Missing_Goals_Pct'].values[0] if len(home_health) > 0 else 0
    away_missing_key = away_health['Missing_Key_Players'].values[0] if len(away_health) > 0 else 0
    away_missing_impact = away_health['Missing_Impact_Pct'].values[0] if len(away_health) > 0 else 0
    away_missing_goals = away_health['Missing_Goals_Pct'].values[0] if len(away_health) > 0 else 0

    # 4. Calculate Differentials and H2H
    form_diff = home_stats[0] - away_stats[0]
    offense_diff = home_stats[10] - away_stats[10]
    rest_diff = home_rest_days - away_rest_days
    missing_key_diff = home_missing_key - away_missing_key
    missing_impact_diff = home_missing_impact - away_missing_impact
    
    # 4.1. Calculate Implied Market Probabilities
    prob_h_raw = 1 / b365h
    prob_d_raw = 1 / b365d
    prob_a_raw = 1 / b365a
    overround = prob_h_raw + prob_d_raw + prob_a_raw
    market_prob_h = prob_h_raw / overround
    market_prob_d = prob_d_raw / overround
    market_prob_a = prob_a_raw / overround

    # 4.2. Get Referee Stats
    ref_history = df[df['Referee'] == referee_name]
    if not ref_history.empty:
        ref_avg_cards = ref_history['Ref_Avg_Cards'].iloc[-1]
        ref_avg_fouls = ref_history['Ref_Avg_Fouls'].iloc[-1]
    else:
        # Default to global averages if referee is new
        ref_avg_cards = df['Ref_Avg_Cards'].mean()
        ref_avg_fouls = df['Ref_Avg_Fouls'].mean()

    # 4.3. Get Categorical Codes
    # We recreate the factorize mapping from the training data
    home_code = pd.Categorical(df['HomeTeam']).categories.get_loc(home_team) if home_team in df['HomeTeam'].values else -1
    away_code = pd.Categorical(df['AwayTeam']).categories.get_loc(away_team) if away_team in df['AwayTeam'].values else -1
    referee_code = pd.Categorical(df['Referee']).categories.get_loc(referee_name) if referee_name in df['Referee'].values else -1
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

    xg_form_diff = home_stats[11] - away_stats[11]
    tilt_diff = home_stats[13] - away_stats[13]    # EMA_Field_Tilt
    ppda_diff = home_stats[14] - away_stats[14]    # PPDA

    # 5. Construct the feature array exactly how the model was trained
    match_features = pd.DataFrame([[
        home_code, away_code, referee_code,
        b365h, b365d, b365a,
        market_prob_h, market_prob_d, market_prob_a,
        ref_avg_cards, ref_avg_fouls,
        home_stats[11], home_stats[12],              # Home xG Form
        away_stats[11], away_stats[12],              # Away xG Form
        xg_form_diff,
        home_stats[13], away_stats[13], tilt_diff,   # Field Tilt
        home_stats[14], away_stats[14], ppda_diff,   # PPDA
        home_stats[0], home_stats[1], home_stats[2], home_stats[3],  # Home Form
        home_stats[4], home_stats[5], home_stats[6],  # Home Dominance
        home_stats[7], home_stats[8], home_stats[9],  # Home Defense
        away_stats[0], away_stats[1], away_stats[2], away_stats[3],  # Away Form
        away_stats[4], away_stats[5], away_stats[6],  # Away Dominance
        away_stats[7], away_stats[8], away_stats[9],  # Away Defense
        home_stats[10], away_stats[10],                # Advanced Expected Offense Indices
        home_rest_days, away_rest_days,
        home_missing_key, away_missing_key,          # Squad Health
        home_missing_impact, away_missing_impact,
        home_missing_goals, away_missing_goals,
        form_diff, offense_diff, rest_diff,          # Explicit Differentials
        missing_key_diff, missing_impact_diff,
        h2h_win_rate                                 # H2H bias
    ]], columns=[
        'Home_Code', 'Away_Code', 'Referee_Code',
        'B365H', 'B365D', 'B365A',
        'Market_Prob_H', 'Market_Prob_D', 'Market_Prob_A',
        'Ref_Avg_Cards', 'Ref_Avg_Fouls',
        'Home_EMA_xG_Created', 'Home_EMA_xG_Conceded',
        'Away_EMA_xG_Created', 'Away_EMA_xG_Conceded',
        'xG_Form_Diff',
        'Home_EMA_Field_Tilt', 'Away_EMA_Field_Tilt', 'Tilt_Diff',
        'Home_PPDA', 'Away_PPDA', 'PPDA_Diff',
        'Home_EMA_Points', 'Home_EMA_GS', 'Home_EMA_GC', 'Home_EMA_GoalDiff',
        'Home_EMA_Shots', 'Home_EMA_ShotsOnTarget', 'Home_EMA_Corners',
        'Home_EMA_ShotsConceded', 'Home_EMA_SOTConceded', 'Home_EMA_CornersConceded',
        'Away_EMA_Points', 'Away_EMA_GS', 'Away_EMA_GC', 'Away_EMA_GoalDiff',
        'Away_EMA_Shots', 'Away_EMA_ShotsOnTarget', 'Away_EMA_Corners',
        'Away_EMA_ShotsConceded', 'Away_EMA_SOTConceded', 'Away_EMA_CornersConceded',
        'Home_Expected_Offense', 'Away_Expected_Offense',
        'Home_Days_Rest', 'Away_Days_Rest',
        'Home_Missing_Key_Players', 'Away_Missing_Key_Players',
        'Home_Missing_Impact_Pct', 'Away_Missing_Impact_Pct',
        'Home_Missing_Goals_Pct', 'Away_Missing_Goals_Pct',
        'Form_Diff', 'Offense_Diff', 'Rest_Diff',
        'Missing_Key_Diff', 'Missing_Impact_Diff',
        'H2H_Home_Win_Rate'
    ])

    # 6. Make Prediction
    probabilities = model.predict_proba(match_features)[0]
    prediction = model.predict(match_features)[0]

    outcomes = {0: "Away Win", 1: "Draw", 2: "Home Win"}

    print(f"\n==============================================")
    print(f"  MATCH PREDICTION: {home_team} vs {away_team} ")
    print(f"==============================================")
    print(f"Current Home xG+xA Index : {home_stats[10]:.2f}")
    print(f"Current Away xG+xA Index : {away_stats[10]:.2f}")
    print(f"----------------------------------------------")
    print(f"Squad Health:")
    print(f"  [{home_team}] Missing {int(home_missing_key)} key players ({home_missing_impact:.1f}% playing time, {home_missing_goals:.1f}% goals)")
    print(f"  [{away_team}] Missing {int(away_missing_key)} key players ({away_missing_impact:.1f}% playing time, {away_missing_goals:.1f}% goals)")
    print(f"----------------------------------------------")
    print(f"Win Probabilities:")
    print(f"[{home_team}] Home Win : {probabilities[2] * 100:.1f}%")
    print(f"Draw              : {probabilities[1] * 100:.1f}%")
    print(f"[{away_team}] Away Win : {probabilities[0] * 100:.1f}%")
    print(f"\n Model Prediction: {outcomes[prediction].upper()} \n")
    print(f"==============================================\n")

if __name__ == "__main__":
    # Test matchups with Betting Odds and Referees!
    # Format: home, away, home_rest, away_rest, b365h, b365d, b365a, referee
    predict_match("Girona", "Mallorca", 6, 11, 2.00, 3.50, 3.75, "Francisco Hernandez")
    predict_match("Villarreal", "Levante", 6, 5, 1.72, 4.35, 4.76, "Mario Melero")
