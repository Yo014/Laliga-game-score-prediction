import pandas as pd
import numpy as np

def calculate_h2h(df):
    """
    Calculates the historical win percentage of the Home Team against the Away Team.
    """
    df = df.sort_values('Date')
    h2h_win_rates = []
    
    for i, row in df.iterrows():
        # Look at all past matches between these exact two teams
        past_matches = df.iloc[:i]
        matchups = past_matches[
            ((past_matches['HomeTeam'] == row['HomeTeam']) & (past_matches['AwayTeam'] == row['AwayTeam'])) |
            ((past_matches['HomeTeam'] == row['AwayTeam']) & (past_matches['AwayTeam'] == row['HomeTeam']))
        ]
        
        if len(matchups) == 0:
            h2h_win_rates.append(0.5) # No history, assume 50/50
        else:
            # How many times did the current Home Team win?
            wins = len(matchups[(matchups['HomeTeam'] == row['HomeTeam']) & (matchups['FTR'] == 'H')]) + \
                   len(matchups[(matchups['AwayTeam'] == row['HomeTeam']) & (matchups['FTR'] == 'A')])
            h2h_win_rates.append(wins / len(matchups))
            
    df['H2H_Home_Win_Rate'] = h2h_win_rates
    return df

def calculate_ema_form(df, span=5):
    """
    Calculates the Exponential Moving Average (EMA) for team stats.
    Using EMA means recent matches have a higher weight in the 'form' calculation.
    """
    # Create a dataframe to track every team's individual match history
    home_stats = df[['Date', 'HomeTeam', 'FTHG', 'FTAG', 'HS', 'HST', 'HC']].rename(
        columns={'HomeTeam': 'Team', 'FTHG': 'GoalsScored', 'FTAG': 'GoalsConceded', 'HS': 'Shots', 'HST': 'ShotsOnTarget', 'HC': 'Corners'}
    )
    home_stats['IsHome'] = 1
    home_stats['Points'] = np.where(home_stats['GoalsScored'] > home_stats['GoalsConceded'], 3, 
                           np.where(home_stats['GoalsScored'] == home_stats['GoalsConceded'], 1, 0))

    away_stats = df[['Date', 'AwayTeam', 'FTAG', 'FTHG', 'AS', 'AST', 'AC']].rename(
        columns={'AwayTeam': 'Team', 'FTAG': 'GoalsScored', 'FTHG': 'GoalsConceded', 'AS': 'Shots', 'AST': 'ShotsOnTarget', 'AC': 'Corners'}
    )
    away_stats['IsHome'] = 0
    away_stats['Points'] = np.where(away_stats['GoalsScored'] > away_stats['GoalsConceded'], 3, 
                           np.where(away_stats['GoalsScored'] == away_stats['GoalsConceded'], 1, 0))

    # Combine and sort chronologically
    team_matches = pd.concat([home_stats, away_stats]).sort_values(['Team', 'Date'])

    # Calculate Goal Difference for each match
    team_matches['GoalDiff'] = team_matches['GoalsScored'] - team_matches['GoalsConceded']
    
    # Calculate Proxy Match xG (True Form Indicator)
    # Formula: (SOT * 0.3) + (Non-SOT Shots * 0.07) + (Corners * 0.05)
    team_matches['Match_xG'] = (pd.to_numeric(team_matches['ShotsOnTarget'], errors='coerce') * 0.3) + \
                               ((pd.to_numeric(team_matches['Shots'], errors='coerce') - pd.to_numeric(team_matches['ShotsOnTarget'], errors='coerce')) * 0.07) + \
                               (pd.to_numeric(team_matches['Corners'], errors='coerce') * 0.05)
    
    # Fill any NaNs in xG calculation with 0 (for games with missing shot data)
    team_matches['Match_xG'] = team_matches['Match_xG'].fillna(0)

    # Calculate EMA. The shift(1) is critical: it prevents data leakage
    team_matches['EMA_Points'] = team_matches.groupby('Team')['Points'].transform(lambda x: x.shift(1).ewm(span=span, adjust=False).mean())
    team_matches['EMA_GoalsScored'] = team_matches.groupby('Team')['GoalsScored'].transform(lambda x: x.shift(1).ewm(span=span, adjust=False).mean())
    team_matches['EMA_GoalsConceded'] = team_matches.groupby('Team')['GoalsConceded'].transform(lambda x: x.shift(1).ewm(span=span, adjust=False).mean())
    team_matches['EMA_GoalDiff'] = team_matches.groupby('Team')['GoalDiff'].transform(lambda x: x.shift(1).ewm(span=span, adjust=False).mean())
    
    # Offensive dominance metrics
    team_matches['EMA_Shots'] = team_matches.groupby('Team')['Shots'].transform(lambda x: pd.to_numeric(x, errors='coerce').shift(1).ewm(span=span, adjust=False).mean())
    team_matches['EMA_ShotsOnTarget'] = team_matches.groupby('Team')['ShotsOnTarget'].transform(lambda x: pd.to_numeric(x, errors='coerce').shift(1).ewm(span=span, adjust=False).mean())
    team_matches['EMA_Corners'] = team_matches.groupby('Team')['Corners'].transform(lambda x: pd.to_numeric(x, errors='coerce').shift(1).ewm(span=span, adjust=False).mean())

    # Defensive dominance metrics (How much they allow)
    team_matches['EMA_ShotsConceded'] = team_matches.groupby('Team')['GoalsConceded'].transform(lambda x: pd.to_numeric(x, errors='coerce').shift(1).ewm(span=span, adjust=False).mean())
    team_matches['EMA_SOTConceded'] = team_matches.groupby('Team')['GoalsConceded'].transform(lambda x: pd.to_numeric(x, errors='coerce').shift(1).ewm(span=span, adjust=False).mean())
    team_matches['EMA_CornersConceded'] = team_matches.groupby('Team')['GoalsConceded'].transform(lambda x: pd.to_numeric(x, errors='coerce').shift(1).ewm(span=span, adjust=False).mean())

    # TRUE FORM METRICS: EMA of Match xG (Created and Conceded)
    team_matches['EMA_xG_Created'] = team_matches.groupby('Team')['Match_xG'].transform(lambda x: x.shift(1).ewm(span=span, adjust=False).mean())
    
    # To get EMA_xG_Conceded, we need to know what xG the opponent created
    # We'll merge this back later, but for now we'll calculate it by looking at the Match_xG allowed
    # Actually, a simpler way in this combined dataframe is to group by match and take the other team's xG
    # But since team_matches is a long-form table of EVERY team performance, we can just use the match link.

    return team_matches[['Date', 'Team', 'EMA_Points', 'EMA_GoalsScored', 'EMA_GoalsConceded', 'EMA_GoalDiff', 
                         'EMA_Shots', 'EMA_ShotsOnTarget', 'EMA_Corners', 
                         'EMA_ShotsConceded', 'EMA_SOTConceded', 'EMA_CornersConceded',
                         'EMA_xG_Created']]

def calculate_referee_stats(df, span=20):
    """
    Calculates historical card and foul averages for each referee.
    Using a longer span (20) to capture their long-term 'personality'.
    """
    df = df.sort_values('Date')
    
    # Clean numeric columns
    for col in ['HY', 'AY', 'HR', 'AR', 'HF', 'AF']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    df['Total_Cards'] = df['HY'] + df['AY'] + df['HR'] + df['AR']
    df['Total_Fouls'] = df['HF'] + df['AF']
    
    # Calculate rolling averages per referee
    # Shift(1) to avoid leakage
    df['Ref_Avg_Cards'] = df.groupby('Referee')['Total_Cards'].transform(lambda x: x.shift(1).ewm(span=span, adjust=False).mean())
    df['Ref_Avg_Fouls'] = df.groupby('Referee')['Total_Fouls'].transform(lambda x: x.shift(1).ewm(span=span, adjust=False).mean())
    
    # Fill first-time referee stats with global averages
    df['Ref_Avg_Cards'] = df['Ref_Avg_Cards'].fillna(df['Total_Cards'].mean())
    df['Ref_Avg_Fouls'] = df['Ref_Avg_Fouls'].fillna(df['Total_Fouls'].mean())
    
    return df[['Date', 'HomeTeam', 'AwayTeam', 'Ref_Avg_Cards', 'Ref_Avg_Fouls']]
def get_rest_days(df):
    """
    Calculates the number of rest days a team has had since their last match.
    Caps rest at 14 days (representing a full recovery / international break).
    """
    home_dates = df[['Date', 'HomeTeam']].rename(columns={'HomeTeam': 'Team'})
    away_dates = df[['Date', 'AwayTeam']].rename(columns={'AwayTeam': 'Team'})
    
    all_dates = pd.concat([home_dates, away_dates]).sort_values(['Team', 'Date']).reset_index(drop=True)
    
    # Calculate the difference in days using shift(1) to look at the previous game
    all_dates['Prev_Date'] = all_dates.groupby('Team')['Date'].shift(1)
    all_dates['Days_Rest'] = (all_dates['Date'] - all_dates['Prev_Date']).dt.days
    
    # Cap the rest days at 14 
    all_dates['Days_Rest'] = all_dates['Days_Rest'].fillna(14) 
    all_dates['Days_Rest'] = np.where(all_dates['Days_Rest'] > 14, 14, all_dates['Days_Rest'])
    
    return all_dates.drop_duplicates(subset=['Team', 'Date'])
def build_advanced_strength_index():
    """
    Aggregates Expected Goals (xG) and Expected Assists (xA) from your master datasets
    to create a true 'Expected Offensive Index' for each team per season.
    """
    # Load your updated master files
    scorers = pd.read_csv('/Users/santomukiza/Desktop/Github/LaligaPrediction/Laliga-game-score-prediction/Processed_Scorers.csv')
    assists = pd.read_csv('/Users/santomukiza/Desktop/Github/LaligaPrediction/Laliga-game-score-prediction/Processed_Assists.csv')
    
    # Calculate total Expected Goals (xG) per team per season
    team_xg = scorers.groupby(['Season', 'Team'])['xG'].sum().reset_index(name='Total_xG_Power')
    
    # Calculate total Expected Assists (xA) per team per season
    team_xa = assists.groupby(['Season', 'Team'])['xA'].sum().reset_index(name='Total_xA_Power')
    
    # Merge them together into an Ultimate Strength Index
    strength_index = pd.merge(team_xg, team_xa, on=['Season', 'Team'], how='outer').fillna(0)
    
    # The new index combines the team's underlying ability to CREATE and FINISH chances
    strength_index['Expected_Offensive_Index'] = strength_index['Total_xG_Power'] + strength_index['Total_xA_Power']
    
    return strength_index

def main():
    print("--- Starting Advanced Feature Engineering ---")
    
    # 1. Load Match Data
    matches = pd.read_csv('/Users/santomukiza/Desktop/Github/LaligaPrediction/Laliga-game-score-prediction/Processed_Matches.csv')
    matches['Date'] = pd.to_datetime(matches['Date'])
    
    print("Calculating Head-to-Head history...")
    matches = calculate_h2h(matches)
    
    # Map the match date back to its specific season
    matches['Season'] = matches['Date'].apply(
        lambda x: f"{str(x.year)[-2:]}-{str(x.year+1)[-2:]}" if x.month >= 8 else f"{str(x.year-1)[-2:]}-{str(x.year)[-2:]}"
    )

    # 2. Calculate Rolling Form (EMA)
    print("Calculating exponential moving averages for team form...")
    form_df = calculate_ema_form(matches, span=5)

    matches = pd.merge(matches, form_df, left_on=['Date', 'HomeTeam'], right_on=['Date', 'Team'], how='left')
    matches = matches.rename(columns={
        'EMA_Points': 'Home_EMA_Points', 'EMA_GoalsScored': 'Home_EMA_GS', 'EMA_GoalsConceded': 'Home_EMA_GC',
        'EMA_GoalDiff': 'Home_EMA_GoalDiff', 'EMA_Shots': 'Home_EMA_Shots', 'EMA_ShotsOnTarget': 'Home_EMA_ShotsOnTarget', 'EMA_Corners': 'Home_EMA_Corners',
        'EMA_ShotsConceded': 'Home_EMA_ShotsConceded', 'EMA_SOTConceded': 'Home_EMA_SOTConceded', 'EMA_CornersConceded': 'Home_EMA_CornersConceded',
        'EMA_xG_Created': 'Home_EMA_xG_Created'
    }).drop('Team', axis=1)

    matches = pd.merge(matches, form_df, left_on=['Date', 'AwayTeam'], right_on=['Date', 'Team'], how='left')
    matches = matches.rename(columns={
        'EMA_Points': 'Away_EMA_Points', 'EMA_GoalsScored': 'Away_EMA_GS', 'EMA_GoalsConceded': 'Away_EMA_GC',
        'EMA_GoalDiff': 'Away_EMA_GoalDiff', 'EMA_Shots': 'Away_EMA_Shots', 'EMA_ShotsOnTarget': 'Away_EMA_ShotsOnTarget', 'EMA_Corners': 'Away_EMA_Corners',
        'EMA_ShotsConceded': 'Away_EMA_ShotsConceded', 'EMA_SOTConceded': 'Away_EMA_SOTConceded', 'EMA_CornersConceded': 'Away_EMA_CornersConceded',
        'EMA_xG_Created': 'Away_EMA_xG_Created'
    }).drop('Team', axis=1)

    # Calculate xG Conceded (which is just the opponent's xG Created)
    matches['Home_EMA_xG_Conceded'] = matches['Away_EMA_xG_Created']
    matches['Away_EMA_xG_Conceded'] = matches['Home_EMA_xG_Created']

    # 3. Add Rest Days Context
    print("Calculating player fatigue and rest days...")
    rest_df = get_rest_days(matches)

    matches = pd.merge(matches, rest_df[['Date', 'Team', 'Days_Rest']], left_on=['Date', 'HomeTeam'], right_on=['Date', 'Team'], how='left')
    matches = matches.rename(columns={'Days_Rest': 'Home_Days_Rest'}).drop('Team', axis=1)

    matches = pd.merge(matches, rest_df[['Date', 'Team', 'Days_Rest']], left_on=['Date', 'AwayTeam'], right_on=['Date', 'Team'], how='left')
    matches = matches.rename(columns={'Days_Rest': 'Away_Days_Rest'}).drop('Team', axis=1)

    # 3. Add Advanced Expected Strength Index
    print("Integrating Expected Goals (xG) and Expected Assists (xA)...")
    strength_idx = build_advanced_strength_index()
    
    matches = pd.merge(matches, strength_idx[['Season', 'Team', 'Expected_Offensive_Index']], left_on=['Season', 'HomeTeam'], right_on=['Season', 'Team'], how='left')
    matches = matches.rename(columns={'Expected_Offensive_Index': 'Home_Expected_Offense'}).drop('Team', axis=1)
    
    matches = pd.merge(matches, strength_idx[['Season', 'Team', 'Expected_Offensive_Index']], left_on=['Season', 'AwayTeam'], right_on=['Season', 'Team'], how='left')
    matches = matches.rename(columns={'Expected_Offensive_Index': 'Away_Expected_Offense'}).drop('Team', axis=1)

    # Fill in zeroes for promoted teams missing from the top scorers list
    matches['Home_Expected_Offense'] = matches['Home_Expected_Offense'].fillna(0)
    matches['Away_Expected_Offense'] = matches['Away_Expected_Offense'].fillna(0)

    # 5. Add Squad Health Data (only for the current 25-26 season)
    print("Integrating squad health / injury data...")
    squad_health_path = '/Users/santomukiza/Desktop/Github/LaligaPrediction/Laliga-game-score-prediction/current_squad_health.csv'
    squad_health = pd.read_csv(squad_health_path)

    # Tag squad health with the current season so it only merges with 25-26 matches
    squad_health['Season'] = '25-26'

    # Merge for Home team (by Season + Team)
    matches = pd.merge(matches, squad_health[['Season', 'Team', 'Missing_Key_Players', 'Missing_Impact_Pct', 'Missing_Goals_Pct']],
                       left_on=['Season', 'HomeTeam'], right_on=['Season', 'Team'], how='left')
    matches = matches.rename(columns={
        'Missing_Key_Players': 'Home_Missing_Key_Players',
        'Missing_Impact_Pct': 'Home_Missing_Impact_Pct',
        'Missing_Goals_Pct': 'Home_Missing_Goals_Pct'
    }).drop('Team', axis=1)

    # Merge for Away team (by Season + Team)
    matches = pd.merge(matches, squad_health[['Season', 'Team', 'Missing_Key_Players', 'Missing_Impact_Pct', 'Missing_Goals_Pct']],
                       left_on=['Season', 'AwayTeam'], right_on=['Season', 'Team'], how='left')
    matches = matches.rename(columns={
        'Missing_Key_Players': 'Away_Missing_Key_Players',
        'Missing_Impact_Pct': 'Away_Missing_Impact_Pct',
        'Missing_Goals_Pct': 'Away_Missing_Goals_Pct'
    }).drop('Team', axis=1)

    # Fill 0 for teams/seasons without squad health data (all historical seasons)
    for col in ['Home_Missing_Key_Players', 'Away_Missing_Key_Players',
                'Home_Missing_Impact_Pct', 'Away_Missing_Impact_Pct',
                'Home_Missing_Goals_Pct', 'Away_Missing_Goals_Pct']:
        matches[col] = matches[col].fillna(0)

    # 6. Add Betting Odds Implied Probabilities
    # Prob = 1 / Odds. We normalize these to remove the bookmaker's margin (overround).
    matches['B365H'] = pd.to_numeric(matches['B365H'], errors='coerce').fillna(2.0)
    matches['B365D'] = pd.to_numeric(matches['B365D'], errors='coerce').fillna(3.0)
    matches['B365A'] = pd.to_numeric(matches['B365A'], errors='coerce').fillna(3.0)
    
    # Calculate raw probabilities
    matches['Prob_H_Raw'] = 1 / matches['B365H']
    matches['Prob_D_Raw'] = 1 / matches['B365D']
    matches['Prob_A_Raw'] = 1 / matches['B365A']
    
    # Normalize (Divide by the sum of probabilities to make them sum to 1.0)
    overround = matches['Prob_H_Raw'] + matches['Prob_D_Raw'] + matches['Prob_A_Raw']
    matches['Market_Prob_H'] = matches['Prob_H_Raw'] / overround
    matches['Market_Prob_D'] = matches['Prob_D_Raw'] / overround
    matches['Market_Prob_A'] = matches['Prob_A_Raw'] / overround

    # 7. Add Referee Stats
    print("Calculating referee historical stats (cards/fouls)...")
    ref_stats = calculate_referee_stats(matches)
    matches = pd.merge(matches, ref_stats, on=['Date', 'HomeTeam', 'AwayTeam'], how='left')

    # Calculate Explicit Differentials
    matches['Form_Diff'] = matches['Home_EMA_Points'] - matches['Away_EMA_Points']
    matches['Offense_Diff'] = matches['Home_Expected_Offense'] - matches['Away_Expected_Offense']
    matches['Rest_Diff'] = matches['Home_Days_Rest'] - matches['Away_Days_Rest']
    matches['Missing_Key_Diff'] = matches['Home_Missing_Key_Players'] - matches['Away_Missing_Key_Players']
    matches['Missing_Impact_Diff'] = matches['Home_Missing_Impact_Pct'] - matches['Away_Missing_Impact_Pct']
    matches['xG_Form_Diff'] = matches['Home_EMA_xG_Created'] - matches['Away_EMA_xG_Created']

    # 4. Target Variable Setup
    # 0 = Away Win, 1 = Draw, 2 = Home Win
    matches['Target'] = np.where(matches['FTR'] == 'H', 2, np.where(matches['FTR'] == 'D', 1, 0))

    final_dataset = matches.dropna().reset_index(drop=True)
    features_to_keep = [
        'Date', 'HomeTeam', 'AwayTeam', 'FTR', 'Target', 'Referee',
        'B365H', 'B365D', 'B365A',
        'Market_Prob_H', 'Market_Prob_D', 'Market_Prob_A',
        'Ref_Avg_Cards', 'Ref_Avg_Fouls',
        'Home_EMA_Points', 'Home_EMA_GS', 'Home_EMA_GC', 'Home_EMA_GoalDiff',
        'Home_EMA_xG_Created', 'Home_EMA_xG_Conceded',
        'Away_EMA_xG_Created', 'Away_EMA_xG_Conceded',
        'xG_Form_Diff',
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
    ]
    
    # Fill missing values for betting odds and referee before saving
    matches['B365H'] = pd.to_numeric(matches['B365H'], errors='coerce').fillna(1.0)
    matches['B365D'] = pd.to_numeric(matches['B365D'], errors='coerce').fillna(1.0)
    matches['B365A'] = pd.to_numeric(matches['B365A'], errors='coerce').fillna(1.0)
    matches['Referee'] = matches['Referee'].fillna('Unknown')

    final_dataset = matches[features_to_keep].dropna().reset_index(drop=True)
    # Use your absolute path so you know exactly where it saves
    save_path = '/Users/santomukiza/Desktop/Github/LaligaPrediction/Laliga-game-score-prediction/ml_ready_data.csv'
    final_dataset.to_csv(save_path, index=False)
    
    print(f"Feature engineering complete! Final dataset shape: {final_dataset.shape}")
    print(f"Saved as {save_path}")

if __name__ == "__main__":
    main()
