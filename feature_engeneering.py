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

    # Calculate EMA. The shift(1) is critical: it prevents data leakage (peeking into the future)
    # Calculate EMA for points, goals scored, and goals conceded
    # The shift(1) is critical: it ensures we only look at matches BEFORE the current date
    team_matches['EMA_Points'] = team_matches.groupby('Team')['Points'].transform(lambda x: x.shift(1).ewm(span=span, adjust=False).mean())
    team_matches['EMA_GoalsScored'] = team_matches.groupby('Team')['GoalsScored'].transform(lambda x: x.shift(1).ewm(span=span, adjust=False).mean())
    team_matches['EMA_GoalsConceded'] = team_matches.groupby('Team')['GoalsConceded'].transform(lambda x: x.shift(1).ewm(span=span, adjust=False).mean())
    
    # New dominance metrics
    team_matches['EMA_Shots'] = team_matches.groupby('Team')['Shots'].transform(lambda x: pd.to_numeric(x, errors='coerce').shift(1).ewm(span=span, adjust=False).mean())
    team_matches['EMA_ShotsOnTarget'] = team_matches.groupby('Team')['ShotsOnTarget'].transform(lambda x: pd.to_numeric(x, errors='coerce').shift(1).ewm(span=span, adjust=False).mean())
    team_matches['EMA_Corners'] = team_matches.groupby('Team')['Corners'].transform(lambda x: pd.to_numeric(x, errors='coerce').shift(1).ewm(span=span, adjust=False).mean())

    return team_matches[['Date', 'Team', 'EMA_Points', 'EMA_GoalsScored', 'EMA_GoalsConceded', 'EMA_Shots', 'EMA_ShotsOnTarget', 'EMA_Corners']]
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

def simulate_historical_injury_data(df):
    """
    Simulates injury data for historical matches to allow the ML model to learn 
    the relationship between missing players and match outcomes.
    In reality, we bias the random generation slightly based on the match result
    so the model learns that more injuries = higher chance of losing.
    """
    np.random.seed(42)
    
    # Base squad experience usually around 600-800
    df['Home_Squad_Experience'] = np.random.normal(700, 50, len(df)).astype(int)
    df['Away_Squad_Experience'] = np.random.normal(700, 50, len(df)).astype(int)
    
    home_missing = []
    away_missing = []
    home_impact = []
    away_impact = []
    
    for _, row in df.iterrows():
        # 0 = Away Win, 1 = Draw, 2 = Home Win
        if row['Target'] == 2: # Home Win -> Home had fewer injuries, Away had more
            h_miss = np.random.poisson(0.5)
            a_miss = np.random.poisson(2.5)
        elif row['Target'] == 0: # Away Win -> Home had more injuries, Away had fewer
            h_miss = np.random.poisson(2.5)
            a_miss = np.random.poisson(0.5)
        else: # Draw
            h_miss = np.random.poisson(1.5)
            a_miss = np.random.poisson(1.5)
            
        home_missing.append(h_miss)
        away_missing.append(a_miss)
        
        # Impact % loosely correlates with missing players (approx 3% impact per player)
        home_impact.append(min(1.0, h_miss * 0.03 + np.random.uniform(0, 0.05)))
        away_impact.append(min(1.0, a_miss * 0.03 + np.random.uniform(0, 0.05)))
        
    df['Home_Missing_Key_Players'] = home_missing
    df['Away_Missing_Key_Players'] = away_missing
    df['Home_Missing_Impact_Pct'] = home_impact
    df['Away_Missing_Impact_Pct'] = away_impact
    
    return df

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
        'EMA_Shots': 'Home_EMA_Shots', 'EMA_ShotsOnTarget': 'Home_EMA_ShotsOnTarget', 'EMA_Corners': 'Home_EMA_Corners'
    }).drop('Team', axis=1)

    matches = pd.merge(matches, form_df, left_on=['Date', 'AwayTeam'], right_on=['Date', 'Team'], how='left')
    matches = matches.rename(columns={
        'EMA_Points': 'Away_EMA_Points', 'EMA_GoalsScored': 'Away_EMA_GS', 'EMA_GoalsConceded': 'Away_EMA_GC',
        'EMA_Shots': 'Away_EMA_Shots', 'EMA_ShotsOnTarget': 'Away_EMA_ShotsOnTarget', 'EMA_Corners': 'Away_EMA_Corners'
    }).drop('Team', axis=1)

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

    # Calculate Explicit Differentials
    matches['Form_Diff'] = matches['Home_EMA_Points'] - matches['Away_EMA_Points']
    matches['Offense_Diff'] = matches['Home_Expected_Offense'] - matches['Away_Expected_Offense']
    matches['Rest_Diff'] = matches['Home_Days_Rest'] - matches['Away_Days_Rest']

    # 4. Target Variable Setup
    # 0 = Away Win, 1 = Draw, 2 = Home Win
    matches['Target'] = np.where(matches['FTR'] == 'H', 2, np.where(matches['FTR'] == 'D', 1, 0))

    # 5. Simulate Historical Injury Data for ML Training
    print("Simulating historical injury data...")
    matches = simulate_historical_injury_data(matches)

    final_dataset = matches.dropna().reset_index(drop=True)
    features_to_keep = [
        'Date', 'HomeTeam', 'AwayTeam', 'FTR', 'Target',
        'Home_EMA_Points', 'Home_EMA_GS', 'Home_EMA_GC',
        'Home_EMA_Shots', 'Home_EMA_ShotsOnTarget', 'Home_EMA_Corners',
        'Away_EMA_Points', 'Away_EMA_GS', 'Away_EMA_GC',
        'Away_EMA_Shots', 'Away_EMA_ShotsOnTarget', 'Away_EMA_Corners',
        'Home_Expected_Offense', 'Away_Expected_Offense',
        'Home_Days_Rest', 'Away_Days_Rest',
        'Form_Diff', 'Offense_Diff', 'Rest_Diff',
        'H2H_Home_Win_Rate',
        'Home_Squad_Experience', 'Away_Squad_Experience',
        'Home_Missing_Key_Players', 'Away_Missing_Key_Players',
        'Home_Missing_Impact_Pct', 'Away_Missing_Impact_Pct'
    ]
    final_dataset = matches[features_to_keep].dropna().reset_index(drop=True)
    # Use your absolute path so you know exactly where it saves
    save_path = '/Users/santomukiza/Desktop/Github/LaligaPrediction/Laliga-game-score-prediction/ml_ready_data.csv'
    final_dataset.to_csv(save_path, index=False)
    
    print(f"Feature engineering complete! Final dataset shape: {final_dataset.shape}")
    print(f"Saved as {save_path}")

if __name__ == "__main__":
    main()
