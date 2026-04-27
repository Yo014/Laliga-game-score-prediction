import pandas as pd
import numpy as np  

def calculate_ema_form(df, span=5):
    """
    Calculates the Exponential Moving Average (EMA) for team stats.
    Using EMA means recent matches have a higher weight in the 'form' calculation.
    """
    # Create a dataframe to track every team's individual match history
    home_stats = df[['Date', 'HomeTeam', 'FTHG', 'FTAG']].rename(
        columns={'HomeTeam': 'Team', 'FTHG': 'GoalsScored', 'FTAG': 'GoalsConceded'}
    )
    home_stats['IsHome'] = 1
    home_stats['Points'] = np.where(home_stats['GoalsScored'] > home_stats['GoalsConceded'], 3, 
                           np.where(home_stats['GoalsScored'] == home_stats['GoalsConceded'], 1, 0))

    away_stats = df[['Date', 'AwayTeam', 'FTAG', 'FTHG']].rename(
        columns={'AwayTeam': 'Team', 'FTAG': 'GoalsScored', 'FTHG': 'GoalsConceded'}
    )
    away_stats['IsHome'] = 0
    away_stats['Points'] = np.where(away_stats['GoalsScored'] > away_stats['GoalsConceded'], 3, 
                           np.where(away_stats['GoalsScored'] == away_stats['GoalsConceded'], 1, 0))

    # Combine and sort chronologically
    team_matches = pd.concat([home_stats, away_stats]).sort_values(['Team', 'Date'])

    # Calculate EMA for points, goals scored, and goals conceded
    # The shift(1) is critical: it ensures we only look at matches BEFORE the current date
    team_matches['EMA_Points'] = team_matches.groupby('Team')['Points'].transform(lambda x: x.shift(1).ewm(span=span, adjust=False).mean())
    team_matches['EMA_GoalsScored'] = team_matches.groupby('Team')['GoalsScored'].transform(lambda x: x.shift(1).ewm(span=span, adjust=False).mean())
    team_matches['EMA_GoalsConceded'] = team_matches.groupby('Team')['GoalsConceded'].transform(lambda x: x.shift(1).ewm(span=span, adjust=False).mean())

    return team_matches[['Date', 'Team', 'EMA_Points', 'EMA_GoalsScored', 'EMA_GoalsConceded']]

def build_team_strength_index():
    """
    Aggregates the Coefficients from your scraped top scorers and assisters 
    to create an 'Offensive Power' rating for each team per season.
    """
    scorers = pd.read_csv('/Users/santomukiza/Desktop/Github/LaligaPrediction/Laliga-game-score-prediction/LaligaScorers/Processed Scorers.csv')
    assists = pd.read_csv('/Users/santomukiza/Desktop/Github/LaligaPrediction/Laliga-game-score-prediction/LaligaAssist/Processed Assists.csv')
    
    # Sum the coefficients of top scorers for each team per season
    team_scoring_power = scorers.groupby(['Season', 'Team'])['Coefficient'].sum().reset_index(name='Scoring_Power')
    team_assist_power = assists.groupby(['Season', 'Team'])['Coefficient'].sum().reset_index(name='Assist_Power')
    
    # Merge and create total offensive index
    strength_index = pd.merge(team_scoring_power, team_assist_power, on=['Season', 'Team'], how='outer').fillna(0)
    strength_index['Offensive_Index'] = strength_index['Scoring_Power'] + strength_index['Assist_Power']
    
    return strength_index

def main():
    print("--- Starting Feature Engineering ---")
    
    # 1. Load Data
    matches = pd.read_csv('/Users/santomukiza/Desktop/Github/LaligaPrediction/Laliga-game-score-prediction/LaligaSeasons/Processed Matches.csv')
    matches['Date'] = pd.to_datetime(matches['Date'])
    
    # Determine the season based on the month (August to July)
    matches['Season'] = matches['Date'].apply(
        lambda x: f"{str(x.year)[-2:]}-{str(x.year+1)[-2:]}" if x.month >= 8 else f"{str(x.year-1)[-2:]}-{str(x.year)[-2:]}"
    )

    # 2. Calculate Rolling Form (EMA)
    print("Calculating exponential moving averages for team form...")
    form_df = calculate_ema_form(matches, span=5)

    # Merge Home Team Form
    matches = pd.merge(matches, form_df, left_on=['Date', 'HomeTeam'], right_on=['Date', 'Team'], how='left')
    matches = matches.rename(columns={'EMA_Points': 'Home_EMA_Points', 'EMA_GoalsScored': 'Home_EMA_GS', 'EMA_GoalsConceded': 'Home_EMA_GC'})
    matches = matches.drop('Team', axis=1)

    # Merge Away Team Form
    matches = pd.merge(matches, form_df, left_on=['Date', 'AwayTeam'], right_on=['Date', 'Team'], how='left')
    matches = matches.rename(columns={'EMA_Points': 'Away_EMA_Points', 'EMA_GoalsScored': 'Away_EMA_GS', 'EMA_GoalsConceded': 'Away_EMA_GC'})
    matches = matches.drop('Team', axis=1)

    # 3. Add Team Strength Index
    print("Integrating player coefficient data...")
    strength_idx = build_team_strength_index()
    
    # Merge Home Team Strength
    matches = pd.merge(matches, strength_idx[['Season', 'Team', 'Offensive_Index']], left_on=['Season', 'HomeTeam'], right_on=['Season', 'Team'], how='left')
    matches = matches.rename(columns={'Offensive_Index': 'Home_Offensive_Index'}).drop('Team', axis=1)
    
    # Merge Away Team Strength
    matches = pd.merge(matches, strength_idx[['Season', 'Team', 'Offensive_Index']], left_on=['Season', 'AwayTeam'], right_on=['Season', 'Team'], how='left')
    matches = matches.rename(columns={'Offensive_Index': 'Away_Offensive_Index'}).drop('Team', axis=1)

    # Fill missing offensive indices with 0 (e.g., promoted teams with no top players from the previous year)
    matches['Home_Offensive_Index'] = matches['Home_Offensive_Index'].fillna(0)
    matches['Away_Offensive_Index'] = matches['Away_Offensive_Index'].fillna(0)

    # 4. Target Variable Setup
    # 0 = Away Win, 1 = Draw, 2 = Home Win (Standard classification format)
    matches['Target'] = np.where(matches['FTR'] == 'H', 2, np.where(matches['FTR'] == 'D', 1, 0))

    # Drop rows with NaN values in EMA (usually the first game of a team's dataset)
    final_dataset = matches.dropna().reset_index(drop=True)

    save_path = '/Users/santomukiza/Desktop/Github/LaligaPrediction/Laliga-game-score-prediction/ml_ready_data.csv'
    final_dataset.to_csv(save_path, index=False)
    
    print(f"Feature engineering complete! Final dataset shape: {final_dataset.shape}")
    print(f"Saved as {save_path}")
    print("--- Feature Engineering Complete ---")
