import pandas as pd
import numpy as np
import os



# Map Squad folder names to match data team names
SQUAD_NAME_MAP = {
    'Alavés': 'Alaves', 'Athletic Bilbao': 'Ath Bilbao',
    'Atlético Madrid': 'Ath Madrid', 'Barcelona': 'Barcelona',
    'Celta Vigo': 'Celta', 'Elche': 'Elche',
    'Espanyol': 'Espanol', 'Getafe': 'Getafe',
    'Girona': 'Girona', 'Levante': 'Levante',
    'Mallorca': 'Mallorca', 'Osasuna': 'Osasuna',
    'Rayo Vallecano': 'Vallecano', 'Real Betis': 'Betis',
    'Real Madrid': 'Real Madrid', 'Real Oviedo': 'Oviedo',
    'Real Sociedad': 'Sociedad', 'Sevilla': 'Sevilla',
    'Valencia': 'Valencia', 'Villarreal': 'Villarreal'
}

def parse_market_value(val_str):
    """Parses market value strings like '€40.00m' or '€900k' into float numbers."""
    if pd.isna(val_str) or val_str == "Unknown":
        return 0
    try:
        val_str = str(val_str).replace("€", "").strip().lower()
        if "m" in val_str:
            return float(val_str.replace("m", "")) * 1_000_000
        if "k" in val_str:
            return float(val_str.replace("k", "")) * 1_000
        return float(val_str)
    except (ValueError, TypeError):
        return 0

def load_squad_value_data():
    """
    Aggregates player market values from the 'Laliga Squads' directory.
    Returns a DataFrame with 'Team' and 'Total_Squad_Value'.
    """
    base_dir = '/Users/santomukiza/Desktop/Github/LaligaPrediction/Laliga-game-score-prediction/Laliga Squads'
    squad_values = []
    
    if not os.path.exists(base_dir):
        print(f"Warning: Squad directory not found at {base_dir}")
        return pd.DataFrame(columns=['Team', 'Total_Squad_Value'])

    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        if os.path.isdir(folder_path):
            file_path = os.path.join(folder_path, 'player_data.csv')
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                if 'Market Value' in df.columns:
                    total_value = df['Market Value'].apply(parse_market_value).sum()
                    match_name = SQUAD_NAME_MAP.get(folder_name, folder_name)
                    squad_values.append({'Team': match_name, 'Total_Squad_Value': total_value})
    
    return pd.DataFrame(squad_values)


def calculate_h2h(df):
    """
    Calculates the historical win percentage of the Home Team against the Away Team.
    Optimized to run in O(N) using an accumulated state dictionary.
    """
    df = df.sort_values('Date').copy()
    h2h_win_rates = []
    
    # Store wins and matches as: (team_a, team_b): {'matches': 0, team_a: 0, team_b: 0}
    # team_a is always the alphabetically first team
    history = {}
    
    for idx, row in df.iterrows():
        home = row['HomeTeam']
        away = row['AwayTeam']
        ftr = row['FTR']
        
        team_a, team_b = sorted([home, away])
        key = (team_a, team_b)
        
        if key not in history:
            h2h_win_rates.append(0.5)
            history[key] = {'matches': 0, team_a: 0, team_b: 0}
        else:
            match_data = history[key]
            if match_data['matches'] == 0:
                h2h_win_rates.append(0.5)
            else:
                home_wins = match_data[home]
                h2h_win_rates.append(home_wins / match_data['matches'])
                
        # Update history AFTER appending to prevent data leakage
        history[key]['matches'] += 1
        if ftr == 'H':
            history[key][home] += 1
        elif ftr == 'A':
            history[key][away] += 1
            
    df['H2H_Home_Win_Rate'] = h2h_win_rates
    return df

def calculate_ema_form(df, span=5):
    """
    Calculates the Exponential Moving Average (EMA) for team stats.
    """
    # Create a dataframe to track every team's individual match history
    # Also need opponent shots/corners for Field Tilt calculation
    home_stats = df[['Date', 'HomeTeam', 'FTHG', 'FTAG', 'HS', 'HST', 'HC', 'AS', 'AST', 'AC', 'Home_PPDA']].rename(
        columns={'HomeTeam': 'Team', 'FTHG': 'GoalsScored', 'FTAG': 'GoalsConceded',
                 'HS': 'Shots', 'HST': 'ShotsOnTarget', 'HC': 'Corners',
                 'AS': 'OppShots', 'AST': 'OppShotsOnTarget', 'AC': 'OppCorners',
                 'Home_PPDA': 'Match_PPDA'}
    )
    home_stats['IsHome'] = 1
    home_stats['Points'] = np.where(home_stats['GoalsScored'] > home_stats['GoalsConceded'], 3, 
                           np.where(home_stats['GoalsScored'] == home_stats['GoalsConceded'], 1, 0))

    away_stats = df[['Date', 'AwayTeam', 'FTAG', 'FTHG', 'AS', 'AST', 'AC', 'HS', 'HST', 'HC', 'Away_PPDA']].rename(
        columns={'AwayTeam': 'Team', 'FTAG': 'GoalsScored', 'FTHG': 'GoalsConceded',
                 'AS': 'Shots', 'AST': 'ShotsOnTarget', 'AC': 'Corners',
                 'HS': 'OppShots', 'HST': 'OppShotsOnTarget', 'HC': 'OppCorners',
                 'Away_PPDA': 'Match_PPDA'}
    )
    away_stats['IsHome'] = 0
    away_stats['Points'] = np.where(away_stats['GoalsScored'] > away_stats['GoalsConceded'], 3, 
                           np.where(away_stats['GoalsScored'] == away_stats['GoalsConceded'], 1, 0))

    # Combine and sort chronologically
    team_matches = pd.concat([home_stats, away_stats]).sort_values(['Team', 'Date'])

    # Calculate Goal Difference for each match
    team_matches['GoalDiff'] = team_matches['GoalsScored'] - team_matches['GoalsConceded']
    
    # Calculate Proxy Match xG (True Form Indicator)
    shots = pd.to_numeric(team_matches['Shots'], errors='coerce').fillna(0)
    sot = pd.to_numeric(team_matches['ShotsOnTarget'], errors='coerce').fillna(0)
    corners = pd.to_numeric(team_matches['Corners'], errors='coerce').fillna(0)
    opp_shots = pd.to_numeric(team_matches['OppShots'], errors='coerce').fillna(0)
    opp_corners = pd.to_numeric(team_matches['OppCorners'], errors='coerce').fillna(0)
    
    team_matches['Match_xG'] = (sot * 0.3) + ((shots - sot) * 0.07) + (corners * 0.05)
    team_matches['Match_xG'] = team_matches['Match_xG'].fillna(0)
    
    # Calculate Field Tilt Proxy: proportion of territory controlled
    total_shots = shots + opp_shots
    total_corners = corners + opp_corners
    total_activity = total_shots + total_corners
    team_matches['Field_Tilt'] = np.where(total_activity > 0,
                                          (shots + corners) / total_activity, 0.5)

    # Calculate EMA. The shift(1) is critical: it prevents data leakage
    # We consolidate the groupby to a single operation for massive speedup
    cols_to_ema = ['Points', 'GoalsScored', 'GoalsConceded', 'GoalDiff', 
                   'Shots', 'ShotsOnTarget', 'Corners', 
                   'OppShots', 'OppShotsOnTarget', 'OppCorners',
                   'Match_xG', 'Field_Tilt', 'Match_PPDA']
                   
    for col in cols_to_ema:
        if col == 'Match_PPDA':
            team_matches[col] = pd.to_numeric(team_matches[col], errors='coerce').fillna(10.0)
        else:
            team_matches[col] = pd.to_numeric(team_matches[col], errors='coerce').fillna(0)
                   
    grouped = team_matches.groupby('Team')[cols_to_ema]
    ema_df = grouped.transform(lambda x: x.shift(1).ewm(span=span, adjust=False).mean())
    
    # Assign EMA columns
    team_matches['EMA_Points'] = ema_df['Points']
    team_matches['EMA_GoalsScored'] = ema_df['GoalsScored']
    team_matches['EMA_GoalsConceded'] = ema_df['GoalsConceded']
    team_matches['EMA_GoalDiff'] = ema_df['GoalDiff']
    
    # Offensive dominance metrics
    team_matches['EMA_Shots'] = ema_df['Shots']
    team_matches['EMA_ShotsOnTarget'] = ema_df['ShotsOnTarget']
    team_matches['EMA_Corners'] = ema_df['Corners']

    # Defensive dominance metrics (How much they allow)
    team_matches['EMA_ShotsConceded'] = ema_df['OppShots']
    team_matches['EMA_SOTConceded'] = ema_df['OppShotsOnTarget']
    team_matches['EMA_CornersConceded'] = ema_df['OppCorners']

    # TRUE FORM METRICS: EMA of Match xG and Field Tilt
    team_matches['EMA_xG_Created'] = ema_df['Match_xG']
    team_matches['EMA_Field_Tilt'] = ema_df['Field_Tilt']
    team_matches['EMA_PPDA'] = ema_df['Match_PPDA']

    # Defensve/Offensive reliability
    team_matches['Clean_Sheet'] = np.where(team_matches['GoalsConceded'] == 0, 1, 0)
    team_matches['Failed_To_Score'] = np.where(team_matches['GoalsScored'] == 0, 1, 0)
    
    team_matches['EMA_Clean_Sheet'] = team_matches.groupby('Team')['Clean_Sheet'].transform(lambda x: x.shift(1).ewm(span=span*2, adjust=False).mean())
    team_matches['EMA_Failed_To_Score'] = team_matches.groupby('Team')['Failed_To_Score'].transform(lambda x: x.shift(1).ewm(span=span*2, adjust=False).mean())

    return team_matches[['Date', 'Team', 'EMA_Points', 'EMA_GoalsScored', 'EMA_GoalsConceded', 'EMA_GoalDiff', 
                         'EMA_Shots', 'EMA_ShotsOnTarget', 'EMA_Corners', 
                         'EMA_ShotsConceded', 'EMA_SOTConceded', 'EMA_CornersConceded',
                         'EMA_xG_Created', 'EMA_Field_Tilt', 'EMA_PPDA', 'EMA_Clean_Sheet', 'EMA_Failed_To_Score']]

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
    # Shift(1) to avoid leakage. Consolidated groupby for speed.
    grouped = df.groupby('Referee')[['Total_Cards', 'Total_Fouls']]
    ema_df = grouped.transform(lambda x: x.shift(1).ewm(span=span, adjust=False).mean())
    
    df['Ref_Avg_Cards'] = ema_df['Total_Cards']
    df['Ref_Avg_Fouls'] = ema_df['Total_Fouls']
    
    # Referee Home Bias: Percentage of home wins
    df['Is_Home_Win'] = np.where(df['FTR'] == 'H', 1, 0)
    df['Ref_Home_Win_Rate'] = df.groupby('Referee')['Is_Home_Win'].transform(lambda x: x.shift(1).expanding().mean())
    
    # Fill first-time referee stats with global averages
    df['Ref_Avg_Cards'] = df['Ref_Avg_Cards'].fillna(df['Total_Cards'].mean())
    df['Ref_Avg_Fouls'] = df['Ref_Avg_Fouls'].fillna(df['Total_Fouls'].mean())
    df['Ref_Home_Win_Rate'] = df['Ref_Home_Win_Rate'].fillna(df['Is_Home_Win'].mean())
    
    return df[['Date', 'HomeTeam', 'AwayTeam', 'Ref_Avg_Cards', 'Ref_Avg_Fouls', 'Ref_Home_Win_Rate']]
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
        'EMA_xG_Created': 'Home_EMA_xG_Created', 'EMA_Field_Tilt': 'Home_EMA_Field_Tilt', 'EMA_PPDA': 'Home_EMA_PPDA',
        'EMA_Clean_Sheet': 'Home_Clean_Sheet_Rate', 'EMA_Failed_To_Score': 'Home_FTS_Rate'
    }).drop('Team', axis=1)

    matches = pd.merge(matches, form_df, left_on=['Date', 'AwayTeam'], right_on=['Date', 'Team'], how='left')
    matches = matches.rename(columns={
        'EMA_Points': 'Away_EMA_Points', 'EMA_GoalsScored': 'Away_EMA_GS', 'EMA_GoalsConceded': 'Away_EMA_GC',
        'EMA_GoalDiff': 'Away_EMA_GoalDiff', 'EMA_Shots': 'Away_EMA_Shots', 'EMA_ShotsOnTarget': 'Away_EMA_ShotsOnTarget', 'EMA_Corners': 'Away_EMA_Corners',
        'EMA_ShotsConceded': 'Away_EMA_ShotsConceded', 'EMA_SOTConceded': 'Away_EMA_SOTConceded', 'EMA_CornersConceded': 'Away_EMA_CornersConceded',
        'EMA_xG_Created': 'Away_EMA_xG_Created', 'EMA_Field_Tilt': 'Away_EMA_Field_Tilt', 'EMA_PPDA': 'Away_EMA_PPDA',
        'EMA_Clean_Sheet': 'Away_Clean_Sheet_Rate', 'EMA_Failed_To_Score': 'Away_FTS_Rate'
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

    # 5b. Add Total Squad Market Value (Proxy for raw talent/depth)
    print("Integrating Total Squad Market Value (Transfermarkt)...")
    squad_values = load_squad_value_data()
    
    matches = pd.merge(matches, squad_values, left_on='HomeTeam', right_on='Team', how='left')
    matches = matches.rename(columns={'Total_Squad_Value': 'Home_Squad_Value'}).drop('Team', axis=1)
    
    matches = pd.merge(matches, squad_values, left_on='AwayTeam', right_on='Team', how='left')
    matches = matches.rename(columns={'Total_Squad_Value': 'Away_Squad_Value'}).drop('Team', axis=1)
    
    # Fill missing values with league median
    median_val = squad_values['Total_Squad_Value'].median() if not squad_values.empty else 100_000_000
    matches['Home_Squad_Value'] = matches['Home_Squad_Value'].fillna(median_val)
    matches['Away_Squad_Value'] = matches['Away_Squad_Value'].fillna(median_val)

    # Tag squad health with the current season so it only merges with 25-26 matches
    squad_health['Season'] = '25-26'

    # Merge for Home team (by Season + Team)
    matches = pd.merge(matches, squad_health[['Season', 'Team', 'Missing_Key_Players', 'Missing_Impact_Pct', 'Missing_Goals_Pct', 'Missing_Assists_Pct', 'Missing_NP_Goals_Pct', 'Missing_Yellows_Pct', 'Missing_Reds_Pct']],
                       left_on=['Season', 'HomeTeam'], right_on=['Season', 'Team'], how='left')
    matches = matches.rename(columns={
        'Missing_Key_Players': 'Home_Missing_Key_Players',
        'Missing_Impact_Pct': 'Home_Missing_Impact_Pct',
        'Missing_Goals_Pct': 'Home_Missing_Goals_Pct',
        'Missing_Assists_Pct': 'Home_Missing_Assists_Pct',
        'Missing_NP_Goals_Pct': 'Home_Missing_NP_Goals_Pct',
        'Missing_Yellows_Pct': 'Home_Missing_Yellows_Pct',
        'Missing_Reds_Pct': 'Home_Missing_Reds_Pct'
    }).drop('Team', axis=1)

    # Merge for Away team (by Season + Team)
    matches = pd.merge(matches, squad_health[['Season', 'Team', 'Missing_Key_Players', 'Missing_Impact_Pct', 'Missing_Goals_Pct', 'Missing_Assists_Pct', 'Missing_NP_Goals_Pct', 'Missing_Yellows_Pct', 'Missing_Reds_Pct']],
                       left_on=['Season', 'AwayTeam'], right_on=['Season', 'Team'], how='left')
    matches = matches.rename(columns={
        'Missing_Key_Players': 'Away_Missing_Key_Players',
        'Missing_Impact_Pct': 'Away_Missing_Impact_Pct',
        'Missing_Goals_Pct': 'Away_Missing_Goals_Pct',
        'Missing_Assists_Pct': 'Away_Missing_Assists_Pct',
        'Missing_NP_Goals_Pct': 'Away_Missing_NP_Goals_Pct',
        'Missing_Yellows_Pct': 'Away_Missing_Yellows_Pct',
        'Missing_Reds_Pct': 'Away_Missing_Reds_Pct'
    }).drop('Team', axis=1)

    # Fill 0 for teams/seasons without squad health data (all historical seasons)
    for col in ['Home_Missing_Key_Players', 'Away_Missing_Key_Players',
                'Home_Missing_Impact_Pct', 'Away_Missing_Impact_Pct',
                'Home_Missing_Goals_Pct', 'Away_Missing_Goals_Pct',
                'Home_Missing_Assists_Pct', 'Away_Missing_Assists_Pct',
                'Home_Missing_NP_Goals_Pct', 'Away_Missing_NP_Goals_Pct',
                'Home_Missing_Yellows_Pct', 'Away_Missing_Yellows_Pct',
                'Home_Missing_Reds_Pct', 'Away_Missing_Reds_Pct']:
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
    matches['Missing_Goals_Diff'] = matches['Home_Missing_Goals_Pct'] - matches['Away_Missing_Goals_Pct']
    matches['Missing_Assists_Diff'] = matches['Home_Missing_Assists_Pct'] - matches['Away_Missing_Assists_Pct']
    matches['Missing_NP_Goals_Diff'] = matches['Home_Missing_NP_Goals_Pct'] - matches['Away_Missing_NP_Goals_Pct']
    matches['Missing_Yellows_Diff'] = matches['Home_Missing_Yellows_Pct'] - matches['Away_Missing_Yellows_Pct']
    matches['Missing_Reds_Diff'] = matches['Home_Missing_Reds_Pct'] - matches['Away_Missing_Reds_Pct']
    matches['xG_Form_Diff'] = matches['Home_EMA_xG_Created'] - matches['Away_EMA_xG_Created']
    matches['PPDA_Diff'] = matches['Home_EMA_PPDA'] - matches['Away_EMA_PPDA']
    matches['Tilt_Diff'] = matches['Home_EMA_Field_Tilt'] - matches['Away_EMA_Field_Tilt']
    matches['Value_Diff'] = matches['Home_Squad_Value'] - matches['Away_Squad_Value']

    # 4. Target Variable Setup
    # 0 = Away Win, 1 = Draw, 2 = Home Win
    matches['Target'] = np.where(matches['FTR'] == 'H', 2, np.where(matches['FTR'] == 'D', 1, 0))

    final_dataset = matches.dropna().reset_index(drop=True)
    features_to_keep = [
        'Date', 'HomeTeam', 'AwayTeam', 'FTR', 'Target', 'Referee',
        'B365H', 'B365D', 'B365A',
        'Market_Prob_H', 'Market_Prob_D', 'Market_Prob_A',
        'Ref_Avg_Cards', 'Ref_Avg_Fouls', 'Ref_Home_Win_Rate',
        'Home_EMA_Points', 'Home_EMA_GS', 'Home_EMA_GC', 'Home_EMA_GoalDiff',
        'Home_Clean_Sheet_Rate', 'Home_FTS_Rate',
        'Home_EMA_xG_Created', 'Home_EMA_xG_Conceded',
        'Away_EMA_xG_Created', 'Away_EMA_xG_Conceded',
        'Away_Clean_Sheet_Rate', 'Away_FTS_Rate',
        'xG_Form_Diff',
        'Home_EMA_Field_Tilt', 'Away_EMA_Field_Tilt', 'Tilt_Diff',
        'Home_EMA_PPDA', 'Away_EMA_PPDA', 'PPDA_Diff',
        'Home_Squad_Value', 'Away_Squad_Value', 'Value_Diff',
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
        'Home_Missing_Assists_Pct', 'Away_Missing_Assists_Pct',
        'Home_Missing_NP_Goals_Pct', 'Away_Missing_NP_Goals_Pct',
        'Home_Missing_Yellows_Pct', 'Away_Missing_Yellows_Pct',
        'Home_Missing_Reds_Pct', 'Away_Missing_Reds_Pct',
        'Form_Diff', 'Offense_Diff', 'Rest_Diff',
        'Missing_Key_Diff', 'Missing_Impact_Diff',
        'Missing_Goals_Diff', 'Missing_Assists_Diff',
        'Missing_NP_Goals_Diff', 'Missing_Yellows_Diff', 'Missing_Reds_Diff',
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
