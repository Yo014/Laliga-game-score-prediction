"""
aggregate_player_data.py
Reads each team's player_data.csv from the Laliga Squads directory and
computes squad health metrics for use in match predictions.

Outputs: current_squad_health.csv
"""

import pandas as pd
import os

SQUADS_DIR = '/Users/santomukiza/Desktop/Github/LaligaPrediction/Laliga-game-score-prediction/Laliga Squads'
OUTPUT_PATH = '/Users/santomukiza/Desktop/Github/LaligaPrediction/Laliga-game-score-prediction/current_squad_health.csv'

# Map folder names to names used in Processed_Matches.csv
TEAM_NAME_MAP = {
    'Real Madrid':       'Real Madrid',
    'Barcelona':         'Barcelona',
    'Atlético Madrid':   'Ath Madrid',
    'Athletic Bilbao':   'Ath Bilbao',
    'Villarreal':        'Villarreal',
    'Real Betis':        'Betis',
    'Rayo Vallecano':    'Vallecano',
    'Celta Vigo':        'Celta',
    'Osasuna':           'Osasuna',
    'Mallorca':          'Mallorca',
    'Real Sociedad':     'Sociedad',
    'Valencia':          'Valencia',
    'Getafe':            'Getafe',
    'Alavés':            'Alaves',
    'Sevilla':           'Sevilla',
    'Espanyol':          'Espanol',
    'Levante':           'Levante',
    'Elche':             'Elche',
    'Real Oviedo':       'Oviedo',
    'Girona':            'Girona',
}

KEY_PLAYER_APPEARANCES_THRESHOLD = 15  # Players with >= this many apps are "key players"
KEY_PLAYER_GOALS_THRESHOLD = 5 #players with >= this many goals are "key players"
records = []

for folder_name, match_name in TEAM_NAME_MAP.items():
    csv_path = os.path.join(SQUADS_DIR, folder_name, 'player_data.csv')
    if not os.path.exists(csv_path):
        print(f"  [WARNING] Missing: {csv_path}")
        continue

    df = pd.read_csv(csv_path)

    # Ensure correct column types
    df['Appearances'] = pd.to_numeric(df['Appearances'], errors='coerce').fillna(0)
    df['Injuries'] = pd.to_numeric(df['Injuries'], errors='coerce').fillna(0)

    total_apps = df['Appearances'].sum()
    injured_mask = df['Injuries'] == 1
    injured_apps = df.loc[injured_mask, 'Appearances'].sum()
    
    # Key players = regular starters with significant appearances
    key_players_mask = (df['Appearances'] >= KEY_PLAYER_APPEARANCES_THRESHOLD)
    missing_key_players = int((injured_mask & key_players_mask).sum())

    # Missing impact %: what fraction of the squad's total "experience" is unavailable
    missing_impact_pct = (injured_apps / total_apps * 100) if total_apps > 0 else 0.0

    # Total injured players count
    total_injured = int(injured_mask.sum())

    records.append({
        'Team':                  match_name,
        'Total_Apps':            int(total_apps),
        'Missing_Key_Players':   missing_key_players,
        'Total_Injured':         total_injured,
        'Missing_Impact_Pct':    round(missing_impact_pct, 2),
    })
    print(f"  {match_name:20s} | Key Players Missing: {missing_key_players} | Impact%: {missing_impact_pct:.1f}%")

health_df = pd.DataFrame(records)
health_df.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved current squad health to: {OUTPUT_PATH}")
print(health_df.to_string(index=False))
