"""
build_squad_health.py
Reads all Laliga Squads/<Team>/player_data.csv files and produces a single
current_squad_health.csv with per-team injury aggregates.
"""

import os
import pandas as pd
from datetime import datetime

# Maps squad folder names → canonical model team names (from Data_processing.py)
FOLDER_TO_MODEL_NAME = {
    'Alavés': 'Alaves',
    'Athletic Bilbao': 'Ath Bilbao',
    'Atlético Madrid': 'Ath Madrid',
    'Barcelona': 'Barcelona',
    'Celta Vigo': 'Celta',
    'Elche': 'Elche',
    'Espanyol': 'Espanol',
    'Getafe': 'Getafe',
    'Girona': 'Girona',
    'Levante': 'Levante',
    'Mallorca': 'Mallorca',
    'Osasuna': 'Osasuna',
    'Rayo Vallecano': 'Vallecano',
    'Real Betis': 'Betis',
    'Real Madrid': 'Real Madrid',
    'Real Oviedo': 'Oviedo',
    'Real Sociedad': 'Sociedad',
    'Sevilla': 'Sevilla',
    'Valencia': 'Valencia',
    'Villarreal': 'Villarreal',
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SQUADS_DIR = os.path.join(BASE_DIR, 'Laliga Squads')
OUTPUT_PATH = os.path.join(BASE_DIR, 'current_squad_health.csv')


def parse_appearances(val):
    """Convert appearances to int, treating 'Not used during this season' as 0."""
    if isinstance(val, str) and 'not used' in val.lower():
        return 0
    try:
        return int(val)
    except (ValueError, TypeError):
        return 0


def is_currently_injured(row, today=None):
    """
    A player is currently injured if:
      - Injuries > 0, AND
      - Expected Return is in the future OR 'Unknown'
    """
    if today is None:
        today = datetime.now()

    try:
        injuries = int(row['Injuries'])
    except (ValueError, TypeError):
        return False

    if injuries == 0:
        return False

    expected_return = str(row.get('Expected Return', 'N/A')).strip()
    if expected_return in ('N/A', ''):
        return False
    if expected_return.lower() == 'unknown':
        return True  # still out, no known return date

    try:
        return_date = pd.to_datetime(expected_return, dayfirst=True)
        return return_date >= today
    except Exception:
        return True  # can't parse → assume still injured


def build_squad_health():
    """Build team-level injury aggregates from individual player data."""
    today = datetime.now()
    rows = []

    for folder_name in sorted(os.listdir(SQUADS_DIR)):
        folder_path = os.path.join(SQUADS_DIR, folder_name)
        csv_path = os.path.join(folder_path, 'player_data.csv')
        if not os.path.isfile(csv_path):
            continue

        model_name = FOLDER_TO_MODEL_NAME.get(folder_name, folder_name)
        df = pd.read_csv(csv_path)

        # Parse appearances and goals
        df['Apps'] = df['Appearances'].apply(parse_appearances)
        df['Goals'] = pd.to_numeric(df['Goals Scored'], errors='coerce').fillna(0).astype(int)

        total_apps = df['Apps'].sum()
        total_goals = df['Goals'].sum()

        # Identify currently injured players
        df['Currently_Injured'] = df.apply(lambda r: is_currently_injured(r, today), axis=1)
        injured = df[df['Currently_Injured']]

        total_injured = len(injured)

        # "Key player" = ≥15 appearances this season
        key_threshold = 15
        injured_key = injured[injured['Apps'] >= key_threshold]
        missing_key_players = len(injured_key)

        # Missing Impact % = sum of injured key players' appearances / total team appearances
        missing_apps = injured_key['Apps'].sum()
        missing_impact_pct = round((missing_apps / total_apps * 100), 2) if total_apps > 0 else 0.0

        # Missing Goals % = sum of injured players' goals / total team goals
        missing_goals = injured['Goals'].sum()
        missing_goals_pct = round((missing_goals / total_goals * 100), 2) if total_goals > 0 else 0.0

        rows.append({
            'Team': model_name,
            'Total_Apps': total_apps,
            'Missing_Key_Players': missing_key_players,
            'Total_Injured': total_injured,
            'Missing_Impact_Pct': missing_impact_pct,
            'Missing_Goals_Pct': missing_goals_pct,
        })

    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_PATH, index=False)
    print(f"Squad health saved to {OUTPUT_PATH}")
    print(f"  Teams processed: {len(result)}")
    print(result.to_string(index=False))
    return result


if __name__ == '__main__':
    build_squad_health()
