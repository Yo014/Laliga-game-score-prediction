import pandas as pd
import glob
import os
import difflib

# 1. Load the scraped PPDA data
ppda_file = 'scratch/all_seasons_ppda.csv'
ppda_df = pd.read_csv(ppda_file)
# Normalize dates: understat has '2015-08-21 22:30:00'. We just need '2015-08-21'.
ppda_df['Date'] = pd.to_datetime(ppda_df['Date']).dt.date
understat_teams = set(ppda_df['Team'].unique())

# 2. Find all LaligaSeasons CSV files
season_files = glob.glob('LaligaSeasons/La Liga Season *.csv')
season_files.sort()

# Manual mapping (LaLigaSeasons -> Understat)
manual_map = {
    'Ath Madrid': 'Atletico Madrid',
    'Espanol': 'Espanyol',
    'La Coruna': 'Deportivo La Coruna',
    'Sociedad': 'Real Sociedad',
    'Vallecano': 'Rayo Vallecano',
    'Ath Bilbao': 'Athletic Club',
    'Betis': 'Real Betis',
    'Celta': 'Celta Vigo',
    'Sp Gijon': 'Sporting Gijon',
    'Alaves': 'Alaves',
    'Leganes': 'Leganes',
    'Girona': 'Girona',
    'Valladolid': 'Real Valladolid',
    'Huesca': 'Leganes', # Will be overwritten if found dynamically
    'Mallorca': 'Real Mallorca',
    'Osasuna': 'Mallorca',
    'Cadiz': 'Cadiz',
    'Elche': 'Elche',
    'Almeria': 'Almeria'
}

all_laliga_teams = set()
for f in season_files:
    df = pd.read_csv(f)
    if 'HomeTeam' in df.columns:
        all_laliga_teams.update(df['HomeTeam'].unique())

# Generate dynamic mapping
final_map = {}
for team in all_laliga_teams:
    if team in manual_map and manual_map[team] in understat_teams:
        final_map[team] = manual_map[team]
    elif team in understat_teams:
        final_map[team] = team
    else:
        # try to find closest match
        matches = difflib.get_close_matches(team, understat_teams, n=1, cutoff=0.4)
        if matches:
            final_map[team] = matches[0]
        else:
            final_map[team] = team # fallback

print("Team Mapping (LaligaSeasons -> Understat):")
for k, v in sorted(final_map.items()):
    print(f"  {k} -> {v}")

