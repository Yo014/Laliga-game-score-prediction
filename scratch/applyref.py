import pandas as pd
import datetime
import os

# 1. Load the scraped Referee data and target dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
ref_file = os.path.join(script_dir, 'all_seasons_referees.csv')

# Use robust fallbacks for the target dataset filename
main_file = os.path.join(script_dir, 'all_seasons_ppda_and_ref.csv')
if not os.path.exists(main_file):
    main_file = os.path.join(script_dir, 'all_seasons_ppda.csv')
    
output_file = os.path.join(script_dir, 'all_seasons_final.csv')

if not os.path.exists(ref_file):
    print(f"Error: Scraped referee file {ref_file} not found! Please run fetch_ref.py first.")
    exit(1)
    
if not os.path.exists(main_file):
    print(f"Error: Target dataset file not found under 'all_seasons_ppda_and_ref.csv' or 'all_seasons_ppda.csv'!")
    exit(1)

print(f"Loading target dataset from {os.path.basename(main_file)}...")
ref_df = pd.read_csv(ref_file)
df = pd.read_csv(main_file)

# Normalize dates: FBref dates YYYY-MM-DD
ref_df['Date'] = pd.to_datetime(ref_df['Date']).dt.date
# Target dataset dates are YYYY-MM-DD HH:MM:SS
df['MatchDate'] = pd.to_datetime(df['Date']).dt.date

# 2. Team mapping (Main Dataset -> FBref)
manual_map = {
    'Alaves': 'Alavés',
    'Almeria': 'Almería',
    'Athletic Club': 'Athletic Club',
    'Atletico Madrid': 'Atlético Madrid',
    'Barcelona': 'Barcelona',
    'Cadiz': 'Cádiz',
    'Celta Vigo': 'Celta Vigo',
    'Deportivo La Coruna': 'La Coruña',
    'Eibar': 'Eibar',
    'Elche': 'Elche',
    'Espanyol': 'Espanyol',
    'Getafe': 'Getafe',
    'Girona': 'Girona',
    'Granada': 'Granada',
    'Las Palmas': 'Las Palmas',
    'Leganes': 'Leganés',
    'Levante': 'Levante',
    'Malaga': 'Málaga',
    'Mallorca': 'Mallorca',
    'Osasuna': 'Osasuna',
    'Rayo Vallecano': 'Rayo Vallecano',
    'Real Betis': 'Real Betis',
    'Real Madrid': 'Real Madrid',
    'Real Oviedo': 'Oviedo',
    'Real Sociedad': 'Real Sociedad',
    'Real Valladolid': 'Valladolid',
    'SD Huesca': 'Huesca',
    'Sevilla': 'Sevilla',
    'Sporting Gijon': 'Sporting Gijón',
    'Valencia': 'Valencia',
    'Villarreal': 'Villarreal'
}

referee_list = []
missing_count = 0

print("Applying referee matching...")
# 3. Iterate over dataset and merge matching values
for idx, row in df.iterrows():
    team = row['Team']
    match_date = row['MatchDate']
    
    # Translate target team name to FBref representation
    fbref_team = manual_map.get(team, team)
    
    # Define acceptable match names in scraped FBref data to handle historical variations
    if fbref_team == 'Real Betis':
        team_names = ['Real Betis', 'Betis']
    elif fbref_team == 'Real Sociedad':
        team_names = ['Real Sociedad', 'Sociedad']
    elif fbref_team == 'Real Oviedo':
        team_names = ['Real Oviedo', 'Oviedo']
    elif fbref_team == 'Real Valladolid':
        team_names = ['Real Valladolid', 'Valladolid']
    elif fbref_team == 'SD Huesca':
        team_names = ['SD Huesca', 'Huesca']
    elif fbref_team == 'Deportivo La Coruña' or fbref_team == 'La Coruña':
        team_names = ['Deportivo La Coruña', 'La Coruña']
    else:
        team_names = [fbref_team]
    
    # Match by date (+/- 1 day for timezone safety) and Team (found as either Home or Away)
    mask = (
        (ref_df['HomeTeam'].isin(team_names) | ref_df['AwayTeam'].isin(team_names)) & 
        ((ref_df['Date'] == match_date) | 
         (ref_df['Date'] == match_date + datetime.timedelta(days=1)) | 
         (ref_df['Date'] == match_date - datetime.timedelta(days=1)))
    )
    
    match_rows = ref_df[mask]
    
    if len(match_rows) > 0:
        # Get the referee name
        referee = match_rows.iloc[0]['Referee']
        referee_list.append(referee)
    else:
        referee_list.append(None)
        missing_count += 1
        if missing_count <= 10:
            print(f"  Warning: No referee found for match of {team} on {match_date} (Mapped to FBref as '{fbref_team}')")

# 4. Save the compiled results        
df['Referee'] = referee_list
df.drop(columns=['MatchDate'], inplace=True) # Cleanup

df.to_csv(output_file, index=False, encoding='utf-8')
success_rate = (len(df) - missing_count) / len(df) * 100
print(f"\nMerge Complete!")
print(f"Successfully matched: {len(df) - missing_count} / {len(df)} rows ({success_rate:.2f}%)")
print(f"Missing: {missing_count} rows")
print(f"Results saved to {output_file}")