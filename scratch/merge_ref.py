import pandas as pd
import glob
import os
import datetime

# 1. Load the scraped Referee data
script_dir = os.path.dirname(os.path.abspath(__file__))
ref_file = os.path.join(script_dir, 'all_seasons_referees.csv')

if not os.path.exists(ref_file):
    print(f"Error: Scraped referee file {ref_file} not found! Please run fetch_ref.py first.")
    exit(1)

print("Loading scraped referee dataset...")
ref_df = pd.read_csv(ref_file)
# Normalize dates: FBref dates YYYY-MM-DD
ref_df['Date'] = pd.to_datetime(ref_df['Date']).dt.date

# 2. Team mapping (LaligaSeasons -> FBref)
manual_map = {
    'Ath Madrid': 'Atlético Madrid',
    'Espanol': 'Espanyol',
    'La Coruna': 'Deportivo La Coruña',
    'Sociedad': 'Real Sociedad',
    'Vallecano': 'Rayo Vallecano',
    'Ath Bilbao': 'Athletic Club',
    'Betis': 'Real Betis',
    'Celta': 'Celta Vigo',
    'Sp Gijon': 'Sporting Gijón',
    'Huesca': 'Huesca',
    'Oviedo': 'Oviedo',
    'Real Oviedo': 'Oviedo',
    'Valladolid': 'Valladolid',
    'Alaves': 'Alavés',
    'Almeria': 'Almería',
    'Barcelona': 'Barcelona',
    'Cadiz': 'Cádiz',
    'Eibar': 'Eibar',
    'Elche': 'Elche',
    'Getafe': 'Getafe',
    'Girona': 'Girona',
    'Granada': 'Granada',
    'Las Palmas': 'Las Palmas',
    'Leganes': 'Leganés',
    'Levante': 'Levante',
    'Malaga': 'Málaga',
    'Mallorca': 'Mallorca',
    'Osasuna': 'Osasuna',
    'Real Madrid': 'Real Madrid',
    'Sevilla': 'Sevilla',
    'Valencia': 'Valencia',
    'Villarreal': 'Villarreal'
}

# 3. Process each LaligaSeasons CSV file in-place
seasons_pattern = os.path.join(os.path.dirname(script_dir), 'LaligaSeasons', 'La Liga Season *.csv')
season_files = glob.glob(seasons_pattern)
season_files.sort()

if not season_files:
    print(f"Error: No season CSV files found matching {seasons_pattern}")
    exit(1)

for f in season_files:
    print(f"\nProcessing {os.path.basename(f)}...")
    
    # Robust file reading to handle encoding differences
    try:
        df = pd.read_csv(f, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(f, encoding='latin-1')
        
    if 'Date' not in df.columns or 'HomeTeam' not in df.columns:
        print(f"  Skipping (missing required Date or HomeTeam columns)")
        continue
        
    # Parse dates carefully (supporting DD/MM/YY, DD/MM/YYYY, or standard format)
    try:
        parsed_dates = pd.to_datetime(df['Date'], format='%d/%m/%y').dt.date
    except ValueError:
        try:
            parsed_dates = pd.to_datetime(df['Date'], format='%d/%m/%Y').dt.date
        except ValueError:
            parsed_dates = pd.to_datetime(df['Date']).dt.date
            
    df['parsed_date'] = parsed_dates
    
    referee_list = []
    missing_count = 0
    
    for idx, row in df.iterrows():
        home_team = row['HomeTeam']
        match_date = row['parsed_date']
        
        if pd.isna(home_team) or pd.isna(match_date):
            referee_list.append(None)
            continue
            
        # Translate HomeTeam to FBref representation
        fbref_team = manual_map.get(home_team, home_team)
        
        # Define acceptable match names in scraped FBref data to handle variations (Betis, La Coruña, Huesca)
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
            
        # Match by date (+/- 1 day window) and Team
        mask = (
            (ref_df['HomeTeam'].isin(team_names) | ref_df['AwayTeam'].isin(team_names)) & 
            ((ref_df['Date'] == match_date) | 
             (ref_df['Date'] == match_date + datetime.timedelta(days=1)) | 
             (ref_df['Date'] == match_date - datetime.timedelta(days=1)))
        )
        
        match_rows = ref_df[mask]
        
        if len(match_rows) > 0:
            referee = match_rows.iloc[0]['Referee']
            referee_list.append(referee)
        else:
            referee_list.append(None)
            missing_count += 1
            if missing_count <= 5:
                print(f"  Warning: No referee found for match of {home_team} on {match_date} (Mapped to FBref as '{fbref_team}')")
                
    df['Referee'] = referee_list
    df = df.drop(columns=['parsed_date'])
    
    # Save back in-place
    df.to_csv(f, index=False, encoding='utf-8')
    success_rate = (len(df) - missing_count) / len(df) * 100
    print(f"  Saved in-place. Matched: {len(df) - missing_count} / {len(df)} matches ({success_rate:.2f}%)")

print("\nAll LaLigaSeasons CSV files updated with referee data successfully!")
