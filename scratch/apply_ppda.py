import pandas as pd
import glob
import os
import datetime

# 1. Load the scraped PPDA data
ppda_file = 'scratch/all_seasons_ppda.csv'
ppda_df = pd.read_csv(ppda_file)

# Understat dates are 'YYYY-MM-DD HH:MM:SS'
ppda_df['Date'] = pd.to_datetime(ppda_df['Date']).dt.date

# 2. Team mapping (LaligaSeasons -> Understat)
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
    'Huesca': 'SD Huesca',
    'Oviedo': 'Real Oviedo',
    'Valladolid': 'Real Valladolid',
    'Alaves': 'Alaves',
    'Almeria': 'Almeria',
    'Barcelona': 'Barcelona',
    'Cadiz': 'Cadiz',
    'Eibar': 'Eibar',
    'Elche': 'Elche',
    'Getafe': 'Getafe',
    'Girona': 'Girona',
    'Granada': 'Granada',
    'Las Palmas': 'Las Palmas',
    'Leganes': 'Leganes',
    'Levante': 'Levante',
    'Malaga': 'Malaga',
    'Mallorca': 'Mallorca',
    'Osasuna': 'Osasuna',
    'Real Madrid': 'Real Madrid',
    'Sevilla': 'Sevilla',
    'Valencia': 'Valencia',
    'Villarreal': 'Villarreal'
}

# 3. Process each LaligaSeasons CSV
season_files = glob.glob('LaligaSeasons/La Liga Season *.csv')
season_files.sort()

for f in season_files:
    print(f"Processing {f}...")
    df = pd.read_csv(f)
    
    if 'Date' not in df.columns or 'HomeTeam' not in df.columns:
        print(f"  Skipping {f} (missing columns)")
        continue
    
    # Parse dates carefully (format is DD/MM/YY or DD/MM/YYYY)
    try:
        parsed_dates = pd.to_datetime(df['Date'], format='%d/%m/%y').dt.date
    except ValueError:
        try:
            parsed_dates = pd.to_datetime(df['Date'], format='%d/%m/%Y').dt.date
        except ValueError:
            parsed_dates = pd.to_datetime(df['Date']).dt.date
            
    df['parsed_date'] = parsed_dates
    
    home_ppda_list = []
    away_ppda_list = []
    
    for idx, row in df.iterrows():
        home_team = row['HomeTeam']
        match_date = row['parsed_date']
        
        # Ensure we have a valid row
        if pd.isna(home_team) or pd.isna(match_date):
            home_ppda_list.append(None)
            away_ppda_list.append(None)
            continue
            
        understat_home = manual_map.get(home_team, home_team)
        
        # Look up in ppda_df
        # We allow a date window of +/- 1 day because some games kick off at 00:00 midnight UTC vs local time
        mask = (
            (ppda_df['Team'] == understat_home) & 
            ((ppda_df['Date'] == match_date) | 
             (ppda_df['Date'] == match_date + datetime.timedelta(days=1)) | 
             (ppda_df['Date'] == match_date - datetime.timedelta(days=1)))
        )
        match_rows = ppda_df[mask]
        
        if len(match_rows) > 0:
            # Understat PPDA is passes allowed. For the home team, 'PPDA' is their PPDA, 'OPPDA' is the opponent's PPDA
            # BUT wait: we must ensure this is the HOME game for the team in Understat.
            # Sometimes 'h_a' is 'h'. If they are away on Understat due to some weird reason, we should use the appropriate metric.
            m_row = match_rows.iloc[0]
            if m_row['h_a'] == 'h':
                h_ppda = m_row['PPDA']
                a_ppda = m_row['OPPDA']
            else:
                h_ppda = m_row['OPPDA']
                a_ppda = m_row['PPDA']
            
            home_ppda_list.append(round(h_ppda, 3) if not pd.isna(h_ppda) else None)
            away_ppda_list.append(round(a_ppda, 3) if not pd.isna(a_ppda) else None)
        else:
            home_ppda_list.append(None)
            away_ppda_list.append(None)
            
    df['Home_PPDA'] = home_ppda_list
    df['Away_PPDA'] = away_ppda_list
    
    missing = df['Home_PPDA'].isna().sum()
    if missing > 0:
        print(f"  Warning: {missing} matches could not be matched with PPDA data.")
    
    # Clean up and save
    df = df.drop(columns=['parsed_date'])
    df.to_csv(f, index=False)
    print(f"  Saved {f} with Home_PPDA and Away_PPDA columns.")

print("All files updated successfully.")
