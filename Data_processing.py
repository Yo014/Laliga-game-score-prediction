import pandas as pd
import glob
import re

def clean_and_combine_data():
    # Standardize team names exactly as you had them
    TEAM_MAPPING = {
        'FC Barcelona': 'Barcelona', 'BAR': 'Barcelona',
        'Real Madrid': 'Real Madrid', 'RMA': 'Real Madrid', 'R. Madrid': 'Real Madrid',
        'Atlético de Madrid': 'Ath Madrid', 'Ath Madrid': 'Ath Madrid', 'ATM': 'Ath Madrid',
        'Sevilla': 'Sevilla', 'SEV': 'Sevilla',
        'Villarreal': 'Villarreal', 'VIL': 'Villarreal',
        'Real Sociedad': 'Sociedad', 'Sociedad': 'Sociedad', 'RSO': 'Sociedad',
        'Athletic': 'Ath Bilbao', 'Ath Bilbao': 'Ath Bilbao', 'ATH': 'Ath Bilbao',
        'Real Betis': 'Betis', 'Betis': 'Betis', 'BET': 'Betis',
        'Celta Vigo': 'Celta', 'Celta': 'Celta', 'CEL': 'Celta',
        'Valencia': 'Valencia', 'VAL': 'Valencia',
        'Getafe': 'Getafe', 'GET': 'Getafe',
        'Girona FC': 'Girona', 'Girona': 'Girona', 'GIR': 'Girona',
        'Osasuna': 'Osasuna', 'OSA': 'Osasuna',
        'Mallorca': 'Mallorca', 'MLL': 'Mallorca',
        'Deportivo Alavés': 'Alaves', 'Alaves': 'Alaves', 'ALA': 'Alaves',
        'Rayo Vallecano': 'Vallecano', 'Vallecano': 'Vallecano', 'RAY': 'Vallecano',
        'UD Las Palmas': 'Las Palmas', 'Las Palmas': 'Las Palmas',
        'Leganés': 'Leganes', 'Leganes': 'Leganes',
        'Espanyol': 'Espanol', 'Espanol': 'Espanol', 'ESP': 'Espanol',
        'Real Valladolid': 'Valladolid', 'Valladolid': 'Valladolid',
        'Eibar': 'Eibar', 'Granada': 'Granada',
        'Levante': 'Levante', 'LEV': 'Levante',
        'Elche': 'Elche', 'ELC': 'Elche',
        'Cádiz': 'Cadiz', 'Cadiz': 'Cadiz',
        'Almería': 'Almeria', 'Almeria': 'Almeria',
        'Deportivo La Coruña': 'La Coruna', 'Deportivo': 'La Coruna', 'RC Deportivo': 'La Coruna', 'La Coruna': 'La Coruna',
        'Málaga': 'Malaga', 'Malaga': 'Malaga',
        'Sporting Gijón': 'Sp Gijon', 'Real Sporting': 'Sp Gijon', 'Sp Gijon': 'Sp Gijon',
        'Huesca': 'Huesca',
        'Real Oviedo': 'Oviedo', 'OVI': 'Oviedo'
    }

    def clean_team_name(name):
        if pd.isna(name): return name
        return TEAM_MAPPING.get(str(name).strip(), str(name).strip())

    def get_season(filename):
        """Extracts the season string (e.g., '15-16') directly from the filename."""
        nums = re.findall(r'\d+', filename)
        if len(nums) >= 2:
            return f"{nums[0][-2:]}-{nums[1][-2:]}"
        elif len(nums) == 1 and len(nums[0]) == 4:
            return f"{nums[0][0:2]}-{nums[0][2:4]}"
        return "Unknown"

    # Gather raw files (ignoring previously processed files)
    all_files = [f for f in glob.glob("**/*.csv", recursive=True) if "Processed" not in f and "cleaned_" not in f]
    scorers_files = sorted([f for f in all_files if 'scorers' in f.lower()])
    assists_files = sorted([f for f in all_files if 'assists' in f.lower()])
    matches_files = sorted([f for f in all_files if 'la liga' in f.lower() and 'scorer' not in f.lower() and 'assist' not in f.lower()])

# --- 1. COMBINE SCORERS ---
    scorers_list = []
    for f in scorers_files:
        df = pd.read_csv(f)
        # Added 'Expected Goals (xG)': 'xG' to the rename mapping
        df.rename(columns={'Player Name': 'Player', 'Goals Scored (GS)': 'Goals', 'Games Played (GP)': 'Matches Played', 'Average (OG)': 'Coefficient', 'Expected Goals (xG)': 'xG'}, inplace=True)
        if 'Team' in df.columns: df['Team'] = df['Team'].apply(clean_team_name)
        df['Season'] = get_season(f)
        
        # If the season didn't have xG data, fill the column with empty values
        if 'xG' not in df.columns:
            df['xG'] = pd.NA
            
        scorers_list.append(df[['Player', 'Team', 'Goals', 'Matches Played', 'Coefficient', 'xG', 'Season']])
    
    pd.concat(scorers_list, ignore_index=True).to_csv("Processed_Scorers.csv", index=False)

    # --- 2. COMBINE ASSISTS ---
    assists_list = []
    for f in assists_files:
        df = pd.read_csv(f)
        # Added 'Expected Assists (xA)': 'xA' to the rename mapping
        df.rename(columns={'Player Name': 'Player', 'Assists (ASS)': 'Assists', 'Games Played (GP)': 'Matches Played', 'Average (APM)': 'Coefficient', 'Expected Assists (xA)': 'xA'}, inplace=True)
        if 'Team' in df.columns: df['Team'] = df['Team'].apply(clean_team_name)
        df['Season'] = get_season(f)
        
        # If the season didn't have xA data, fill the column with empty values
        if 'xA' not in df.columns:
            df['xA'] = pd.NA
            
        assists_list.append(df[['Player', 'Team', 'Assists', 'Matches Played', 'Coefficient', 'xA', 'Season']])
        
    pd.concat(assists_list, ignore_index=True).to_csv("Processed_Assists.csv", index=False)

    # --- 3. COMBINE MATCHES --
    matches_list = []
    for f in matches_files:
        df = pd.read_csv(f)
        
        # Handle newer formats (e.g., 25/26 season)
        if 'home_team' in df.columns:
            df.rename(columns={'home_team': 'HomeTeam', 'away_team': 'AwayTeam', 'date': 'Date'}, inplace=True)
        if 'result' in df.columns:
            mask = df['result'].notna() & df['result'].astype(str).str.contains('-')
            df.loc[mask, 'FTHG'] = pd.to_numeric(df.loc[mask, 'result'].apply(lambda x: str(x).split('-')[0].strip()), errors='coerce')
            df.loc[mask, 'FTAG'] = pd.to_numeric(df.loc[mask, 'result'].apply(lambda x: str(x).split('-')[1].strip()), errors='coerce')
            df.loc[df['FTHG'] > df['FTAG'], 'FTR'] = 'H'
            df.loc[df['FTHG'] < df['FTAG'], 'FTR'] = 'A'
            df.loc[df['FTHG'] == df['FTAG'], 'FTR'] = 'D'
            
        df['HomeTeam'] = df['HomeTeam'].apply(clean_team_name)
        df['AwayTeam'] = df['AwayTeam'].apply(clean_team_name)
        df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=True, errors='coerce').dt.strftime('%Y-%m-%d')
        
        cols_to_keep = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HF', 'AF', 'HY', 'AY', 'HR', 'AR', 'Referee', 'B365H', 'B365D', 'B365A']
        for c in cols_to_keep:
            if c not in df.columns: df[c] = None # Fill gracefully if a column doesn't exist
            
        df = df[cols_to_keep].dropna(subset=['Date', 'HomeTeam', 'AwayTeam'])
        matches_list.append(df)

    pd.concat(matches_list, ignore_index=True).to_csv("Processed_Matches.csv", index=False)
    print("Files successfully combined into Processed_Matches.csv, Processed_Assists.csv, and Processed_Scorers.csv!")

if __name__ == "__main__":
    clean_and_combine_data()