import os
import glob
import re
import unicodedata
import pandas as pd
import db_manager

def get_season(filename):
    """Extracts the season string (e.g., '15-16') directly from the filename."""
    nums = re.findall(r'\d+', filename)
    if len(nums) >= 2:
        return f"{nums[0][-2:]}-{nums[1][-2:]}"
    elif len(nums) == 1 and len(nums[0]) == 4:
        return f"{nums[0][0:2]}-{nums[0][2:4]}"
    return "Unknown"

def populate_squad_players():
    print("Populating squad_players...")
    base_dir = 'Laliga Squads'
    all_players = []
    
    if not os.path.exists(base_dir):
        print(f"Warning: Squad directory not found at {base_dir}")
        return

    for folder_name in os.listdir(base_dir):
        folder_name_nfc = unicodedata.normalize('NFC', folder_name)
        folder_path = os.path.join(base_dir, folder_name)
        if os.path.isdir(folder_path):
            file_path = os.path.join(folder_path, 'player_data.csv')
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df['Team'] = folder_name_nfc  # Keep NFC normalized team name
                all_players.append(df)
                
    if all_players:
        combined_df = pd.concat(all_players, ignore_index=True)
        db_manager.save_to_db(combined_df, 'squad_players')
        print(f"  Successfully inserted {len(combined_df)} player records.")

def populate_assists():
    print("Populating assists...")
    files = glob.glob('LaligaAssist/*.csv')
    all_assists = []
    for f in files:
        df = pd.read_csv(f)
        df.rename(columns={'Player Name': 'Player', 'Assists (ASS)': 'Assists', 'Games Played (GP)': 'Matches Played', 'Average (APM)': 'Coefficient', 'Expected Assists (xA)': 'xA'}, inplace=True)
        df['Season'] = get_season(f)
        all_assists.append(df)
        
    if all_assists:
        combined_df = pd.concat(all_assists, ignore_index=True)
        db_manager.save_to_db(combined_df, 'assists')
        print(f"  Successfully inserted {len(combined_df)} assist records.")

def populate_scorers():
    print("Populating scorers...")
    files = glob.glob('Laligascoring/*.csv')
    all_scorers = []
    for f in files:
        df = pd.read_csv(f)
        df.rename(columns={'Player Name': 'Player', 'Goals Scored (GS)': 'Goals', 'Games Played (GP)': 'Matches Played', 'Average (OG)': 'Coefficient', 'Expected Goals (xG)': 'xG'}, inplace=True)
        df['Season'] = get_season(f)
        all_scorers.append(df)
        
    if all_scorers:
        combined_df = pd.concat(all_scorers, ignore_index=True)
        db_manager.save_to_db(combined_df, 'scorers')
        print(f"  Successfully inserted {len(combined_df)} scorer records.")

def populate_seasons():
    print("Populating seasons matches...")
    files = glob.glob('LaligaSeasons/*.csv')
    all_matches = []
    for f in files:
        df = pd.read_csv(f)
        df['Season'] = get_season(f)
        all_matches.append(df)
        
    if all_matches:
        combined_df = pd.concat(all_matches, ignore_index=True)
        db_manager.save_to_db(combined_df, 'seasons')
        print(f"  Successfully inserted {len(combined_df)} seasons matches.")

def main():
    print("--- Starting SQLite Database Seeding ---")
    populate_squad_players()
    populate_assists()
    populate_scorers()
    populate_seasons()
    
    # Run simple index creations
    db_manager.init_db()
    print("Database seeding completed successfully!")

if __name__ == "__main__":
    main()
