import pandas as pd
import os

# Paths
sp1_path = "/Users/santomukiza/Desktop/Github/LaligaPrediction/Laliga-game-score-prediction/LaligaSeasons/SP1 25-26.csv"
target_path = "/Users/santomukiza/Desktop/Github/LaligaPrediction/Laliga-game-score-prediction/LaligaSeasons/La Liga Season 2526.csv"

# Load data
# Note: SP1 CSV has a BOM character at the start, so using utf-8-sig
df_sp1 = pd.read_csv(sp1_path, encoding='utf-8-sig')
df_target = pd.read_csv(target_path)

# Ensure columns to keep exist in SP1
cols_to_keep = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR', 'B365H', 'B365D', 'B365A']

# Identify new matches
# We'll use a combination of Date, HomeTeam, and AwayTeam as a unique identifier
df_sp1['Match_ID'] = df_sp1['Date'] + "_" + df_sp1['HomeTeam'] + "_" + df_sp1['AwayTeam']
df_target['Match_ID'] = df_target['Date'] + "_" + df_target['HomeTeam'] + "_" + df_target['AwayTeam']

new_matches = df_sp1[~df_sp1['Match_ID'].isin(df_target['Match_ID'])].copy()

if new_matches.empty:
    print("No new matches found to update.")
else:
    print(f"Found {len(new_matches)} new matches.")
    
    # Add Referee column as 'Unknown'
    new_matches['Referee'] = "Unknown"
    
    # Reorder columns to match target
    final_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR', 'Referee', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR', 'B365H', 'B365D', 'B365A']
    
    new_matches_clean = new_matches[final_cols]
    
    # Append to target
    updated_target = pd.concat([df_target.drop(columns=['Match_ID']), new_matches_clean], ignore_index=True)
    
    # Save back
    updated_target.to_csv(target_path, index=False)
    print(f"Successfully updated {target_path} with {len(new_matches)} new matches.")
