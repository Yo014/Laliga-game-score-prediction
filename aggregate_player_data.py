import os
import pandas as pd

def main():
    base_dir = '/Users/santomukiza/Desktop/Github/LaligaPrediction/Laliga-game-score-prediction/Laliga Squads'
    output_path = '/Users/santomukiza/Desktop/Github/LaligaPrediction/Laliga-game-score-prediction/current_squad_health.csv'
    
    # Map the folder names to the names used in your ML pipeline
    name_map = {
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
        'Villarreal': 'Villarreal'
    }
    
    health_data = []
    
    if not os.path.exists(base_dir):
        print(f"Error: Directory {base_dir} not found.")
        return
        
    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        
        if not os.path.isdir(folder_path):
            continue
            
        csv_path = os.path.join(folder_path, 'player_data.csv')
        if not os.path.exists(csv_path):
            continue
            
        df = pd.read_csv(csv_path)
        
        # Ensure appearances are numeric (replace '-' or NaN with 0)
        df['Appearances'] = pd.to_numeric(df['Appearances'], errors='coerce').fillna(0)
        df['Injuries'] = pd.to_numeric(df['Injuries'], errors='coerce').fillna(0)
        
        # Metrics Calculation
        total_squad_apps = df['Appearances'].sum()
        
        # Injured players
        injured_df = df[df['Injuries'] == 1]
        
        # Missing Key Players (Injured AND played > 10 games last season)
        missing_key_players = len(injured_df[injured_df['Appearances'] > 10])
        
        # Missing Impact Percentage
        missing_apps = injured_df['Appearances'].sum()
        missing_impact_pct = (missing_apps / total_squad_apps) if total_squad_apps > 0 else 0
        
        team_ml_name = name_map.get(folder_name, folder_name)
        
        health_data.append({
            'Team': team_ml_name,
            'Squad_Experience': total_squad_apps,
            'Missing_Key_Players': missing_key_players,
            'Missing_Impact_Pct': missing_impact_pct
        })
        
    final_df = pd.DataFrame(health_data)
    final_df.to_csv(output_path, index=False)
    print(f"Aggregated current squad health data saved to {output_path}")
    print(final_df.head(20))

if __name__ == "__main__":
    main()
