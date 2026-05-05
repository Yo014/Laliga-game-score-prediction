import pandas as pd
ppda_df = pd.read_csv('scratch/all_seasons_ppda.csv')
understat_teams = sorted(set(ppda_df['Team'].unique()))
print(understat_teams)
