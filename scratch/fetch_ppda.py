import urllib.request
import urllib.error
import ssl
import re
import json
import pandas as pd
import os

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

seasons = {
    '15-16': '2015',
    '16-17': '2016',
    '17-18': '2017',
    '1819': '2018',
    '1920': '2019',
    '2021': '2020',
    '21-22': '2021',
    '22-23': '2022',
    '23-24': '2023',
    '2425': '2024',
    '2526': '2025'
}

all_match_data = []

for season_str, year in seasons.items():
    print(f"Fetching data for season {season_str} (Year {year})...")
    url = f"https://understat.com/main/getLeagueData/La_liga/{year}"
    headers = {
        'User-Agent': 'Mozilla/5.0',
        'X-Requested-With': 'XMLHttpRequest'
    }
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, context=ctx) as response:
            if response.info().get('Content-Encoding') == 'gzip':
                import gzip
                json_text = gzip.decompress(response.read()).decode('utf-8')
            else:
                json_text = response.read().decode('utf-8')
        
        data = json.loads(json_text)
        if 'teams' in data:
            teams_data = data['teams']
            for team_id, team_info in teams_data.items():
                team_name = team_info['title']
                for game in team_info['history']:
                    all_match_data.append({
                        'Season': season_str,
                        'Team': team_name,
                        'Date': game['date'],
                        'h_a': game['h_a'],
                        'PPDA': game['ppda']['att'] / game['ppda']['def'] if game['ppda']['def'] > 0 else 0,
                        'OPPDA': game['ppda_allowed']['att'] / game['ppda_allowed']['def'] if game['ppda_allowed']['def'] > 0 else 0
                    })
        else:
            print(f"Failed to find 'teams' in JSON for {year}")
    except Exception as e:
        print(f"Error fetching {year}: {e}")

df = pd.DataFrame(all_match_data)
output_path = os.path.join(os.path.dirname(__file__), 'all_seasons_ppda.csv')
df.to_csv(output_path, index=False)
print(f"Saved PPDA data to {output_path}")
