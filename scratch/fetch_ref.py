import time
import os
import pandas as pd
import requests
from bs4 import BeautifulSoup

# Map your season strings to the FBref year format (YYYY-YYYY)
seasons = {
    '15-16': '2015-2016',
    '16-17': '2016-2017',
    '17-18': '2017-2018',
    '1819': '2018-2019',
    '1920': '2019-2020',
    '2021': '2020-2021',
    '21-22': '2021-2022',
    '22-23': '2022-2023',
    '23-24': '2023-2024',
    '2425': '2024-2025',
    '2526': '2025-2026'
}

all_match_data = []

# A complete set of headers mimicking a real Windows/Chrome user
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
    'Connection': 'keep-alive'
}

script_dir = os.path.dirname(os.path.abspath(__file__))

for season_str, year_str in seasons.items():
    html = None
    
    # 1. Try local file for 2526
    if season_str == '2526':
        local_path = os.path.join(script_dir, 'fbref_debug.html')
        if os.path.exists(local_path):
            print(f"Reading season {season_str} from local debug file...")
            with open(local_path, 'r', encoding='utf-8', errors='ignore') as f:
                html = f.read()
                
    # 2. If not read locally, fetch from Wayback Machine to bypass 403 Forbidden blocks
    if not html:
        orig_url = f"https://fbref.com/en/comps/12/{year_str}/schedule/{year_str}-La-Liga-Scores-and-Fixtures"
            
        url = f"http://web.archive.org/web/{orig_url}"
        print(f"Fetching data for season {season_str} (Year {year_str}) via Wayback Machine...")
        
        # Retry logic with exponential backoff to handle rate limits or network issues
        max_retries = 5
        retry_delay = 2
        for attempt in range(1, max_retries + 1):
            try:
                response = requests.get(url, headers=headers, timeout=15)
                if response.status_code == 200:
                    # Explicitly set encoding to utf-8 to preserve Spanish accent characters
                    response.encoding = 'utf-8'
                    html = response.text
                    break
                else:
                    print(f"  [Attempt {attempt}/{max_retries}] HTTP Status {response.status_code}. Retrying...")
            except Exception as e:
                print(f"  [Attempt {attempt}/{max_retries}] Connection Error: {e}. Retrying...")
            
            time.sleep(retry_delay)
            retry_delay *= 2 # Exponential backoff
            
    if not html:
        print(f"Failed to obtain HTML for season {season_str} (Year {year_str})")
        continue

    try:
        soup = BeautifulSoup(html, 'html.parser')
        
        # Look for stats_table class
        table = soup.find('table', {'class': 'stats_table'})
        if not table:
            print(f"Schedule table not found on page for {season_str}.")
            continue
            
        tbody = table.find('tbody')
        if not tbody:
            print(f"Tbody not found on table for {season_str}.")
            continue
            
        rows = tbody.find_all('tr')
        season_matches = 0
        
        for row in rows:
            if 'spacer' in row.get('class', []) or 'partial_table' in row.get('class', []):
                continue
                
            date_td = row.find(['td', 'th'], {'data-stat': 'date'})
            home_td = row.find('td', {'data-stat': 'home_team'})
            away_td = row.find('td', {'data-stat': 'away_team'})
            ref_td = row.find('td', {'data-stat': 'referee'})
            
            if date_td and home_td and away_td and ref_td:
                date_str = date_td.get_text(strip=True)
                home_team = home_td.get_text(strip=True)
                away_team = away_td.get_text(strip=True)
                referee = ref_td.get_text(strip=True)
                
                if date_str and home_team and away_team:
                    all_match_data.append({
                        'Season': season_str,
                        'Date': date_str,
                        'HomeTeam': home_team,
                        'AwayTeam': away_team,
                        'Referee': referee
                    })
                    season_matches += 1
                    
        print(f"Successfully scraped {season_matches} matches for season {season_str}.")
        
        # Be a good citizen, sleep slightly between network fetches
        time.sleep(2)
        
    except Exception as e:
        print(f"Error parsing data for season {season_str}: {e}")

# Save the dataset
if all_match_data:
    df = pd.DataFrame(all_match_data)
    output_path = os.path.join(script_dir, 'all_seasons_referees.csv')
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\nDone. Successfully saved {len(df)} referee rows to {output_path}")
else:
    print("\nError: No match data scraped!")