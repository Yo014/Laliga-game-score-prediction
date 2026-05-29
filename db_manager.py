import sqlite3
import pandas as pd
import os

DB_PATH = 'laliga.db'

def get_connection():
    """Returns a sqlite3 connection to the database."""
    return sqlite3.connect(DB_PATH)

def init_db():
    """Initializes database tables if they do not exist."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # We will let pandas define schemas dynamically upon data insertion,
    # but we can initialize tables or indices here to ensure database robustness.
    try:
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_squad_players_team ON squad_players(Team);")
    except sqlite3.OperationalError:
        pass
        
    try:
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_seasons_date ON seasons(Date);")
    except sqlite3.OperationalError:
        pass
        
    try:
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_processed_matches_date ON processed_matches(Date);")
    except sqlite3.OperationalError:
        pass
    
    conn.commit()
    conn.close()

def save_to_db(df, table_name, if_exists='replace'):
    """Saves a pandas DataFrame to a specified SQLite table."""
    conn = get_connection()
    df.to_sql(table_name, conn, if_exists=if_exists, index=False)
    conn.close()

def query_db(query, params=None):
    """Executes a query and returns a pandas DataFrame."""
    conn = get_connection()
    if params is None:
        params = []
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df

def get_squad_value_data():
    """
    Simulates load_squad_value_data using SQL.
    Returns a DataFrame with 'Team' and 'Total_Squad_Value'.
    """
    # Load all players to aggregate values
    players_df = query_db("SELECT Team, [Market Value] FROM squad_players;")
    
    # Inline parse_market_value functionality to simulate feature_engeneering.py
    from feature_engeneering import parse_market_value, SQUAD_NAME_MAP
    
    players_df['Total_Squad_Value'] = players_df['Market Value'].apply(parse_market_value)
    
    # Group by Team and sum
    squad_values = players_df.groupby('Team')['Total_Squad_Value'].sum().reset_index()
    squad_values['Team'] = squad_values['Team'].apply(lambda x: SQUAD_NAME_MAP.get(x, x))
    return squad_values
