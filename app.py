import os
import sys
import subprocess
import threading
import pandas as pd
from flask import Flask, jsonify, request, render_template, send_from_directory
import db_manager
import predict
import build_squad_health

app = Flask(__name__, template_folder='templates', static_folder='static')

# Map canonical model name to Laliga Squads folder name
MODEL_TO_FOLDER_NAME = {
    'Alaves': 'Alavés',
    'Ath Bilbao': 'Athletic Bilbao',
    'Ath Madrid': 'Atlético Madrid',
    'Barcelona': 'Barcelona',
    'Celta': 'Celta Vigo',
    'Elche': 'Elche',
    'Espanol': 'Espanyol',
    'Getafe': 'Getafe',
    'Girona': 'Girona',
    'Levante': 'Levante',
    'Mallorca': 'Mallorca',
    'Osasuna': 'Osasuna',
    'Vallecano': 'Rayo Vallecano',
    'Betis': 'Real Betis',
    'Real Madrid': 'Real Madrid',
    'Oviedo': 'Real Oviedo',
    'Sociedad': 'Real Sociedad',
    'Sevilla': 'Sevilla',
    'Valencia': 'Valencia',
    'Villarreal': 'Villarreal',
}

# Real-time process logging store
script_logs = {}
script_status = {}
processes = {}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SQUADS_DIR = os.path.join(BASE_DIR, 'Laliga Squads')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/teams', methods=['GET'])
def get_teams():
    """Return team names and their aggregated squad health metrics."""
    squad_health_path = os.path.join(BASE_DIR, 'current_squad_health.csv')
    try:
        if os.path.exists(squad_health_path):
            df = pd.read_csv(squad_health_path)
            # Ensure all numeric cols are serializable
            df = df.fillna(0)
            teams_data = df.to_dict(orient='records')
            return jsonify({
                "success": True,
                "teams": teams_data
            })
        else:
            return jsonify({
                "success": False,
                "error": "current_squad_health.csv not found. Please run the injury aggregation script first."
            })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/referees', methods=['GET'])
def get_referees():
    """Extract list of historical referees from database or processed matches CSV."""
    try:
        # Attempt DB query first
        try:
            ref_df = db_manager.query_db("SELECT DISTINCT Referee FROM processed_matches ORDER BY Referee;")
            referees = ref_df['Referee'].dropna().tolist()
        except Exception:
            # Fallback to Processed_Matches.csv
            csv_path = os.path.join(BASE_DIR, 'Processed_Matches.csv')
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                referees = df['Referee'].dropna().unique().tolist()
                referees.sort()
            else:
                referees = ["Unknown"]
        
        # Filter out empty strings
        referees = [r for r in referees if str(r).strip() != '']
        if "Unknown" not in referees:
            referees.append("Unknown")
            
        return jsonify({"success": True, "referees": referees})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/predict', methods=['POST'])
def run_prediction():
    """Execute prediction model on user-defined matchup parameters."""
    try:
        data = request.json
        home_team = data.get('home_team')
        away_team = data.get('away_team')
        home_rest = int(data.get('home_rest_days', 6))
        away_rest = int(data.get('away_rest_days', 6))
        b365h = float(data.get('b365h', 2.0))
        b365d = float(data.get('b365d', 3.0))
        b365a = float(data.get('b365a', 3.0))
        referee = data.get('referee', 'Unknown')

        if not home_team or not away_team:
            return jsonify({"success": False, "error": "Both Home Team and Away Team must be provided."})

        if home_team == away_team:
            return jsonify({"success": False, "error": "Home and Away teams must be different."})

        # Run prediction
        result = predict.predict_match(
            home_team=home_team,
            away_team=away_team,
            home_rest_days=home_rest,
            away_rest_days=away_rest,
            b365h=b365h,
            b365d=b365d,
            b365a=b365a,
            referee_name=referee
        )

        if result:
            return jsonify({"success": True, "data": result})
        else:
            return jsonify({"success": False, "error": "Prediction failed. Verify team names spelling or check model status."})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/squad/<team_name>', methods=['GET'])
def get_squad(team_name):
    """Retrieve player-level roster data for a specific team."""
    folder_name = MODEL_TO_FOLDER_NAME.get(team_name)
    if not folder_name:
        return jsonify({"success": False, "error": f"Squad folder mapping not found for team: {team_name}"})

    csv_path = os.path.join(SQUADS_DIR, folder_name, 'player_data.csv')
    try:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # Replaces NaNs with empty string or logical defaults
            df = df.fillna({
                'Appearances': 'Not used during this season',
                'Goals Scored': 0,
                'Injuries': 0,
                'Day Injured': '',
                'Missed Games': 0,
                'Expected Return': '',
                'Market Value': 'Unknown',
                'Gls': 0,
                'Ast': 0,
                'G+A': 0,
                'G-PK': 0,
                'PK': 0,
                'PKatt': 0,
                'CrdY': 0,
                'CrdR': 0
            })
            players = df.to_dict(orient='records')
            return jsonify({
                "success": True,
                "team": team_name,
                "players": players
            })
        else:
            return jsonify({"success": False, "error": f"player_data.csv for {team_name} does not exist at {csv_path}"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/squad/<team_name>/update', methods=['POST'])
def update_squad(team_name):
    """Save inline edits of player roster (injuries, return dates) and trigger health re-aggregation."""
    folder_name = MODEL_TO_FOLDER_NAME.get(team_name)
    if not folder_name:
        return jsonify({"success": False, "error": f"Squad folder mapping not found for team: {team_name}"})

    csv_path = os.path.join(SQUADS_DIR, folder_name, 'player_data.csv')
    try:
        updated_players = request.json.get('players', [])
        if not updated_players:
            return jsonify({"success": False, "error": "No player data provided for update."})

        if not os.path.exists(csv_path):
            return jsonify({"success": False, "error": "Target player_data.csv file not found."})

        # Load existing file to preserve other unmodified columns
        df = pd.read_csv(csv_path)

        # Update player columns
        for player_update in updated_players:
            p_name = player_update.get('Player')
            # Locate player row index
            idx = df[df['Player'] == p_name].index
            if not idx.empty:
                # Update specific editable fields
                df.at[idx[0], 'Injuries'] = int(player_update.get('Injuries', 0))
                df.at[idx[0], 'Day Injured'] = player_update.get('Day Injured', '')
                df.at[idx[0], 'Missed Games'] = int(player_update.get('Missed Games', 0))
                df.at[idx[0], 'Expected Return'] = player_update.get('Expected Return', '')

        # Write back to CSV
        df.to_csv(csv_path, index=False)

        # Trigger squad health re-aggregation
        build_squad_health.build_squad_health()

        return jsonify({
            "success": True,
            "message": f"Squad player details updated and squad health re-aggregated for {team_name}."
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/run-script', methods=['POST'])
def run_script():
    """Spawn a background subprocess executing a chosen pipeline script."""
    script_name = request.json.get('script')
    allowed_scripts = [
        'populate_db.py',
        'Data_processing.py',
        'build_squad_health.py',
        'feature_engeneering.py',
        'train_model.py'
    ]

    if not script_name or script_name not in allowed_scripts:
        return jsonify({"success": False, "error": "Invalid or unauthorized script name."})

    # If already running, prevent spawning another
    if script_status.get(script_name) == 'running':
        return jsonify({"success": False, "error": f"Script {script_name} is already executing."})

    script_path = os.path.join(BASE_DIR, script_name)
    if not os.path.exists(script_path):
        return jsonify({"success": False, "error": f"Script {script_name} not found at {script_path}."})

    # Start script async
    def run_process():
        script_status[script_name] = 'running'
        script_logs[script_name] = []
        
        script_logs[script_name].append(f"=== Starting {script_name} ===\n")
        cmd = [sys.executable, script_path]
        
        try:
            # Set environment variables if needed
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1' # Ensure stdout flushed immediately
            
            p = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=BASE_DIR,
                env=env
            )
            processes[script_name] = p
            
            for line in iter(p.stdout.readline, ''):
                script_logs[script_name].append(line)
                
            p.stdout.close()
            return_code = p.wait()
            
            if return_code == 0:
                script_status[script_name] = 'completed'
                script_logs[script_name].append(f"\n=== Finished {script_name} Successfully ===\n")
            else:
                script_status[script_name] = 'failed'
                script_logs[script_name].append(f"\n=== Script Failed with exit code {return_code} ===\n")
                
        except Exception as err:
            script_status[script_name] = 'failed'
            script_logs[script_name].append(f"\nError executing script: {str(err)}\n")

    t = threading.Thread(target=run_process)
    t.daemon = True
    t.start()

    return jsonify({"success": True, "message": f"{script_name} execution started in background."})

@app.route('/api/script-logs/<script_name>', methods=['GET'])
def get_script_logs(script_name):
    """Retrieve full logs and current process status for a script."""
    logs = script_logs.get(script_name, [])
    status = script_status.get(script_name, 'idle')
    return jsonify({
        "success": True,
        "status": status,
        "logs": "".join(logs)
    })

if __name__ == '__main__':
    # Launch local GUI
    print("--------------------------------------------------")
    print("La Liga Prediction GUI Dashboard Local Web Server")
    print("Address: http://127.0.0.1:5000")
    print("Press Ctrl+C to terminate the local server.")
    print("--------------------------------------------------")
    
    # Optional auto-browser launch (standard utility)
    import webbrowser
    def open_browser():
        webbrowser.open_new('http://127.0.0.1:5000/')
        
    threading.Timer(1.5, open_browser).start()
    
    # Run server
    app.run(host='127.0.0.1', port=5000, debug=False)
