import app
import json

def test_endpoints():
    print("--- Starting Local App Endpoint Tests ---")
    
    # 1. Test teams endpoint
    client = app.app.test_client()
    res = client.get('/api/teams')
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    data = json.loads(res.data)
    assert data['success'] is True, "Expected success to be True"
    assert len(data['teams']) > 0, "Expected at least one team in the list"
    print(f"✔ Teams API verified. Retained {len(data['teams'])} teams.")

    # 2. Test referees endpoint
    res = client.get('/api/referees')
    assert res.status_code == 200
    data = json.loads(res.data)
    assert data['success'] is True
    assert len(data['referees']) > 0
    print(f"✔ Referees API verified. Retained {len(data['referees'])} refs.")

    # 3. Test squad lookup
    test_team = "Barcelona"
    res = client.get(f'/api/squad/{test_team}')
    assert res.status_code == 200
    data = json.loads(res.data)
    assert data['success'] is True
    assert len(data['players']) > 0
    print(f"✔ Squad Roster API verified for {test_team}. Found {len(data['players'])} players.")

    # 4. Test prediction logic
    payload = {
        "home_team": "Barcelona",
        "away_team": "Real Madrid",
        "home_rest_days": 6,
        "away_rest_days": 6,
        "b365h": 2.10,
        "b365d": 3.40,
        "b365a": 3.20,
        "referee": "Unknown"
    }
    res = client.post('/api/predict', data=json.dumps(payload), content_type='application/json')
    assert res.status_code == 200
    data = json.loads(res.data)
    assert data['success'] is True, f"Prediction failed: {data.get('error')}"
    pred_data = data['data']
    assert 'probabilities' in pred_data
    assert 'prediction' in pred_data
    print(f"✔ Prediction Model API verified. Predicted: {pred_data['prediction']}")
    print(f"  probabilities: {pred_data['probabilities']}")

    print("\n--- All GUI API tests passed successfully! ---")

if __name__ == '__main__':
    test_endpoints()
