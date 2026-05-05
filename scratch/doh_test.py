import requests
doh_url = "https://dns.google/resolve?name=understat.com&type=A"
try:
    resp = requests.get(doh_url)
    print("DoH response:", resp.json())
except Exception as e:
    print("Error:", e)
