import requests
import re
import json

url = "https://understat.com/team/Real_Madrid/2015"
response = requests.get(url)
match = re.search(r"var datesData \= JSON\.parse\('([^']+)'\)", response.text)
if match:
    decoded_data = match.group(1).encode('utf-8').decode('unicode_escape')
    data = json.loads(decoded_data)
    print("Total matches:", len(data))
    if len(data) > 0:
        print("First match sample:")
        print(json.dumps(data[0], indent=2))
        print("PPDA keys:", data[0].get("ppda"))
else:
    print("Could not find datesData")
