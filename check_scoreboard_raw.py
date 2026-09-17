import requests

url = "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json"
headers = {"User-Agent": "Mozilla/5.0"}
resp = requests.get(url, headers=headers)
print("Status code:", resp.status_code)
print("Response length:", len(resp.text))
print("First 500 chars of response:")
print(resp.text[:500])