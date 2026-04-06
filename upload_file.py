import requests
import json

file_path = "deprem_vizyon_v6.zip"
url = "https://file.io"

try:
    with open(file_path, 'rb') as f:
        response = requests.post(url, files={'file': f})
        data = response.json()
        print("--- UPLOAD SUCCESS ---")
        print(f"LINK: {data.get('link')}")
        print(f"EXPIRY: {data.get('expiry')}")
except Exception as e:
    print(f"--- UPLOAD FAILED ---")
    print(e)
