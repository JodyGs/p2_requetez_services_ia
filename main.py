from dotenv import load_dotenv
import os
import requests


load_dotenv()

token = os.getenv("HF_API_TOKEN")

headers = {"Authorization": f"Bearer {token}"}
response = requests.get("https://huggingface.co/api/whoami-v2", headers=headers)

if response.status_code == 200:
    print("✅ Token valide !")
    print("Détails :", response.json())
else:
    print("❌ Token invalide ou expiré.")
    print("Code :", response.status_code)
    print("Message :", response.text)