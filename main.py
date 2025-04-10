from dotenv import load_dotenv
import os
import requests
from PIL import Image

load_dotenv()
token = os.getenv("HF_API_TOKEN")

headers = {"Authorization": f"Bearer {token}"}
image_folder = "asset/IMG/"
image_files = [f for f in os.listdir(image_folder) if f.endswith(('png', 'jpg', 'jpeg'))]

def send_image(image_path):
    with open(image_path, 'rb') as img_file:
        image_data = img_file.read()

    url = "https://api-inference.huggingface.co/models/mattmdjaga/segformer_b2_clothes"
    response = requests.post(url, headers=headers, data=image_data)  # <-- data et non files

    if response.status_code == 200:
        print(f"[+] Image {image_path} traitée avec succès!")
        print("Réponse :", response.json())
    else:
        print(f"[-] Erreur lors de l'envoi de l'image {image_path}: {response.status_code}")
        print("Message d'erreur :", response.text)


for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    send_image(image_path)
