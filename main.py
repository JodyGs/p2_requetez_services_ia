import os
import requests
from PIL import Image
from io import BytesIO
import base64
from dotenv import load_dotenv
import time

load_dotenv()

token = os.getenv("HF_API_TOKEN")
headers = {"Authorization": f"Bearer {token}"}

image_folder = "asset/IMG"
image_files = [f for f in os.listdir(image_folder) if f.endswith(".png")]

def send_image(image_path):
    with open(image_path, 'rb') as img_file:
        image_data = img_file.read()

    url = "https://api-inference.huggingface.co/models/mattmdjaga/segformer_b2_clothes"
    response = requests.post(url, headers=headers, data=image_data)

    time.sleep(5)

    if response.status_code == 200:
        print(f"[+] Image {image_path} traitée avec succès!")

        result = response.json()
        print(f"[+] Nombre de masques : {len(result)}")

        output_dir = "asset/IMG/result"
        os.makedirs(output_dir, exist_ok=True)

        for i, item in enumerate(result):
            if "mask" in item:
                mask_b64 = item["mask"]
                mask_bytes = base64.b64decode(mask_b64)
                mask_image = Image.open(BytesIO(mask_bytes))

                base_name = os.path.basename(image_path)
                name, _ = os.path.splitext(base_name)
                label = item.get('label', f"mask_{i}")
                output_path = os.path.join(output_dir, f"{name}_{label}.png")

                mask_image.save(output_path)
                print(f"[+] Masque {i} enregistré sous : {output_path}")
            else:
                print(f"[!] Aucun masque trouvé dans l'élément {i} :", item)
    else:
        print(f"[-] Erreur lors de l'envoi de l'image {image_path}: {response.status_code}")
        print("Message d'erreur :", response.text)

for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    send_image(image_path)


