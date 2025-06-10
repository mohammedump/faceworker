# convert_pgm_to_png.py

import os
import cv2

input_folder = "data/faces"
output_folder = "data/faces_png"

os.makedirs(output_folder, exist_ok=True)

for root, _, files in os.walk(input_folder):
    for file in files:
        if file.endswith(".pgm"):
            img = cv2.imread(os.path.join(root, file), cv2.IMREAD_GRAYSCALE)

            # Créer le sous-dossier dans faces_png (ex: s1, s2...)
            relative_path = os.path.relpath(root, input_folder)
            target_dir = os.path.join(output_folder, relative_path)
            os.makedirs(target_dir, exist_ok=True)

            # Nom final du fichier .png
            name = os.path.splitext(file)[0] + ".png"
            output_path = os.path.join(target_dir, name)

            # Sauvegarder l’image convertie
            cv2.imwrite(output_path, img)

print("✅ Conversion PGM → PNG terminée.")
