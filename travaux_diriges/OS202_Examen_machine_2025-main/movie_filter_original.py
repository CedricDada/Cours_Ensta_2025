from PIL import Image
import os
import numpy as np
from scipy import signal
from time import time

# Fonction pour appliquer un filtre de netteté à une image
def apply_filter(image):
    start_load = time()
    img = Image.open(image)
    img = img.convert('HSV')
    end_load = time()
    print(f"Chargement {image} en {end_load - start_load:.3f} s")

    start_process = time()
    img = np.repeat(np.repeat(np.array(img), 2, axis=0), 2, axis=1)
    img = np.array(img, dtype=np.double) / 255.

    # Filtre de flou gaussien
    mask = np.array([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]]) / 16.
    blur_image = np.zeros_like(img, dtype=np.double)
    for i in range(3):
        blur_image[:, :, i] = signal.convolve2d(img[:, :, i], mask, mode='same')

    # Filtre de netteté
    mask = np.array([[0., -1., 0.], [-1., 5., -1.], [0., -1., 0.]])
    sharpen_image = np.zeros_like(img)
    sharpen_image[:, :, :2] = blur_image[:, :, :2]
    sharpen_image[:, :, 2] = np.clip(signal.convolve2d(blur_image[:, :, 2], mask, mode='same'), 0., 1.)

    sharpen_image *= 255.
    sharpen_image = sharpen_image.astype(np.uint8)
    end_process = time()
    print(f"Traitement {image} en {end_process - start_process:.3f} s")

    return Image.fromarray(sharpen_image, 'HSV').convert('RGB')

start_total = time()
path = "datas/perroquets/"
if not os.path.exists("sorties/perroquets"):
    os.makedirs("sorties/perroquets")
out_path = "sorties/perroquets/"

output_images = []
for i in range(37):
    image = path + "Perroquet{:04d}.jpg".format(i+1)
    output_images.append(apply_filter(image))
    print(f"Image {i+1} traitée")

start_save = time()
for i, img in enumerate(output_images):
    img.save(out_path + "Perroquet{:04d}.jpg".format(i+1))
end_save = time()
print(f"Sauvegarde des images en {end_save - start_save:.3f} s")

end_total = time()
print(f"Temps total d'exécution : {end_total - start_total:.3f} s")
