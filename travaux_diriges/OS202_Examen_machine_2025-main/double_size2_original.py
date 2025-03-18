from PIL import Image
import os
import numpy as np
from scipy import signal
from time import time

# Fonction pour doubler la taille d'une image sans trop la pixeliser
def double_size(image):
    start_load = time()
    img = Image.open(image)
    img = img.convert('HSV')
    end_load = time()
    print(f"Chargement {image} en {end_load - start_load:.3f} s")

    start_process = time()
    img = np.array(img, dtype=np.double)
    img = np.repeat(np.repeat(img, 2, axis=0), 2, axis=1) / 255.

    # Filtre de flou gaussien pour H et S
    mask = np.array([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]]) / 16.
    blur_image = np.zeros_like(img, dtype=np.double)
    for i in range(2):
        blur_image[:, :, i] = signal.convolve2d(img[:, :, i], mask, mode='same')

    blur_image[:, :, 2] = img[:, :, 2]

    # Filtre 5x5 pour la luminance
    mask = -np.array([[1., 4., 6., 4., 1.], [4., 16., 24., 16., 4.], [6., 24., -476., 24., 6.], [4., 16., 24., 16., 4.], [1., 4., 6., 4., 1.]]) / 256
    blur_image[:, :, 2] = np.clip(signal.convolve2d(blur_image[:, :, 2], mask, mode='same'), 0., 1.)

    blur_image = (255. * blur_image).astype(np.uint8)
    end_process = time()
    print(f"Traitement {image} en {end_process - start_process:.3f} s")

    return Image.fromarray(blur_image, 'HSV').convert('RGB')

start_total = time()
path = "datas/"
image = path + "paysage.jpg"
doubled_image = double_size(image)

start_save = time()
if not os.path.exists("sorties"):
    os.makedirs("sorties")
doubled_image.save("sorties/paysage_double_2.jpg")
end_save = time()
print(f"Sauvegarde de l'image en {end_save - start_save:.3f} s")

end_total = time()
print(f"Temps total d'exécution : {end_total - start_total:.3f} s")
