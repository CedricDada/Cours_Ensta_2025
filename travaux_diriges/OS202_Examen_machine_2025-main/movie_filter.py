# Ce programme va charger n images et y appliquer un filtre de netteté
# puis les sauvegarder dans un dossier de sortie

from PIL import Image
import os
import numpy as np
from scipy import signal
from mpi4py import MPI
from time import time

# Initialisation de MPI
comm   = MPI.COMM_WORLD
rank   = comm.Get_rank()
nbp    = comm.Get_size()
NBRE_FRAMES = 37


# Fonction pour appliquer un filtre de netteté à une image
def apply_filter(image):
    # On charge l'image
    img = Image.open(image)
    print(f"Taille originale {img.size}")
    # Conversion en HSV :
    img = img.convert('HSV')
    # On convertit l'image en tableau numpy et on normalise
    img = np.repeat(np.repeat(np.array(img), 2, axis=0), 2, axis=1)
    img = np.array(img, dtype=np.double)/255.
    # Tout d'abord, on crée un masque de flou gaussien
    mask = np.array([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]]) / 16.
    # On applique le filtre de flou
    blur_image = np.zeros_like(img, dtype=np.double)
    for i in range(3):
        blur_image[:,:,i] = signal.convolve2d(img[:,:,i], mask, mode='same')
    # On crée un masque de netteté
    mask = np.array([[0., -1., 0.], [-1., 5., -1.], [0., -1., 0.]])
    # On applique le filtre de netteté
    sharpen_image = np.zeros_like(img)
    sharpen_image[:,:,:2] = blur_image[:,:,:2]
    sharpen_image[:,:,2] = np.clip(signal.convolve2d(blur_image[:,:,2], mask, mode='same'), 0., 1.)

    sharpen_image *= 255.
    sharpen_image = sharpen_image.astype(np.uint8)
    # On retourne l'image modifiée
    return Image.fromarray(sharpen_image, 'HSV').convert('RGB')


path = "datas/perroquets/"
# On crée un dossier de sortie
if not os.path.exists("sorties/perroquets"):
    os.makedirs("sorties/perroquets")
out_path = "sorties/perroquets/"

output_local_images = []

# Découpage de l’image par lignes
chunk = NBRE_FRAMES // nbp
rest  = NBRE_FRAMES % nbp
start = rank * chunk + min(rank, rest)
end   = start + chunk + (1 if rank < rest else 0)
deb_local = time()
for i in range(start, end):
    image = path + "Perroquet{:04d}.jpg".format(i+1)
    sharpen_image = apply_filter(image)
    # On sauvegarde l'image modifiée
    output_local_images.append(sharpen_image)
    print(f"Image {i+1} traitée")
print("Traitement terminé")

# On sauvegarde les images modifiées
for i, img in enumerate(output_local_images):
    img.save(out_path + "Perroquet{:04d}.jpg".format(i+1+start))
print("Images sauvegardées")
fin_local = time()
duration_local = fin_local - deb_local
durations = comm.gather(duration_local, root=0)
if rank == 0:
    print("\nTemps de calcul par processus (secondes) :")
    for i, d in enumerate(durations):
        print(f"  - Processus {i}: {d:.3f}")
    max_time = max(durations)
    min_time = min(durations)
    imbalance_ratio = (max_time - min_time) / max_time * 100
    print(f"\nDéséquilibre max/min : {max_time:.3f} s vs {min_time:.3f} s")
    print(f"Ratio de déséquilibrage : {imbalance_ratio:.1f}%")
