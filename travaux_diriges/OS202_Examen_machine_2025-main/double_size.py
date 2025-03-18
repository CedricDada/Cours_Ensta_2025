from mpi4py import MPI
from PIL import Image
import numpy as np
from scipy import signal
import os
from time import time

# Fonction qui double la taille d'un morceau d'image (en array) sans trop la pixeliser.
def double_size_array(img_array):
    # On double la taille : répétition des pixels sur chaque axe
    # (la division par 255. est effectuée après la répétition, comme dans le code original)
    doubled = np.repeat(np.repeat(img_array, 2, axis=0), 2, axis=1) / 255.
    print("Taille du morceau après doublement :", doubled.shape)
    
    # Application du flou gaussien
    mask = np.array([[1., 2., 1.],
                     [2., 4., 2.],
                     [1., 2., 1.]]) / 16.
    blur_image = np.zeros_like(doubled, dtype=np.double)
    for i in range(3):
        blur_image[:, :, i] = signal.convolve2d(doubled[:, :, i], mask, mode='same')
    
    # Application du filtre de netteté sur le canal de luminance (ici, on traite le 3ème canal)
    mask_sharpen = np.array([[0, -1, 0],
                             [-1, 5, -1],
                             [0, -1, 0]])
    sharpen_image = np.zeros_like(blur_image, dtype=np.double)
    sharpen_image[:, :, :2] = blur_image[:, :, :2]
    sharpen_image[:, :, 2] = np.clip(signal.convolve2d(blur_image[:, :, 2], mask_sharpen, mode='same'), 0., 1.)
    
    # Revenir à une échelle [0,255] et au type uint8
    sharpen_image = (255. * sharpen_image).astype(np.uint8)
    # Conversion de l'image du format HSV en RGB
    return Image.fromarray(sharpen_image, 'HSV').convert('RGB')

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Le processus 0 charge l'image et la découpe par lignes
    if rank == 0:
        # Chargement de l'image et conversion en HSV
        img = Image.open("datas/paysage.jpg")
        print("Taille originale de l'image :", img.size)
        img = img.convert('HSV')
        img_array = np.array(img, dtype=np.double)
        
        # Découpage de l'image par lignes
        height = img_array.shape[0]
        # Calcul des tailles de tranche pour chaque processus
        counts = [height // size + (1 if i < (height % size) else 0) for i in range(size)]
        starts = [sum(counts[:i]) for i in range(size)]
        chunks = [img_array[starts[i]:starts[i]+counts[i], :, :] for i in range(size)]
    else:
        chunks = None

    # Diffusion (scatter) des morceaux d'image
    local_chunk = comm.scatter(chunks, root=0)

    # Chaque processus applique la fonction double_size sur sa partie
    deb = time()
    local_result_img = double_size_array(local_chunk)
    fin = time()
    print(f"Processus {rank} a traité son morceau en {fin-deb:.3f} s")
    
    # Conversion du résultat en array pour le rassemblement
    local_result_array = np.array(local_result_img)

    # Rassemblement des résultats sur le processus 0
    gathered = comm.gather(local_result_array, root=0)

    if rank == 0:
        # Reconstitution de l'image finale par concaténation verticale
        final_array = np.vstack(gathered)
        final_image = Image.fromarray(final_array)
        # Création du dossier de sortie si nécessaire
        output_dir = "sorties"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        out_path = os.path.join(output_dir, "paysage_double_2.jpg")
        final_image.save(out_path)
        print("Image sauvegardée dans", out_path)

if __name__ == "__main__":
    main()
