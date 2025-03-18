from mpi4py import MPI
from PIL import Image
import numpy as np
from scipy import signal
import os
from time import time

def process_chunk(chunk):
    """
    Applique le filtrage correspondant à double_size2 sur un morceau d'image (en array).
    Le morceau est déjà doublé en taille et normalisé (valeurs entre 0 et 1).
    """
    # On prépare un tableau pour stocker le résultat
    processed = np.zeros_like(chunk, dtype=np.double)
    
    # Masque de flou gaussien 3x3 pour la teinte et la saturation (H et S)
    mask3 = np.array([[1., 2., 1.],
                      [2., 4., 2.],
                      [1., 2., 1.]]) / 16.
    
    # Application du flou sur les deux premiers canaux
    for i in range(2):
        processed[:, :, i] = signal.convolve2d(chunk[:, :, i], mask3, mode='same')
    # Pour le troisième canal (luminance), on copie d'abord les valeurs
    processed[:, :, 2] = chunk[:, :, 2]
    
    # Masque de filtre 5x5 pour le canal de luminance
    mask5 = -np.array([[1., 4., 6., 4., 1.],
                       [4., 16., 24., 16., 4.],
                       [6., 24., -476., 24., 6.],
                       [4., 16., 24., 16., 4.],
                       [1., 4., 6., 4., 1.]]) / 256.
    # Application du filtre sur le canal 2
    processed[:, :, 2] = np.clip(signal.convolve2d(processed[:, :, 2], mask5, mode='same'), 0., 1.)
    
    # Revenir à l'échelle [0,255] et conversion en uint8
    processed = (255. * processed).astype(np.uint8)
    return processed

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    start_total = time()  # Début du programme
    
    # Seul le processus 0 charge l'image, la convertit, la double et la découpe en tranches
    if rank == 0:
        start_load = time()
        image_path = "datas/paysage.jpg"
        img = Image.open(image_path)
        print("Taille originale de l'image :", img.size)
        img = img.convert('HSV')
        img_array = np.array(img, dtype=np.double)
        # Doublement de la taille et normalisation (valeurs entre 0 et 1)
        doubled = np.repeat(np.repeat(img_array, 2, axis=0), 2, axis=1) / 255.
        print("Nouvelle taille après doublement :", doubled.shape)
        end_load = time()
        print(f"Temps de chargement et prétraitement : {end_load - start_load:.3f} s")
        
        # Découpage par lignes en autant de morceaux que de processus
        total_rows = doubled.shape[0]
        counts = [total_rows // size + (1 if i < (total_rows % size) else 0) for i in range(size)]
        starts = [sum(counts[:i]) for i in range(size)]
        chunks = [doubled[starts[i]:starts[i]+counts[i], :, :] for i in range(size)]
    else:
        chunks = None

    comm.barrier()
    start_scatter = time()
    # Distribution (scatter) des morceaux à tous les processus
    local_chunk = comm.scatter(chunks, root=0)
    comm.barrier()
    end_scatter = time()
    
    if rank == 0:
        print(f"Temps de distribution des données (scatter) : {end_scatter - start_scatter:.3f} s")

    # Chaque processus applique le traitement sur sa tranche
    start_processing = time()
    local_processed = process_chunk(local_chunk)
    end_processing = time()
    
    print(f"Processus {rank} a traité {local_chunk.shape[0]} lignes en {end_processing - start_processing:.3f} s")
    
    comm.barrier()
    start_gather = time()
    # Rassemblement (gather) des résultats sur le processus 0
    gathered = comm.gather(local_processed, root=0)
    comm.barrier()
    end_gather = time()
    
    if rank == 0:
        print(f"Temps de rassemblement des données (gather) : {end_gather - start_gather:.3f} s")

        # Reconstitution de l'image finale par concaténation verticale
        final_array = np.vstack(gathered)
        # Conversion de l'image depuis le mode HSV vers RGB
        final_img = Image.fromarray(final_array, 'HSV').convert('RGB')
        
        start_save = time()
        # Création du dossier de sortie si nécessaire
        output_dir = "sorties"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, "paysage_double_2.jpg")
        final_img.save(output_path)
        end_save = time()
        
        print("Image sauvegardée dans", output_path)
        print(f"Temps de sauvegarde de l'image : {end_save - start_save:.3f} s")
    
    end_total = time()
    if rank == 0:
        print(f"Temps total d'exécution : {end_total - start_total:.3f} s")

if __name__ == "__main__":
    main()
