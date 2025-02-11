#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char* argv[]) {
    int rank, nbp, d;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nbp);

    // Determination de la dimension d :
    // Si un argument est fourni, on l'utilise ; sinon, on deduit d a partir du nb de processus.
    if (argc > 1) {
        d = atoi(argv[1]);
    } else {
        // Deduire d si nbp est une puissance de 2.
        d = 0;
        int tmp = nbp;
        while (tmp > 1) {
            if (tmp % 2 != 0) {
                if (rank == 0)
                    fprintf(stderr, "Erreur : Le nombre de processus doit etre une puissance de 2.\n");
                MPI_Finalize();
                exit(EXIT_FAILURE);
            }
            tmp /= 2;
            d++;
        }
    }

    // Verification : le nombre de processus doit etre 2^d.
    if (nbp != (1 << d)) {
        if (rank == 0)
            fprintf(stderr, "Erreur : Pour un hypercube de dimension %d, il faut 2^d processus (ici %d processus).\n", d, nbp);
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    int token;  // le jeton a diffuser
    double tdeb, tfin;
    
    // Mesurer le temps de diffusion
    tdeb = MPI_Wtime();

    // 1. Cas de l'hypercube de dimension 1 (2 processus) :
    //    - Si d vaut 1, alors seul 2 processus sont utilises.
    //    - Le processus 0 initialise et envoie au processus 1.
    // Ce meme algorithme fonctionne pour d = 2, d = 3 et d = d general.
    if (rank == 0) {
        token = 12345;  // valeur choisie par le programmeur
    }
    
    // Diffusion dans un hypercube en d etapes.
    // Pour chaque etape i, chaque processus echange avec son partenaire defini par rank xor (1 << i).
    for (int i = 0; i < d; i++) {
        int partner = rank ^ (1 << i);
        if (rank < partner) {
            // Le processus avec un rang plus faible envoie d'abord
            MPI_Send(&token, 1, MPI_INT, partner, 0, MPI_COMM_WORLD);
        } else {
            // Celui avec le rang plus eleve recoit
            MPI_Recv(&token, 1, MPI_INT, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    
    tfin = MPI_Wtime();

    // Affichage : chaque processus affiche la valeur recue et le temps de diffusion.
    printf("Processus %d : token = %d, temps de diffusion = %f secondes\n", rank, token, tfin - tdeb);

    MPI_Finalize();
    return 0;
}
