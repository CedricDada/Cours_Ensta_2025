#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, nbp, token;

    /* Initialisation de l'environnement MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nbp);

    /* On verifie qu'il y a au moins 2 processus */
    if(nbp < 2) {
        if(rank == 0)
            fprintf(stderr, "Ce programme necessite au moins 2 processus.\n");
        MPI_Finalize();
        return 1;
    }

    /* Processus de rang 0 initialise le jeton et l'envoie au processus de rang 1 */
    if (rank == 0) {
        token = 1;  // initialisation du jeton
        printf("Processus %d envoie le jeton %d vers le processus 1\n", rank, token);
        MPI_Send(&token, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        
        /* Processus 0 attend ensuite de recevoir le jeton venant du dernier processus */
        MPI_Recv(&token, 1, MPI_INT, nbp - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Processus %d a recu le jeton %d depuis le processus %d\n", rank, token, nbp - 1);
    }
    else {
        /* Tous les autres processus recoivent le jeton du processus precedent */
        MPI_Recv(&token, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        token++;  // incrementation du jeton
        printf("Processus %d a recu le jeton et l'incremente a %d\n", rank, token);

        /* Envoi du jeton au processus suivant.
            Le processus nbp-1 envoie vers le processus 0 pour fermer l'anneau. */
        int dest = (rank + 1) % nbp;
        MPI_Send(&token, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}