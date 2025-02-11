        /* pi_mpi.c */
        #include <stdio.h>
        #include <stdlib.h>
        #include <time.h>
        #include <math.h>
        #include <mpi.h>
        
        int main(int argc, char *argv[]) {
            int rank, nbp;
            MPI_Init(&argc, &argv);
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            MPI_Comm_size(MPI_COMM_WORLD, &nbp);
        
            unsigned long nbSamplesTotal = 10000000UL;
            /* Chaque processus traite une portion des echantillons */
            unsigned long nbSamples = nbSamplesTotal / nbp;
            unsigned long nbDarts_local = 0;
        
            /* Utiliser une graine differente pour chaque processus */
            unsigned int seed = time(NULL) ^ rank;
        
            double tdeb = MPI_Wtime();
            for (unsigned long i = 0; i < nbSamples; i++) {
                double x = (double)rand_r(&seed) / RAND_MAX * 2.0 - 1.0;
                double y = (double)rand_r(&seed) / RAND_MAX * 2.0 - 1.0;
                if (x*x + y*y <= 1.0)
                    nbDarts_local++;
            }
        
            unsigned long nbDarts_total = 0;
            MPI_Reduce(&nbDarts_local, &nbDarts_total, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
            double tfin = MPI_Wtime();
        
            if (rank == 0) {
                double ratio = (double)nbDarts_total / (nbSamples * nbp);
                double pi = 4.0 * ratio;
                printf("Approximation de pi (MPI) = %f\n", pi);
                printf("Temps d'execution (MPI) = %f secondes\n", tfin - tdeb);
            }
        
            MPI_Finalize();
            return 0;
        }  