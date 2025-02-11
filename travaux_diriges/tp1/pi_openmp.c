/* pi_openmp.c */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>

double approximate_pi(unsigned long nbSamples) {
    unsigned long nbDarts = 0;

    #pragma omp parallel
    {
        unsigned int seed = time(NULL) ^ omp_get_thread_num(); 
        unsigned long localCount = 0;
        #pragma omp for
        for (unsigned long i = 0; i < nbSamples; i++) {
            double x = (double)rand_r(&seed) / RAND_MAX * 2.0 - 1.0;
            double y = (double)rand_r(&seed) / RAND_MAX * 2.0 - 1.0;
            if (x*x + y*y <= 1.0)
                localCount++;
        }
        #pragma omp atomic
        nbDarts += localCount;
    }
    double ratio = (double)nbDarts / nbSamples;
    return 4.0 * ratio;
}

int main(void) {
    unsigned long nbSamples = 10000000UL;
    double tdeb = omp_get_wtime();
    double pi = approximate_pi(nbSamples);
    double tfin = omp_get_wtime();
    printf("Approximation de pi (OpenMP) = %f\n", pi);
    printf("Temps d'execution = %f secondes\n", tfin - tdeb);
    return 0;
}