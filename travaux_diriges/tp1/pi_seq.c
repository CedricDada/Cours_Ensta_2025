/* pi_seq.c */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

double approximate_pi(unsigned long nbSamples) {
    unsigned long nbDarts = 0;
    for (unsigned long i = 0; i < nbSamples; i++) {
        /* Generer un point dans [-1, 1] */
        double x = (double)rand() / RAND_MAX * 2.0 - 1.0;
        double y = (double)rand() / RAND_MAX * 2.0 - 1.0;
        if (x*x + y*y <= 1.0)
            nbDarts++;
    }
    double ratio = (double)nbDarts / nbSamples;
    return 4.0 * ratio;
}

int main(void) {
    unsigned long nbSamples = 10000000UL; 
    srand(time(NULL));
    clock_t tdeb = clock();
    double pi = approximate_pi(nbSamples);
    clock_t tfin = clock();
    double temps = (double)(tfin - tdeb) / CLOCKS_PER_SEC;
    printf("Approximation de pi (sequentiel) = %f\n", pi);
    printf("Temps d'execution = %f secondes\n", temps);
    return 0;
}
