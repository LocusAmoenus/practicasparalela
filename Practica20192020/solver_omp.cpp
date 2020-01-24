#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <sys/time.h>
#include "wtime.h"
#include "definitions.h"
#include "energy_struct.h"
#include "solver.h"

/**
* Funcion que implementa la solvatacion en openmp
*/
extern void forces_OMP_AU (int atoms_r, int atoms_l, int nlig, float *rec_x, float *rec_y, float *rec_z, float *lig_x, float *lig_y, float *lig_z, float *ql ,float *qr, float *energy, int nconformations){
		
	float electro, distancia, total_electro = 0, atomo[3];
	int totalLig = nconformations * nlig;

	printf("Utilizando OPENMP con 12 hilos\n");
	omp_set_num_threads(12);
	//#pragma omp parallel for schedule(guided) private(electro,distancia,atomo) reduction(+:total_electro)
	#pragma omp parallel for private(electro,distancia,atomo) reduction(+:total_electro)
	for (int a=0; a < totalLig; a+=nlig) {
		for(int b=0;b<atoms_l;b++){
			atomo[0] = *(lig_x + a + b);
			atomo[1] = *(lig_y + a + b);
			atomo[2] = *(lig_z + a + b);
			for(int c=0;c<atoms_r;c++){
				electro = 0;
				distancia=calculaDistancia(rec_x[c], rec_y[c], rec_z[c], atomo[0], atomo[1], atomo[2]);
				electro = (ql[b]*qr[c])/distancia;
				total_electro += electro;
			}
		}
		energy[a/nlig] = total_electro;
		total_electro = 0;
	}
	printf("Valor del termino electrostatico %f\n", energy[0]);
}


