#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <sys/time.h>
#include "wtime.h"
#include "definitions.h"
#include "energy_struct.h"
#include "cuda_runtime.h"
#include "solver.h"

using namespace std;

/**
* Kernel del calculo de la solvation. Se debe anadir los parametros 
*/
__global__ void escalculation (int atoms_r, int atoms_l, int nlig, float *rec_x_d, float *rec_y_d, float *rec_z_d, float *lig_x_d, float *lig_y_d, float *lig_z_d, float *ql_d,float *qr_d, float *energy_d, int nconformations){
    
    float atomo[3], distancia, electro;
    int indiceR,indiceL;
    //Calculamos los índices de los atomos actuales
    indiceR= blockIdx.x * blockDim.x + threadIdx.x; 
    indiceL= blockIdx.y * blockDim.y + threadIdx.y;
    
    //Ca´lculo electroestático para cada conformación
    for(int k=0;k<nconformations;++k)
    {
       atomo[0] = *(lig_x_d + indiceL + (k*nlig));
       atomo[1] = *(lig_y_d + indiceL + (k*nlig));
       atomo[2] = *(lig_z_d + indiceL + (k*nlig));
       distancia=calculaDistancia(rec_x_d[indiceR], rec_y_d[indiceR], rec_z_d[indiceR], atomo[0], atomo[1], atomo[2]);
       electro = (ql_d[indiceL] * qr_d[indiceR]) / distancia;
       atomicAdd (&energy_d[k], electro);
    }
}


/**
* Funcion para manejar el lanzamiento de CUDA 
*/
void forces_GPU_AU (int atoms_r, int atoms_l, int nlig, float *rec_x, float *rec_y, float *rec_z, float *lig_x, float *lig_y, float *lig_z, float *ql ,float *qr, float *energy, int nconformations){

	cudaError_t cudaStatus; //variable para recoger estados de cuda

	//seleccionamos device
	cudaSetDevice(0); //0 - Tesla K40 vs 1 - Tesla K230

	//creamos memoria para los vectores para GPU _d (device)
	float *rec_x_d, *rec_y_d, *rec_z_d, *qr_d, *lig_x_d, *lig_y_d, *lig_z_d, *ql_d, *energy_d;

	//reservamos memoria para GPU
	cudaMalloc(&energy_d, nconformations * sizeof(float));
	cudaMalloc(&rec_x_d, atoms_r * sizeof(float));
	cudaMalloc(&rec_y_d, atoms_r * sizeof(float));
	cudaMalloc(&rec_z_d, atoms_r * sizeof(float));
	cudaMalloc(&qr_d, atoms_r * sizeof(float));
	cudaMalloc(&lig_x_d, nconformations * atoms_l * sizeof(float));
	cudaMalloc(&lig_y_d, nconformations * atoms_l * sizeof(float));
	cudaMalloc(&lig_z_d, nconformations * atoms_l * sizeof(float));
	cudaMalloc(&ql_d, atoms_l * sizeof(float));

	//pasamos datos de host to device
	cudaMemcpy(energy_d, energy, nconformations* sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(rec_x_d, rec_x, atoms_r * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(rec_y_d, rec_y, atoms_r * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(rec_z_d, rec_z, atoms_r * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(qr_d, qr, atoms_r * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(lig_x_d, lig_x, nconformations * atoms_l * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(lig_y_d, lig_y, nconformations * atoms_l * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(lig_z_d, lig_z, nconformations * atoms_l * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(ql_d, ql, atoms_l * sizeof(float), cudaMemcpyHostToDevice);

	//Definir numero de hilos y bloques
	int hilos_x_bloque=16;
	int R_bloques=(int)ceil(atoms_r/hilos_x_bloque);
	int L_bloques=(int)ceil(atoms_l/hilos_x_bloque);

	dim3 block (R_bloques,L_bloques);
	dim3 thread (hilos_x_bloque,hilos_x_bloque);
	printf("hilos por bloque: %d\n",hilos_x_bloque);

	//llamamos a kernel
	escalculation <<< block,thread>>> (atoms_r, atoms_l, nlig, rec_x_d, rec_y_d, rec_z_d, lig_x_d, lig_y_d, lig_z_d, ql_d, qr_d, energy_d, nconformations);

	//control de errores kernel
	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	if(cudaStatus != cudaSuccess) fprintf(stderr, "Error en el kernel %d\n", cudaStatus); 

	//Traemos info al host
	  cudaMemcpy(energy, energy_d, nconformations * sizeof(float), cudaMemcpyDeviceToHost);

	// para comprobar que la ultima conformacion tiene el mismo resultado que la primera
	printf("Termino electrostatico de conformacion %d es: %f\n", nconformations-1, energy[nconformations-1]); 

	//resultado varia repecto a SECUENCIAL y CUDA en 0.000002 por falta de precision con float
	//posible solucion utilizar double, probablemente bajara el rendimiento -> mas tiempo para calculo
	printf("Termino electrostatico %f\n", energy[0]);

	//Liberamos memoria reservada para GPU
	  cudaFree(energy_d);
	  cudaFree(rec_x_d);
	  cudaFree(rec_y_d);
	  cudaFree(rec_z_d);
	  cudaFree(qr_d);
	  cudaFree(lig_x_d);
	  cudaFree(lig_y_d);
	  cudaFree(lig_z_d);
	  cudaFree(ql_d);
}

/**
* Distancia euclidea compartida por funcion CUDA y CPU secuencial
*/
__device__ __host__ extern float calculaDistancia (float rx, float ry, float rz, float lx, float ly, float lz) {
	  float distx=rx-lx;
	  float disty=ry-ly;
	  float distz=rz-lz;
	  return sqrtf((distx*distx)+(disty*disty)+(distz*distz));
}


/**
 * Funcion que implementa el termino electrostático en CPU
 */
void forces_CPU_AU (int atoms_r, int atoms_l, int nlig, float *rec_x, float *rec_y, float *rec_z, float *lig_x, float *lig_y, float *lig_z, float *ql ,float *qr, float *energy, int nconformations){

	float dist, total_elec = 0, miatomo[3], elecTerm;
	int totalAtomLig = nconformations * nlig;

	for (int k=0; k < totalAtomLig; k+=nlig){
		for(int i=0;i<atoms_l;i++){					
			miatomo[0] = *(lig_x + k + i);
			miatomo[1] = *(lig_y + k + i);
			miatomo[2] = *(lig_z + k + i);
			for(int j=0;j<atoms_r;j++){				
				elecTerm = 0;
        			dist=calculaDistancia (rec_x[j], rec_y[j], rec_z[j], miatomo[0], miatomo[1], miatomo[2]);
				elecTerm = (ql[i]* qr[j]) / dist;
				total_elec += elecTerm;
			}
		}
		energy[k/nlig] = total_elec;
		total_elec = 0;
	}
	printf("Termino electrostatico %f\n", energy[0]);
}


extern void solver_AU(int mode, int atoms_r, int atoms_l,  int nlig, float *rec_x, float *rec_y, float *rec_z, float *lig_x, float *lig_y, float *lig_z, float *ql, float *qr, float *energy_desolv, int nconformaciones) {

	double elapsed_i, elapsed_o;
	
	switch (mode) {
		case 0://Sequential execution
			printf("\* CALCULO ELECTROSTATICO EN CPU *\n");
			printf("**************************************\n");			
			printf("Conformations: %d\t Mode: %d, CPU\n",nconformaciones,mode);			
			elapsed_i = wtime();
			forces_CPU_AU (atoms_r,atoms_l,nlig,rec_x,rec_y,rec_z,lig_x,lig_y,lig_z,ql,qr,energy_desolv,nconformaciones);
			elapsed_o = wtime() - elapsed_i;
			printf ("CPU Processing time: %f (seg)\n", elapsed_o);
			break;
		case 1: //OpenMP execution
			printf("\* CALCULO ELECTROSTATICO EN OPENMP *\n");
			printf("**************************************\n");			
			printf("**************************************\n");			
			printf("Conformations: %d\t Mode: %d, CMP\n",nconformaciones,mode);			
			elapsed_i = wtime();
			forces_OMP_AU (atoms_r,atoms_l,nlig,rec_x,rec_y,rec_z,lig_x,lig_y,lig_z,ql,qr,energy_desolv,nconformaciones);
			elapsed_o = wtime() - elapsed_i;
			printf ("OpenMP Processing time: %f (seg)\n", elapsed_o);
			break;
		case 2: //CUDA exeuction
			printf("\* CALCULO ELECTROSTATICO EN CUDA *\n");
			printf("**************************************\n");
			printf("Conformaciones: %d\t Mode: %d, GPU\n",nconformaciones,mode);
			elapsed_i = wtime();
			forces_GPU_AU (atoms_r,atoms_l,nlig,rec_x,rec_y,rec_z,lig_x,lig_y,lig_z,ql,qr,energy_desolv,nconformaciones);
			elapsed_o = wtime() - elapsed_i;
			printf ("GPU Processing time: %f (seg)\n", elapsed_o);			
			break; 	
	  	default:
 	    	printf("Wrong mode type: %d.  Use -h for help.\n", mode);
			exit (-1);	
	} 		
}
