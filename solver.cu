#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <sys/time.h>
#include "wtime.h"
#include "definitions.h"
#include "energy_struct.h"
#include "cuda_runtime.h"
#include "solver.h"

#define TAMBLOCK_R 512
#define TAMBLOCK_L 8

using namespace std;

/**
* Kernel del calculo de la solvation. Se debe anadir los parametros 
*/
__global__ void escalculation (int atoms_r, int atoms_l, int nlig, float *rec_x_d, float *rec_y_d, float *rec_z_d, float *lig_x_d, float *lig_y_d, float *lig_z_d, float *ql_d,float *qr_d, float *energy_d, int nconformations){

  __shared__ float idata[1024];
  __shared__ float resultado;

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col =   blockIdx.x * blockDim.x + threadIdx.x;
  int tidx = threadIdx.x;
  int k;
  float dist = 0;
  float temp[1024];

  idata[tidx] = 0;
  temp[tidx] = 0;
  
  if  (col < atoms_r && row < atoms_l) {
    for ( k=0; k < nconformations*nlig; k+=nlig) {

      idata[tidx]=0;
      temp[tidx]=0;

      //Distancia para 1 bloque.
      dist=calculaDistancia (rec_x_d[col], rec_y_d[col], rec_z_d[col], lig_x_d[row+k], lig_y_d[row+k], lig_z_d[row+k]);
      temp[tidx]  += (ql_d[row]* qr_d[col]) / dist;
  
      idata[tidx]= temp[tidx];
      __syncthreads();
  
      //Reduccion para bloques de 2.
      for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tidx < s) {
          idata[tidx] += idata[tidx + s];
        }
        __syncthreads();
      }
  
      //Se obtiene el resultado para los bloques.
      if (tidx == 0) {
        atomicAdd(&energy_d[k/nlig], idata[0]);
      }
    }
  }
}


/**
* Funcion para manejar el lanzamiento de CUDA 
*/
void forces_GPU_AU (int atoms_r, int atoms_l, int nlig, float *rec_x, float *rec_y, float *rec_z, float *lig_x, float *lig_y, float *lig_z, float *ql ,float *qr, float *energy, int nconformations){
	
	cudaError_t cudaStatus; //variable para recoger estados de cuda
  cudaEvent_t ini, fin;
  cudaEvent_t ini_htd, fin_htd;
  cudaEvent_t ini_dth, fin_dth;

  float mil, mil_htd, mil_dth;

  cudaEventCreate(&ini);
  cudaEventCreate(&fin);
  cudaEventCreate(&ini_htd);
  cudaEventCreate(&fin_htd);
  cudaEventCreate(&ini_dth);
  cudaEventCreate(&fin_dth);

  int tam_receptor = atoms_r * sizeof(float);
  int tam_l = atoms_l * sizeof(float);
  int tam_ligando = (atoms_l+nconformations*nlig) * sizeof(float);
  int tam_energy = (nconformations) * sizeof(float);
  int total_hilos;
  int hilos_bloque;
  int ancho_bloque_x;
  int ancho_bloque_y;

  
	//seleccionamos device
	cudaSetDevice(0); //0 - Tesla K40 vs 1 - Tesla K230

	//creamos memoria para los vectores para GPU _d (device)
	float *rec_x_d, *rec_y_d, *rec_z_d, *qr_d, *lig_x_d, *lig_y_d, *lig_z_d, *ql_d, *energy_d;

	//reservamos memoria para GPU
  cudaMalloc( (void **)  &rec_x_d,atoms_r * sizeof(float)); 
  cudaMalloc( (void **)  &rec_y_d,atoms_r * sizeof(float));
  cudaMalloc( (void **)  &rec_z_d,atoms_r * sizeof(float));
  cudaMalloc( (void **)  &qr_d,atoms_r * sizeof(float));
  cudaMalloc( (void **)  &lig_x_d,(atoms_r +nconformations*nlig) * sizeof(float));
  cudaMalloc( (void **)  &lig_y_d,(atoms_r +nconformations*nlig ) * sizeof(float));
  cudaMalloc( (void **)  &lig_z_d,(atoms_r +nconformations*nlig) * sizeof(float));
  cudaMalloc( (void **)  &ql_d,atoms_l * sizeof(float));
  cudaMalloc( (void **)  &energy_d,( nconformations) * sizeof(float));

	//pasamos datos de host to device
	cudaEventRecord(ini_htd, 0);

  cudaStatus = cudaMemcpy(rec_x_d, rec_x, tam_receptor, cudaMemcpyHostToDevice);
  cudaStatus = cudaMemcpy(rec_y_d, rec_y, tam_receptor, cudaMemcpyHostToDevice);
  cudaStatus = cudaMemcpy(rec_z_d, rec_z, tam_receptor, cudaMemcpyHostToDevice);
  cudaStatus = cudaMemcpy(qr_d, qr, tam_receptor, cudaMemcpyHostToDevice);
  cudaStatus = cudaMemcpy(lig_x_d, lig_x, tam_ligando, cudaMemcpyHostToDevice);
  cudaStatus = cudaMemcpy(lig_y_d, lig_y, tam_ligando, cudaMemcpyHostToDevice);
  cudaStatus = cudaMemcpy(lig_z_d, lig_z, tam_ligando, cudaMemcpyHostToDevice);
  cudaStatus = cudaMemcpy(ql_d, ql, tam_l, cudaMemcpyHostToDevice);
  cudaStatus = cudaMemcpy(energy_d, energy, tam_energy, cudaMemcpyHostToDevice);

  cudaEventRecord(fin_htd, 0);
  cudaEventSynchronize(fin_htd);
  mil_htd = 0.0f;
  cudaEventElapsedTime(&mil_htd, ini_htd, fin_htd);
  
	//Definir numero de hilos y bloques
  total_hilos = atoms_r * atoms_l;
  ancho_bloque_x = 128;
  ancho_bloque_y = 1;

  hilos_bloque = ancho_bloque_x * ancho_bloque_y;

  dim3 dimGrid(ceil((atoms_r + ancho_bloque_x - 1) / ancho_bloque_x), ceil((atoms_l + ancho_bloque_y - 1) / ancho_bloque_y));
  dim3 dimBlock(ancho_bloque_x, ancho_bloque_y);

	//printf("bloques: %d\n", (int)ceil(total_hilos/hilos_bloque)+1);
	//printf("hilos por bloque: %d\n", hilos_bloque);

	//llamamos a kernel
  cudaEventRecord(ini, 0);
  escalculation <<<dimGrid, dimBlock>>> (atoms_r, atoms_l, nlig, rec_x_d, rec_y_d, rec_z_d, lig_x_d, lig_y_d, lig_z_d, ql_d, qr_d, energy_d, nconformations);

  cudaEventRecord(fin, 0);
  cudaEventSynchronize(fin);
  mil = 0.0f;

  cudaEventElapsedTime(&mil, ini, fin); 	

	//control de errores kernel
	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	if(cudaStatus != cudaSuccess) fprintf(stderr, "Error en el kernel %d\n", cudaStatus); 

  cudaEventRecord(ini_dth, 0);

	//Traemos info al host
  cudaMemcpy(energy, energy_d, tam_energy, cudaMemcpyDeviceToHost);
  cudaEventRecord(fin_dth, 0);
  cudaEventSynchronize(fin_dth);
  mil_dth = 0.0f;
  cudaEventElapsedTime(&mil_dth, ini_dth, fin_dth);

	// para comprobar que la ultima conformacion tiene el mismo resultado que la primera
	//printf("Termino electrostatico de conformacion %d es: %f\n", nconformations-1, energy[nconformations-1]); 
  printf("-Tiempo calculo: %f\n", mil / 1000);
  printf("-Tiempo host to device: %f\n", mil_htd / 1000);
  printf("-Tiempo device to host: %f\n", mil_dth / 1000);  

	//resultado varia repecto a SECUENCIAL y CUDA en 0.000002 por falta de precision con float
	//posible solucion utilizar double, probablemente bajara el rendimiento -> mas tiempo para calculo
	printf("Termino electrostatico %f\n", energy[0]);

	//Liberamos memoria reservada para GPU
  cudaFree(rec_x_d);
  cudaFree(rec_y_d);
  cudaFree(rec_z_d);
  cudaFree(qr_d);
  cudaFree(lig_x_d);
  cudaFree(lig_y_d);
  cudaFree(lig_z_d);
  cudaFree(ql_d);
  cudaFree(energy_d);
}

/**
* Distancia euclidea compartida por funcion CUDA y CPU secuencial
*/
__device__ __host__ extern float calculaDistancia (float rx, float ry, float rz, float lx, float ly, float lz) {

  return 0; 
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
