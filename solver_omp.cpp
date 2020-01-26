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

float t_ini = omp_get_wtime();
 // printf(" En el fichero solver_omp.cpp se encuentra la funcion forces_omp_au que se debe implementar con la version OpenMP\n");
float dist, total_elec = 0, miatomo[3], elecTerm;
int totalAtomLig = nconformations * nlig;
 
omp_set_num_threads(omp_get_max_threads());

  #pragma omp parallel for private(elecTerm,dist,miatomo) reduction(+:total_elec)
  for (int k=0; k < totalAtomLig; k+=nlig) {
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

  float t_fin = omp_get_wtime()-t_ini;
  
  //Ancho de banda//
  float bucle_i = ((1 + (2 * atoms_l) + (2 * atoms_l)) * (totalAtomLig / nlig));
  float dentro_i = (6*3 + 5*3) * atoms_l * (totalAtomLig / nlig);
  float bucle_j = ((1 + (2 * atoms_r) + (2 * atoms_r)) * atoms_l * (totalAtomLig / nlig));
  float dentro_j_p1 = (38)* atoms_r * atoms_l * (totalAtomLig / nlig);
  float dentro_j_p2 = (4+10)* atoms_r * atoms_l * (totalAtomLig / nlig);
  float dentro_j_p3 = (4+8)* atoms_r * atoms_l * (totalAtomLig / nlig);
  float bucle_k = 1 + (2 * totalAtomLig / nlig) + (3 * (totalAtomLig / nlig));
  float dentro_k = (6+8)* (totalAtomLig / nlig);
  float dentro_calcula = (6*3*4 +  4*4 + 4) * atoms_r * atoms_l * (totalAtomLig / nlig);
  float ancho_banda = bucle_i + dentro_i +  bucle_j + dentro_j_p1 + dentro_j_p2 + dentro_j_p3 + bucle_k + dentro_k + dentro_calcula;

  float rate = ancho_banda / t_fin*1E-9;

  //Rendimiento//
  float precalculo_flops = (12) * atoms_r * atoms_l * totalAtomLig / nlig;

  //Se divide entre el tiempo final//
  float flops = precalculo_flops/t_fin * 1E-9;

  printf("\n***Rendimiento y Ancho de banda***"); 
  printf("\n-Tiempo de ejecución: %f s", t_fin);
  printf("\n-Ancho de banda: %.2f GB/s ", rate);
  printf("\n-Rendimiento computacional: %.2f GFLOPS", flops);
  printf("\nTermino electrostatico %f\n", energy[0]);

}



