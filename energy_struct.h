#ifndef _STRUCT_H_
#define _STRUCT_H_

#include "definitions.h"

//Agrupa la informacion del ligando

typedef struct ligand_t {

	int nlig;		//Numero de atomos
	int atoms;
	float * lig_x;		//Posicion x
	float * lig_y;		//Posicion y
	float * lig_z;		//Posicion 
	int * ligtype;		//Tipo: número asignado al tipo (carbono=0, hidrogeno=1,...)
	float *ql;
	char *subtype;
	int *bonds;
	char *nbonds;
}ligand;

typedef struct receptor_t {

	int nrec;
	int atoms;
	float * rec_x; // Posicion x
	float * rec_y; // Posicion y
	float * rec_z; // Posicion z	
	float * qr;
	int * rectype;
     	int *bonds;
     	char * nbonds;
}receptor;

typedef struct autodock_param_t {
        float epsilon;
        float sigma;
	float vol;
	float asp;
	float qasp;
	float epsilon_h;
	float sigma_h;
} autodock_param;


#endif
