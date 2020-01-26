#ifndef COMMON_H_
#define COMMON_H_
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <cstring>
#include <math.h>
#include <time.h>
#include <ostream>
#include <iomanip>
#include "definitions.h"
#include "energy_struct.h"


// includes, project


extern int getTypeNumber (char* cadena);
extern void readLigand (char *filename, struct ligand_t *ligando);

extern void param_autodock (char *input, struct autodock_param_t *a_param);

extern void readProtein (char* protname, struct receptor_t &proteina);
extern void fill_conformations (int n,float *conformations_x,float *conformations_y, float *conformations_z, struct ligand_t ligando);
extern void writeLigandEnergies (char *filename, int nconformations, float *energy_desolv_CPU);

#endif /* COMMON_H_ */
