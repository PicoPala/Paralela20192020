#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <math.h>
#include "energy_common.h"
#include "definitions.h"
#include "solver.h"
#include "energy_struct.h"


using namespace std;

int main (int argc, char **argv){

  char c;
	int mode;
	char *file_l, *file_p, *file_e, *file_a;
	int nconformations;

	float *energy_desolv_CPU;
	float *conformations_x, *conformations_y, *conformations_z;

	struct ligand_t ligando;
	struct receptor_t proteina;

  // system ("clear");

	file_l = (char *)malloc(sizeof(char)*100);
	file_p = (char *)malloc(sizeof(char)*100);
	file_e = (char *)malloc(sizeof(char)*100);
	file_a = (char *)malloc(sizeof(char)*100);
	nconformations = 1;

	while ((c = getopt (argc, argv, "vr:m:l:a:r:x:g:y:z:s:c:e:n:h")) != -1) {
	  switch (c) {
	    	case 'v':
     			printf("Energy atomic Ligand-Protein Calculation v.1.0\n\n");
				return 0;
			case 'm':
				mode = atoi(optarg);
				break;
			case 'r':
				strcpy(file_p,optarg);
				break;
			case 'l':
				strcpy(file_l,optarg);
				break;
			case 'c':
				nconformations = atoi(optarg);
				break;
			case 'h':
			case '?':
	      		printf("Usage:\tenergy -r fichero_protein.mol2 -l fichero_ligand.mol2 [-c numero_conformaciones] [-h | -? HELP] \n");
				printf("\t<Params>\n");
		    	printf("\t\t-v\t\tOutput version information and exit\n");
	    		return 0;
    	  }
 	  }

  readLigand (file_l,&ligando);
	readProtein (file_p, proteina);

  //CREATE CONFORMATIONS
  conformations_x = (float *)malloc(sizeof(float)*nconformations*ligando.nlig);
  conformations_y = (float *)malloc(sizeof(float)*nconformations*ligando.nlig);
  conformations_z = (float *)malloc(sizeof(float)*nconformations*ligando.nlig);

  //FILL CONFORMATIONS.
  fill_conformations(nconformations,conformations_x,conformations_y,conformations_z,ligando);

	//ENERGY FOR EACH ATOM
	energy_desolv_CPU = (float *)calloc(sizeof(float),nconformations);

	solver_AU(mode, proteina.atoms, ligando.atoms, ligando.nlig,proteina.rec_x,proteina.rec_y,proteina.rec_z,conformations_x,conformations_y,conformations_z,ligando.ql,proteina.qr,energy_desolv_CPU,nconformations);

	return 0;

}

