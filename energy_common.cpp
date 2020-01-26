#include "energy_common.h"
#include "energy_struct.h"
#include "definitions.h"

using namespace std;


extern void writeLigandEnergies (char *filename, int nconformations, float *energy_desolv_CPU)
{
        float desolv_conformation_CPU;
        ofstream desolvfile(filename, ios::out | ios::trunc);

        if (!desolvfile.good()) {
                cout << "can't open file " << filename << endl;
                cout << "Will not write ligand energies file" << endl;
                return;
        }
        desolvfile << "DESOLVATION TERM" << endl;
        desolvfile << "CONF." << "\t\t" << "E_CPU" << endl;
        desolvfile << endl;
        for (int i = 0; i < nconformations; i++)
        {
                desolv_conformation_CPU = energy_desolv_CPU[i];
                desolvfile << i+1 << "\t\t  " << desolv_conformation_CPU << endl;
        }
        desolvfile.close();
}


extern void fill_conformations (int n,float *conformations_x,float *conformations_y, float  *conformations_z, struct ligand_t ligando) {

        for (int i=0;i<(n*ligando.nlig);i+=ligando.nlig)
        {
                        memcpy(conformations_x+i,ligando.lig_x,sizeof(float)*ligando.nlig);
                        memcpy(conformations_y+i,ligando.lig_y,sizeof(float)*ligando.nlig);
                        memcpy(conformations_z+i,ligando.lig_z,sizeof(float)*ligando.nlig);
        }

}


extern int skip_to_token(FILE* file,char* token)
{
        char c;
        unsigned int count=0;

        while ((c = fgetc(file)) != EOF)
        {
                if (c==token[count])
                        count++;
                else
                        count=0;
                if (count == strlen(token))
                        return 0;
        }
        return 1;
}


extern void param_autodock (char *input, struct autodock_param_t * a_params) {

        float f1,f2,f3,f4,f5,f6,f7;
        int type;
        char header1[100];

        FILE *fp;
        if ((fp =  fopen(input,"r")) == NULL) {
                printf("ReadInput(): Can't open file \"%s\"\n", input);
                exit(1);
        }

         for (int i1=0;i1 < 22;i1++) {
                a_params[i1].sigma = 0;
		a_params[i1].epsilon = 0;
                a_params[i1].asp = 0;
                a_params[i1].qasp = 0;
                a_params[i1].vol = 0;
        }

        while (!feof(fp))
        {
                fscanf(fp,"%s %f %f %f %f %f %f", header1,&f1,&f2,&f3,&f4,&f5,&f6);
                //printf("atom type %s\n",header1);
                type = getTypeNumber(header1);
                a_params[type].sigma = f1;
                a_params[type].epsilon = f2;
                a_params[type].vol = f3;
                a_params[type].asp = f4;
                a_params[type].sigma_h = f5;
                a_params[type].epsilon_h = f6;
        }

        a_params[HBOND] = a_params[HID];
        a_params[NIT2] = a_params[NIT];
}


extern int getTypeNumber (char* cadena){

        if ((cadena[0] == 'H' || cadena[0] == 'h')) {   //HIDROGENO
                //&& (cadena[1] != 'G' && cadena[1] != 'g')){ //No es mercurio
                return HID;
        }
        else if (cadena[0] == 'C' || cadena[0] == 'c'){

                if (cadena[1] == 'L' || cadena[1] == 'l')
                        return CLO;     //CLORO
                else if (cadena[1] == 'A' || cadena[1] == 'a')
                        return CAL;     //CALCIO
                else if (cadena[1] == '.' && cadena[2] == 'a')
                        return AR;
                return CAR;
        }
        else if (cadena[0] == 'O' || cadena[0] == 'o') //OXIGENO
                return OXI;
        else if (cadena[0] == 'S' || cadena[0] == 's') //AZUFRE
                return AZU;
        else if (cadena[0] == 'N' || cadena[0] == 'n'){ //NITROGENO
                if (cadena[1] == 'A' || cadena[1] == 'a')
                        return SOD;
                return NIT;
                }
        else if (cadena[0] == 'I' || cadena[0] == 'I') //IRIDIO
                return IRI;
        else if (cadena[0] == 'F' || cadena[0] == 'f') {
                if (cadena[1] == 'e' || cadena[1] == 'E')
                        return FER;//HIERRO
                return FLU; //FLUOR
                }
        else if ((cadena[0] == 'B' || cadena[0] == 'b') && //BROMO
                (cadena[1] == 'R' || cadena[1] == 'r'))
                return BRO;
        else if ((cadena[0] == 'L' || cadena[0] == 'l') && //LITIO
                (cadena[1] == 'I' || cadena[1] == 'i'))
                return LIT;

        else if (cadena[0] == 'K' || cadena[0] == 'k') //POTASIO
                return POT;
        else if ((cadena[0] == 'M' || cadena[0] == 'm') && //MAGNESIO
                (cadena[1] == 'G' || cadena[1] == 'g'))
                return MAG;
        else if ((cadena[0] == 'Z' || cadena[0] == 'z') && //ZINC
                (cadena[1] == 'N' || cadena[1] == 'n'))
                return ZIN;
        else{
                return DUM; //DUMMY
        }

}

extern int skip_to_eol(FILE* file)
{
	int c;
	int nl = '\n';

	while ((c = fgetc(file)) != EOF)
		if (c == nl) return 0;
	return 1;
}


extern void readProtein(char* protname, struct receptor_t &proteina)
{
	FILE *fp;
	unsigned int qrSize,id,id_1;
	unsigned int recSize;
	char cadena[5],cadena_1[5];
	char header[100];
	int nAtom, nBond, dum1, dum2, dum3, ind1, ind2, padding;


        if ((fp = fopen(protname,"r")) == NULL) {
                printf("ReadInput(): Can't open file \"%s\"\n", protname);
                exit(1);
        }

//
        if (skip_to_token(fp,"@<TRIPOS>MOLECULE")) {
                exit(1);
                return;
        }
        fscanf(fp,"%s", header);
        fscanf(fp,"%d %d %d %d %d", &nAtom, &nBond, &dum1, &dum2,&dum3);

	proteina.nrec = nAtom;

	qrSize = proteina.nrec * sizeof(float);
	recSize = proteina.nrec * sizeof(float);


	proteina.rec_x = (float*) malloc (recSize);
	proteina.rec_y = (float*) malloc (recSize);
	proteina.rec_z = (float*) malloc (recSize);
	proteina.qr = (float *) malloc (recSize);
	proteina.rectype = (int *) malloc (proteina.nrec * sizeof(int));

	if (skip_to_token(fp,"@<TRIPOS>ATOM")) {
                exit(1);
                return;
        }

	for (int k = 0; k < nAtom; k++) {

		//int k = i;
		fscanf(fp, "%d %s %f %f %f %s %d %s %f", &id,header,&proteina.rec_x[k],&proteina.rec_y[k],&proteina.rec_z[k],cadena,&id_1,cadena_1,&proteina.qr[k]);
		proteina.rectype[k] = getTypeNumber(cadena);

		if (skip_to_eol(fp)) {
			exit(1);
			return;
		}
		//printf("Pasada %d %f %f %f\n",k,proteina.rec_x[k],proteina.rec_y[k],proteina.rec_z[k]);
	}
	proteina.atoms = proteina.nrec;

	for (int i=proteina.nrec ; i < proteina.nrec + padding ; i++){
		proteina.rectype[i] = 18;
		proteina.rec_x[i]  = 0.0;
		proteina.rec_y[i]= 0.0;
		proteina.rec_z[i]= 0.0;
	}
	fclose(fp);

}


extern void readLigand (char *filename, struct ligand_t *ligando){

	int nAtom,nBond,aux,ind1, ind2,dum1,dum2,dum3;

	//Auxiliar variables to read useless information
	char auxtipo[5];
	char cadena[80], cadena1[80], cadena2[80];

	FILE *fp;

        if ((fp = fopen(filename,"r")) == NULL) {
                printf("ReadInput(): Can't open file \"%s\"\n", filename);
                exit(1);
        }

        if (skip_to_token(fp,"@<TRIPOS>MOLECULE")) {
                exit(1);
                return;
        }
        fscanf(fp,"%s", cadena);
        fscanf(fp,"%d %d %d %d %d", &nAtom, &nBond, &dum1, &dum2,&dum3);


	ligando->nlig = nAtom;
	ligando->atoms = nAtom;


	ligando->lig_x = (float*) malloc(ligando->nlig * sizeof(float));
	ligando->lig_y = (float*) malloc(ligando->nlig * sizeof(float));
	ligando->lig_z = (float*) malloc(ligando->nlig * sizeof(float));
	ligando->ql = (float *) malloc(ligando->nlig * sizeof(float));
	ligando->ligtype = (int*) malloc (ligando->nlig * sizeof(int));

	if (skip_to_token(fp,"@<TRIPOS>ATOM")) {
                exit(1);
                return;
        }

        for (int i=0; i < nAtom; i++){
                //Read type of atom
                fscanf(fp, "%d %s %f %f %f %s %s %s %f",&aux,auxtipo,&ligando->lig_x[i],&ligando->lig_y[i],&ligando->lig_z[i],cadena,cadena1,cadena2,&ligando->ql[i]);

		ligando->ligtype[i] = getTypeNumber(auxtipo);

	}
	fclose(fp);

	for (int i=nAtom ; i < ligando->nlig ; i++){
		ligando->ligtype[i] = 18;
		ligando->lig_x[i]  = 0.0;
		ligando->lig_y[i]= 0.0;
		ligando->lig_z[i]= 0.0;
	}

}
