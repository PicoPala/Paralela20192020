#!/bin/bash

# Si los argumentos introducidos son incorrectos
if [ $# -ne 2 ]
then
	echo -e "\nUsage: ./lanzador.sh M C (M=Modo. 0-CPU, 1-OMP, 2-GPU -- C=Conformaciones)\n"

# Si los argumentos introducidos estan correctos
else
	# Genera 2 arrays con los ligandos y receptores
	ligandos=(`ls input/*lig*`)
	receptores=(`ls input/*rec*`)

	echo -e "\nLANZADOR -- TERMINO DE DESOLVATACION"
	echo -e "Modo: $1, Conformaciones: $2, Simulaciones: ${#ligandos[*]}\n"

	# Para cada ligando y su respectivo receptor calcula el termino con
	# los parametros indicados
	for l in ${!ligandos[*]}
	do
		echo -e "\nSimulacion $((l+1)) (${ligandos[$l]##*/} | ${receptores[$l]##*/})\n"
		./energy -l ${ligandos[$l]} -r ${receptores[$l]} -m $1 -c $2
		sleep 5
	done

	echo -e
fi
