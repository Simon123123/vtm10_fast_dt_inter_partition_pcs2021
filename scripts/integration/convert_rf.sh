#!/bin/bash

for i in 64 32 16 8
do
	for j in 64 32 16 8
	do
		porter /path_rf_model/hv_${i}_${j}.pkl --c --pipe > /output_path/hv_${i}_${j}.c
	done
done


porter /path_rf_model/hv_128_128.pkl --c --pipe > /output_path/hv_128_128.c


for i in 128 64 32 16
do
	porter /path_rf_model/qm_${i}_${i}.pkl --c --pipe > /output_path/qm_${i}_${i}.c
done 