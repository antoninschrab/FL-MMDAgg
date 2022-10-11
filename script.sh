#!/bin/bash
trap "exit" INT
datasets=('mnist' 'cifar10')
shift_types=('small_gn_shift' 'medium_gn_shift' 'large_gn_shift' 'adversarial_shift' 'ko_shift' 'small_image_shift' 'medium_image_shift' 'large_image_shift' 'medium_img_shift+ko_shift' 'only_zero_shift+medium_img_shift')
for dataset in "${datasets[@]}"
do 
    for shift_type in "${shift_types[@]}"
    do
	python pipeline.py $dataset $shift_type mmdagg
    done
done
