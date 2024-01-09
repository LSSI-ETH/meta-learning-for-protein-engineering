#!/bin/bash

cd ../../


for b in cnn transformer
do
    for m in fine-tune l2rw mlc metaset_baseline
    do 
        for n in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
        do
            python main.py --base_model=$b --top_model=$m --data_type=4d5_syn --noise_type=flip --noise_fraction=$n
        done
    done
done


for b in cnn transformer
do
    for m in fine-tune l2rw mlc metaset_baseline
    do 
        for n in 0 1 2 3
        do
            for e in 4 5 6 7
            do
                python main.py --base_model=$b --top_model=$m --data_type=4d5_exp --meta_set_number=$n --edit_distance=$e
            done
        done
    done
done

for b in cnn transformer
do
    for m in fine-tune l2rw mlc metaset_baseline
    do 
        for n in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8
        do
            python main.py --base_model=$b --top_model=$m --data_type=5a12_PUL_syn --alpha=$n
        done
    done
done


for b in cnn transformer
do
    for m in fine-tune l2rw mlc metaset_baseline
    do 
        for n in 0 1 2 3
        do
            for e in 4 5 6 7
            do
                python main.py --base_model=$b --top_model=$m --data_type=5a12_PUL_exp --meta_set_number=$n --edit_distance=$e
            done
        done
    done
done

for b in cnn transformer
do
    for m in fine-tune l2rw mlc metaset_baseline
    do 
        for n in 0 1 2 3
        do
            for e in 4 5 6 7
            do
                python main.py --base_model=$b --top_model=$m --data_type=5a12_2ag --meta_set_number=$n --edit_distance=$e
            done
        done
    done
done