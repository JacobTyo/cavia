#!/usr/bin/env bash

anum_fns=($1)
anum_inner_updates=(10)
atasks_per_metaupdate=(25)


for num_fns in "${anum_fns[@]}"; do
  for num_inner_updates in "${anum_inner_updates[@]}"; do
    for tasks_per_metaupdate in "${atasks_per_metaupdate[@]}"; do
      tsk='ftinc_lrin001_'${num_fns}__${num_inner_updates}_${tasks_per_metaupdate}
      # cavia
      python ./main.py --n_iter 100000 --num_fns ${num_fns} --tasks_per_metaupdate ${tasks_per_metaupdate} \
        --num_inner_updates ${num_inner_updates}  --id "${tsk}" --lr_inner 0.001
    done
  done
done
