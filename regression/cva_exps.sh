#!/usr/bin/env bash

anum_fns=($1)
anum_eval_updates=(1 10 100)
anum_inner_updates=(0)
atasks_per_metaupdate=(25)


for num_fns in "${anum_fns[@]}"; do
  for num_eval_updates in "${anum_eval_updates[@]}"; do
    for num_inner_updates in "${anum_inner_updates[@]}"; do
      for tasks_per_metaupdate in "${atasks_per_metaupdate[@]}"; do
        # normal
        tsk='100k_'${num_fns}_${num_eval_updates}_${num_inner_updates}_${tasks_per_metaupdate}
        python ./main.py --cva --num_fns ${num_fns} --lr_inner 0.001 --lr_meta 0.001 \
          --tasks_per_metaupdate ${tasks_per_metaupdate} --num_inner_updates ${num_inner_updates} \
          --num_eval_updates ${num_eval_updates} --id "${tsk}"
        # reinitialize embeddings
        tsk='remb100k_'${num_fns}_${num_eval_updates}_${num_inner_updates}_${tasks_per_metaupdate}
        python ./main.py --cva --num_fns ${num_fns} --lr_inner 0.001 --lr_meta 0.001 \
          --tasks_per_metaupdate ${tasks_per_metaupdate} --num_inner_updates ${num_inner_updates} \
          --num_eval_updates ${num_eval_updates} --id "${tsk}" --reinit_emb
      done
    done
  done
done
