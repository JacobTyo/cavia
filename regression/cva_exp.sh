#!/usr/bin/env bash

anum_fns=($1)
anum_eval_updates=(40)
anum_inner_updates=(0)
atasks_per_metaupdate=(10)
alr_inner=(0.001)
alr_meta=(0.001)
alr_scheduler_period=(100 5000)
alr_model_decay=(0.99)
alr_emb_decay=(0.9)
apdropout=(0.2)
areset_emb=(10)
areset_emb_decay=(1)

for num_fns in "${anum_fns[@]}"; do
  for num_eval_updates in "${anum_eval_updates[@]}"; do
    for num_inner_updates in "${anum_inner_updates[@]}"; do
      for tasks_per_metaupdate in "${atasks_per_metaupdate[@]}"; do
        for lr_inner in "${alr_inner[@]}"; do
          for lr_meta in "${alr_meta[@]}"; do
            for lr_scheduler_period in "${alr_scheduler_period[@]}"; do
              for lr_model_decay in "${alr_model_decay[@]}"; do
                for lr_emb_decay in "${alr_model_decay[@]}"; do
                  for pdropout in "${apdropout[@]}"; do
                    for reset_emb in "${areset_emb[@]}"; do
                      for reset_emb_decay in "${areset_emb_decay[@]}"; do

                        tsk="selected_nf"${num_fns}"_lri"${lr_inner}"_lrm"${lr_meta}"_lrs"${lr_scheduler_period}"_lrmd"${lr_model_decay}"_lred"${lr_emb_decay}"_tpu"${tasks_per_metaupdate}"_neu"${num_eval_updates}"_dop"${pdropout}"_re"${reset_emb}"_red"${reset_emb_decay}

                        echo "starting " ${tsk}

                        python ./main.py \
                          --cva \
                          --id "${tsk}" \
                          --n_iter 50000 \
                          --num_fns ${num_fns} \
                          --lr_inner ${lr_inner} \
                          --lr_meta ${lr_meta} \
                          --lr_scheduler_period ${lr_scheduler_period} \
                          --lr_model_decay ${lr_model_decay} \
                          --lr_emb_decay ${lr_emb_decay} \
                          --tasks_per_metaupdate ${tasks_per_metaupdate} \
                          --num_inner_updates ${num_inner_updates} \
                          --num_eval_updates ${num_eval_updates} \
                          --pdropout ${pdropout} \
                          --reset_emb ${reset_emb} \
                          --reset_emb_decay ${reset_emb_decay}

                      done
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done
