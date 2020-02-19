#!/usr/bin/env bash

iterations=50000
anum_fns=(1000)
anum_eval_updates=(40)
anum_inner_updates=(0)
atasks_per_metaupdate=(1)
alr_inner=(0.001)
alr_meta=(0.001)
alr_scheduler_period=(5000)
alr_model_decay=(1)
alr_emb_decay=(1)
apdropout=(0)
areset_emb=(-1)
areset_emb_decay=(1)
anum_tasks_check=(100) # 10 50 100 1000
anum_points_check=(10)

for num_fns in "${anum_fns[@]}"; do
  for num_eval_updates in "${anum_eval_updates[@]}"; do
    for num_inner_updates in "${anum_inner_updates[@]}"; do
      for tasks_per_metaupdate in "${atasks_per_metaupdate[@]}"; do
        for lr_inner in "${alr_inner[@]}"; do
          for lr_meta in "${alr_meta[@]}"; do
            for lr_scheduler_period in "${alr_scheduler_period[@]}"; do
              for lr_model_decay in "${alr_model_decay[@]}"; do
                for lr_emb_decay in "${alr_emb_decay[@]}"; do
                  for pdropout in "${apdropout[@]}"; do
                    for reset_emb in "${areset_emb[@]}"; do
                      for reset_emb_decay in "${areset_emb_decay[@]}"; do
                        for num_tasks_check in "${anum_tasks_check[@]}"; do
                          for num_points_check in "${anum_points_check[@]}"; do

                            tsk="tpg_nf${num_fns}_lri${lr_inner}_lrm${lr_meta}_lrs${lr_scheduler_period}_lrmd${lr_model_decay}_lred${lr_emb_decay}_tpu${tasks_per_metaupdate}_neu${num_eval_updates}_dop${pdropout}_re${reset_emb}_red${reset_emb_decay}_ntc${num_tasks_check}_npc${num_points_check}"

                            echo "starting " ${tsk}

                            python ./main.py \
                              --cva \
                              --tpg \
                              --num_tasks_check "${num_tasks_check}" \
                              --num_points_check "${num_points_check}" \
                              --id "${tsk}" \
                              --n_iter "${iterations}" \
                              --num_fns "${num_fns}" \
                              --lr_inner "${lr_inner}" \
                              --lr_meta "${lr_meta}" \
                              --lr_scheduler_period "${lr_scheduler_period}" \
                              --lr_model_decay "${lr_model_decay}" \
                              --lr_emb_decay "${lr_emb_decay}" \
                              --tasks_per_metaupdate "${tasks_per_metaupdate}" \
                              --num_inner_updates "${num_inner_updates}" \
                              --num_eval_updates "${num_eval_updates}" \
                              --pdropout "${pdropout}" \
                              --reset_emb "${reset_emb}" \
                              --reset_emb_decay "${reset_emb_decay}"

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
  done
done

