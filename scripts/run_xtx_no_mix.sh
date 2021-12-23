LOG_FOLDER='xtx_no_mix_zork1'
GAME='zork1.z5'
SEED=0
JERICHO_SEED=$SEED # set to -1 if you want stochastic version
MODEL_NAME='xtx'
JERICHO_ADD_WT='add_wt' # change to 'no_add_wt' if you don't want to add extra actions to game grammar

# NOTE: r_for below corresponds to alpha_1 in the paper

python3 -m scripts.train_rl --output_dir logs/${LOG_FOLDER} \
                    --rom_path games/${GAME} \
                    --seed ${SEED} \
                    --jericho_seed ${JERICHO_SEED} \
                    --model_name ${MODEL_NAME} \
                    --eval_freq 10000000 \
                    --memory_size 10000 \
                    --T 1 \
                    --w_inv 1 \
                    --r_for 1 \
                    --w_act 1 \
                    --graph_num_explore_steps 50 \
                    --graph_rescore_freq 1000000 \
                    --env_step_limit 50 \
                    --graph_score_temp 1 \
                    --graph_q_temp 10000 \
                    --graph_alpha 0 \
                    --log_top_blue_acts_freq 500 \
                    --use_action_model 1 \
                    --action_model_update_freq 500 \
                    --action_model_type transformer \
                    --il_max_context 512 \
                    --max_acts 2 \
                    --il_vocab_size 2000 \
                    --il_k 10 \
                    --il_temp 3 \
                    --use_il 1 \
                    --il_batch_size 64 \
                    --il_max_num_epochs 40 \
                    --il_len_scale 1 \
                    --use_il_graph_sampler 0 \
                    --use_il_buffer_sampler 1 \
                    --il_top_p 1 \
                    --il_use_dropout 0 \
                    --traj_dropout_prob 0.0 \
                    --jericho_add_wt ${JERICHO_ADD_WT}