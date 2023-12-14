CUDA_VISIBLE_DEVICES=$1 python pdt_meta.py --seed 123 --max_iters 100000 --n_layer 6 --n_head 8 \
    --test_eval_interval 1000 --sparsity 0.5  --num_eval_episodes 10 \
    --mask_change_ratio 0.0001 --conflict_thres 0. --merge_thres 25 --mask_interval 100  \
    --test_eval_interval 1000 --test_eval_seperate_interval 2000 \
    --prefix_name MT50