CUDA_VISIBLE_DEVICES=$1 python pdt_meta.py --seed 123 --max_iters 1000000 --n_layer 6 --n_head 8 \
    --test_eval_interval 5000  --num_eval_episodes 10 --test_eval_seperate_interval 10000\
    --mask_interval 500 --sparsity 0.5 --mask_change_ratio 0.0001 --conflict_thres 0. --merge_thres 25 \
    --prefix_name MT50