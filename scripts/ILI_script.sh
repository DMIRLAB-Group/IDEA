python -u run.py \
  --root_path ./dataset/illness/    \
  --data_path national_illness.csv   \
  --data custom    \
  --features M     \
  --enc_in 7   \
  --dec_in 7   \
  --c_out 7    \
  --seq_len 36  \
  --label_len 0  \
  --pred_len  12  \
  --model IDEA   \
  --zd_dim 3    \
  --hidden_dim 512  \
  --hidden_layers 3  \
  --dropout 0     \
  --activation relu  \
  --is_bn   \
  --learning_rate 0.001


python -u run.py \
  --root_path ./dataset/illness/    \
  --data_path national_illness.csv   \
  --data custom    \
  --features M     \
  --enc_in 7   \
  --dec_in 7   \
  --c_out 7    \
  --seq_len 72  \
  --label_len 0  \
  --pred_len  24  \
  --model IDEA   \
  --zd_dim 3    \
  --hidden_dim 512  \
  --hidden_layers 3  \
  --dropout 0     \
  --activation relu  \
  --is_bn   \
  --learning_rate 0.001


python -u run.py \
  --root_path ./dataset/illness/    \
  --data_path national_illness.csv   \
  --data custom    \
  --features M     \
  --enc_in 7   \
  --dec_in 7   \
  --c_out 7    \
  --seq_len 108  \
  --label_len 0  \
  --pred_len  36  \
  --model IDEA   \
  --zd_dim 3    \
  --hidden_dim 512  \
  --hidden_layers 3  \
  --dropout 0     \
  --activation relu  \
  --is_bn   \
  --learning_rate 0.001



python -u run.py \
  --root_path ./dataset/illness/    \
  --data_path national_illness.csv   \
  --data custom    \
  --features M     \
  --enc_in 7   \
  --dec_in 7   \
  --c_out 7    \
  --seq_len 144  \
  --label_len 0  \
  --pred_len  48  \
  --model IDEA   \
  --zd_dim 3    \
  --hidden_dim 640  \
  --hidden_layers 2  \
  --dropout 0.2     \
  --activation relu  \
  --is_bn   \
  --learning_rate 0.001   \
  --zd_kl_weight 0.0001  \
  --zc_kl_weight 0.0001   \
  --hmm_weight 0.0001