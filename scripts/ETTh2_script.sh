python -u run.py \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --data ETTh2  \
  --features M  \
  --enc_in 7 \
  --dec_in 7  \
  --c_out 7  \
  --seq_len 36  \
  --label_len 0  \
  --pred_len  12 \
  --model IDEA  \
  --zd_dim 3  \
  --hidden_dim 512  \
  --hidden_layers 2  \
  --dropout 0  \
  --activation ide  \
  --learning_rate 0.001


python -u run.py \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --data ETTh2  \
  --features M  \
  --enc_in 7 \
  --dec_in 7  \
  --c_out 7  \
  --seq_len 72  \
  --label_len 0  \
  --pred_len  24 \
  --model IDEA  \
  --zd_dim 3  \
  --hidden_dim 512  \
  --hidden_layers 2  \
  --dropout 0  \
  --activation ide  \
  --learning_rate 0.001


python -u run.py \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --data ETTh2  \
  --features M  \
  --enc_in 7 \
  --dec_in 7  \
  --c_out 7  \
  --seq_len 144  \
  --label_len 0  \
  --pred_len  48 \
  --model IDEA  \
  --zd_dim 3  \
  --hidden_dim 512  \
  --hidden_layers 2  \
  --dropout 0  \
  --activation ide  \
  --learning_rate 0.001

python -u run.py \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --data ETTh2  \
  --features M  \
  --enc_in 7 \
  --dec_in 7  \
  --c_out 7  \
  --seq_len 216  \
  --label_len 0  \
  --pred_len  72 \
  --model IDEA  \
  --zd_dim 3  \
  --hidden_dim 256  \
  --hidden_layers 2  \
  --dropout 0  \
  --activation ide  \
  --learning_rate 0.001

