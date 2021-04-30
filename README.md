# mirror-nmt

## Requirements: 
* **Pytorch-1.1**
* Python 2.7
* CUDA 9.2+ (For GPU)

#### The main function for training is in train.py
To process a new dataset, run:
```
$ python -u preprocess.py -train_src $train_src -train_tgt $train_tgt -train_ctx $train_ctx -valid_src $valid_src -valid_tgt $valid_tgt -valid_ctx $valid_ctx  -save_data /daily_dialog/pair_daily -dynamic_dict -share_vocab -src_seq_length 45 -ctx_seq_length 100 -tgt_seq_length 45
```
$train_tgt is the path of target response in training set (utt at step t); $train_src is the path of source utteracne in training set (utt at step t-1); $train_ctx is the context utt (utt at step t-2).

To train a new model, just run:

```
$ python train.py -data data/mirror_dailydialog -save_model /model_dir/model_prefix -gpuid 0 -encoder_type rnn -param_init 0.08 -batch_size 128 -learning_rate 0.001 -optim adam -max_grad_norm 2 -word_vec_size 300 -enc_layers 2 -dec_layers 2 -rnn_size 1000 -epochs 65 -learning_rate_decay 0.98 -z_dim 100  -start_decay_at 3  -kl_balance 0.2  -mirror_type mirror -share_embeddings 
```

More hyper-parameter options can be found in file opts.py. Since this code is based on the open-nmt framework, you can go through the open-nmt instructions or the file README_OpenNMT.md if you want. 


To generate the responses with a trained model, just run:
```
python -u translate.py -model $model -src $test_src -tgt $test_tgt -ctx $test_ctx -output $mirror_output_2 -attn_debug -beam_size 10 -n_best 1 -batch_size 1 -verbose -gpu 0 -use_dc -1
```
$mirror_output_2 is for the file to save generated dialogues.


