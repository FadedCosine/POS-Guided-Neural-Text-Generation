


# F2-Softmax : Diversifying Neural Text Generation via Frequency Factorized Softmax


pytorch implementation

| Table of Contents |
|-|
| [Setup](#setup)|
| [Training](#training)|
| [Evaluation](#evaluation)|
| [Result](#result)|


## Setup
### Dependencies

Install other dependecies:
```bash
conda create -n f2_softmax_test python=3.6
conda activate f2_softmax_test
conda install cudatoolkit=10.1 -c pytorch -n f2_softmax_test 

pip install -r requirement.txt
mkdir data
mkdir data/checkpoint
```


We implemented with mixed precision using [apex](https://github.com/NVIDIA/apex) (pytorch-extension library)

```bash
cd ../
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir ./
cd ../EMNLP2020_submission
```





### Data preprocess
A script(`script/prepare_lm.sh`) performs preprocess data.
Wiki103 dataset will be preprocessed via torchtext library  in ```script/prepare_lm.sh``` 

```bash
sh script/prepare_lm.sh
```


### Training 

\*We tested these scripts using Titan RTX 24GB gpu in single and training with mixed precision.
If you get OOM errors, try decreasing ```batch_size``` in `config/training.yaml`.\
Each Loss function is implemented in a single script:

| Loss type                                                 | Script      | Description                                                  |
| ------------------------------------------------------------ | ----------- | ------------------------------------------------------------ |
| MLE                                                         | script/emnlp_train_mle.sh       | Basic |
| [FACE](<https://arxiv.org/pdf/1902.09191.pdf>) | script/emnlp_train_face.sh       | finetune:: need inital checkpoints from mle loss                                                              |
| [UL-token](<https://arxiv.org/pdf/1908.04319.pdf>)| script/emnlp_train_ul_token.sh     | |
| [UL-token-seq](<https://arxiv.org/pdf/1908.04319.pdf>) | script/emnlp_train_ul_token_seq.sh   | finetune:: need inital checkpoints from unlikelihood-token loss |
| \*F2-softmax (our)  | script/emnlp_train_f2_softmax.sh |  |





## Evaluation
### Calculate Perplexity
A script (`script/emnlp_evalutate.sh`) is for calculating perplexity.
Please specify `Data`, `saved-path` and `Mode`(loss)\


```bash
#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:./"
echo $PYTHONPATH

Mode=plain
Data=wiki103
CUDA_VISIBLE_DEVICES=0 python lm_ppl_eval.py \
    --saved-path data/$Data/... \ # specify model checkpoint path
    --dataset $Data \
    --loss-type $Mode \
    --root ./data \
    --encoder-class SPBPE \
    --vocab-size 30000;
```



### Sampling
A script (`script/emnlp_sample.sh`) performs sampling.
Please specify `Data` and `Mode`(loss)
Please set K (top-k) and S(temperature) in loop
you can choose top-1(greedy-deterministic) or top-k (stochastic) sampling.


```bash
#!/bin/bash

echo $PYTHONPATH

Mode=experimental
Data=wiki103
for K in 1 3 
do
  CUDA_VISIBLE_DEVICES=0 python lm_sample.py \
        --saved-path data/$Data/... \ # specify model checkpoint path
        --dataset $Data \
        --loss-type $Mode \ 
        --top-k $K \
        --sampling-mode 2 \
        --root ./data \
        --nprefix 50 \
        --ngenerate 100 ;
done
```


### Evaluation

A script (`script/corpuswise_eval.sh`) performs evaulation at 6 metrics

Please specify sampled file at ```$path```

```bash
#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH

for path in data/sampled/wiki103/* ; do
  CUDA_VISIBLE_DEVICES=0 python eval_corpus_wise_metric.py \
    --folderpath $path data/sampled/..
done
```

