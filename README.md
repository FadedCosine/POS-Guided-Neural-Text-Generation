


# Diversifying Neural Text Generation with Part-of-Speech Guided Softmax and Sampling


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
conda create -n POS_Softmax python=3.6
conda activate POS_Softmax
conda env create -f environment.yaml
```

We implemented with mixed precision using [apex](https://github.com/NVIDIA/apex) (pytorch-extension library)

```bash
cd ../
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir ./
cd ../POS-Guided-Neural-Text-Generation
```


### Data preprocess
First, download [wikitext-103](https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip) to `data/wikitext-103` and [ParaNMT-50M](https://drive.google.com/file/d/1rbF3daJjCsa1-fu2GANeJd2FBXos1ugD/view) to `data/paraNMT` .


Next, download the [Stanford CoreNLP](http://nlp.stanford.edu/software/stanford-corenlp-full-2018-02-27.zip). And start a local Stanford CoreNLP Server.
```java
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9876 -timeout 15000
```
For paraphrase generation, we first filter the [ParaNMT-50M](https://drive.google.com/file/d/1rbF3daJjCsa1-fu2GANeJd2FBXos1ugD/view) dataset with a script(`scripts/filter_paraNMT_data.sh`). You need to specify the path where you install the Stanford CoreNLP in this script.
```bash
sh scripts/filter_paraNMT_data.sh
```

Then, we use StanfordCoreNLP to annotate the text with POS taggers.
```bash
sh scripts/pos_tagging.sh
```

After all above finished, A script(`scripts/prepare_dataset.sh`) is used to preprocess data, including coverting tokens to ids, count the token frequency for F2-Softmax, etc.

```bash
sh scripts/prepare_dataset.sh
```


### Training 

\*We tested these scripts using Titan RTX 24GB gpu in single and training with mixed precision.
If you get OOM errors, try decreasing ```batch_size``` in `config/training.yaml`.\
Each Loss function is implemented in a single script:

| Loss type                                                 | Script      | Description                                                  |
| ------------------------------------------------------------ | ----------- | ------------------------------------------------------------ |
| MLE                                                         | scripts/train_mle.sh       | Basic |
| [FACE](<https://arxiv.org/pdf/1902.09191.pdf>) | scripts/train_face.sh       | finetune:: need inital checkpoints from mle loss                                                              |
| [UL](<https://arxiv.org/pdf/1908.04319.pdf>)| scripts/train_ul.sh     | |
| [F2-Softmax](<https://www.aclweb.org/anthology/2020.emnlp-main.737/>)  | scripts/train_f2_softmax.sh |  |
| \*POS guided Softmax (ours) | scripts/train_pos_softmax.sh |  |





## Evaluation
### Calculate Perplexity
A script (`scripts/evalutate_ppl.sh`) is for calculating perplexity.
Please specify `Data`, `saved-path` and `Mode`(loss)\

```bash
#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:./"
echo $PYTHONPATH

Mode=MLE
Data=wikitext-103
CUDA_VISIBLE_DEVICES=0 python lm_ppl_eval.py \
    --saved-path data/$Data/... \ # specify model checkpoint path
    --dataset $Data \
    --loss-type $Mode \
    --root ./data \
    --vocab-size 200000;
```

### Sampling
A script (`scripts/sample.sh`) performs sampling for baselines.
Please specify `Data` and `Mode`(loss)
Please set P (nucleus sampling for token sampling) in loop.

```bash
#!/bin/bash

echo $PYTHONPATH

Mode=UL
Data=wikitext-103
for P in 0.8 0.9 0.7 0.6 0.5 0.4 0.3
do
  CUDA_VISIBLE_DEVICES=0 python sample.py \
        --saved-path data/$Data/... \ # specify model checkpoint path
        --dataset $Data \
        --loss-type $Mode \ 
        --top-p $K \
        --sampling-mode 2 \
        --root ./data \
        --nprefix 50 \  # for paraphrase generation, no need to specify this argument
        --ngenerate 100 ; # for paraphrase generation, no need to specify this argument
done
```

A script (`scripts/pos_sample.sh`) performs sampling for our POS guided Sampling. You can specify `pos-top-p` and `pos-top-k` to compare different sampling strategis in POS sampling. Note that only one of `pos-top-p` or `pos-top-k` should be specified.

```bash
#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH

Mode=POS
Data=wikitext-103

for P in 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
  CUDA_VISIBLE_DEVICES=0 python sample.py \
      --saved-path data/checkpoint/$Data/_$Mode''_layer_6_lr_0.0001_cutoffs_17_core_epoch_8 \
      --dataset $Data \
      --loss-type $Mode \
      --top-p $P \ 
      --pos-top-p ... \ # specify a sampling hyper-parameter for POS's nucleus sampling
      --pos-top-k ... \ # specify a sampling hyper-parameter for POS's top-k sampling. Note that only one of --pos-top-p or --pos-top-k should be specified.
      --sampling-mode 3 \
      --root ./data \
      --nprefix 50 \
      --ngenerate 100 \
      --vocab-size 100000;
done

```
### Evaluation

Script (`scripts/evaluate_lm.sh`) and (`scripts/evaluate_para.sh`) are used to evaulate the automatic metrics for language modeling task and paraphrase generation, respectively.

Please specify dataset at ```$Data``` and sampling hyper-parameters: ```top-p``` and ```top-k```.

```bash
#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH

Mode=MLE
Data=wikitext-103

for P in 0.5
do
    CUDA_VISIBLE_DEVICES=4 python eval_LM_metric.py \
        --folderpath data/sampled/$Data/prefix-50_nsample-100 \
        --top-p $P \
        --top-k 0
done
```

Besides, we also calculate the [BERTScore](https://github.com/Tiiiger/bert_score) in paraphrase generation task.

## Credits

The code in this repository and portions of this README are based on [F2-Softmax](<https://www.aclweb.org/anthology/2020.emnlp-main.737/>) 

## **Citation**
If you find our paper useful to your work, please kindly cite our paper:

```
@inproceedings{yang-etal-2022-diversifying,
    title = "Diversifying Neural Text Generation with Part-of-Speech Guided Softmax and Sampling",
    author = "Yang, Zhixian  and
      Xu, Pengxuan  and
      Wan, Xiaojun",
    booktitle = "Proceedings of the 29th International Conference on Computational Linguistics",
    month = oct,
    year = "2022",
    address = "Gyeongju, Republic of Korea",
    publisher = "International Committee on Computational Linguistics",
    url = "https://aclanthology.org/2022.coling-1.570",
    pages = "6547--6563",
    abstract = "Neural text generation models are likely to suffer from the low-diversity problem. Various decoding strategies and training-based methods have been proposed to promote diversity only by exploiting contextual features, but rarely do they consider incorporating syntactic structure clues. In this work, we propose using linguistic annotation, i.e., part-of-speech (POS), to guide the text generation. In detail, we introduce POS Guided Softmax to explicitly model two posterior probabilities: (i) next-POS, and (ii) next-token from the vocabulary of the target POS. A POS Guided Sampling strategy is further proposed to address the low-diversity problem by enriching the diversity of POS. Extensive experiments and human evaluations show that, compared with existing state-of-the-art methods, our POS Guided Softmax and Sampling (POSG) can generate more diverse text while maintaining comparable quality.",
}
```
