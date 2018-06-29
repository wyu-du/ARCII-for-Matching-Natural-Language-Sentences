# ARCII-for-Matching-Natural-Language-Sentences
A simple version of ARC-II model implemented in Keras.<br>
Please reference paper：<a href='https://arxiv.org/abs/1503.03244'>Convolutional Neural Network Architectures for Matching Natural Language Sentences</a>

## Quick Glance
1. Input Data Format
* Train set:
```
label	|q1	|q2<br>
1	|Amrozi accused his brother, whom he called "the witness", of deliberately distorting his evidence.	|Referring to him as only "the witness", Amrozi accused his brother of deliberately distorting his evidence.<br>
0	|Yucaipa owned Dominick's before selling the chain to Safeway in 1998 for $2.5 billion.	|Yucaipa bought Dominick's in 1995 for $693 million and sold it to Safeway for $1.8 billion in 1998.<br>
```

* Test set:
```
q1	|q2<br>
Amrozi accused his brother, whom he called "the witness", of deliberately distorting his evidence.	|Referring to him as only "the witness", Amrozi accused his brother of deliberately distorting his evidence.<br>
Yucaipa owned Dominick's before selling the chain to Safeway in 1998 for $2.5 billion.	|Yucaipa bought Dominick's in 1995 for $693 million and sold it to Safeway for $1.8 billion in 1998.<br>
```

* Word Embedding:
```
word	|embedding (300-dimension)<br>
Amrozi |-0.54645991 2.28509140 ... -0.34052843 -2.01874685
chief |-9.01635551 -3.80108356 ... 1.86873138 2.14706421
```

2. Train the model
```
$ python arcii.py
```

3. Loss and Accuracy


## Requirements
* Python 3.5
* TensorFlow 1.8.0
* Keras 2.1.6

## To do list
[] Negative Sampling<br>
[] Mask zero inputs<br>

