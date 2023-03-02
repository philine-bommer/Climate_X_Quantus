</p>
<p align="center">

# Climate X Quantus

This repository contains the code and supplementary packages for the paper **["Finding the right XAI Method --- A Guide for the Evaluation and Ranking of Explainable AI Methods in Climate Science](https://arxiv.org/abs/2303.00652)**  by Bommer et. al.

![Python version](https://upload.wikimedia.org/wikipedia/commons/f/fc/Blue_Python_3.7_Shield_Badge.svg) ![Conda - License](https://img.shields.io/conda/l/conda-forge/setuptools)

This code is currently under active development and will be continuously updated including further tutorials and version which will provide fully running 
experiment version. Further requirements such as preprocessed download are currently necessary to reproduce paper results.

## Citation

If you find this work or papers for included methods interesting or useful in your research, use the following Bibtex annotation to cite us:

```bibtex
@article{bommer2023finding,
 author = {Philine Bommer, Marlene Kretschmer, Anna Hedström, Dilyara Bareeva, Marina M.-C. Höhne},
  title = {Finding the right XAI method -- A Guide for the Evaluation and Ranking of Explainable AI Methods in Climate Science},
  doi = {10.48550/arXiv.2303.00652},
  url = {https://arxiv.org/abs/2303.00652},
  publisher = {arXiv},
}
```

Further citations referring to used other used and referenced packages will be provided in the reference section and according to their introduction in the instructions.

## Table of Contents
1. [Motivation](#motivation)
2. [Additional Libraries](#library)
3. [Data Instructions](#data)
   
   3.1.[Data Download](#download)
   
   3.2.[Data Preprocessing](#preprocesing)
   
4. [Experiment Instructions](#Experiments)
   
    4.1 [Training and Explanation](#training)
   
    4.2 [Baseline Test](#Baseline)
   
    4.3 [Network Comparison](#Network)
   
6. [Additional Plots](#plots)

7. [Further references](#Refs)

## Motivation

Explainable artificial intelligence (XAI) methods shed light on the predictions of deep neural networks (DNNs). 
Several different approaches exist and have partly already been successfully applied in climate science. 
However, the often missing ground truth explanations complicate their evaluation and validation, subsequently compounding the choice of the XAI
method. Therefore, here, we introduce XAI evaluation in the context of climate research and assess different desired explanation properties, namely, robustness, faithfulness, randomization,
complexity, and localization, as provided by [Quantus by Hedström et. al. 2022](https://www.jmlr.org/papers/volume24/22-0142/22-0142.pdf).
</p>
<p align="center">
  <img width="600" src="https://github.com/philine-bommer/Climate_X_Quantus/blob/main/FinalFirstGraph_v1.png">


Building upon previous research **[Labe et. al. 2021](https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/2021MS002464)** this repository includes experiments to train an MLP and a CNN based on 2m-temperature maps to predict the respective decade class (see step 1 in Figure above which illustrates this workflow). 
To make the decision comprehensible, several explanation methods are applied, which vary in their explained evidence and might lead to different conclusions (step 2 in the Figure above). 
Therefore, in two experiment you can apply XAI evaluation metrics from [Quantus by Hedström et. al. 2022](https://www.jmlr.org/papers/volume24/22-0142/22-0142.pdf) to quantitatively measure the performance of the different XAI methods (step 3 in Figure above). 
We provide experiments to score the different explanation methods and compare the scores to the score achieved by a Random Baseline drawn from a uniform distribution U[0,1]. The scores are also ranked and we provide means to plot the normalized scores 
as well as the ranked scores in a spyder plot (based on [Quantus by Hedström et. al. 2022](https://www.jmlr.org/papers/volume24/22-0142/22-0142.pdf)) to provide statements about the respective suitability for the underlying climate task.
With this work hope to supports climate researchers in the selection of a suitable XAI method.