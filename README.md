</p>
<p align="center">

# Climate X Quantus

This repository contains the code and supplementary packages for the paper **["Finding the right XAI Method --- A Guide for the Evaluation and Ranking of Explainable AI Methods in Climate Science](https://arxiv.org/abs/2303.00652)**  by Bommer et. al.

![Version](https://img.shields.io/badge/version-0.0.1-green)

![Python version](https://upload.wikimedia.org/wikipedia/commons/f/fc/Blue_Python_3.7_Shield_Badge.svg)  ![Tensorflow](https://img.shields.io/badge/Tensorflow%20-1.15-orange) ![INNvestigate](https://img.shields.io/badge/INNvestigate-1.0.9-orange)

####This code is currently under active development and will be continuously updated including further tutorials and version which will provide fully running experiments and plots. Further requirements such as preprocessed download are currently necessary to reproduce paper results.
**Please open issues to let the authors know about bugs and necessary fixes**

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
3. [Instructions](#data)
   
   3.1.[Data](#download)
   
   3.2.[Data Preprocessing](#preprocesing)
   
4. [Experiment Instructions](#Experiments)
   
    4.1 [Training and Explanation](#training)
   
    4.2 [Baseline Test](#Baseline)
   
    4.3 [Comprehensive Evaluation](#Network)
   
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


Building upon previous research [Labe et. al. 2021](https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/2021MS002464) this repository includes experiments to train an MLP and a CNN based on 2m-temperature maps to predict the respective decade class (see step 1 in Figure above which illustrates this workflow). 
To make the decision comprehensible, several explanation methods are applied, which vary in their explained evidence and might lead to different conclusions (step 2 in the Figure above). 
Therefore, in two experiment you can apply XAI evaluation metrics from [Quantus by Hedström et. al. 2022](https://www.jmlr.org/papers/volume24/22-0142/22-0142.pdf) to quantitatively measure the performance of the different XAI methods (step 3 in Figure above). 
We provide experiments to score the different explanation methods and compare the scores to the score achieved by a Random Baseline drawn from a uniform distribution U[0,1]. The scores are also ranked and we provide means to plot the normalized scores 
as well as the ranked scores in a spyder plot (based on [Quantus by Hedström et. al. 2022](https://www.jmlr.org/papers/volume24/22-0142/22-0142.pdf)) to provide statements about the respective suitability for the underlying climate task.
With this work hope to supports climate researchers in the selection of a suitable XAI method.

## Additional Libraries

The code offered in this repository and provided experiments are partly based on and extensions of exsisting code libraries. 
We want to acknowledge the following contributions:

* **CphXAI-Package**: Data preprocessing, network structure and training code is based on [Code](https://zenodo.org/record/4890496#.ZACRqS8w1pQ) by [Labe et. al. 2021](https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/2021MS002464)
  

* **NoiseGrad-Package**: NoiseGrad and FusionGrad extensions for TF 1 are based on [Package](https://github.com/understandable-machine-intelligence-lab/NoiseGrad) by [Bykov et. al. 2022](https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/2021MS002464)


* **[QuantusClimate](https://github.com/philine-bommer/QuantusClimate)**: Fork with TF1 models and parallelized computation of Faithfulness, Robustness and Randomisation metrics, of the [Quantus Package](https://github.com/understandable-machine-intelligence-lab/Quantus) by [Hedström et. al. 2022](https://www.jmlr.org/papers/volume24/22-0142/22-0142.pdf)


## Instructions

You can download a local copy (and then, access the folder):

```setup
git clone https://github.com/philine-bommer/Climate_X_Quantus.git
cd Climate_X_Quantus
```
Make sure to install the requirements.
```setup
pip install -r requirements.txt
```

#### Package requirements

The package requirements are as follows:
```
python<=3.7.0
```

As well as the 3 included packages, each by changing to the according package folder (CCxai, CphXAI, NoiseGrad, QuantusClimate):
```setup
cd folder
pip install . --user
```
### Data 

For the data used in this work we acknowledge the [CESM Large Ensemble Community Project](https://www.cesm.ucar.edu/projects/community-projects/LENS/data-sets.html)
If you build upon this work or use the according raw model data, cite as follows:
```bibtex
@article { TheCommunityEarthSystemModelCESMLargeEnsembleProjectACommunityResourceforStudyingClimateChangeinthePresenceofInternalClimateVariability,
      author = "J. E. Kay and C. Deser and A. Phillips and A. Mai and C. Hannay and G. Strand and J. M. Arblaster and S. C. Bates and G. Danabasoglu and J. Edwards and M. Holland and P. Kushner and J.-F. Lamarque and D. Lawrence and K. Lindsay and A. Middleton and E. Munoz and R. Neale and K. Oleson and L. Polvani and M. Vertenstein",
      title = "The Community Earth System Model (CESM) Large Ensemble Project: A Community Resource for Studying Climate Change in the Presence of Internal Climate Variability",
      journal = "Bulletin of the American Meteorological Society",
      year = "2015",
      publisher = "American Meteorological Society",
      address = "Boston MA, USA",
      volume = "96",
      number = "8",
      doi = "10.1175/BAMS-D-13-00255.1",
      pages=      "1333 - 1349",
      url = "https://journals.ametsoc.org/view/journals/bams/96/8/bams-d-13-00255.1.xml"
}
```
For the observational data please make sure to cite the following article by [Slivinski et. al. 2019](https://rmets.onlinelibrary.wiley.com/action/showCitFormats?doi=10.1002%2Fqj.3598) and acknowledge the usage. 

#### Raw data
The workflow of the experiments includes 4 steps, starting with **Experiments/data_preparation.py**. This script runs the data preprocessing including
creating annual averages, standardization, creating class labels and splitting the data for the network training and testing.

**Model Data:** For the script to run you have to download the 40-member Large Ensemble data using the RCP8.5 setting as detailed in the [data instructions](https://www.cesm.ucar.edu/community-projects/lens/instructions). 
The corresponding nc.-files should be stored in the **'Climate_X_Quantus/Data/Raw/LENS/monthly/'** folder and each file should be renamed to  'CESM1A_All1-start-end.nc', with start marking the start year and 
end marking the end year as stated in the original file names.

**Observation Data:** You have also download the monthly 2m Air Temperture of the [20th century Reanalysis data (V3)](https://www.psl.noaa.gov/data/gridded/data.20thC_ReanV3.html) as Monthly values for 1836/01 to 2015/12. 
The corresponding nc.-files should be stored in the **'Climate_X_Quantus/Data/Raw/20CRv3/monthly/'** folder and the file should be renamed to  'T2M_1836-2015.nc'.

### Data Preprocessing
After the data is saved open **'Experiments/Data_config.yaml'** and make the following changes in the config file:
```setup
dirhome: 'path/Climate_X_Quanus/'
params:
  net: 'network'
```
with path - home path of cloned repo and network = (CNN, MLP) according to the experiment you want to run. 

#### Downloading preprocessed data: 
**Note that this step can be skipped to reproduce paper results.**
The preprocessed data for the paper results will be made available on [zenodo]() within the next week and links will be included.
The downloaded data called 'Preprocessed_datat_NET_CESM1_obs_20CRv3.npz' (with two files for NET=CNN/MLP) should be saved in **/Climate_X_Quantus/Data/Training/**.
To skip the data preparation adjust the data_config.yaml as described above and set up the **'NET_data.yaml'** corresponding to the NET you want to run:
```setup
dirhome: /path/Climate_x_Quantus/
diroutput: /path/Climate_x_Quantus/Data/Training/
```
with path - home path of cloned repo

## Experimental Instructions

To start the experiments, either the preprocessed data should have been downloaded or the raw data preprocessed. **Note that** at the moment 
the python-files included in experiments should be run consecutively. Furthermore the set network configuration (MLP or CNN) in the 'data_config.yaml' determines 
which network explanations are evaluated. 

### Training and Explanation
To reproduce paper experiemnts no further adjustments are needed and you can run **'training_and_explanation.py'** to 
train a network and calculate the explanations for Gradients, SmoothGrad, InputGradients, Integrated Gradients, LRPz, LRPab, LRPcomposite(CNN), NoiseGrad and FusionGrad.

You can also adjust hyperparameters, such as batch size, regularization and learning rate in the **'NET_config.yaml'** under 
```setup
params: 
      lr: 
train: 
   ridge_penalty:
```
make sure to maintain the format of each adjusted parameter e.g. if '[lr]' maintain list format. 
The trained network is saved und '/Climate_X_Quantus/Network/' and the explanations in '/Climate_X_Quantus/Data/Training/NET/' for both NET configurations.

#### Postprocessing:
In order to prepare the data for evaluation please run the 'data_postprocessing.py' also contained in 'Climate_X_Quantus/Experiments/'. No further adjustments of the 'Post_config.yaml' are needed.

### Baseline Test
The Baseline test **'QuantusExperiment_BaselineTest.py'** runs the experiment according to section 4.2 in **[Bommer et. al 2023](https://arxiv.org/abs/2303.00652)**. Here you can choose to create results for 
two representative metrics of either the robustness ("Robustness") property, faithfulness ("Faithfulness") or complexity with randomisation and localisation ("Complexity") by adjusting the **'plot_config.yaml'** as follows: 
```setup
base: 1
property: 'Robustness'
```
with 'Robustness' as an example.

The resulting mean scores across 50 explanations and standard error of the mean (SEM) will be saved as individual csv.-files alongside the ranked scores. All results can be found in 
**'/Climate_X_Quantus/Data/Quantus/Baseline/'**

### Comprehensive Evaluation
The Comprehensive Evaluation **'QuantusExperiment_Comprehensive.py'** runs the experiment according to section 4.3 in **[Bommer et. al 2023](https://arxiv.org/abs/2303.00652)**.
You can decide to include the random baseline as comparitive value into the results and if enabled (see below base set to 1), the spyder plot will include the baseline value and conist of the 
scores instead of the ranks as in Fig. 8 of **[Bommer et. al 2023](https://arxiv.org/abs/2303.00652)**.
Settings are the following:
```setup
base: 1
```
with base disabled resulting in a a spyder plot of the ranked scores.

The resulting mean scores across 50 explanations and standard error of the mean (SEM) will be saved as individual csv.-files alongside the ranked scores. All results can be found in
**'/Climate_X_Quantus/Data/Quantus/Comprehensive/'**. The spyder plot in the chosen settings will be saved in '/Climate_X_Quantus/Figures/'

## Additional Plots

**Please note that we will soon update this repository to include more relevant plotting routines using the CCxai package.**

## Further references

* [Quantus Tutorials](https://github.com/understandable-machine-intelligence-lab/Quantus)
* [Explanation Tutorial](https://github.com/albermax/innvestigate)

**Please note that end of april we will link here to a TF2 tutorial on climate XAI evaluation with Quantus **