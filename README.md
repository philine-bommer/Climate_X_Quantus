</p>
<p align="center">

# Climate X Quantus

This repository contains the code and supplementary packages for the paper **["Finding the right XAI Method --- A Guide for the Evaluation and Ranking of Explainable AI Methods in Climate Science](https://arxiv.org/abs/2303.00652)**  by Bommer et. al.

![Version](https://img.shields.io/badge/version-0.0.1-green)

![Python version](https://upload.wikimedia.org/wikipedia/commons/f/fc/Blue_Python_3.7_Shield_Badge.svg)  ![Tensorflow](https://img.shields.io/badge/Tensorflow%20-1.15-orange) ![INNvestigate](https://img.shields.io/badge/INNvestigate-1.0.9-orange)

####This code is currently under active developments and further requirements such as preprocessed download are currently necessary to reproduce paper results. The library code is only meant to provide reproducability for the publication (code will not be updated to TF2). For own research we refer user to tutorials (for TF 2 and Pytorch based code):
* [Quantus X Climate Tutorial - CCAI](https://www.climatechange.ai/tutorials?search=id:quantus-x-climate) **(Tutorial for this publication)**
* [Quantus Tutorials](https://github.com/understandable-machine-intelligence-lab/Quantus) (for pytorch version)

*Please open issues to let the authors know about bugs and necessary fixes*
 
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
Create conda virtual env from sepc file. (Note that, innvestigate v.1.0.9 might has to be installed via pip)
```setup
conda create --name myenv --file spec-file.txt
```

or create and install packages from spec-file. (Note that, innvestigate v.1.0.9 might has to be installed via pip)

```setup
conda create --name myenv python=3.7.11
conda activate myenv
conda install --name myenv --file spec-file.txt
```

[comment]: <> (Make sure to install the requirements in the conda virtual env.)

[comment]: <> (```setup)

[comment]: <> (while read requirement; do conda install --yes $requirement; done < requirements.txt)

[comment]: <> (while read requirement; do conda install -c conda-forge --yes $requirement; done < requirements.txt)

[comment]: <> (```)

#### Package requirements

The package requirements are as follows:
```
python<=3.7.0
innvestigate==1.0.9
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
creating annual averages, standardisation, creating class labels and splitting the data for the network training and testing.

**Model Data:** For the script to run you have to download the 40-member Large Ensemble data using the RCP8.5 setting as detailed in the [data instructions](https://www.cesm.ucar.edu/community-projects/lens/instructions). 
The corresponding nc.-files should be stored in the **'Climate_X_Quantus/Data/Raw/LENS/monthly/'** folder and each file should be renamed to  'CESM1A_All1-start-end.nc', with start marking the start year and 
end marking the end year as stated in the original file names.

**Observation Data:** You have also download the monthly 2m Air Temperature of the [20th century Reanalysis data (V3)](https://www.psl.noaa.gov/data/gridded/data.20thC_ReanV3.html) as Monthly values for 1836/01 to 2015/12. 
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
The preprocessed data for the paper results is available on [zenodo](https://zenodo.org/record/7715398#.ZAsvSS8w1zU) with technical details and file documentation provided in the datasets **Readme.md** .
The downloaded data called 'Preprocessed_datat_NET_CESM1_obs_20CRv3.npz' (with two files for NET=CNN/MLP) should unpacked and be saved in **/Climate_X_Quantus/Data/Training/**.
To skip the data preparation adjust the data_config.yaml as described above and set up the **'NET_data.yaml'** with corresponding to the NET you want to run as follows:
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

### Explanation method comparison (Section 4b)
The explanation method comparison experiment according to section 4b in **[Bommer et. al 2023](https://arxiv.org/abs/2303.00652)** can be performed by running **'QuantusExperiment_skill.py'**. Here you can choose to create results for 
two representative metrics of either the robustness ("Robustness") property, faithfulness ("Faithfulness") or complexity with randomisation and localisation ("Complexity") by adjusting the **'plot_config.yaml'** as follows: 
```setup
base: 1
property: 'Robustness'
```
with 'Robustness' as an example. 
*Note that to perform the experiment for the MLP as in the paper please in Data_config.yaml set net = 'MLP' and run data_postprocessing.py first.*

The resulting mean skill scores across 50 explanations and standard error of the mean (SEM) will be saved as individual pkl.-files and npz.-files to enable plotting as provided in **Plot_Tables.ipynb**. All results can be found in 
**'/Climate_X_Quantus/Data/Quantus/Baseline/'**

### Comprehensive Evaluation
The comparison across network architecture according to section 4c in **[Bommer et. al 2023](https://arxiv.org/abs/2303.00652)** can be performed by running **'QuantusExperiment_skill.py'** for the CNN.
To perform the experiment for the CNN as in the paper please in Data_config.yaml set net = 'CNN' and run data_postprocessing.py first. Maintain the same settings in **'plot_config.yaml'**.

The resulting mean skill scores across 50 explanations and standard error of the mean (SEM) will be saved as individual pkl.-files and npz.-files to enable plotting as provided in **Plot_Tables.ipynb**, which includes
the plot script for Figure 8, 9, and 10a (the spyder plot for the MLP). All results can be found in 
**'/Climate_X_Quantus/Data/Quantus/Baseline/'**

### DeepShap
The calculations have been seperated into a Colab python notebook due to version conflicts with innvestigate v.1.0.9. Thus, the evaluation protocol 
for Deep Shap following evaluation procedure and skill score calculation described in **[Bommer et. al 2023](https://arxiv.org/abs/2303.00652)** can be found in **'Experiment_DeepShap.ipynb'**. 
*We suggest running the notebook using Google Colab.*


## Additional Plots

### Figures 8 - 10a
The plots for Figures 8 - 10a from Section 4 in **[Bommer et. al 2023](https://arxiv.org/abs/2303.00652)** can be reproduced by running the python notebook. 
*We highly suggest running the notebooks via Google Colab.*

Follow the instructions as details in the notebook. All figures will be saved in '/Climate_X_Quantus/Figures/'.

### Figures B4 and B5
The plot routine to plot the temporal average of the explanation maps across all XAI methods as displayed in Figure B4 and B5 in **[Bommer et. al 2023](https://arxiv.org/abs/2303.00652)** run **'Plot_temporalAverageMaps.py'**.
The DeepShap explanations have to be generated for both networks by running **'Explanation_DeepShap.ipynb'**. Before running the plot routine, adjust the **'plot_config.yaml'** as follows:
```setup
base: net
```
with net = 'MLP' to plot the explanation of the MLP predictions or net = 'CNN' to plot the explanation amps of the CNN predictions. 



## Further references

* [Quantus Tutorials](https://github.com/understandable-machine-intelligence-lab/Quantus)
* [Explanation Tutorial](https://github.com/albermax/innvestigate)

**Please note that end of april we will link here to a TF2 tutorial on climate XAI evaluation with Quantus **