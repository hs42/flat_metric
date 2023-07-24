# About
This code implements the computation of the flat distance between two distributions. Paper link: ToDo

# Installation
* Create a new conda environment and activate it:
    ```
    conda create -n fd
    conda activate fd
    ```
    
* Install PyTorch, following instructions in [PyTorch](https://pytorch.org). 

* Install torchnet by:
    ```
    pip install git+https://github.com/pytorch/tnt.git@master
    ```

* Navigate to the root of the project. Install the package, along with requirements:
    ```
    python setup.py install
    ```
    
* Add project root to PYTHONPATH. One way to do this: 
    ```
    export PYTHONPATH="${PYTHONPATH}:`pwd`"
    ``` 
    Usually, you want to add such an export of the enviromtal variable to your `.bashrc` file such that you don't have to re-run this command after every reboot.

# Usage
## General
The code provides a way to compute the flat distance between two distributions. Generally, these distributions are specified either by their law or by empirical samples (i.e. datasets). Both methods are explained in the `experiments/tutorial*` directories of this repository. 

Furthermore, all the experiments carried out in the paper can be found in the `experiments/paper_experiments` folder. There, you will also find scripts re-creating the paper's plots.

## Running
Each experiment is configured in a json file. There, the user may input the data to be analyzed, how many epochs to train for, how many layers to use, etc. The `experiments/tutorial*` sections provide a wrapper and a break-down of the most important of these properties. To get started, run for instance


```
python ./experiments/tutorial_compare_datasets/wrapper.py
```

If you configured your own experiment (and provided your own data files or distributions), you can start the estimation by

```
python ./lnets/tasks/dualnets/mains/train_dual.py <your_experiment.json>
```

In all of our paper scripts, this line is invoked in wrapper scripts corresponding to the respective experiments.

## Specifying input distributions
As mentioned, there are two ways in which the user may give input data. They can either simply provide a text file containing the data set (2 files for the 2 distributions in question). This is especially useful when analyzing experimental data; just make sure that the files are readable by np.loadtxt().

Alternatively, the user may specify a distribution by an analytical expression such that samples can be drawn from this law.

Kindly refer to the `experiments/tutorial*` directories to see how to actually use both methods.

## A note on specifying the masses of the measures
This implementation handles the masses of the measures by taking differently many samples into account for each distribution. For instance, if distribution 1 should hold double the mass of distribution 2, then the former could be approximated by e.g. 600 data points, while the latter one only counts 300 data points. 

This is configured as the "sample_size" entries in the corresponding .json file. For input via the dataset file this number should simply match the number of individual data points contained in the data set (if you measured 300 cells then chances are that the distribution you want to analyze corresponds to this amount).

## Running on GPU
If a GPU is available, we strongly encourage the users to turn on GPU training by turning on the related json field in
the experiment configs. In all experiments, set  `"cuda": true` or the `use_cuda = True` flag in the wrapper scripts.

By default, GPU support is disabled such that the user does not run into errors if their graphics card is not supported by Torch.


# File structure
The original name of the root directory LNets (=Lipschitz neural Networks) was kept for compatibility reasons.

Interesting directories or files include
* `lnets/tasks/dualnets/configs`: sample json configuration file. You can find both, illustrive examples (e.g. `absolute_value_experiment.json` or `Gaussian_uniform.json`) and the default architectures used for the paper experiments (`default_*.json`)
* `lnets/tasks/dualnets/distrib`: some generator scripts for drawing samples according to a law (e.g. Gaussian, uniform, Diracs, etc.). They can be used in the experiments by specifying them in the json files.
* `lnets/tasks/dualnets/mains/train_dual.py`: the script which actually trains a neural network. It expects a json file containing the configuration as a parameter

# Credits
This code is a fork from https://github.com/cemanil/LNets.git, which is the code used for "Sorting out Lipschitz function approximation" by Anil, Lucas, and Grosse (https://doi.org/10.48550/arXiv.1811.05381). This repository's commit 61d2f01 only
contains minimal changes to the original.
