"""
This is a wrapper script to easily access the most important parameters when computing the flat distance between
two measures as given by empirical samples

Typically, the syntax would be
python ./lnets/tasks/dualnets/mains/train_dual.py <path to experiment configuration json file>,
The configuration json file hereby specifies both, the datasets in question, and the architecture of the neural net.

This wrapper script takes care of that, such that you only need to execute this wrapper.py file
That means, it will automatically load a json file; in our case that is the default_datasets.json file found in this directory.
It specifies where the files for the datasets can be found. In our case, the dataset for distribution 1 consits of samples from
a uniform distribution centered around -3 and distribution 2, which is a uniform distribution centered around 3.
For details, there is a small Readme file in the data subfodler 
"""

import json
import subprocess
import os
import numpy as np

__basedir__ = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir)
__file_dir__ =os.path.dirname(os.path.abspath(__file__))

"""
Firstly, read in the configuration file
"""
#read data
with open(os.path.join(__file_dir__, 'default_datasets.json')) as f:
   data = json.load(f)

"""
Adapt parameters.
The json file specifies all finer points needed for training. In the following, we will adapt the most important of those parameters.
This should prove illustrive to the reader such that they can pick up the use of the programm from here and put it to their use.
"""

"""
Most important parameters
"""

#specify where the training data should be saved
out_path = os.path.join(__basedir__, 'out', 'compare_datasets')
data['output_root'] = out_path #save to config

"""
Specify distributions
"""
#When we want to compare datasets, we simply specify where a text file containing the samples is stored. Note that this
#text file must be readable by np.loadtxt()

#We are able to pass information about the masses of the distributions as the amount of samples to consider.
#E.g. if dist1 has sample_size=300 but dist2 has sample_size=100, then dist1 is made to have three times the probability mass of dist2
#Usually, a canonical choice is to set this number to the available samples. For instance, if you measure cells and cell type 1 shows 300 cells
#but cell type 2 only 100 than this encodes already the information about differing normalizations of their respective measure and you would
#set sample_size=300 for dist1 and sample_size=100 for dist1.
#You as a user can of course also set different values for the sample sizes, but they must reflect the the ratio of the normalizations. Otherwise,
#computing the flat distance is futile and will yield no meaninful insight.

#specify distribution 1
data['distrib1']['path_train'] = os.path.join(__file_dir__, 'data', 'train_samples_1') #specify where the training data file can be found
data['distrib1']['sample_size'] = 300 #how many samples to consider

data['distrib1']['path_val'] = os.path.join(__file_dir__, 'data', 'val_samples_1') #Optional - where data for validation after each epoch can be found. Can be left empty
data['distrib1']['test_sample_size'] = 50 #how many samples should be used for testing

data['distrib1']['path_test'] = os.path.join(__file_dir__, 'data', 'test_samples_1') #Optional - where data for testing after all of the training. Can be left empty

#specify distribution 2
data['distrib2']['path_train'] = os.path.join(__file_dir__, 'data', 'train_samples_2') #specify where the training data file can be found
data['distrib2']['sample_size'] = 120 #how many samples to consider

data['distrib2']['path_val'] = os.path.join(__file_dir__, 'data', 'val_samples_2') #Optional - where data for validation after each epoch can be found. Can be left empty
data['distrib2']['test_sample_size'] = 20 #how many samples should be used for testing

data['distrib2']['path_test'] = os.path.join(__file_dir__, 'data', 'test_samples_2') #Optional - where data for testing after all of the training. Can be left empty

#Note: a few hundred datapoints are most often rather too little to train a neural network on 5-dimensional input data. We remedied this by reducing the
#size of the network. In the default config it possess two hidden layers of 64 neurons each. You may want to play around with that yourself to get a feeling


"""
Misc parameters
"""
use_cuda = False #whether to use GPU or not
data['cuda'] = use_cuda

save_best_model = False #whether or not to store the best model for each training. These will be stored in the training output directory under 'checkpoints
data['logging']['save_best'] = save_best_model

epochs = 5000 #how many epochs to train for
data['optim']['epochs'] = epochs

"""
Save configuration
Now that the sample json was read, we want to save the modified experimental entries. 
We do so, by dumping everything into a second json file, which will be passed to the flat metric computatution
"""
config_To_be_written = 'temporary_experiment_config.json' 

#write new config file
with open(config_To_be_written, "w") as write_file:
    json.dump(data, write_file, indent=4)

"""
Actually call the script for computing the flat distance
"""
subprocess.call("python " + os.path.join(__basedir__, "lnets", "tasks", "dualnets", "mains","train_dual.py") + " " + config_To_be_written, shell=True)


"""
Hooray! Everything was computed and stored in the directory specified in out_path.
Howeverm as we didn't redturn any values, we have no idea as of yet, what the computed flat distance actually is.
So, we will read in the generated log files and print these
"""

dirs = os.listdir(out_path) #there should be only 1 entry corresponding to the 1 one run of the experiment

#Now, read in the losses incurred during training
distance_losses = np.loadtxt(os.path.join(out_path, dirs[-1], 'logs{s}train_loss_W.log'.format(s=os.sep)), skiprows=1, delimiter=',')

#We assume that at the end, training has converged. We thus take the mean of the last 50 epochs as our estimate for the flat distance
flat_metric = -np.mean(distance_losses[:-50:-1,1]) #Note the minus sign, taking care of switching from loss term to distance estimate

#Output to user
print(f'\n\n\nThe two given datasetshave a flat distance of approximately {flat_metric:.2f}.')

