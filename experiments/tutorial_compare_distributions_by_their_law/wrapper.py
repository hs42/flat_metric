"""
This is a wrapper script to easily access the most important parameters when computing the flat distance between
two measures as given by their laws.

Typically, the syntax would be
python ./lnets/tasks/dualnets/mains/train_dual.py <path to experiment configuration json file>,
The configuration json file hereby specifies both, the distributions in question, and the architecture of the neural net.

This wrapper script takes care of that, such that you only need to execute this wrapper.py file
That means, it will automatically load a json file; in our case that is the Gaussian_uniform.json file found in this directory.
It specifies that in this example we shall compare a (1 dimensional) Gauss distribution with a (1 dimensional) uniform distribution.
"""

import json
import subprocess
import os
import numpy as np

"""
Firstly, read in the configuration file
"""
#read data
with open('Gaussian_uniform.json') as f:
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
__basedir__ = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir)
out_path = os.path.join(__basedir__, 'out', 'compare_laws')
data['output_root'] = out_path #save to config

#specify distribution 1

#whenever we want to compare laws, we specify a Python class, which implements an generator for empirical samples of said law.
#In 'name' we provide the name of this class and in 'filepath' we specify where it can be found
#Note that the sample_size property is the way we encode different masses. I.e. if dist1 has sample_size=100 but dist2 has
#sample_size=300, then dist3 is made to have three times the probability mass of dist1
data['distrib1']['name'] = 'Gauss'
data['distrib1']['filepath'] = 'lnets/tasks/dualnets/distrib/Gauss.py' #note this is relative to the base directory, from where the process will be started later on

data['distrib1']['dim'] = 1 #dimensionality of the normal distrubtion. This parameter will be read by the 'Gauss' generator class
data['distrib1']['sigma'] = 1#std of the normal distrubtion. This parameter will be read by the 'Gauss' generator class
data['distrib1']['mu'] = 0#mean of the normal distrubtion. This parameter will be read by the 'Gauss' generator class

data['distrib1']['sample_size'] = 200 #how many samples should be generated in each epoch for training
data['distrib1']['test_sample_size'] = 50 #how many samples should be used for testing

#specify distribution 2
data['distrib2']['name'] = 'Uniform'
data['distrib2']['filepath'] = 'lnets/tasks/dualnets/distrib/uniform.py' #note this is relative to the base directory, from where the process will be started later on

data['distrib2']['dim'] = 1 #dimensionality of the uniform distrubtion. This parameter will be read by the 'Uniform' generator class
data['distrib2']['supportlength'] = 2#length of the interval, where we have a non-vanishing PDF. This parameter will be read by the 'Uniform' generator class
data['distrib2']['center'] = 5#where the uniform distribution should be centered. This parameter will be read by the 'Uniform' generator class

data['distrib2']['sample_size'] = 100 #how many samples should be generated in each epoch for training
data['distrib2']['test_sample_size'] = 25 #how many samples should be used for testing


"""
Misc parameters
"""
use_cuda = True #whether to use GPU or not
data['cuda'] = use_cuda

save_best_model = False #whether or not to store the best model for each training. These will be stored in the training output directory under 'checkpoints
data['logging']['save_best'] = save_best_model

epochs = 2000 #how many epochs to train for
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
print('\n\n\nThe two given distributions,', data['distrib1']['name'], 'and', data['distrib2']['name'], f', have a flat distance of approximately {flat_metric:.2f}.')

