
"""
This is the script to reproduce the "Experiments - Testing unequal masses and the effectiveness of adaptive penalties" section of the paper.
Compute the flat distance between mu sitting at zero (mu=sample_size * m * delta(0)) and nu, which describes a 
sphere of radius r (mu=sample_size * n * delta(sphere)). R is changed to the values specified in radii_to_test, and the masses m and n
are probed according to m_to_test and n_to_test
"""

import json
import numpy as np
import subprocess
import os


__basedir__ = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, os.pardir)
__filedir__ = os.path.dirname(os.path.abspath(__file__))


"""
quick / most important parameters
"""
use_cuda = True #whether to use GPU or not
save_best_model = False #whether or not to store the best model for each training. These will be stored in the training output directory under 'checkpoints

dim = 4
few_samples = True

if few_samples:
    sample_size_factor =  2**dim 
else:
    sample_size_factor = 6 * 2**dim #account for need for more training data in higher dimensions


#radii_to_test = np.arange(radius_start, radius_stop, radius_step)
radii_to_test = list(np.arange(0.01, 3, 0.3)) + list(range(3,18,5)) #the radius which are to be tested in this script.
m_to_test = sample_size_factor*np.array([5, 10, 20], dtype=int) #the masses of measure mu to probe
n_to_test = sample_size_factor*np.array([10, 20, 30, 40, 50, 60, 70, 80], dtype=int) #the masses of measure nu to probe

linear_layer_type = 'spectral_normal'

#ask user where the training results shall be strored
root = tk.Tk()
root.withdraw()
out_path_parent = askdirectory(title='Select empty folder for the output of this experiment', initialdir=os.path.join(__basedir__, 'out'))
# if out_path already exists from a previous experiment
if len(os.listdir(out_path_parent)) != 0:
    raise RuntimeError('output directory is not empty. Please choose another one', out_path_parent)
out_path = os.path.join(out_path_parent, 'training{s}{n}_samples{s}dimension_{d}'.format(d=dim, s=os.sep, n=['many', 'few'][few_samples]))



"""
path configurations
"""
tempdir = tempfile.TemporaryDirectory() #create temporary dir where text files of the the currently celltypes under question are stored
path_to_save_processed = tempdir.name
if os.path.isdir(path_to_save_processed): #reset
    shutil.rmtree(path_to_save_processed)
os.makedirs(path_to_save_processed)

path_to_default =      os.path.join(__basedir__, 'lnets{s}tasks{s}dualnets{s}configs{s}default_2_diracs.json'.format(s=os.sep))
config_To_be_written = os.path.join(tempdir.name, '2_diracs.json')


#read data
with open(path_to_default) as f:
   data = json.load(f)

#correct center_x if dimensionality does not match
if len(data['distrib1']['center_x']) == 1 and dim > 1:
    data['distrib1']['center_x'] = dim*data['distrib1']['center_x']#repeat value in list
elif len(data['distrib1']['center_x']) != dim:
    print('Error: give enough coordinates for center_x in distrib1')
if len(data['distrib2']['center_x']) == 1 and dim > 1:
    data['distrib2']['center_x'] = dim*data['distrib2']['center_x'] #repeat value in list
elif len(data['distrib2']['center_x']) != dim:
    print('Error: give enough coordinates for center_x in distrib2')

#set certain parameters
data['model']['linear']['type'] = linear_layer_type

data['distrib1']['dim'] = dim 
data['distrib2']['dim'] = dim
data['output_root'] = out_path
data['cuda'] = use_cuda
data['logging']['save_best'] = save_best_model

"""
enter loop and sweep through different values for the radius
"""
for m in m_to_test:
    for n in n_to_test:
        for r in radii_to_test:
            data['distrib2']['radius'] = r

            data['distrib1']['sample_size'] = int(m)
            data['distrib2']['sample_size'] = int(n)
            data['distrib1']['test_sample_size'] = max(1,int(0.1*m))
            data['distrib2']['test_sample_size'] = max(1,int(0.1*n))

            #write new config file
            with open(config_To_be_written, "w") as write_file:
                json.dump(data, write_file, indent=4)

            #compute flat metric
            subprocess.call("python " + os.path.join(__basedir__, "lnets", "tasks", "dualnets", "mains","train_dual.py") + " " + config_To_be_written, shell=True)


#clean up
tempdir.cleanup()