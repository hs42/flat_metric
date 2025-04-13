import numpy as np

"""
These are the found parameters (found by fitting against experimental groundtruth)
They will be used later correct by a log-normal distribution.
Specifically, each parameter comes with two values (encoded in the array), to interpolate
its value for arbitrary dimensions by a linear model: specific_para = dim * para_value[0] + para_value[1]
"""
found_params ={'a': np.array([-0.00408901, -0.10763544]),
 'b': np.array([9.58512253e-05, 5.74280898e-03]),
 'c': np.array([-0.00174557,  0.05173382]),
 'mu': np.array([-0.00199061,  0.19272778]),
 'sigma': np.array([0.00616927, 0.23810255])}

def linear(x, a, b):
    return a*x + b

def post_process(array_of_training_distances, dim, mass_ratio):
    #Refine the parameters of the log-normal correction to fit the problem's dimensionality
    this_a = linear(dim, *found_params['a'])
    this_b = linear(dim, *found_params['b'])
    this_c = linear(dim, *found_params['c'])
    this_mu = linear(dim, *found_params['mu'])
    this_sigma = linear(dim, *found_params['sigma'])

    x = mass_ratio
    #perform the log-normal correction: first calculate the expected relative error given the dimension and the mass_ratio
    lognormal = (np.exp(-(np.log(x) - this_mu)**2 / (2 * this_sigma**2)) / (x * this_sigma * np.sqrt(2 * np.pi)))
    expected_rel_error = this_a*lognormal + this_b*x + this_c

    corrected_estimates = array_of_training_distances / (1.0 + expected_rel_error)

    return corrected_estimates

def return_distance(array_of_training_distances, dim, mass_ratio, data_points_to_consider=50):
    #print('before (dim ={d}, n/m={mr}): '.format(d=dim, mr=mass_ratio), -np.mean(array_of_training_distances[:-data_points_to_consider:-1,1]))
    array_of_training_distances = post_process(array_of_training_distances, dim, mass_ratio)

    flat_distance = -np.mean(array_of_training_distances[:-data_points_to_consider:-1,1]) #mean of last data_points_to_consider entries
    std = np.std(array_of_training_distances[:-data_points_to_consider:-1,1]) / np.sqrt(data_points_to_consider) #error of the mean

    #print('after (dim ={d}, n/m={mr}): '.format(d=dim, mr=mass_ratio), flat_distance)
    return (flat_distance, std)
