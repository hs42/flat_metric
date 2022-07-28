import numpy as np 
import numpy.ma as ma
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import json

dim = 2

path = './out/sweep_n_m_r_all_at_sphere_dim={d}/'.format(d=dim)

dirs = os.listdir(path)
number_of_dirs = int(len(dirs))


#r = np.zeros(n)
#flat_metric = np.zeros(n, 2) #for value and uncertainties

#loss_ratios = {'many_samples': np.zeros((len(dirs),2)), 'few_samples' : np.zeros((len(dirs),2))}

#put in prior knowledge of number of tests
#otherwise: first scan all configs, then store data

distance_estimates = np.zeros((2, 8, 8, 2)) #m,n,r,[mean, std]
relative_errors = np.zeros((2, 8, 8, 2)) #m,n,r,[mean, std]

actual_penalties = np.zeros((2, 8, 8, 2))   #m,n,r,[mean, std]

r_values = [0.1, 0.5, 1.0, 1.5] + list(range(2,22,5))
m_values = [5, 10]
n_values = [10, 20, 30, 40, 50, 60, 70, 80]


for i, dir in enumerate(dirs):
    with open(os.path.join(path, dir, 'logs/config.json')) as f:
        data = json.load(f)

    #r = abs(data['distrib2']['radius'] - data['distrib1']['radius'])
    r = data['distrib2']['radius']
    n = data['distrib2']['sample_size']
    m = data['distrib1']['sample_size']

    dim = data['distrib2']['dim']
    linear_type = data['model']['linear']['type']

    idx = [m_values.index(m), n_values.index(n), r_values.index(r)]

    distance_train = np.loadtxt(os.path.join(path, dir, 'logs/train_loss_W.log'), skiprows=1, delimiter=',')
    loss_flat = np.loadtxt(os.path.join(path, dir, 'logs/train_loss_flat.log'), skiprows=1, delimiter=',')
    lambda_penalties = np.loadtxt(os.path.join(path, dir, 'logs/lambda.log'), skiprows=1)

    distance_estimates[tuple(idx + [0])] = -np.mean(distance_train[:-10:-1,1]) #mean of last 10 entries
    distance_estimates[tuple(idx + [1])] = np.std(distance_train[:-10:-1,1]) #mean of last 10 entries

    
    groundtruth = min(2.0, r) + np.abs(n-m)/min(n, m)
    relative_errors[tuple(idx + [0])] = np.abs(distance_estimates[tuple(idx + [0])] / groundtruth) / 1#distance_estimates[tuple(idx + [0])]
    relative_errors[tuple(idx + [1])] = 42
    

    actual_penalties[tuple(idx + [0])] = np.mean(lambda_penalties[:-10:-1,1]) #mean of last 10 entries of actual_penalty
    actual_penalties[tuple(idx + [1])] = np.std(lambda_penalties[:-10:-1,1]) #mean of last 10 entries of actual_penalty


fig, ax = plt.subplots(3,2, figsize=(16, 16))

n_ax = 8
x_label_list = n_values
y_label_list = r_values

min_penalty = 0
max_penalty = 0.1

img = []
img.append(ax[0,0].imshow(relative_errors[0,:,:,0], extent=[0,n_ax,0,n_ax], vmin=0, vmax=2,  cmap='hot'))
img.append(ax[0,1].imshow(actual_penalties[0,:,:,0], extent=[0,n_ax,0,n_ax], vmin=min_penalty, vmax=max_penalty,  cmap='hot'))
img.append(ax[1,0].imshow(relative_errors[1,:,:,0], extent=[0,n_ax,0,n_ax], vmin=0, vmax=2,  cmap='hot'))
img.append(ax[1,1].imshow(actual_penalties[1,:,:,0], extent=[0,n_ax,0,n_ax], vmin=min_penalty, vmax=max_penalty,  cmap='hot'))

for m in range(2):
    for j, name in enumerate(['estimate ratio', 'actual bound penalty']):        
        ax[m,j].set_title(name + 'for m=' + str(m_values[m]))
        ax[m,j].set_xticks(np.arange(n_ax) + 0.5)
        ax[m,j].set_yticks(np.arange(n_ax) + 0.5)
        ax[m,j].set_xticklabels(x_label_list)
        ax[m,j].set_yticklabels(y_label_list)
        ax[m,j].set_xlabel('n = sample size of distr2')
        ax[m,j].set_ylabel('r')

        divider = make_axes_locatable(ax[m,j])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(img[2*m + j], cax=cax, orientation='vertical')
ax[2,0].hist(relative_errors[0,:,:,0], bins=np.arange(0.8, 2.2, 0.1))
ax[2,1].hist(relative_errors[1,:,:,0], bins=np.arange(0.8, 2.2, 0.1))

#   cbar = plt.colorbar(img,cax=ax[0,0])

fig.savefig('relative_errors_test.png', format='PNG')