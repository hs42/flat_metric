import numpy as np 
import numpy.ma as ma
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import json

dim = 2

path = './out/few_hp_large_ds_sweep_m_n_l_dim={d}_with_intermediate_check_of_penalty_Adam_decaying_lr_penalty_002_long/'.format(d=dim)
#path = './out/sweep_m_n_l_dim={d}/'.format(d=dim)


dirs = os.listdir(path)
dirs.sort()
dirs.pop(-1) #remove the groundtruth.npy file (which is sadly still listed in os.listdir())
number_of_dirs = int(len(dirs))

#r = np.zeros(n)
#flat_metric = np.zeros(n, 2) #for value and uncertainties

#loss_ratios = {'many_samples': np.zeros((len(dirs),2)), 'few_samples' : np.zeros((len(dirs),2))}

#put in prior knowledge of number of tests
#otherwise: first scan all configs, then store data

distance_estimates = np.zeros((2, 8, 11, 2)) #m,n,l,[mean, std]
relative_errors = np.zeros((2, 8, 11, 2)) #m,n,l,[mean, std]

actual_penalties = np.zeros((2, 8, 11, 2))   #m,n,l,[mean, std]

#m_values = [5, 10]
#n_values = [10, 20, 30, 40, 50, 60, 70, 80]
m_values = [20, 40]
n_values = [100, 200, 300, 400, 500, 600, 700, 800]
l_fraction_values = np.arange(0, 1.1, 0.1).tolist()

groundtruth = np.load(path + 'groundtruth.npy')

for i, dir in enumerate(dirs):
    with open(os.path.join(path, dir, 'logs/config.json')) as f:
        data = json.load(f)

    n = data['distrib2']['sample_size']
    m = data['distrib1']['sample_size']
    l_f = data['distrib2']['l_fraction']

    dim = data['distrib2']['dim']
    linear_type = data['model']['linear']['type']

    idx = [m_values.index(m), n_values.index(n), l_fraction_values.index(l_f)]

    distance_train = np.loadtxt(os.path.join(path, dir, 'logs/train_loss_W.log'), skiprows=1, delimiter=',')
    loss_flat = np.loadtxt(os.path.join(path, dir, 'logs/train_loss_flat.log'), skiprows=1, delimiter=',')
    lambda_penalties = np.loadtxt(os.path.join(path, dir, 'logs/lambda.log'), skiprows=1)

    distance_estimates[tuple(idx + [0])] = -np.mean(distance_train[:-10:-1,1]) #mean of last 10 entries
    distance_estimates[tuple(idx + [1])] = np.std(distance_train[:-10:-1,1]) #mean of last 10 entries

    groundtruth[tuple(idx)] = groundtruth[tuple(idx)] / min(n,m)
    
    relative_errors[tuple(idx + [0])] = np.abs(distance_estimates[tuple(idx + [0])] / groundtruth[tuple(idx)]) #/ distance_estimates[tuple(idx + [0])]    
    relative_errors[tuple(idx + [1])] = relative_errors[tuple(idx + [0])] * distance_estimates[tuple(idx + [1])]/distance_estimates[tuple(idx + [0])]


    actual_penalties[tuple(idx + [0])] = np.mean(lambda_penalties[:-10:-1,1]) #mean of last 10 entries of actual_penalty
    actual_penalties[tuple(idx + [1])] = np.std(lambda_penalties[:-10:-1,1]) #mean of last 10 entries of actual_penalty

    if(actual_penalties[tuple(idx + [1])] > 0.03):
        print('High bound penalty for run in ', dir)

fig, ax = plt.subplots(3,2, figsize=(16, 16))

#relative_errors = relative_errors - np.mean(relative_errors, axis=(1,2), keepdims=True)

n_ax_x = len(n_values)
n_ax_y = len(l_fraction_values)
x_label_list = n_values
y_label_list = l_fraction_values


min_penalty = 0
max_penalty = 0.05

a=0
b=2

img = []
img.append(ax[0,0].imshow(relative_errors[0,:,:,0], extent=[0,n_ax_x,0,n_ax_y], vmin=a, vmax=b,  cmap='hot'))
img.append(ax[0,1].imshow(actual_penalties[0,:,:,0], extent=[0,n_ax_x,0,n_ax_y], vmin=min_penalty, vmax=max_penalty,  cmap='hot'))
img.append(ax[1,0].imshow(relative_errors[1,:,:,0], extent=[0,n_ax_x,0,n_ax_y], vmin=a, vmax=b,  cmap='hot'))
img.append(ax[1,1].imshow(actual_penalties[1,:,:,0], extent=[0,n_ax_x,0,n_ax_y], vmin=min_penalty, vmax=max_penalty,  cmap='hot'))
print('std of ratios for m=5',  np.std(relative_errors[0,1:,:,0]))
print('std of ratios for m=10', np.std(relative_errors[1,1:,:,0]))

print('mean error for of ratios for m=5', np.mean(relative_errors[0,1:,:,1]))
print('mean error for of ratios for m=10', np.mean(relative_errors[1,1:,:,1]))

print('max ratio, m=5: ', np.max(relative_errors[0,:,:,0]))
print('min ratio, m=5: ', np.min(relative_errors[0,:,:,0]))


for m in range(2):
    for j, name in enumerate(['ratio of distance estimates', 'actual bound penalty']):        
        ax[m,j].set_title(name + 'for m=' + str(m_values[m]))
        ax[m,j].set_xticks(np.arange(n_ax_x) + 0.5)
        ax[m,j].set_yticks(np.arange(n_ax_y) + 0.5)
        ax[m,j].set_xticklabels(x_label_list)
        ax[m,j].set_yticklabels(y_label_list)
        ax[m,j].set_xlabel('n = sample size of distr2')
        ax[m,j].set_ylabel('l/n')

        divider = make_axes_locatable(ax[m,j])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(img[2*m + j], cax=cax, orientation='vertical')
ax[2,0].hist(relative_errors[0,:,:,0], bins=np.arange(0.8, 2.2, 0.1))
ax[2,1].hist(relative_errors[1,:,:,0], bins=np.arange(0.8, 2.2, 0.1))

#   cbar = plt.colorbar(img,cax=ax[0,0])

fig.savefig('few_hp_large_ds_relative_errors_for_ball_with_intermediate_check_of_penalty_Adam_decaying_lr_penalty_002_long.png', format='PNG')