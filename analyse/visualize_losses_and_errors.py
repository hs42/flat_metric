import numpy as np 
import numpy.ma as ma
from matplotlib import pyplot as plt
import os
import json

dim = 2

path = './out/examine_loss_and_errors/test_11-09-FM_2_Diracs_d=2_few_samples' 

dirs = os.listdir(path)
dirs.sort()
r = np.zeros(len(dirs))
flat_metric = {'many_samples': np.zeros((len(dirs),2)), 'few_samples' : np.zeros((len(dirs),2))}

loss_ratios = {'many_samples': np.zeros((len(dirs),2)), 'few_samples' : np.zeros((len(dirs),2))}

for key in flat_metric:
    path = './out/examine_loss_and_errors/FM_2_Diracs_d={d}_{b}'.format(d=dim, b=key) 
    dirs = os.listdir(path)
    dirs.sort()

    for i, dir in enumerate(dirs):
        with open(os.path.join(path, dir, 'logs/config.json')) as f:
            data = json.load(f)
        r[i] = abs(data['distrib2']['radius'] - data['distrib1']['radius'])
        dim = data['distrib2']['dim']
        linear_type = data['model']['linear']['type']
        distance_train = np.loadtxt(os.path.join(path, dir, 'logs/train_loss_W.log'), skiprows=1, delimiter=',')
        loss_flat = np.loadtxt(os.path.join(path, dir, 'logs/train_loss_flat.log'), skiprows=1, delimiter=',')

        flat_metric[key][i,0] = -np.mean(distance_train[:-10:-1,1]) #mean of last 10 entries
        flat_metric[key][i,1] = np.std(distance_train[:-10:-1,1]) #mean of last 10 entries

        loss_ratios[key][i,0] = np.mean(loss_flat[:-10:-1,1]) / np.mean(distance_train[:-10:-1,1]) #ratio of mean of last 10 entries
        
        with np.errstate(divide='ignore'): #ignore 'divide by 0' and resulting NaNs for now
            loss_ratios[key][i,1] =  np.sqrt(np.var(distance_train[:-10:-1,1])/np.mean(loss_flat[:-10:-1,1])**2 + np.var(loss_flat[:-10:-1,1])/np.mean(distance_train[:-10:-1,1])**2) #error propagation part 1


    loss_ratios[key][:,1] = np.where(np.isfinite(loss_ratios[key][:,1]), loss_ratios[key][:,1], 0)
    loss_ratios[key][:,1] = loss_ratios[key][:,0] * loss_ratios[key][:,1] #error propagation part 2



truth_y = [min(x,2.0) for x in r]


fig, axs = plt.subplots(2)

for key in flat_metric:

    axs[0].errorbar(r, flat_metric[key][:,0]/truth_y, yerr=flat_metric[key][:,1]/truth_y, fmt="o", label='flat metric: exp./ground truth for ' + key)
    axs[0].set_ylim([-0.2, 2.4])
    axs[0].set_ylabel('value of flat metric')
    axs[0].legend()

    axs[1].errorbar(r, loss_ratios[key][:,0], yerr=loss_ratios[key][:,1], fmt="o", label='loss ratio: flat/W  for ' + key)
    axs[1].set_ylim([-0.2, 0.2])
    axs[1].set_ylabel('ratio of losses: flat/W')
    axs[1].legend()

#plt.plot(truth_x, truth_y, label='true solution')
plt.xlabel('r')

#plt.title('Flat metric between two Diracs for dim={a} and {b} normalization'.format(a=dim, b=linear_type))
fig.savefig('losses_and_errors_over_r_d={a}.png'.format(a=dim), format='PNG')