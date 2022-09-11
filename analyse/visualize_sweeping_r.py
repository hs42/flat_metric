import numpy as np 
from matplotlib import pyplot as plt
import os
import json

dim = 2

path = './out/examine_loss_and_errors/just_a_simple_test'

dirs = os.listdir(path)
r = np.zeros(len(dirs))
flat_metric = np.zeros((len(dirs),2))



for i, dir in enumerate(dirs):
    with open(os.path.join(path, dir, 'logs/config.json')) as f:
        data = json.load(f)
    r[i] = abs(data['distrib2']['radius'] - data['distrib1']['radius'])
    dim = data['distrib2']['dim']
    linear_type = data['model']['linear']['type']
    distance_train = np.loadtxt(os.path.join(path, dir, 'logs/train_loss_W.log'), skiprows=1, delimiter=',')
    flat_metric[i,0] = -np.mean(distance_train[:-10:-1,1]) #mean of last 10 entries
    flat_metric[i,1] = np.std(distance_train[:-10:-1,1]) #mean of last 10 entries

truth_x = np.linspace(min(r), max(r))
truth_y = [min(x,2.0) for x in truth_x]
plt.errorbar(r, flat_metric[:,0], yerr=flat_metric[:,1], fmt="o", label='experiment')
plt.plot(truth_x, truth_y, label='true solution')
plt.ylim([-0.1,2.3])
plt.legend()
plt.xlabel('r')
plt.ylabel('value of flat metric')
plt.title('Flat metric between two Diracs for dim={a} and {b} normalization'.format(a=dim, b=linear_type))
plt.savefig('vary_r_dim={a}.png'.format(a=dim), format='PNG')