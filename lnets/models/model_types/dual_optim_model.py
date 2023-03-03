import torch
import numpy as np
from torch.autograd import Variable

from lnets.models.model_types.base_model import ExperimentModel


class DualOptimModel(ExperimentModel):
    def _init_meters(self):
        super(DualOptimModel, self)._init_meters()


    def loss(self, sample, test=False):
        # d1 stands for distribution 1.
        # d2 stands for distribution 2.

        samples_from_d1 = Variable(sample[0])
        samples_from_d2 = Variable(sample[1])

        potentials_from_d1 = self.model.forward(samples_from_d1)
        potentials_from_d2 = self.model.forward(samples_from_d2)

        assert potentials_from_d1.shape[1] == 1
        assert potentials_from_d2.shape[1] == 1

        loss = -1 * (torch.mean(potentials_from_d1) - torch.mean(potentials_from_d2))

        return [loss], (potentials_from_d1, potentials_from_d2)

    def add_to_meters(self, state):
        self.meters['loss'].add(state['loss'].item())
        #type(state['model'].meters['loss']) is torchnet.meter.averagevaluemeter.AverageValueMeter -> builds average automatically


class DualOptimModel_flat_norm(ExperimentModel):
    def _init_meters(self):
        super(DualOptimModel_flat_norm, self)._init_meters()


    def loss(self, sample, test=False):
        # d1 stands for distribution 1.
        # d2 stands for distribution 2.

        samples_from_d1 = Variable(sample[0])
        samples_from_d2 = Variable(sample[1])
        potentials_from_d1 = self.model.forward(samples_from_d1)
        potentials_from_d2 = self.model.forward(samples_from_d2)

        assert potentials_from_d1.shape[1] == 1
        assert potentials_from_d2.shape[1] == 1

        normalization_constant = 0

        if test:
            normalization_constant = min((self.model.config.distrib1.test_sample_size, self.model.config.distrib2.test_sample_size))
        else:
            normalization_constant = min((self.model.config.distrib1.sample_size, self.model.config.distrib2.sample_size))

        #normalize the losses by dividing by smallest sample size. This way we can incorporate measures which are normalized to
        #different constants and not only compare probability measures with each other
        loss_W = -1 * (torch.sum(potentials_from_d1) - torch.sum(potentials_from_d2)) / normalization_constant

        all_samples = torch.concat((potentials_from_d1[:,0], potentials_from_d2[:,0]), dim=0)
        #print('Pot from d1: ', potentials_from_d1[:,0])
        #print('Pot from d2: ', potentials_from_d2[:,0])

        #print('mean of samples:', torch.abs(all_samples - torch.mean(all_samples)))
        #term_flat = torch.maximum(torch.abs(all_samples) - self.model.config.model.upper_bound, torch.zeros(len(all_samples)))
        tmp1 = torch.zeros(len(potentials_from_d1[:,0]), device=potentials_from_d1.device) #array of zeros on same device (GPU vs CPU) as data
        tmp2 = torch.zeros(len(potentials_from_d2[:,0]), device=potentials_from_d1.device)


        term_flat1 = torch.maximum(torch.abs(potentials_from_d1[:,0]) - self.model.config.model.bound.upper_bound, tmp1)
        term_flat2 = torch.maximum(torch.abs(potentials_from_d2[:,0]) - self.model.config.model.bound.upper_bound, tmp2)

        a = torch.inner(term_flat1, term_flat1) / len(term_flat1) #normalize for number of samples
        b = torch.inner(term_flat2, term_flat2) / len(term_flat2) #normalize for number of samples
        #print('Die beiden bound loss terme: ',  len(term_flat2), a.item(), b.item())
        loss_flat = a+b
        my_lambda = self.model.config.model.bound.lambda_current 
        #print('d1: ',all_samples.size())
        loss = loss_W + my_lambda * loss_flat

        return [loss, loss_W, my_lambda * loss_flat], (potentials_from_d1, potentials_from_d2)

    def add_to_meters(self, state):
        self.meters['loss'].add(state['loss'].item())
        self.meters['loss_W'].add(state['loss_W'].item())
        self.meters['loss_flat'].add(state['loss_flat'].item())

        #type(state['model'].meters['loss']) is torchnet.meter.averagevaluemeter.AverageValueMeter -> builds average automatically

    def update_lambda_bound(self, state, mode='linear'):
        start = int(self.model.config.model.bound.lambda_loss_start_at * self.model.config.optim.epochs)
        stop1 = int(self.model.config.model.bound.lambda_loss_stop_at * self.model.config.optim.epochs)
        stop2 = int(0.8 * self.model.config.optim.epochs)
 
        l = 0
        if mode == 'linear':
            if state['epoch'] < start:
                l = self.model.config.model.bound.lambda_init
            elif state['epoch'] == start:
                '''
                Basically, set the final lambda (which we slowly attain) as the Wasserstein after some reasonable amounts of epochs,
                multiplied by some coefficient self.model.config.model.bound.lambda_coefficient 
                '''

                #T = min(3, len(state['recent_d_estimate']))
                #first_evals = np.mean(state['recent_d_estimate'][0:T])

                current_eval = state['recent_d_estimate'][-1]

                #difference_in_W_loss = np.abs(0*first_evals - current_eval)

                #self.model.config.model.bound.lambda_final = self.model.config.model.bound.lambda_coefficient * difference_in_W_loss
                self.model.config.model.bound.lambda_final = self.model.config.model.bound.lambda_coefficient * current_eval

                l = self.model.config.model.bound.lambda_init

            elif state['epoch'] > start and state['epoch'] < stop1:
                l = (self.model.config.model.bound.lambda_init - self.model.config.model.bound.lambda_final)/(start - stop1) * (state['epoch'] - start) + self.model.config.model.bound.lambda_init 

            elif state['epoch'] == stop1:
                #print('Wert von lambda und flat loss', self.model.config.model.bound.lambda_current, self.meters['loss_flat'].value()[0])
                current_actual_penalty = self.meters['loss_flat'].value()[0] / self.model.config.model.bound.lambda_current
                self.model.config.model.bound.lambda_init = self.model.config.model.bound.lambda_final #save needed value
                self.model.config.model.bound.lambda_final = self.model.config.model.bound.lambda_final * (current_actual_penalty / 0.02)
                l = self.model.config.model.bound.lambda_init
                #l = self.model.config.model.bound.lambda_final
                #self.model.config.model.bound.lambda_init = self.model.config.model.bound.lambda_final

            elif state['epoch'] > stop1 and state['epoch'] < stop2:
                l = (self.model.config.model.bound.lambda_init - self.model.config.model.bound.lambda_final)/(stop1 - stop2) * (state['epoch'] - stop1) + self.model.config.model.bound.lambda_init 

            else:
                l = self.model.config.model.bound.lambda_final
        #l = self.model.config.model.bound.lambda_init
        self.model.config.model.bound.lambda_current = l
