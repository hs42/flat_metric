import torch
import numpy as np
from torch.autograd import Variable

from lnets.models.model_types.base_model import ExperimentModel


class DualOptimModel_flat_norm(ExperimentModel):
    """
    Our implementation of a model, approxmiamting the flat norm. Its loss is given in accordance to the optimization problem when computing the flat norm. See below for details
    """
    def _init_meters(self):
        super(DualOptimModel_flat_norm, self)._init_meters()


    def loss(self, sample, test=False):
        """
        compute the loss such that the flat norm is approximated. As we want to solve argmax_{f}( E[f(d1)] - E[f(d2)]), where f is a bounded 1-Lipschitz function,
        we set the loss to L = -( E[f(d1)] - E[f(d2)]). This is augmented by a penalty for when f is not bounded L -> L + L_{bound}. L_{bound} is defined by equation (3.2)
        in the paper and is given by the quadratic deviations between f and its allowed band [-M, M] = [-1, 1] on the domain where there is input data. 
        Here M = self.model.config.model.bound.upper_bound = 1. 
        """

        # d1 stands for distribution 1.
        # d2 stands for distribution 2.
        
        # put samples into a PyTorch Variable object, which we can differentiate after
        samples_from_d1 = Variable(sample[0])
        samples_from_d2 = Variable(sample[1])
        potentials_from_d1 = self.model.forward(samples_from_d1)
        potentials_from_d2 = self.model.forward(samples_from_d2)

        assert potentials_from_d1.shape[1] == 1
        assert potentials_from_d2.shape[1] == 1



        #Divide the metric contribution of the loss by normalization_constant. We set normalization_constant as the smallest sample size of distribution 1 or 2.
        #This means, that we artifically scale both measures such that the mass of the smaller distribution is 1.
        #This is rather for convenience and could be scrapped (making sure that the relative balance between both loss contributions still makes sense)
        normalization_constant = 0

        if test:
            normalization_constant = min((self.model.config.distrib1.test_sample_size, self.model.config.distrib2.test_sample_size))
        else:
            normalization_constant = min((self.model.config.distrib1.sample_size, self.model.config.distrib2.sample_size))
        #normalize the losses by dividing by smallest sample size. 
        loss_W = -1 * (torch.sum(potentials_from_d1) - torch.sum(potentials_from_d2)) / normalization_constant


        #Now, compute the bound loss.
        #First, we init an array of constant 0 in the appropriate length. Surely, there is a more elegant way for that
        tmp1 = torch.zeros(len(potentials_from_d1[:,0]), device=potentials_from_d1.device) #array of zeros on same device (GPU vs CPU) as data
        tmp2 = torch.zeros(len(potentials_from_d2[:,0]), device=potentials_from_d1.device)


        #check whether the evaluations f(d1) or f(d2) are in the allowed band of [-M, M] = [-1,1], where M = self.model.config.model.bound.upper_bound = 1. 
        #term_flat1, term_flat2 contain the absolute deviations from this band.
        term_flat1 = torch.maximum(torch.abs(potentials_from_d1[:,0]) - self.model.config.model.bound.upper_bound, tmp1)
        term_flat2 = torch.maximum(torch.abs(potentials_from_d2[:,0]) - self.model.config.model.bound.upper_bound, tmp2)

        #Now, take the quadratic deviations, i.e. <term_flat1, term_flat1>. For term_flat2 likewise.
        #Also, normalizee by the number of samples in d1 and d2 -- this is such that the bound loss behaves invariant if we approximate a distribution by, for instance,
        #len(d1) = 100 or len(d1) = 3000 data points. The bound loss should not depend on how high the resolution of the empirical measures are, but only on how often in
        #relative terms the evaluations are beyond the allowed boundary
        a = torch.inner(term_flat1, term_flat1) / len(term_flat1) #normalize for number of samples
        b = torch.inner(term_flat2, term_flat2) / len(term_flat2) #normalize for number of samples
        loss_flat = a+b

        #get current value for adaptive penalty
        my_lambda = self.model.config.model.bound.lambda_current 
        
        #put everything together
        loss = loss_W + my_lambda * loss_flat

        #return everything, which is useful. The gradient descent step will be done wrt loss
        return [loss, loss_W, my_lambda * loss_flat], (potentials_from_d1, potentials_from_d2)

    def add_to_meters(self, state):
        self.meters['loss'].add(state['loss'].item())
        self.meters['loss_W'].add(state['loss_W'].item())
        self.meters['loss_flat'].add(state['loss_flat'].item())


    def update_lambda_bound(self, state, mode='linear'):
        """
        update_lambda_bound() implements the adaptive penalty, i.e. here we compute the scaling factor lambda which balances the both contributions
        L_{metric} and L_{bound} in the loss and store it in self.model.config.model.bound.lambda_current.

        Lambda is altered as a piecewise linear function over the training epochs, with changes of the slope occuring at start, stop1, and stop2.
        Usually, this happens after start = 20%, stop1 = 50%, and stop2 = 80% of all training epochs.

        The details of how the new slopes are computed can be found in eq (3.4) of the chapter "Robustness and comparability" in the paper. Essentially,
        we first set lambda to a starting value of lambda=10, then set its aim value to double the metric loss at stop1, such that both losses should range
        in the same order of magnitude. Afterwards, we want to compare the acual bound penalty loss_flat to a set value of 0.02 and make lambda accordingly 
        bigger or smaller.
        """
        start = int(self.model.config.model.bound.lambda_loss_start_at * self.model.config.optim.epochs)
        stop1 = int(self.model.config.model.bound.lambda_loss_stop_at * self.model.config.optim.epochs)
        stop2 = int(0.8 * self.model.config.optim.epochs)
 
        l = 0
        if mode == 'linear':
            if state['epoch'] < start:
                #here nothing should happen
                l = self.model.config.model.bound.lambda_init
            elif state['epoch'] == start:
                '''
                Basically, set the final lambda (which we slowly attain) as the Wasserstein after some reasonable amounts of epochs,
                multiplied by some coefficient self.model.config.model.bound.lambda_coefficient 
                '''

                current_eval = state['recent_d_estimate'][-1]

                self.model.config.model.bound.lambda_final = self.model.config.model.bound.lambda_coefficient * current_eval

                l = self.model.config.model.bound.lambda_init

            elif state['epoch'] > start and state['epoch'] < stop1:
                #linear interpolation
                l = (self.model.config.model.bound.lambda_init - self.model.config.model.bound.lambda_final)/(start - stop1) * (state['epoch'] - start) + self.model.config.model.bound.lambda_init 

            elif state['epoch'] == stop1:
                #get current penalty of how much f violates the bound requiremen
                current_actual_penalty = self.meters['loss_flat'].value()[0] / self.model.config.model.bound.lambda_current

                self.model.config.model.bound.lambda_init = self.model.config.model.bound.lambda_final #save needed value
                #remain continous: at epoch stop1, we want to keep the value of lambda from the previous epoch
                l = self.model.config.model.bound.lambda_init

                #change lambda such that the bound penalty aims at approximating 0.02 in the following epochs
                self.model.config.model.bound.lambda_final = self.model.config.model.bound.lambda_final * (current_actual_penalty / 0.02)

            elif state['epoch'] > stop1 and state['epoch'] < stop2:
                #linear interpolation
                l = (self.model.config.model.bound.lambda_init - self.model.config.model.bound.lambda_final)/(stop1 - stop2) * (state['epoch'] - stop1) + self.model.config.model.bound.lambda_init 

            else:
                #remain constant such that some relaxation of the net wrt to the newly scaled energy landscape can occur
                l = self.model.config.model.bound.lambda_final
        
        #save computed value of lambda in the configuration dictionary
        self.model.config.model.bound.lambda_current = l


class DualOptimModel(ExperimentModel):
    """
    This is the original class for solving the (dual) Wasserstein optimization problem by Cem Anil, James Lucas, Roger Grosse.
    Keep it here, in case you want to play with the Wasserstein metric and not the flat metric
    """
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

