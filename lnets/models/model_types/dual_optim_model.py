import torch
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

        loss_W = -1 * (torch.mean(potentials_from_d1) - torch.mean(potentials_from_d2))
        all_samples = torch.concat((potentials_from_d1, potentials_from_d2), dim=0)
        term_flat = torch.maximum(torch.abs(all_samples - torch.mean(all_samples)) - self.model.config.model.upper_bound, torch.zeros(len(all_samples)))
        loss_flat = 0.1*torch.linalg.norm(term_flat, ord=1)

        loss = loss_W + loss_flat

        #print('flat loss: ', potentials_from_d2)


        #if(loss_flat > 0):
        #    print('Nicht verschwindender flat loss:', loss_flat)

        return [loss, loss_W, loss_flat], (potentials_from_d1, potentials_from_d2)

    def add_to_meters(self, state):
        self.meters['loss'].add(state['loss'].item())
        self.meters['loss_W'].add(state['loss_W'].item())
        self.meters['loss_flat'].add(state['loss_flat'].item())

        #type(state['model'].meters['loss']) is torchnet.meter.averagevaluemeter.AverageValueMeter -> builds average automatically

