import torch
import numpy as np
class SwapNoiseMasker(object):
    def __init__(self, hparams: dict):
        self.num_numerical = hparams['num_numerical']
        self.num_category = hparams['num_category']
        self.swap_category = hparams['swap_category']
        self.swap_numerical = hparams['swap_numerical']
        self.probas = torch.tensor(self.num_numerical * [self.swap_numerical] + self.num_category * [self.swap_category])
        

    def apply(self, input):
        should_swap = torch.bernoulli(self.probas * torch.ones(input.shape))
        input_noise = torch.cat([input[torch.randint(input.shape[0],(1,1)).item()][torch.randperm(input.shape[1])].unsqueeze(0) for i in range(input.shape[0])])
        corrupted_input = torch.where(should_swap == 1, input_noise, input)
        mask = (corrupted_input != input).float()
        return corrupted_input, mask