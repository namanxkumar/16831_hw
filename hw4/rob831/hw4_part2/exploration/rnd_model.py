from rob831.hw4_part2.infrastructure import pytorch_util as ptu
from .base_exploration_model import BaseExplorationModel
import torch.optim as optim
from torch import nn
import torch

def init_method_1(model):
    if hasattr(model, 'weight') and model.weight is not None:
        model.weight.data.uniform_()
    if hasattr(model, 'bias') and model.bias is not None:
        model.bias.data.uniform_()

def init_method_2(model):
    if hasattr(model, 'weight') and model.weight is not None:
        model.weight.data.normal_()
    if hasattr(model, 'bias') and model.bias is not None:
        model.bias.data.normal_()


class RNDModel(nn.Module, BaseExplorationModel):
    def __init__(self, hparams, optimizer_spec, **kwargs):
        super().__init__(**kwargs)
        self.ob_dim = hparams['ob_dim']
        self.output_size = hparams['rnd_output_size']
        self.n_layers = hparams['rnd_n_layers']
        self.size = hparams['rnd_size']
        self.optimizer_spec = optimizer_spec

        self.f = ptu.build_mlp(
            input_size=self.ob_dim,
            output_size=self.output_size,
            n_layers=self.n_layers,
            size=self.size,
        )
        self.f.to(ptu.device)
        self.f.apply(init_method_1)

        self.f_hat = ptu.build_mlp(
            input_size=self.ob_dim,
            output_size=self.output_size,
            n_layers=self.n_layers,
            size=self.size,
        )
        self.f_hat.to(ptu.device)
        self.f_hat.apply(init_method_2)

        self.optimizer = self.optimizer_spec.constructor(
            self.f_hat.parameters(),
            **self.optimizer_spec.optim_kwargs
        )
        self.learning_rate_scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            self.optimizer_spec.learning_rate_schedule,
        )

    def forward(self, ob_no):
        target = self.f(ob_no).detach()
        prediction = self.f_hat(ob_no)
        error = torch.mean((prediction - target) ** 2, dim=-1)
        return error

    def forward_np(self, ob_no):
        ob_no = ptu.from_numpy(ob_no)
        error = self(ob_no)
        return ptu.to_numpy(error)

    def update(self, ob_no):
        ob_no = ptu.from_numpy(ob_no)
        error = self(ob_no)
        loss = error.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.learning_rate_scheduler.step()
        return loss.item()
