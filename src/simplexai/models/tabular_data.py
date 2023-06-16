import torch
import torch.nn as nn
import torch.nn.functional as F

from simplexai.models.base import BlackBox


class MortalityPredictor(BlackBox):
    def __init__(self, n_cont: int = 264, input_feature_num=289) -> None:
        """
        Mortality predictor MLP
        :param n_cont: number of continuous features among the output features
        """
        super().__init__()
        self.n_cont = n_cont
        self.lin1 = nn.Linear(input_feature_num, 200)
        self.lin2 = nn.Linear(200, 50)
        self.lin3 = nn.Linear(50, 2)
        if self.n_cont > 0:
            self.bn1 = nn.BatchNorm1d(self.n_cont)
        self.drops = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.latent_representation(x)
        x = self.lin3(x)
        x = F.log_softmax(x, dim=-1)
        return x

    def latent_representation(self, x: torch.Tensor) -> torch.Tensor:
        x_cont, x_disc = x[:, : self.n_cont], x[:, self.n_cont :]
        print(x_cont, x_disc)
        if self.n_cont > 0:
            x_cont = self.bn1(x_cont)
        print(x_cont, x_disc)
        x = torch.cat([x_cont, x_disc], 1)
        print(x)
        x = F.relu(self.lin1(x))
        print(x)
        x = self.drops(x)
        print(x)
        x = F.relu(self.lin2(x))
        print(x)
        x = self.drops(x)
        print(x)
        print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx')
        return x

    def probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the class probabilities for the input x
        :param x: input features
        :return: probabilities
        """
        x = self.latent_representation(x)
        x = self.lin3(x)
        x = F.softmax(x, dim=-1)
        return x

    def latent_to_presoftmax(self, h: torch.Tensor) -> torch.Tensor:
        """
        Maps a latent representation to a preactivation output
        :param h: latent representations
        :return: presoftmax activations
        """
        h = self.lin3(h)
        return h

    def cont_batchnorm_output(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Returns the batchnormalization outputs for given input.
        '''
        x_cont, x_disc = x[:, : self.n_cont], x[:, self.n_cont :]
        if self.n_cont > 0:
            x_cont = self.bn1(x_cont)
        return x_cont
