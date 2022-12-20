import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pandas import DataFrame

from models.basemodel_torch import BaseModelTorch

'''
    Custom implementation for the standard multi-layer perceptron
'''


class TORCHRLN(BaseModelTorch):

    def __init__(self, params, args, experiment=None):
        super().__init__(params, args, experiment)

        self.dataset = args.dataset
        lr = self.params['learning_rate']

        self.model = MLP_ModelRLN(n_layers=self.params["n_layers"], input_dim=self.args.num_features,
                               hidden_dim=self.params["hidden_dim"], output_dim=self.args.num_classes,
                               task=self.args.objective)
        self.to_device()



        self.rln_callback = RLNCallback(self.model.module.layers[0], norm=self.params["norm"],
                                  avg_reg=self.params["theta"], learning_rate=lr)

    def fit(self, X, y, X_val=None, y_val=None):
        X = np.array(X, dtype=np.float)
        X_val = np.array(X_val, dtype=np.float)

        self.rln_callback.on_train_begin()
        return  super(TORCHRLN, self).fit(X, y, X_val, y_val)

    def run_batch(self, batch_X, batch_y, loss_func, loss_history, optimizer):
        super(TORCHRLN, self).run_batch(batch_X, batch_y, loss_func, loss_history, optimizer)
        self.rln_callback.on_batch_end()

    def predict_helper(self, X):
        X = np.array(X, dtype=np.float)
        return super().predict_helper(X)

    @classmethod
    def define_trial_parameters(cls, trial, args):
        params = {
            "hidden_dim": trial.suggest_int("hidden_dim", 10, 100),
            "n_layers": trial.suggest_int("n_layers", 2, 5),
            "learning_rate": trial.suggest_float("learning_rate", 0.0005, 0.001),
            "norm": trial.suggest_categorical("norm", [1, 2]),
            "theta": trial.suggest_int("theta", -12, -8),
        }
        return params



class MLP_ModelRLN(nn.Module):

    def __init__(self, n_layers, input_dim, hidden_dim, output_dim, task):
        super().__init__()

        self.task = task

        self.layers = nn.ModuleList()

        # Input Layer (= first hidden layer)
        self.input_layer = nn.Linear(input_dim, hidden_dim)

        # Hidden Layers (number specified by n_layers)
        self.layers.extend([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers - 1)])

        # Output Layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.input_layer(x))

        # Use ReLU as activation for all hidden layers
        for layer in self.layers:
            x = F.relu(layer(x))

        # No activation function on the output
        x = self.output_layer(x)

        if self.task == "classification":
            x = F.softmax(x, dim=1)

        return x



class RLNCallback(object):
    def __init__(self, layer, norm=1, avg_reg=-7.5, learning_rate=6e5):
        """
        An implementation of Regularization Learning, described in https://arxiv.org/abs/1805.06440, as a Keras
        callback.
        :param layer: The Keras layer to which we apply regularization learning.
        :param norm: Norm of the regularization. Currently supports only l1 and l2 norms. Best results were obtained
        with l1 norm so far.
        :param avg_reg: The average regularization coefficient, Theta in the paper.
        :param learning_rate: The learning rate of the regularization coefficients, nu in the paper. Note that since we
        typically have many weights in the network, and we optimize the coefficients in the log scale, optimal learning
        rates tend to be large, with best results between 10^4-10^6.
        """
        super(RLNCallback, self).__init__()
        self._layer = layer
        self._prev_weights, self._weights, self._prev_regularization = [None] * 3
        self._avg_reg = avg_reg
        self._shape = torch.t(self._layer.weight).shape
        self._lambdas = DataFrame(np.ones(self._shape) * self._avg_reg)
        self._lr = learning_rate
        assert norm in [1, 2], "Only supporting l1 and l2 norms at the moment"
        self.norm = norm

    def on_train_begin(self, logs=None):
        self._update_values()

    def on_batch_end(self, logs=None):
        self._prev_weights = self._weights
        self._update_values()
        gradients = self._weights - self._prev_weights

        # Calculate the derivatives of the norms of the weights
        if self.norm == 1:
            norms_derivative = np.sign(self._weights)
        else:
            norms_derivative = self._weights * 2

        if self._prev_regularization is not None:
            # This is not the first batch, and we need to update the lambdas
            lambda_gradients = gradients.multiply(self._prev_regularization)
            self._lambdas -= self._lr * lambda_gradients

            # Project the lambdas onto the simplex \sum(lambdas) = Theta
            translation = (self._avg_reg - self._lambdas.mean().mean())
            self._lambdas += translation

        # Clip extremely large lambda values to prevent overflow
        max_lambda_values = np.log(np.abs(self._weights / norms_derivative)).fillna(np.inf)
        self._lambdas = self._lambdas.clip(upper=max_lambda_values)

        # Update the weights
        regularization = norms_derivative.multiply(np.exp(self._lambdas))
        self._weights -= regularization

        with torch.no_grad():
            self._layer.weight = nn.Parameter(torch.t(torch.Tensor(self._weights.values)))

        self._prev_regularization = regularization

    def _update_values(self):
        self._weights = DataFrame(torch.t(self._layer.weight.detach()))
