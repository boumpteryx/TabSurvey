import torch
import numpy as np
import keras
import tensorflow as tf

### Keras / Tensorflow loss

class BalancedBCELossKeras(keras.losses.Loss):
    """
    Args:
      pos_weight: Scalar to affect the positive labels of the loss function.
      weight: Scalar to affect the entirety of the loss function.
      from_logits: Whether to compute loss from logits or the probability.
      reduction: Type of tf.keras.losses.Reduction to apply to loss.
      name: Name of the loss function.
    """
    def __init__(self, dataset, from_logits=False,
                 reduction=tf.keras.losses.Reduction.AUTO,
                 name='weighted_binary_crossentropy'):
        super().__init__(reduction=reduction, name=name)
        if dataset in ["url", "malware", "ctu_13_neris", "lcld_v2_time"]:
            from constrained_attacks import datasets
            _, y = datasets.load_dataset(dataset).get_x_y()
            y = np.array(y)
            y_class, y_occ = np.unique(y, return_counts=True)
            y_occ = (y_occ.max() - y_occ + y_occ.mean())
            self.weights = dict(zip(y_class, y_occ / y_occ.max()))
        self.from_logits = from_logits

    def call(self, target, input):

        if self.weights is None:
            ce = tf.losses.binary_crossentropy(target, input, from_logits=self.from_logits)

        else:

            loss = tf.losses.binary_crossentropy(target, input, from_logits=self.from_logits)
            positive_loss = target * loss
            negative_loss = (1 - target) * loss
            ce = positive_loss * self.weights.get(1) + negative_loss * self.weights.get(0)

        return ce


### Pytorch loss
class BalancedBCELossPytorch(torch.nn.BCEWithLogitsLoss):

    def __init__(self, dataset, **args):
        self.weights = None
        if dataset in ["url", "malware", "ctu_13_neris", "lcld_v2_time"]:
            from constrained_attacks import datasets
            _, y = datasets.load_dataset(dataset).get_x_y()
            y = np.array(y)
            y_class, y_occ = np.unique(y, return_counts=True)
            y_occ = (y_occ.max() -y_occ + y_occ.mean())
            self.weights = dict(zip(y_class, y_occ/y_occ.max()))

        super(BalancedBCELossPytorch, self).__init__(**args)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        if self.weights is None:
            return super(BalancedBCELossPytorch, self).forward(input, target)

        negative_inputs_mask = (target == 0)
        positive_inputs_mask = (target == 1)

        positive_inputs, positive_targets = input[positive_inputs_mask], input[positive_inputs_mask]
        positive_loss = super(BalancedBCELossPytorch, self).forward(positive_inputs, positive_targets)
        negative_inputs, negative_targets = input[negative_inputs_mask], input[negative_inputs_mask]
        negative_loss = super(BalancedBCELossPytorch, self).forward(negative_inputs, negative_targets)

        return positive_loss * self.weights.get(1) + negative_loss * self.weights.get(0)

