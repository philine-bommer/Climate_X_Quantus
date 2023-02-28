from typing import Callable
import keras
import keras.backend as K
import numpy as np
from tqdm import tqdm
import tensorflow as tf



class NoiseGrad:
    def __init__(
        self,
        model,
        mean: float = 1.0,
        std: float = 0.2,
        n: int = 25,
        noise_type: str = "multiplicative",
        verbose: bool = True,
    ):
        """
        Initialize the explanation-enhancing method: NoiseGrad.
        Paper:

        Args:
            model (torch model): a trained model
            weights (dict):
            mean (float): mean of the distribution (often referred to as mu)
            std (float): standard deviation of the distribution (often referred to as sigma)
            n (int): number of Monte Carlo rounds to sample models
            noise_type (str): the type of noise to add to the model parameters, either additive or multiplicative
            verbose (bool): print the progress of explanation enchancement (default True)
        """

        self.std = std
        self.mean = mean

        self.model = keras.models.clone_model(model)
        self.basemodel = model
        self.n = n
        self.noise_type = noise_type
        self.verbose = verbose



        print("NoiseGrad initialized.")

    def sample(self):

        # If std is not zero, loop over each layer and add Gaussian noise.
        if not self.std == 0.0:
            # with tf.stop_gradient():
            for layer in self.model.layers:
                weight = []
                if layer.get_weights():
                    for weights in layer.get_weights():
                        if len(weights.shape) == 1:
                            noise_mat = (np.random.randn(weights.shape[0], ) + self.mean) * self.std
                        else:
                            noise_mat = (np.random.randn(*weights.shape) + self.mean) * self.std
                        if self.noise_type == "additive":
                            weight.append(np.add(weights, noise_mat))
                        elif self.noise_type == "multiplicative":
                            weight.append(np.multiply(weights, noise_mat))
                        else:
                            print(
                                "Set NoiseGrad attribute 'noise_type' to either 'additive' or 'multiplicative' (str)."
                            )

                    layer.set_weights(weight)

    def enhance_explanation(self, inputs, targets, explanation_fn: Callable, **kwargs):
        """Sample explanation."""
        explanation = np.zeros(
            inputs.shape
        )

        for i in (tqdm(range(self.n)) if self.verbose else range(self.n)):
            self.sample()
            explanation = np.add(explanation, explanation_fn(
                self.model, inputs, targets, **kwargs
            ))
            del self.model
            self.model = keras.models.clone_model(self.basemodel)
        return explanation/self.n


class NoiseGradPlusPlus(NoiseGrad):
    def __init__(
        self,
       model,
        mean: float = 1.0,
        std: float = 0.2,
        sg_mean: float = 0.0,
        sg_std: float = 0.4,
        n: int = 10,
        m: int = 10,
        noise_type: str = "multiplicative",
        verbose: bool = True
    ):
        """
        Initialize the explanation-enhancing method: NoiseGrad++.
        Paper:

        Args:
            model (torch model): a trained model
            weights (dict):
            mean (float): mean of the distribution (often referred to as mu)
            std (float): standard deviation of the distribution (often referred to as sigma)
            n (int): number of Monte Carlo rounds to sample models
            noise_type (str): the type of noise to add to the model parameters, either additive or multiplicative
            verbose (bool): print the progress of explanation enchancement (default True)

        Args:
            model:
            weights:
            mean:
            std:
            sg_mean:
            sg_std:
            n:
            m:
            noise_type:
        """

        self.std = std
        self.mean = mean
        self.model = keras.models.clone_model(model)
        self.basemodel = model
        self.n = n
        self.m = m
        self.sg_std = sg_std
        self.sg_mean = sg_mean
        # self.weights = weights
        self.noise_type = noise_type
        self.verbose = verbose



        super(NoiseGrad, self).__init__()
        print("NoiseGrad++ initialized.")

    def sample(self):
        # self.model.trainable_variables(self.weights)
        # If std is not zero, loop over each layer and add Gaussian noise.

        if not self.std == 0.0:

            for layer in self.model.layers:
                weight = []
                if layer.get_weights():
                    for weights in layer.get_weights():
                        if len(weights.shape) == 1:
                            noise_mat = (np.random.randn(weights.shape[0], ) + self.mean) * self.std
                        else:
                            noise_mat = (np.random.randn(*weights.shape) + self.mean) * self.std
                        if self.noise_type == "additive":

                            weight.append(np.add(weights, noise_mat))
                        elif self.noise_type == "multiplicative":

                            weight.append(np.multiply(weights, noise_mat))
                        else:
                            print(
                                "Set NoiseGrad attribute 'noise_type' to either 'additive' or 'multiplicative' (str)."
                            )

                    layer.set_weights(weight)


    def enhance_explanation(self, inputs, targets, explanation_fn: Callable, **kwargs):
        """Sample explanation."""

        explanation = np.zeros(inputs.shape
        )
        explanation_m = np.zeros(inputs.shape
        )


        for i in (tqdm(range(self.n)) if self.verbose else range(self.n)):
            self.sample()
            for j in range(self.m):
                if len(inputs.shape) == 1:
                    noise_inp = (np.random.randn(inputs.shape[0], )* self.sg_std) + self.sg_mean

                else:
                    noise_inp = (np.random.randn(*inputs.shape) * self.sg_std) + self.sg_mean

                inputs_noisy = (
                    inputs + noise_inp
                )
                explanation_m = np.add(explanation_m, explanation_fn(
                    self.model, inputs_noisy, targets,  **kwargs
                ))
            del self.model
            self.model = keras.models.clone_model(self.basemodel)
            explanation = np.add(explanation, explanation_m/self.m)
        return explanation/self.n
