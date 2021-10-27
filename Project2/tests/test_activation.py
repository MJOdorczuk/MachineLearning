import unittest
import torch

import numpy as np
from csnet import activation

class Testactivation(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.activation_functions = activation.Activation()

    def test_sigmoid(self):
        tensor_range = torch.arange(-5,5)
        actual_torch = torch.sigmoid(tensor_range)
        actual_numpy = actual_torch.numpy()

        numpy_range = np.arange(-5,5)
        test_result = self.activation_functions.sigmoid(numpy_range)

        np.testing.assert_almost_equal(actual_numpy, test_result)

    def test_relu(self):
        tensor_range = torch.arange(-5,5)
        actual_torch = torch.nn.ReLU()(tensor_range)
        actual_numpy = actual_torch.numpy()

        numpy_range = np.arange(-5,5)
        test_result = self.activation_functions.relu(numpy_range)

        np.testing.assert_almost_equal(actual_numpy, test_result)

    def test_leaky_relu(self):
        tensor_range = torch.arange(-5,5).type(torch.float)
        actual_torch = torch.nn.LeakyReLU(self.activation_functions.alpha)(tensor_range)
        actual_numpy = actual_torch.numpy()

        numpy_range = np.arange(-5,5)
        test_result = self.activation_functions.leaky_relu(numpy_range)

        np.testing.assert_almost_equal(actual_numpy, test_result)




