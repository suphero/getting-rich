# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://github.com/hudson-and-thames/mlfinlab/blob/master/LICENSE.txt
"""
Tests the data generated by bootstrap methods in data_generation/bootstrap.py
"""

import unittest
import os
import numpy as np
import pandas as pd

from mlfinlab.data_generation.bootstrap import row_bootstrap, pair_bootstrap, block_bootstrap


class TestBootstrap(unittest.TestCase):
    """
    Test the data generated from the bootstrap generation modules.
    """

    def setUp(self):
        """
        Sets the random number generator seed and parameters for functions.
        """
        np.random.seed(2814)
        project_path = os.path.dirname(__file__)
        self.stock_returns_small = (
            pd.read_csv("{}/test_data/stock_prices_2.csv".format(project_path), index_col=0)
            .iloc[:100]
            .pct_change()
            .dropna()
        )
        self.small_shapes = [
            self.stock_returns_small.shape,
            (10, 6),
            (self.stock_returns_small.shape[0], 1),
            (1, self.stock_returns_small.shape[1]),
        ]
        self.pair_shapes = [
            self.stock_returns_small.shape,
            (10, 6),
            (4, 4),
            (1, 3),
        ]
        self.n_samples = [1, 15, 7, 3]
        self.block_size = [(2, 2), (6, 6), self.stock_returns_small.shape, (1, 1), (6, 1), (3, 5)]
        self.n_generated = len(self.small_shapes)

    @staticmethod
    def _data_exists(mats):
        """
        Returns whether mats exists and have elements.
        """

        return mats is not None and len(mats) > 0

    @staticmethod
    def _data_bounded(mats, original_mat):
        """
        Tests that the max and min in original matrix are upper and lower bounds, and that there
        is no value in the generated matrix that it is not in the original matrix.
        """
        mat_is_lower_bounded = (mats >= np.min(original_mat)).all()
        mat_is_upper_bounded = (mats <= np.max(original_mat)).all()
        mat_is_from_original = np.isin(mats.flatten(), original_mat).all()

        return mat_is_lower_bounded and mat_is_upper_bounded and mat_is_from_original

    @staticmethod
    def _corr_mat_correctly_bounded(mats, dim):
        """
        Returns whether a correlation matrix is correctly bounded.
        """

        diag_rows, diag_cols = np.diag_indices(dim)
        diag_is_ones = (mats[:, diag_rows, diag_cols] == 1).all()
        mat_is_lower_bounded = (mats >= -1).all()
        mat_is_upper_bounded = (mats <= 1).all()

        return diag_is_ones and mat_is_lower_bounded and mat_is_upper_bounded

    @staticmethod
    def _corr_mat_is_symmetric(mats, dim):
        """
        Returns whether a correlation matrix is symmetrical.
        """

        result = True
        diag_rows, diag_cols = np.diag_indices(dim)

        for corr_mat in mats:
            result &= (corr_mat[diag_rows, diag_cols] == corr_mat[diag_cols, diag_rows]).all()

        return result

    def test_data_returned(self):
        """
        Tests that the data generated from all methods exists.
        """

        # Test all methods with DataFrame.
        for func in [row_bootstrap, pair_bootstrap, block_bootstrap]:
            bootstrap_mat = func(self.stock_returns_small)
            self.assertTrue(self._data_exists(bootstrap_mat), msg="{} failed".format(func.__name__))

        # Test all methods with numpy array.
        for func in [row_bootstrap, pair_bootstrap, block_bootstrap]:
            bootstrap_mat = func(self.stock_returns_small.values)
            self.assertTrue(self._data_exists(bootstrap_mat), msg="{} failed".format(func.__name__))

    def test_correct_shape(self):
        """
        Tests that the data generated from all methods has the correct shapes.
        """

        # Test all methods.
        for func in [row_bootstrap, pair_bootstrap, block_bootstrap]:
            for i in range(self.n_generated):
                bootstrap_mat = func(
                    self.stock_returns_small, n_samples=self.n_samples[i], size=self.small_shapes[i]
                )
                if func == pair_bootstrap:
                    self.assertTrue(
                        bootstrap_mat.shape
                        == (self.n_samples[i], self.small_shapes[i][1], self.small_shapes[i][1]),
                        msg="{} failed".format(func.__name__),
                    )
                else:
                    self.assertTrue(
                        bootstrap_mat.shape
                        == (self.n_samples[i], self.small_shapes[i][0], self.small_shapes[i][1]),
                        msg="{} failed".format(func.__name__),
                    )

    def test_correctly_bounded(self):
        """
        Tests that the data generated from row and block methods is within bounds (max and min in original
        matrix are upper and lower bounds, and that there is no value in the generated matrix
        that it is not in the original matrix).
        """

        # Test row, block methods.
        for func in [row_bootstrap, block_bootstrap]:
            for i in range(self.n_generated):
                bootstrap_mat = func(
                    self.stock_returns_small, n_samples=self.n_samples[i], size=self.small_shapes[i]
                )
                self.assertTrue(
                    self._data_bounded(bootstrap_mat, self.stock_returns_small.values),
                    msg="{} failed".format(func.__name__),
                )

        # Test for different block sizes.
        for i in range(len(self.block_size)):
            bootstrap_mat = block_bootstrap(self.stock_returns_small, block_size=self.block_size[i])
            self.assertTrue(self._data_bounded(bootstrap_mat, self.stock_returns_small.values))

    def test_pair_correctly_bounded(self):
        """
        Tests that the data generated from the pair method is within bounds and its symmetrical.
        """

        # Test pair methods.
        for i in range(self.n_generated):
            bootstrap_mat = pair_bootstrap(
                self.stock_returns_small, n_samples=self.n_samples[i], size=self.pair_shapes[i]
            )
            self.assertTrue(self._corr_mat_correctly_bounded(bootstrap_mat, self.pair_shapes[i][1]))
            self.assertTrue(self._corr_mat_is_symmetric(bootstrap_mat, self.pair_shapes[i][1]))

    def test_blocks_are_consistent(self):
        """
        Tests that the data generated from the block bootstrap method is bounded by the required blocks.
        """

        sample_mat = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                               [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                               [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
                               [31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
                               [41, 42, 43, 44, 45, 46, 47, 48, 49, 50]])
        blocks = [(2, 2), (3, 3), (2, 5), (4, 3), (2, 10), (5, 2)]
        for n_block in blocks:
            bootstrap_mat = block_bootstrap(sample_mat, block_size=n_block)[0]

            # Test that the start location of each block satisfies the properties:
            # 1) It is equal to its right element - 1
            # 2) It is equal to its bottom element - 10
            # 3) It is equal to its lower diagonal element - 11
            # These tests have to make sure edges of the matrix are accounted for.
            for row in range(0, sample_mat.shape[0], n_block[0]):
                for col in range(0, sample_mat.shape[1], n_block[1]):
                    # Check the right, bottom, diagonal element only if it is not at an edge.
                    if col < sample_mat.shape[1] - (n_block[1] - 1):
                        self.assertTrue(bootstrap_mat[row, col] + 1 == bootstrap_mat[row, col + 1])
                    if row < sample_mat.shape[0] - (n_block[0] - 1):
                        self.assertTrue(bootstrap_mat[row, col] + 10 == bootstrap_mat[row + 1, col])
                    if col < sample_mat.shape[1] - (n_block[1] - 1) and row < sample_mat.shape[0] - (n_block[0] - 1):
                        self.assertTrue(bootstrap_mat[row, col] + 11 == bootstrap_mat[row + 1, col + 1])
