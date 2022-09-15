# -*- coding: utf-8 -*-

import pytest
import dymoval as dmv
import numpy as np
from matplotlib import pyplot as plt
from fixture_data import *  # noqa


class TestdatasetNominal:
    # def test_init(self, good_dataframe):
    #     # Nominal data
    #     df, u_labels, y_labels, fixture = good_dataframe

    #     # Actua value
    #     name_ds = "my_dataset"
    #     ds = dmv.dataset.Dataset(
    #         name_ds, df, u_labels, y_labels, full_time_interval=True
    #     )

    #     # Expected value
    #     u_labels, y_labels = dmv.str2list(u_labels, y_labels)
    #     u_extended_labels = list(zip(["INPUT"] * len(u_labels), u_labels))
    #     y_extended_labels = list(zip(["OUTPUT"] * len(y_labels), y_labels))
    #     df.columns = pd.MultiIndex.from_tuples([*u_extended_labels, *y_extended_labels])

    #     # Check that the passed Dataset is correctly stored.
    #     # Main DataFrame
    #     ds._dataset - df
    #     assert ds._dataset == df

    def test_remove_means(self, sine_dataframe):
        df, u_labels, y_labels, fixture = sine_dataframe

        # Actua value
        name_ds = "my_dataset"
        ds = dmv.dataset.Dataset(
            name_ds, df, u_labels, y_labels, full_time_interval=True
        )

        # Remove means and check if they are removed
        ds.remove_means()
        actual_means = ds.dataset.mean().to_numpy()
        for ii, actual_mean in enumerate(actual_means):
            assert np.allclose(actual_mean, 0.0)

    def test_remove_offset(self, sine_dataframe):
        df, u_labels, y_labels, fixture = sine_dataframe

        # Actua value
        name_ds = "my_dataset"
        ds = dmv.dataset.Dataset(
            name_ds, df, u_labels, y_labels, full_time_interval=True
        )

        u_list = {
            "SISO": ("u1", 2.0),
            "SIMO": ("u1", 2.0),
            "MISO": [("u1", 2.0), ("u2", 2.0), ("u3", 2.0)],
            "MIMO": [("u1", 2.0), ("u2", 2.0), ("u3", 2.0)],
        }

        y_list = {
            "SISO": ("y1", 2.0),
            "SIMO": [("y1", 2.0), ("y2", 1.0), ("y3", 2.0)],
            "MISO": ("y1", 2.0),
            "MIMO": [("y1", 2.0), ("y2", 1.0), ("y3", 2.0)],
        }

        ds.remove_offset(u_list=u_list[fixture], y_list=y_list[fixture])
        actual_means = ds.dataset.mean().to_numpy()
        for ii, actual_mean in enumerate(actual_means):
            assert np.allclose(actual_mean, 0.0)
