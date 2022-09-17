# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 11:43:21 2022

@author: yt75534
"""

import pytest
import dymoval as dmv
from dymoval.dataset import Signal
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random
from .fixture_data import *  # noqa
from typing import Any, Union


class TestSignalValidation:
    def test_name_unicity(self, good_signals: list[Signal]) -> None:
        # Nominal values
        signal_list, _, _, _ = good_signals

        signal_list[1]["name"] = signal_list[0]["name"]
        with pytest.raises(ValueError):
            dmv.signals_validation(signal_list)

    def test_key_not_found(self, good_signals: list[Signal]) -> None:
        # Nominal values
        signal_list, _, _, _ = good_signals

        idx = random.randrange(0, len(signal_list))
        key = "name"
        signal_list[idx].pop(key)
        with pytest.raises(KeyError):
            dmv.signals_validation(signal_list)

    def test_wrong_key(self, good_signals: list[Signal]) -> None:
        # Nominal values
        signal_list, _, _, _ = good_signals

        idx = random.randrange(0, len(signal_list))
        k_new = "potato"
        signal_list[idx][k_new] = signal_list[idx].pop("values")
        with pytest.raises(KeyError):
            dmv.signals_validation(signal_list)

    @pytest.mark.parametrize(
        # The dataype for the annotation can be inferred by the
        # following list.
        "test_input, expected",
        [
            (np.zeros((2, 2)), Exception),
            ("potato", Exception),
            (3, Exception),
            (np.zeros(1), IndexError),
        ],
    )
    def test_wrong_values(
        self,
        good_signals: list[Signal],
        test_input: Any,
        expected: Any,
    ) -> None:
        # Nominal values
        signal_list, _, _, _ = good_signals

        idx = random.randrange(0, len(signal_list))
        signal_list[idx]["values"] = test_input
        with pytest.raises(expected):
            dmv.signals_validation(signal_list)

    @pytest.mark.parametrize(
        "test_input, expected",
        [
            (np.zeros((2, 2)), TypeError),
            ("potato", TypeError),
            (-0.1, ValueError),
        ],
    )
    def test_wrong_sampling_period(
        self, good_signals: list[Signal], test_input: Any, expected: Any
    ) -> None:
        # Nominal values
        signal_list, _, _, _ = good_signals

        idx = random.randrange(0, len(signal_list))
        signal_list[idx]["sampling_period"] = test_input
        with pytest.raises(expected):
            dmv.signals_validation(signal_list)


class TestDataframeValidation:
    def test_there_is_at_least_one_in_and_one_out(
        self, good_dataframe: pd.DataFrame
    ) -> None:
        # Nominal values
        df, u_labels, y_labels, _ = good_dataframe
        u_labels = []
        with pytest.raises(IndexError):
            dmv.dataframe_validation(df, u_labels, y_labels)

    def test_name_unicity(self, good_dataframe: pd.DataFrame) -> None:
        # Nominal values
        df, u_labels, y_labels, fixture = good_dataframe
        u_labels_test = u_labels
        y_labels_test = y_labels
        if fixture == "SISO":  # If SISO the names are obviously unique.
            u_labels = y_labels
            u_labels_test = y_labels
        if fixture == "MISO" or fixture == "MIMO":
            u_labels_test[-1] = u_labels_test[-2]
        if fixture == "SIMO" or fixture == "MIMO":
            y_labels_test[0] = y_labels_test[1]
        with pytest.raises(ValueError):
            dmv.dataframe_validation(df, u_labels_test, y_labels)
        with pytest.raises(ValueError):
            dmv.dataframe_validation(df, u_labels, y_labels_test)
        with pytest.raises(ValueError):
            dmv.dataframe_validation(df, u_labels_test, y_labels_test)

    def test_dataframe_one_level_indices(
        self, good_dataframe: pd.DataFrame
    ) -> None:
        # Nominal values
        df, u_labels, y_labels, _ = good_dataframe
        df_test = df
        df_test.columns = pd.MultiIndex.from_product([df.columns, ["potato"]])
        with pytest.raises(IndexError):
            dmv.dataframe_validation(df_test, u_labels, y_labels)
        df_test = df
        df_test.index = pd.MultiIndex.from_product([df.index, ["potato"]])
        with pytest.raises(IndexError):
            dmv.dataframe_validation(df_test, u_labels, y_labels)

    def test_at_least_two_samples_per_signal(
        self, good_dataframe: pd.DataFrame
    ) -> None:
        # Nominal values
        df, u_labels, y_labels, _ = good_dataframe
        df_test = df.head(1)
        with pytest.raises(IndexError):
            dmv.dataframe_validation(df_test, u_labels, y_labels)

    def test_labels_exist_in_dataframe(
        self, good_dataframe: pd.DataFrame
    ) -> None:
        # Nominal values
        df, u_labels, y_labels, fixture = good_dataframe
        if fixture == "SISO":
            u_labels = [u_labels]
            y_labels = [y_labels]
        if fixture == "MISO":
            y_labels = [y_labels]
        if fixture == "SIMO":
            u_labels = [u_labels]
        u_labels[-1] = "potato"
        with pytest.raises(ValueError):
            dmv.dataframe_validation(df, u_labels, y_labels)

    def test_index_monotonicity(self, good_dataframe: pd.DataFrame) -> None:
        # Nominal values
        df, u_labels, y_labels, _ = good_dataframe
        df.index.values[0:1] = df.index[1]
        with pytest.raises(ValueError):
            dmv.dataframe_validation(df, u_labels, y_labels)

    def test_values_are_float(self, good_dataframe: pd.DataFrame) -> None:
        # Nominal values
        df, u_labels, y_labels, _ = good_dataframe
        df.iloc[0:1, 0:1] = "potato"
        with pytest.raises(TypeError):
            dmv.dataframe_validation(df, u_labels, y_labels)


class TestFixSamplingPeriod:
    def test_excluded_signals_no_args(self, good_signals: list[Signal]) -> None:
        # Nominal values
        (
            signal_list,
            input_signal_names,
            output_signal_names,
            fixture_instance,
        ) = good_signals

        # Test data
        # Always the first two signals to allow the same test also for SISO case.

        signal_list[0]["sampling_period"] = 0.017
        expected_excluded_1 = signal_list[0]["name"]

        signal_list[1]["sampling_period"] = 0.021
        expected_excluded_2 = signal_list[1]["name"]

        # Identify MIMO, SISO, MISO or SIMO case and set the expected exlcuded signal
        if fixture_instance == "SISO":
            expected_excluded = [expected_excluded_1]
        else:
            expected_excluded = [
                expected_excluded_1,
                expected_excluded_2,
            ]
        expected_resampled = [
            s["name"] for s in signal_list if s["name"] not in expected_excluded
        ]

        # Assert
        actual_resampled, actual_excluded = dmv.fix_sampling_periods(
            signal_list
        )
        actual_resampled = [s["name"] for s in actual_resampled]
        assert sorted(actual_excluded) == sorted(expected_excluded)
        assert sorted(actual_resampled) == sorted(expected_resampled)

    def test_excluded_signals(self, good_signals: list[Signal]) -> None:
        # Nominal values
        (
            signal_list,
            input_signal_names,
            output_signal_names,
            fixture,
        ) = good_signals

        test_value = 0.2  # target_sampling_period
        # Test data.
        # Always the first two signals to allow the same test also for SISO case.
        signal_list[0]["sampling_period"] = 0.017
        expected_excluded_1 = signal_list[0]["name"]

        # first_output_idx = len(input_signal_names)
        signal_list[1]["sampling_period"] = 0.023
        expected_excluded_2 = signal_list[1]["name"]

        #
        expected_excluded = [
            expected_excluded_1,
            expected_excluded_2,
        ]
        expected_resampled = [
            s["name"] for s in signal_list if s["name"] not in expected_excluded
        ]

        # Assert
        actual_resampled, actual_excluded = dmv.fix_sampling_periods(
            signal_list, test_value
        )
        actual_resampled = [s["name"] for s in actual_resampled]
        assert sorted(actual_excluded) == sorted(expected_excluded)
        assert sorted(actual_resampled) == sorted(expected_resampled)

    @pytest.mark.parametrize(
        "test_input, expected",
        [("potato", Exception), (-0.1, Exception)],
    )
    def test_excluded_signals_exception(
        self,
        good_signals: list[Signal],
        test_input: tuple[Any, Any],
        expected: tuple[Any, Any],
    ) -> None:
        # Nominal values
        signal_list, _, _, _ = good_signals
        with pytest.raises(expected):
            dmv.signals_validation(signal_list, test_input)


class Test_list_belonging_check:
    # This function returns elements not found.
    @pytest.mark.parametrize(
        "A,B, expected",
        [
            (
                ["dog", "cat", "donkey"],
                ["iguana", "dog", "cat", "donkey", "snake", "cow"],
                [],
            ),
            (
                ["banana", "dog", "cat", "orange", "apple"],
                ["iguana", "dog", "cat", "donkey", "snake", "cow"],
                ["banana", "orange", "apple"],
            ),
            ("iguana", ["iguana", "dog", "cat", "donkey", "snake", "cow"], []),
            (
                "orange",
                ["iguana", "dog", "cat", "donkey", "snake", "cow"],
                ["orange"],
            ),
            (
                ["iguana", "dog", "cat", "donkey", "snake", "cow"],
                "orange",
                ["iguana", "dog", "cat", "donkey", "snake", "cow"],
            ),
            (
                ["iguana", "dog", "cat", "donkey", "snake", "cow"],
                "iguana",
                ["dog", "cat", "donkey", "snake", "cow"],
            ),
        ],
    )
    def test_list_belonging_check(
        self,
        A: Union[str, list[str]],
        B: Union[str, list[str]],
        expected: list[str],
    ) -> None:
        # Nominal
        elements_not_found = dmv.list_belonging_check(A, B)
        assert sorted(elements_not_found) == sorted(expected)

    def test_list_belonging_check_empty_set(self) -> None:
        # Error
        B = [0, 1, 2, 3, 4, 5]
        with pytest.raises(IndexError):
            dmv.list_belonging_check([], B)


class Test_str2list:
    @pytest.mark.parametrize(
        "x, expected",
        [
            ("u1", ["u1"]),  # str input
            (
                ["a", "b", "c", "d"],
                ["a", "b", "c", "d"],
            ),  # list input
        ],
    )
    def test_str2list(
        self, x: Union[str, list[str]], expected: list[str]
    ) -> None:
        actual = dmv.str2list(x)
        assert sorted(actual) == sorted(expected)


class Test_plot:
    def test_plot_nominal(self, good_signals: list[Signal]) -> None:
        # You should just get a plot.
        signal_list, u_labels, y_labels, fixture = good_signals
        dmv.plot_signals(signal_list)
        plt.close("all")

        # Conditional plot
        if fixture == "MIMO":
            del u_labels[-1]
            del y_labels[-1]
        if fixture == "MISO":
            u_labels[-1]
        if fixture == "SIMO":
            del y_labels[-1]
        dmv.plot_signals(signal_list, u_labels, y_labels)
        plt.close("all")

        # With only args
        dmv.plot_signals(signal_list, u_labels)
        plt.close("all")

    def test_plot_raise(self, good_signals: list[Signal]) -> None:
        # Nominal values
        signal_list, u_labels, y_labels, fixture = good_signals
        u_labels_test = u_labels
        y_labels_test = y_labels

        if fixture == "MIMO":
            u_labels_test[0] = "potato"
            y_labels_test[0] = "banana"
        if fixture == "SISO":
            u_labels_test = ["potato"]
            y_labels_test = ["banana"]
        if fixture == "MISO":
            u_labels_test[0] = "potato"
            y_labels_test = ["banana"]
        if fixture == "SIMO":
            u_labels_test = ["potato"]
            y_labels_test[0] = "banana"
        with pytest.raises(KeyError):
            dmv.plot_signals(signal_list, u_labels_test, y_labels)
        with pytest.raises(KeyError):
            dmv.plot_signals(signal_list, u_labels_test, y_labels_test)
        with pytest.raises(KeyError):
            dmv.plot_signals(signal_list, u_labels_test, y_labels_test)
