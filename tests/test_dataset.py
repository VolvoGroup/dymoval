# -*- coding: utf-8 -*-


import dymoval as dmv
import numpy as np
from .fixture_data import *  # noqa
from dymoval.dataset import Signal
from typing import Any
import random
from matplotlib import pyplot as plt


class TestdatasetNominal:
    def test_init(self, good_dataframe: pd.DataFrame) -> None:
        # Check if the passed dataset DataFrame is correctly stored as class attribute.
        # Nominal data
        df_expected, u_labels, y_labels, fixture = good_dataframe

        # Actual value
        name_ds = "my_dataset"
        ds = dmv.dataset.Dataset(
            name_ds, df_expected, u_labels, y_labels, full_time_interval=True
        )

        # Expected value
        u_labels = dmv.str2list(u_labels)
        y_labels = dmv.str2list(y_labels)
        u_extended_labels = list(zip(["INPUT"] * len(u_labels), u_labels))
        y_extended_labels = list(zip(["OUTPUT"] * len(y_labels), y_labels))
        df_expected.columns = pd.MultiIndex.from_tuples(
            [*u_extended_labels, *y_extended_labels]
        )

        print(ds.dataset)
        print(df_expected)
        assert np.allclose(ds.dataset, df_expected)

    def test_remove_means(self, sine_dataframe: pd.DataFrame) -> None:
        df, u_labels, y_labels, fixture = sine_dataframe

        # Actual value
        name_ds = "my_dataset"
        ds = dmv.dataset.Dataset(
            name_ds, df, u_labels, y_labels, full_time_interval=True
        )

        # You should get a dataframe with zero mean.
        df_zero_mean = ds.remove_means()
        # Lets see if it is true
        assert np.allclose(df_zero_mean.mean(), 0.0)

        # inplace test
        ds.remove_means(inplace=True)
        # Lets see if it is true
        assert np.allclose(ds.dataset.mean(), 0.0)

    def test_remove_offset(self, constant_ones_dataframe: pd.DataFrame) -> None:
        df, u_labels, y_labels, fixture = constant_ones_dataframe

        # Test values. OBS! constant_ones_dataframe has 3 input and 3 output.
        u_list = {
            "SISO": ("u1", 2.0),
            "SIMO": ("u1", 2.0),
            "MISO": [("u1", 2.0), ("u2", 2.0), ("u3", 2.0)],
            "MIMO": [("u1", 2.0), ("u2", 2.0), ("u3", 2.0)],
        }

        y_list = {
            "SISO": ("y1", 2.0),
            "SIMO": [("y1", 2.0), ("y2", 1.0), ("y3", -2.0)],
            "MISO": ("y1", 2.0),
            "MIMO": [("y1", 2.0), ("y2", 1.0), ("y3", 2.0)],
        }

        # Expected values. Compare with constant_ones_dataframe and the data above.
        N = 10
        idx = np.linspace(0, 1, N)
        if fixture == "SISO":
            values = -1.0 * np.ones((N, 2))  # Expected
            df_expected = pd.DataFrame(
                index=idx, columns=[u_labels, y_labels], data=values
            )
            df_expected.index.name = "Time (s)"
        if fixture == "SIMO":
            values = np.vstack(
                (
                    -1.0 * np.ones(N),
                    -1.0 * np.ones(N),
                    np.zeros(N),
                    3.0 * np.ones(N),
                )
            ).transpose()  # Expected
            df_expected = pd.DataFrame(
                index=idx, columns=[u_labels, *y_labels], data=values
            )
            df_expected.index.name = "Time (s)"
        if fixture == "MISO":
            values = -1.0 * np.ones((N, 4))  # Expected
            df_expected = pd.DataFrame(
                index=idx, columns=[*u_labels, y_labels], data=values
            )
            df_expected.index.name = "Time (s)"
        if fixture == "MIMO":
            values = np.hstack(
                (
                    -1.0 * np.ones((N, 4)),
                    np.zeros((N, 1)),
                    -1.0 * np.ones((N, 1)),
                )
            )  # Expected
            df_expected = pd.DataFrame(
                index=idx, columns=[*u_labels, *y_labels], data=values
            )
            df_expected.index.name = "Time (s)"

        # Actual value
        name_ds = "my_dataset"
        ds = dmv.dataset.Dataset(
            name_ds, df, u_labels, y_labels, full_time_interval=True
        )

        print(u_list[fixture])
        df_actual = deepcopy(
            ds.remove_offset(u_list=u_list[fixture], y_list=y_list[fixture])
        )

        # Assert
        assert np.allclose(df_actual, df_expected)


class Test_plots:
    def test_plots(self, sine_dataframe: pd.DataFrame) -> None:
        df, u_labels, y_labels, fixture = sine_dataframe

        # Actua value
        name_ds = "my_dataset"
        ds = dmv.dataset.Dataset(
            name_ds, df, u_labels, y_labels, full_time_interval=True
        )

        ds.plot()
        plt.close("all")

        ds.plot_coverage()
        plt.close("all")

        ds.plot_amplitude_spectrum()
        plt.close("all")


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
