# -*- coding: utf-8 -*-


import dymoval as dmv
import numpy as np
from .fixture_data import *  # noqa
from dymoval.dataset import Signal
from dymoval import NUM_DECIMALS
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
        assert np.allclose(df_zero_mean["INPUT"].mean(), 0.0)
        assert np.allclose(df_zero_mean["OUTPUT"].mean(), 0.0)

        # inplace test
        ds.remove_means(inplace=True)
        # Lets see if it is true
        assert np.allclose(ds.dataset["INPUT"].mean(), 0.0)
        assert np.allclose(ds.dataset["OUTPUT"].mean(), 0.0)

    def test_remove_offset_nominal(
        self, constant_ones_dataframe: pd.DataFrame
    ) -> None:
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
        df_actual = deepcopy(
            ds.remove_offset(u_list=u_list[fixture], y_list=y_list[fixture])
        )

        # Assert
        assert np.allclose(df_actual, df_expected)
        # Assert that the internally stored dataset is not overwritten
        assert np.allclose(ds.dataset, df)
        # Test inplace = True
        ds.remove_offset(
            u_list=u_list[fixture], y_list=y_list[fixture], inplace=True
        )
        assert np.allclose(df_actual, ds.dataset)

    def test_remove_offset_only_input(
        self, constant_ones_dataframe: pd.DataFrame
    ) -> None:
        df, u_labels, y_labels, fixture = constant_ones_dataframe

        # Test values. OBS! constant_ones_dataframe has 3 input and 3 output.
        u_list = {
            "SISO": ("u1", 2.0),
            "SIMO": ("u1", 2.0),
            "MISO": [("u1", 2.0), ("u2", 2.0), ("u3", 2.0)],
            "MIMO": [("u1", 2.0), ("u2", 2.0), ("u3", 2.0)],
        }

        # Expected values. Compare with constant_ones_dataframe and the data above.
        N = 10
        idx = np.linspace(0, 1, N)
        if fixture == "SISO":
            values = np.hstack(
                (-1.0 * np.ones((N, 1)), np.ones((N, 1)))
            )  # Expected
            df_expected = pd.DataFrame(
                index=idx, columns=[u_labels, y_labels], data=values
            )
            df_expected.index.name = "Time (s)"
        if fixture == "SIMO":
            values = np.hstack(
                (
                    -1.0 * np.ones((N, 1)),
                    np.ones((N, 3)),
                )
            )
            df_expected = pd.DataFrame(
                index=idx, columns=[u_labels, *y_labels], data=values
            )
            df_expected.index.name = "Time (s)"
        if fixture == "MISO":
            values = np.hstack(
                (-1.0 * np.ones((N, 3)), np.ones((N, 1)))
            )  # Expected
            df_expected = pd.DataFrame(
                index=idx, columns=[*u_labels, y_labels], data=values
            )
            df_expected.index.name = "Time (s)"
        if fixture == "MIMO":
            values = np.hstack(
                (
                    -1.0 * np.ones((N, 3)),
                    np.ones((N, 3)),
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

        df_actual = deepcopy(ds.remove_offset(u_list=u_list[fixture]))

        # Assert
        assert np.allclose(df_actual, df_expected)

    def test_remove_offset_only_output(
        self, constant_ones_dataframe: pd.DataFrame
    ) -> None:
        df, u_labels, y_labels, fixture = constant_ones_dataframe

        # Test values. OBS! constant_ones_dataframe has 3 input and 3 output.
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
            values = np.hstack(
                (np.ones((N, 1)), -1.0 * np.ones((N, 1)))
            )  # Expected
            df_expected = pd.DataFrame(
                index=idx, columns=[u_labels, y_labels], data=values
            )
            df_expected.index.name = "Time (s)"
        if fixture == "SIMO":
            values = np.hstack(
                (
                    np.ones((N, 1)),
                    -1.0 * np.ones((N, 1)),
                    np.zeros((N, 1)),
                    3.0 * np.ones((N, 1)),
                )
            )
            df_expected = pd.DataFrame(
                index=idx, columns=[u_labels, *y_labels], data=values
            )
            df_expected.index.name = "Time (s)"
        if fixture == "MISO":
            values = np.hstack(
                (np.ones((N, 3)), -1.0 * np.ones((N, 1)))
            )  # Expected
            df_expected = pd.DataFrame(
                index=idx, columns=[*u_labels, y_labels], data=values
            )
            df_expected.index.name = "Time (s)"
        if fixture == "MIMO":
            values = np.hstack(
                (
                    np.ones((N, 3)),
                    -1.0 * np.ones((N, 1)),
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

        df_actual = deepcopy(ds.remove_offset(y_list=y_list[fixture]))

        # Assert
        assert np.allclose(df_actual, df_expected)

    def test__signals_exist_raise(self, sine_dataframe: pd.DataFrame) -> None:
        df, u_labels, y_labels, fixture = sine_dataframe

        # Actual value
        name_ds = "my_dataset"
        ds = dmv.dataset.Dataset(
            name_ds, df, u_labels, y_labels, full_time_interval=True
        )

        # Test values. OBS! constant_ones_dataframe has 3 input and 3 output.
        u_list = {
            "SISO": ("potato", 2.0),
            "SIMO": ("u1", 2.0),
            "MISO": [("potato", 2.0), ("u2", 2.0), ("u3", 2.0)],
            "MIMO": [("u1", 2.0), ("u2", 2.0), ("u3", 2.0)],
        }

        y_list = {
            "SISO": ("y1", 2.0),
            "SIMO": [("y1", 2.0), ("potato", 1.0), ("y3", -2.0)],
            "MISO": ("y1", 2.0),
            "MIMO": [("potato", 2.0), ("y2", 1.0), ("y3", 2.0)],
        }

        with pytest.raises(KeyError):
            ds.remove_offset(u_list=u_list[fixture], y_list=y_list[fixture])

    # def test__validate_manipulation_functions_args(
    #     self, sine_dataframe: pd.DataFrame
    # ) -> None:
    #     df, u_labels, y_labels, fixture = sine_dataframe

    #     # Actual value
    #     name_ds = "my_dataset"
    #     ds = dmv.dataset.Dataset(
    #         name_ds, df, u_labels, y_labels, full_time_interval=True
    #     )

    def test_remove_offset_raise(self, sine_dataframe: pd.DataFrame) -> None:
        df, u_labels, y_labels, fixture = sine_dataframe

        # Actua value
        name_ds = "my_dataset"
        ds = dmv.dataset.Dataset(
            name_ds, df, u_labels, y_labels, full_time_interval=True
        )
        with pytest.raises(TypeError):
            ds.remove_offset()

    def test_get_dataset_values(self, sine_dataframe: pd.DataFrame) -> None:
        df, u_labels, y_labels, fixture = sine_dataframe

        # Instantiate dataset
        name_ds = "my_dataset"
        ds = dmv.dataset.Dataset(
            name_ds, df, u_labels, y_labels, full_time_interval=True
        )
        # Expected time vector
        t_expected = df.index.to_numpy().round(NUM_DECIMALS)
        u_expected = df[u_labels].to_numpy().round(NUM_DECIMALS)
        y_expected = df[y_labels].to_numpy().round(NUM_DECIMALS)

        # Actuals
        t_actual, u_actual, y_actual = ds.get_dataset_values()

        # Assert
        assert np.allclose(t_expected, t_actual)
        assert np.allclose(u_expected, u_actual)
        assert np.allclose(y_expected, y_actual)

    def test_get_signal_list(self, sine_dataframe: pd.DataFrame) -> None:
        df, u_labels, y_labels, fixture = sine_dataframe

        # Instantiate dataset
        name_ds = "my_dataset"
        ds = dmv.dataset.Dataset(
            name_ds, df, u_labels, y_labels, full_time_interval=True
        )

        # Expected based on fixture
        expected = {
            "SISO": [("INPUT", "u1"), ("OUTPUT", "y1")],
            "SIMO": [
                ("INPUT", "u1"),
                ("OUTPUT", "y1"),
                ("OUTPUT", "y2"),
                ("OUTPUT", "y3"),
                ("OUTPUT", "y4"),
            ],
            "MISO": [
                ("INPUT", "u1"),
                ("INPUT", "u2"),
                ("INPUT", "u3"),
                ("OUTPUT", "y1"),
            ],
            "MIMO": [
                ("INPUT", "u1"),
                ("INPUT", "u2"),
                ("INPUT", "u3"),
                ("OUTPUT", "y1"),
                ("OUTPUT", "y2"),
                ("OUTPUT", "y3"),
                ("OUTPUT", "y4"),
            ],
        }

        actual = ds.get_signal_list()
        assert expected[fixture] == actual

    def test_replaceNaNs(self, good_signals: list[Signal]) -> None:
        # Nominal data
        (
            signal_list,
            input_signal_names,
            output_signal_names,
            fixture,
        ) = good_signals

        target_sampling_period = 0.1
        ds = dmv.Dataset(
            "potato",
            signal_list,
            input_signal_names,
            output_signal_names,
            target_sampling_period=target_sampling_period,
            plot_raw=True,
            full_time_interval=True,
        )

        # Interpolate
        ds_test = deepcopy(ds)
        df_actual = ds_test.replace_NaNs(method="interpolate")
        # Assert that no NaN:s left
        assert not df_actual.isna().any().any()

        # Fill
        ds_test = deepcopy(ds)
        df_actual = ds_test.replace_NaNs(method="fillna")
        # Assert that no NaN:s left
        assert not df_actual.isna().any().any()

        # Raise
        with pytest.raises(ValueError):
            ds_test.replace_NaNs(method="potato")


class TestDatasetPlots:
    def test_Dataset_plots(self, sine_dataframe: pd.DataFrame) -> None:
        df, u_labels, y_labels, fixture = sine_dataframe

        # Actua value
        name_ds = "my_dataset"
        ds = dmv.dataset.Dataset(
            name_ds,
            df,
            u_labels,
            y_labels,
            overlap=True,
            full_time_interval=True,
        )

        # A is definined for tricking flake8.
        # If you remove it the pipeline for windows will not build.
        A = ds.plot(return_figure=True)
        del A
        plt.close("all")

        ds.plot()
        plt.close("all")

        ds.plot(overlap=True)
        plt.close("all")

        ds.plot_coverage()
        plt.close("all")

        A = ds.plot_coverage(return_figure=True)
        del A
        plt.close("all")

        ds.plot_amplitude_spectrum()
        plt.close("all")

        A = ds.plot_amplitude_spectrum(return_figure=True, overlap=True)
        del A
        plt.close("all")


class Test_plot_Signal:
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

    def test_plot_raise_more(self, good_signals: list[Signal]) -> None:
        # Nominal data
        (
            signal_list,
            input_signal_names,
            output_signal_names,
            fixture,
        ) = good_signals

        # test input
        bad_input_signal_name = "potato"
        bad_output_signal_name = "potato"

        # Assert in error
        with pytest.raises(KeyError):
            dmv.plot_signals(
                signal_list, bad_input_signal_name, output_signal_names
            )

        # Assert in error
        with pytest.raises(KeyError):
            dmv.plot_signals(
                signal_list, input_signal_names, bad_output_signal_name
            )


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

    def test_input_or_output_not_found(
        self, good_dataframe: pd.DataFrame
    ) -> None:
        # Nominal values
        df, u_labels, y_labels, fixture = good_dataframe
        u_labels_test = u_labels
        y_labels_test = y_labels
        if fixture == "SISO":  # If SISO the names are obviously unique.
            u_labels_test = "potato"
        if fixture == "MISO" or fixture == "MIMO":
            u_labels_test[0] = "potato"
        if fixture == "SIMO" or fixture == "MIMO":
            y_labels_test[0] = "potato"
        with pytest.raises(ValueError):
            dmv.dataframe_validation(df, u_labels_test, y_labels)

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

    def test_wrong_sampling_period(self, good_signals: list[Signal]) -> None:
        # Nominal values
        signal_list, _, _, _ = good_signals

        with pytest.raises(ValueError):
            dmv.fix_sampling_periods(
                signal_list, target_sampling_period="potato"
            )

        with pytest.raises(ValueError):
            dmv.fix_sampling_periods(signal_list, target_sampling_period=-0.6)
