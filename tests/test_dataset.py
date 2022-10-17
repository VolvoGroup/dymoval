# -*- coding: utf-8 -*-


import dymoval as dmv
import numpy as np
from .fixture_data import *  # noqa
from dymoval.dataset import Signal
from dymoval import NUM_DECIMALS
from typing import Any
import random
import matplotlib
from matplotlib import pyplot as plt
import os


# import warnings
# show_kw = True
# # show_kw=False
#
# if show_kw:
#     curr_backend = get_backend()
#     # switch to non-Gui, preventing plots being displayed
#     plt.switch_backend("Agg")
#     # suppress UserWarning that agg cannot show plots
#     warnings.filterwarnings("ignore", "Matplotlib is currently using agg")
#


class Test_Dataset_nominal:
    def test_init(self, good_dataframe: pd.DataFrame) -> None:
        # Check if the passed dataset DataFrame is correctly stored as class attribute.
        # Nominal data
        df, u_labels, y_labels, fixture = good_dataframe

        # Expected value. Create a two-levels column from normal DataFrame
        df_expected = deepcopy(df)
        u_labels = dmv.str2list(u_labels)
        y_labels = dmv.str2list(y_labels)
        u_extended_labels = list(zip(["INPUT"] * len(u_labels), u_labels))
        y_extended_labels = list(zip(["OUTPUT"] * len(y_labels), y_labels))
        df_expected.columns = pd.MultiIndex.from_tuples(
            [*u_extended_labels, *y_extended_labels]
        )

        # Actual value. Pass the single-level DataFrame to the Dataset constructor.
        ds = dmv.dataset.Dataset(
            "my_dataset", df, u_labels, y_labels, full_time_interval=True
        )

        assert np.allclose(ds.dataset, df_expected)

    def test__validate_args_no_args(
        self,
        sine_dataframe: pd.DataFrame,
    ) -> None:
        # You pass a list of signal and the function recognizes who is input
        # and who is output

        # Arrange
        df, u_labels, y_labels, fixture = sine_dataframe
        ds = dmv.dataset.Dataset(
            "my_dataset", df, u_labels, y_labels, full_time_interval=True
        )

        expected_u_labels = (
            [u_labels] if isinstance(u_labels, str) else u_labels
        )
        expected_y_labels = (
            [y_labels] if isinstance(y_labels, str) else y_labels
        )
        # Act
        (
            actual_u_labels,
            actual_y_labels,
            actual_u_idx,
            actual_y_idx,
        ) = ds._validate_args()

        if fixture == "SISO":
            expected_u_idx = [0]
            expected_y_idx = [0]
        if fixture == "SIMO":
            expected_u_idx = [0]
            expected_y_idx = [0, 1, 2, 3]
        if fixture == "MISO":
            expected_u_idx = [0, 1, 2]
            expected_y_idx = [0]
        if fixture == "MIMO":
            expected_u_idx = [0, 1, 2]
            expected_y_idx = [0, 1, 2, 3]

        # Assert: if no arguments is passed it returns the whole list of
        # input and output signals
        assert expected_u_labels == actual_u_labels
        assert expected_u_idx == actual_u_idx

        assert expected_y_labels == actual_y_labels
        assert expected_y_idx == actual_y_idx

    @pytest.mark.parametrize(
        # Check if the function can recognize that there is no input or
        # no output
        "test_in, test_out, expected_in_idx, expected_out_idx",
        [
            (["u1", "u2"], ["y1", "y3"], [0, 1], [0, 2]),
            ([], ["y1", "y3"], [], [0, 2]),
            (["u1", "u2"], [], [0, 1], []),
        ],
    )
    def test__validate_args(
        self,
        sine_dataframe: pd.DataFrame,
        test_in: list[str],
        test_out: list[str],
        expected_in_idx: list[int],
        expected_out_idx: list[int],
    ) -> None:
        # You pass a list of signal and the function recognizes who is input
        # and who is output

        # Arrange
        df, u_labels, y_labels, fixture = sine_dataframe
        ds = dmv.dataset.Dataset(
            "my_dataset", df, u_labels, y_labels, full_time_interval=True
        )

        # Expected values
        if fixture == "SISO":
            test_in = [test_in[0]] if test_in else []
            expected_u_labels = [test_in[0]] if test_in else []
            expected_u_idx = [expected_in_idx[0]] if expected_in_idx else []

            test_out = [test_out[0]] if test_out else []
            expected_y_labels = [test_out[0]] if test_out else []
            expected_y_idx = [expected_out_idx[0]] if expected_out_idx else []

        if fixture == "SIMO":
            test_in = [test_in[0]] if test_in else []
            expected_u_labels = [test_in[0]] if test_in else []
            expected_u_idx = [expected_in_idx[0]] if expected_in_idx else []

            expected_y_labels = test_out if test_out else []
            expected_y_idx = expected_out_idx

        if fixture == "MISO":
            expected_u_labels = test_in if test_in else []
            expected_u_idx = expected_in_idx

            test_out = [test_out[0]] if test_out else []
            expected_y_labels = [test_out[0]] if test_out else []
            expected_y_idx = [expected_out_idx[0]] if expected_out_idx else []

        if fixture == "MIMO":
            expected_u_labels = test_in if test_in else []
            expected_u_idx = expected_in_idx

            expected_y_labels = test_out if test_out else []
            expected_y_idx = expected_out_idx

        test_signal_names = test_in + test_out

        # Acual values
        (
            actual_u_labels,
            actual_y_labels,
            actual_u_idx,
            actual_y_idx,
        ) = ds._validate_args(*test_signal_names)

        # Assert
        assert sorted(expected_u_labels) == sorted(actual_u_labels)
        assert sorted(expected_u_idx) == sorted(actual_u_idx)

        assert sorted(expected_y_labels) == sorted(actual_y_labels)
        assert sorted(expected_y_idx) == sorted(actual_y_idx)

    def test__validate_name_value_tuples(self) -> None:
        # Add a test only if you really need it.
        # This function is called so many times so it is implicitly tested
        # and writing tricky tests won't help
        pass

    def test_remove_means(self, sine_dataframe: pd.DataFrame) -> None:
        df, u_labels, y_labels, fixture = sine_dataframe

        # Expected value.
        # If you remove a mean from a signal, then the mean of the reminder
        # signal must be zero.

        # Actual value
        name_ds = "my_dataset"
        ds = dmv.dataset.Dataset(
            name_ds, df, u_labels, y_labels, full_time_interval=True
        )
        # You should get a dataframe with zero mean.
        # Stored dataframe shall be left unchanged.
        ds_expected = ds.remove_means()

        # Lets see if it is true (the mean of a signal with removed mean is 0.0)
        assert np.allclose(ds_expected.dataset.droplevel(0, axis=1).mean(), 0.0)

        # Lets check that the stored DataFrame has not been changed.
        df_actual = ds.dataset.droplevel(0, axis=1)
        assert np.allclose(df_actual.to_numpy(), df.to_numpy())

    def test_remove_offset(self, constant_ones_dataframe: pd.DataFrame) -> None:
        df, u_labels, y_labels, fixture = constant_ones_dataframe

        # Test values, i.e. offset to be removed from the specified signal.
        # The first value of the tuple indicates the signal name, whereas
        # the second value indicated the offset to be removed.
        # OBS! constant_ones_dataframe has 3 input and 3 output.

        # Arrange
        name_ds = "my_dataset"
        ds = dmv.dataset.Dataset(
            name_ds, df, u_labels, y_labels, full_time_interval=True
        )

        test_values = {
            "SISO": [("u1", 2.0), ("y1", 2.0)],
            "SIMO": [("u1", 2.0), ("y1", 2.0), ("y2", 1.0), ("y3", -2.0)],
            "MISO": [("u1", 2.0), ("u2", 2.0), ("u3", 2.0), ("y1", 2.0)],
            "MIMO": [
                ("u1", 2.0),
                ("u2", 2.0),
                ("u3", 2.0),
                ("y1", 2.0),
                ("y2", 1.0),
                ("y3", 2.0),
            ],
        }

        # Expected dataframe.
        N = len(df.index)
        idx = np.linspace(0, df.index[-1], N)
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

        # Act.
        ds_actual = ds.remove_offset(*test_values[fixture])

        # Assert
        assert np.allclose(ds_actual.dataset, df_expected)
        # Assert that the internally stored dataset is not overwritten
        assert np.allclose(ds.dataset, df)

    def test_remove_offset_only_input(
        self, constant_ones_dataframe: pd.DataFrame
    ) -> None:
        # It is the same as above.
        df, u_labels, y_labels, fixture = constant_ones_dataframe

        u_list = {
            "SISO": [("u1", 2.0)],
            "SIMO": [("u1", 2.0)],
            "MISO": [("u1", 2.0), ("u2", 2.0), ("u3", 2.0)],
            "MIMO": [("u1", 2.0), ("u2", 2.0), ("u3", 2.0)],
        }

        # Expected values. Compare with constant_ones_dataframe and the data above.
        N = len(df.index)
        idx = np.linspace(0, df.index[-1], N)
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
        # Function call
        ds_actual = ds.remove_offset(*u_list[fixture])

        # Assert
        assert np.allclose(ds_actual.dataset, df_expected)

        # Assert that the internally stored dataset is not overwritten
        assert np.allclose(ds.dataset, df)

    def test_remove_offset_only_output(
        self, constant_ones_dataframe: pd.DataFrame
    ) -> None:
        # It is the same as above.
        df, u_labels, y_labels, fixture = constant_ones_dataframe

        y_list = {
            "SISO": [("y1", 2.0)],
            "SIMO": [("y1", 2.0), ("y2", 1.0), ("y3", -2.0)],
            "MISO": [("y1", 2.0)],
            "MIMO": [("y1", 2.0), ("y2", 1.0), ("y3", 2.0)],
        }

        # Expected values. Compare with constant_ones_dataframe and the data above.
        N = len(df.index)
        idx = np.linspace(0, df.index[-1], N)
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
        # Function call
        ds_actual = ds.remove_offset(*y_list[fixture])

        # Assert that the offsets are removed
        assert np.allclose(ds_actual.dataset, df_expected)
        # Assert that the internally stored dataset is not overwritten
        assert np.allclose(ds.dataset, df)

    def test_get_dataset_values(self, sine_dataframe: pd.DataFrame) -> None:
        df, u_labels, y_labels, fixture = sine_dataframe

        # Instantiate dataset
        name_ds = "my_dataset"
        ds = dmv.dataset.Dataset(
            name_ds, df, u_labels, y_labels, full_time_interval=True
        )
        # Expected vectors
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

        # TODO: use a standard dataframe instead
        test_sampling_period = 0.1
        # print("signal list = ", signal_list)
        ds = dmv.Dataset(
            "my_dataset",
            signal_list,
            input_signal_names,
            output_signal_names,
            target_sampling_period=test_sampling_period,
            plot_raw=True,
            full_time_interval=True,
        )

        resampled_signals, _ = ds._fix_sampling_periods(
            signal_list, test_sampling_period
        )

        # Interpolate
        ds_actual = ds.replace_NaNs(method="interpolate")
        # Assert that no NaN:s left
        assert not ds_actual.dataset.isna().any().any()

        # Fill
        ds_actual = ds.replace_NaNs(method="fillna")
        # Assert that no NaN:s left
        assert not ds_actual.dataset.isna().any().any()

    def test_lowpass_filter(
        self,
        sine_dataframe: pd.DataFrame,
    ) -> None:
        # Check if the passed dataset DataFrame is correctly stored as class attribute.

        # Arrange
        df, u_labels, y_labels, fixture = sine_dataframe
        ds = dmv.dataset.Dataset(
            "my_dataset", df, u_labels, y_labels, tin=0.0, tout=1.0
        )

        # Filter cutoff frequency
        fc_u = 1
        fc_y = 1.5

        # Computed from Matlab based on the fixture.
        u_expected = np.array(
            [
                0,
                0.2000,
                0.5754,
                0.7841,
                0.7448,
                0.7580,
                0.9857,
                1.2125,
                1.2481,
                1.2195,
            ]
        )

        y_expected = np.array(
            [
                0,
                0.3000,
                0.7165,
                0.9599,
                1.0829,
                1.0056,
                1.2429,
                1.3964,
                1.6907,
                1.5223,
            ]
        )

        # Test
        ds = ds.low_pass_filter(("u1", fc_u), ("y1", fc_y))
        (t, u, y) = ds.get_dataset_values()
        if fixture == "SISO" or fixture == "SIMO":
            u_actual = u
        else:
            u_actual = u[:, 0]

        if fixture == "SISO" or fixture == "MISO":
            y_actual = y
        else:
            y_actual = y[:, 0]

        # Assert if ||y_act-y_exp||**2 < 0.1**2
        assert np.linalg.norm(u_actual - u_expected) ** 2 < 0.1**2
        assert np.linalg.norm(y_actual - y_expected) ** 2 < 0.1**2


class Test_Dataset_raise:
    def test__validate_args_raise(
        self,
        sine_dataframe: pd.DataFrame,
    ) -> None:
        # Check if the passed dataset DataFrame is correctly stored as class attribute.
        # Nominal data
        df, u_labels, y_labels, fixture = sine_dataframe
        ds = dmv.dataset.Dataset(
            "my_dataset", df, u_labels, y_labels, full_time_interval=True
        )

        # Test values
        u_name_test = "potato"
        y_name_test = "y1"

        with pytest.raises(KeyError):
            ds._validate_args(u_name_test, y_name_test)

    def test_replaceNaNs_raise(self, good_signals: list[Signal]) -> None:
        # Nominal data
        (
            signal_list,
            input_signal_names,
            output_signal_names,
            fixture,
        ) = good_signals

        test_sampling_period = 0.1
        # print("signal list = ", signal_list)
        ds = dmv.Dataset(
            "my_dataset",
            signal_list,
            input_signal_names,
            output_signal_names,
            target_sampling_period=test_sampling_period,
            plot_raw=True,
            full_time_interval=True,
        )

        # Raise unexistng replacement method
        with pytest.raises(ValueError):
            ds.replace_NaNs(method="potato")

    def test__validate_name_value_tuples_raise(self, good_signals: Any) -> None:
        signal_list, u_labels, y_labels, fixture = good_signals

        # Actual value
        ds = dmv.dataset.Dataset(
            "my_dataset",
            signal_list,
            u_labels,
            y_labels,
            overlap=True,
            full_time_interval=True,
            verbosity=1,
        )

        if fixture == "SISO" or "MISO":
            test_value = [("potato", 0.3)]
        else:
            test_value = [("u1", 0.3), ("y1", 2), ("y2", 0.0), ("potato", 1)]

        # Try to remove very weird signals.
        with pytest.raises(KeyError):
            ds._validate_name_value_tuples(*test_value)


class Test_Dataset_plots:

    # Use a non-interactive backend
    matplotlib.use("Agg")

    @pytest.mark.plot
    def test_plot_nominal(
        self, good_signals: list[Signal], tmp_path: str
    ) -> None:

        # You should just get a plot.
        signal_list, u_labels, y_labels, fixture = good_signals

        # Actual value
        ds = dmv.dataset.Dataset(
            "my_dataset",
            signal_list,
            u_labels,
            y_labels,
            overlap=True,
            full_time_interval=True,
            verbosity=1,
        )

        # Act
        # =============================
        # plot
        # =============================

        _ = ds.plot()
        plt.close("all")

        _ = ds.plot("u1")
        plt.close("all")

        if fixture == "MIMO":
            _ = ds.plot("u1", "u2", "y1", "y2")
            plt.close("all")

            _ = ds.plot("u1", "u2", "y1", "y2", overlap=True)
            plt.close("all")

            # Test duplicated input
            _ = ds.plot("u1", "u1", "y1", "y2", overlap=True)
            plt.close("all")

            # Test duplicated output
            _ = ds.plot("u1", "u1", "y1", "y1", overlap=True)
            plt.close("all")

        if fixture == "SIMO":
            _ = ds.plot("u1", "y1", "y2", overlap=True)
            plt.close("all")

        if fixture == "MISO":
            _ = ds.plot("u1", "u2", "y1", overlap=True)
            plt.close("all")

        if fixture == "SISO":
            _ = ds.plot("u1", "y1", overlap=True)
            plt.close("all")

        _ = ds.plot(overlap=True)
        plt.close("all")

        # save on disk
        tmp_path_str = str(tmp_path)
        filename = tmp_path_str + "/potato"
        _ = ds.plot(save_as=filename)
        assert os.path.exists(filename + ".png")
        plt.close("all")
        # =============================
        # plot_coverage
        # =============================
        _, _ = ds.plot_coverage()
        plt.close("all")

        _, _ = ds.plot_coverage("u1")
        plt.close("all")

        if fixture == "MIMO":
            _, _ = ds.plot_coverage("u1", "u2", "y1", "y2")
            plt.close("all")

            # Test duplicated input
            _, _ = ds.plot_coverage("u1", "u1", "y1", "y2")
            plt.close("all")

            # Test duplicated output
            _, _ = ds.plot_coverage("u1", "u2", "y1", "y1")
            plt.close("all")

        _ = ds.plot_coverage("y1")
        plt.close("all")

        # save on disk
        tmp_path_str = str(tmp_path)
        filename = tmp_path_str + "/potato"
        _, _ = ds.plot_coverage(save_as=filename)
        plt.close("all")
        assert os.path.exists(filename + "_in.png")
        assert os.path.exists(filename + "_out.png")

    @pytest.mark.plot
    def test_plot_spectrum(
        self,
        good_dataframe: pd.DataFrame,
        good_signals: list[Signal],
        tmp_path: str,
    ) -> None:
        # You should just get a plot.
        df, u_labels, y_labels, fixture = good_dataframe

        # Actua value
        name_ds = "my_dataset"
        ds = dmv.dataset.Dataset(
            name_ds,
            df,
            u_labels,
            y_labels,
            overlap=True,
            tin=0.0,
        )

        # This is only valid if ds does not contain NaN:s, i.e.
        # it is good_dataframe.
        _ = ds.plot_spectrum()
        plt.close("all")

        _ = ds.plot_spectrum("u1")
        plt.close("all")

        _ = ds.plot_spectrum("y1")
        plt.close("all")

        if fixture == "MIMO":
            _ = ds.plot_spectrum("u1", "u2", "y1", "y2", overlap=True)
            plt.close("all")

            # Duplicated input
            _ = ds.plot_spectrum("u1", "u1", "y1", "y1", overlap=True)
            plt.close("all")

            # Duplicated input and output
            _ = ds.plot_spectrum("u1", "u1", "y2", "y2", overlap=True)
            plt.close("all")

            # Duplicated input
            _ = ds.plot_spectrum(
                "u1", "u1", "y1", "y2", kind="amplitude", overlap=True
            )
            plt.close("all")

            # Duplicated input and output
            _ = ds.plot_spectrum(
                "u1", "u1", "y2", "y2", kind="amplitude", overlap=True
            )
            plt.close("all")

            # Duplicated output
            _ = ds.plot_spectrum("u1", "u1", "y2", "y2", overlap=True)
            plt.close("all")

        if fixture == "SIMO":
            _ = ds.plot_spectrum("u1", "y1", "y2", overlap=True)
            plt.close("all")

        if fixture == "MISO":
            _ = ds.plot_spectrum("u1", "u2", "y1", overlap=True)
            plt.close("all")

        if fixture == "SISO":
            _ = ds.plot_spectrum("u1", "y1", overlap=True)
            plt.close("all")

        _ = ds.plot_spectrum(overlap=True)
        plt.close("all")

        _ = ds.plot_spectrum(kind="amplitude")
        plt.close("all")

        _ = ds.plot_spectrum(kind="psd")
        plt.close("all")

        # save on disk
        tmp_path_str = str(tmp_path)
        filename = tmp_path_str + "/potato"
        _ = ds.plot_spectrum(save_as=filename)
        assert os.path.exists(filename + ".png")
        plt.close("all")

        # ======= If NaN:s raise =====================
        # good_signals have some NaN:s
        df, u_labels, y_labels, fixture = good_signals

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

        with pytest.raises(ValueError):
            _ = ds.plot_spectrum(overlap=True)

        with pytest.raises(ValueError):
            _ = ds.plot_spectrum(kind="potato")

    @pytest.mark.plot
    def test_plot_Signals(self, good_signals: list[Signal]) -> None:
        # You should just get a plot.
        signal_list, u_labels, y_labels, fixture = good_signals
        _ = dmv.plot_signals(*signal_list)
        plt.close("all")


class Test_Signal_validation:
    def test_name_unicity(self, good_signals: list[Signal]) -> None:
        # Nominal values
        signal_list, _, _, _ = good_signals

        signal_list[1]["name"] = signal_list[0]["name"]
        with pytest.raises(ValueError):
            dmv.validate_signals(*signal_list)

    def test_key_not_found(self, good_signals: list[Signal]) -> None:
        # Nominal values
        signal_list, _, _, _ = good_signals

        idx = random.randrange(0, len(signal_list))
        key = "name"
        signal_list[idx].pop(key)
        with pytest.raises(KeyError):
            dmv.validate_signals(*signal_list)

    def test_wrong_key(self, good_signals: list[Signal]) -> None:
        # Nominal values
        signal_list, _, _, _ = good_signals

        idx = random.randrange(0, len(signal_list))
        k_new = "potato"
        signal_list[idx][k_new] = signal_list[idx].pop("values")
        with pytest.raises(KeyError):
            dmv.validate_signals(*signal_list)

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
            dmv.validate_signals(*signal_list)

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
            dmv.validate_signals(*signal_list)


class Test_validate_dataframe:
    def test_there_is_at_least_one_in_and_one_out(
        self, good_dataframe: pd.DataFrame
    ) -> None:
        # Nominal values
        df, u_labels, y_labels, _ = good_dataframe
        u_labels = []
        with pytest.raises(IndexError):
            dmv.validate_dataframe(df, u_labels, y_labels)

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
            dmv.validate_dataframe(df, u_labels_test, y_labels)
        with pytest.raises(ValueError):
            dmv.validate_dataframe(df, u_labels, y_labels_test)
        with pytest.raises(ValueError):
            dmv.validate_dataframe(df, u_labels_test, y_labels_test)

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
            dmv.validate_dataframe(df, u_labels_test, y_labels)

    def test_dataframe_one_level_indices(
        self, good_dataframe: pd.DataFrame
    ) -> None:
        # Nominal values
        df, u_labels, y_labels, _ = good_dataframe
        df_test = df
        df_test.columns = pd.MultiIndex.from_product([df.columns, ["potato"]])
        with pytest.raises(IndexError):
            dmv.validate_dataframe(df_test, u_labels, y_labels)
        df_test = df
        df_test.index = pd.MultiIndex.from_product([df.index, ["potato"]])
        with pytest.raises(IndexError):
            dmv.validate_dataframe(df_test, u_labels, y_labels)

    def test_at_least_two_samples_per_signal(
        self, good_dataframe: pd.DataFrame
    ) -> None:
        # Nominal values
        df, u_labels, y_labels, _ = good_dataframe
        df_test = df.head(1)
        with pytest.raises(IndexError):
            dmv.validate_dataframe(df_test, u_labels, y_labels)

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
            dmv.validate_dataframe(df, u_labels, y_labels)

    def test_index_monotonicity(self, good_dataframe: pd.DataFrame) -> None:
        # Nominal values
        df, u_labels, y_labels, _ = good_dataframe
        df.index.values[0:2] = df.index[0]
        with pytest.raises(ValueError):
            dmv.validate_dataframe(df, u_labels, y_labels)

    def test_values_are_float(self, good_dataframe: pd.DataFrame) -> None:
        # Nominal values
        df, u_labels, y_labels, _ = good_dataframe
        df.iloc[0:1, 0:1] = "potato"
        with pytest.raises(TypeError):
            dmv.validate_dataframe(df, u_labels, y_labels)


class Test_fix_sampling_periods:
    def test_excluded_signals_no_args(self, good_signals: list[Signal]) -> None:
        # Nominal values
        (
            signal_list,
            input_signal_names,
            output_signal_names,
            fixture,
        ) = good_signals

        # Test data
        # If more than one input, then we exclude the second input,
        if fixture == "MISO" or fixture == "MIMO":
            signal_list[1]["sampling_period"] = 0.017
            expected_excluded = [signal_list[1]["name"]]

        # If more than one output, then we exclude the second output,
        if fixture == "SIMO" or fixture == "MIMO":
            signal_list[1]["sampling_period"] = 0.021
            expected_excluded = [signal_list[1]["name"]]

        # Build Dataset
        ds = dmv.Dataset(
            "my_dataset",
            signal_list,
            input_signal_names,
            output_signal_names,
            plot_raw=True,
            full_time_interval=True,
        )

        # IS SISO no exclusion. We need at least one IN and one OUT.
        if fixture == "SISO":
            expected_excluded = []

        expected_resampled = [
            s["name"] for s in signal_list if s["name"] not in expected_excluded
        ]

        # Assert. Check that all signals are either re-sampled or excluded.
        actual_excluded = ds.excluded_signals
        actual_resampled = list(ds.dataset.droplevel(level=0, axis=1))

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

        # If more than one input, then we exclude the second input,
        if fixture == "MISO" or fixture == "MIMO":
            signal_list[1]["sampling_period"] = 0.017
            expected_excluded = [signal_list[1]["name"]]

        # If more than one output, then we exclude the second output,
        if fixture == "SIMO" or fixture == "MIMO":
            signal_list[1]["sampling_period"] = 0.021
            expected_excluded = [signal_list[1]["name"]]

        # Test data.
        test_value = 0.2  # target_sampling_period

        # Build Dataset
        ds = dmv.Dataset(
            "my_dataset",
            signal_list,
            input_signal_names,
            output_signal_names,
            target_sampling_period=test_value,
            plot_raw=True,
            full_time_interval=True,
        )

        # IS SISO no exclusion. We need at least one IN and one OUT.
        if fixture == "SISO":
            expected_excluded = []

        expected_resampled = [
            s["name"] for s in signal_list if s["name"] not in expected_excluded
        ]

        # Assert. Check that all signals are either re-sampled or excluded.
        actual_excluded = ds.excluded_signals
        actual_resampled = list(ds.dataset.droplevel(level=0, axis=1))

        assert sorted(actual_excluded) == sorted(expected_excluded)
        assert sorted(actual_resampled) == sorted(expected_resampled)

    @pytest.mark.parametrize(
        "test_input, expected",
        [("potato", Exception), (-0.1, Exception)],
    )
    def test_excluded_signals_raise(
        self,
        good_signals: list[Signal],
        test_input: tuple[Any, Any],
        expected: tuple[Any, Any],
    ) -> None:
        # Nominal values
        signal_list, _, _, _ = good_signals
        with pytest.raises(expected):
            dmv.validate_signals(signal_list, test_input)

    def test_wrong_sampling_period(self, good_signals: list[Signal]) -> None:
        # Nominal values
        (
            signal_list,
            input_signal_names,
            output_signal_names,
            fixture,
        ) = good_signals

        with pytest.raises(ValueError):
            # Build Dataset
            _ = dmv.Dataset(
                "my_dataset",
                signal_list,
                input_signal_names,
                output_signal_names,
                target_sampling_period="potato",
                plot_raw=True,
                full_time_interval=True,
            )

        with pytest.raises(ValueError):
            # Build Dataset
            _ = dmv.Dataset(
                "my_dataset",
                signal_list,
                input_signal_names,
                output_signal_names,
                target_sampling_period=-0.8,
                plot_raw=True,
                full_time_interval=True,
            )
