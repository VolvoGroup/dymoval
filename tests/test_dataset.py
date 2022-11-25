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
        df, u_names, y_names, u_units, y_units, fixture = good_dataframe

        # Expected value. Create a two-levels column from normal DataFrame
        df_expected = deepcopy(df)
        u_names = dmv.str2list(u_names)
        y_names = dmv.str2list(y_names)
        u_extended_labels = list(zip(["INPUT"] * len(u_names), u_names))
        y_extended_labels = list(zip(["OUTPUT"] * len(y_names), y_names))
        df_expected.columns = pd.MultiIndex.from_tuples(
            [*u_extended_labels, *y_extended_labels]
        )

        # Actual value. Pass the single-level DataFrame to the Dataset constructor.
        ds = dmv.dataset.Dataset(
            "my_dataset", df, u_names, y_names, full_time_interval=True
        )

        assert np.allclose(ds.dataset, df_expected, atol=ATOL)

    def test__classify_signals_no_args(
        self,
        sine_dataframe: pd.DataFrame,
    ) -> None:
        # You pass a list of signal and the function recognizes who is input
        # and who is output

        # Arrange
        df, u_names, y_names, u_units, y_units, fixture = sine_dataframe
        ds = dmv.dataset.Dataset(
            "my_dataset", df, u_names, y_names, full_time_interval=True
        )

        expected_u_names = [u_names] if isinstance(u_names, str) else u_names
        expected_y_names = [y_names] if isinstance(y_names, str) else y_names
        expected_u_units = [u_units] if isinstance(u_units, str) else u_units
        expected_y_units = [y_units] if isinstance(y_units, str) else y_units
        # Act
        (
            u_dict,
            y_dict,
        ) = ds._classify_signals()

        actual_u_names = list(u_dict.keys())
        actual_y_names = list(y_dict.keys())

        actual_u_units = list(u_dict.values())
        actual_y_units = list(y_dict.values())

        # Assert: if no arguments is passed it returns the whole list of
        # input and output signals
        assert expected_u_names == actual_u_names
        assert expected_u_units == actual_u_units

        assert expected_y_names == actual_y_names
        assert expected_y_units == actual_y_units

    @pytest.mark.parametrize(
        # Check if the function can recognize that there is no input or
        # no output
        "names_in, names_out,units_in, units_out, expected_in_idx, expected_out_idx",
        [
            (
                ["u1", "u2"],
                ["y1", "y3"],
                ["kPa", "bar"],
                [
                    "deg",
                    "V",
                ],
                [0, 1],
                [0, 2],
            ),
            ([], ["y1", "y3"], [], ["deg", "V"], [], [0, 2]),
            (["u1", "u2"], [], ["kPa", "bar"], [], [0, 1], []),
        ],
    )
    def test__classify_signals(
        self,
        sine_dataframe: pd.DataFrame,
        names_in: list[str],
        names_out: list[str],
        units_in: list[str],
        units_out: list[str],
        expected_in_idx: list[int],
        expected_out_idx: list[int],
    ) -> None:
        # You pass a list of signal and the function recognizes who is input
        # and who is output and it also return the in-out indices

        # Arrange
        df, u_names, y_names, u_units, y_units, fixture = sine_dataframe
        ds = dmv.dataset.Dataset(
            "my_dataset", df, u_names, y_names, full_time_interval=True
        )

        # Expected values
        if fixture == "SISO":
            names_in = [names_in[0]] if names_in else []
            expected_u_names = [names_in[0]] if names_in else []
            expected_u_units = [units_in[0]] if units_in else []

            names_out = [names_out[0]] if names_out else []
            expected_y_names = [names_out[0]] if names_out else []
            expected_y_units = [units_out[0]] if units_out else []

        if fixture == "SIMO":
            names_in = [names_in[0]] if names_in else []
            expected_u_names = [names_in[0]] if names_in else []
            expected_u_units = [units_in[0]] if units_in else []

            expected_y_names = names_out if names_out else []
            expected_y_units = units_out if units_out else []

        if fixture == "MISO":
            expected_u_names = names_in if names_in else []
            expected_u_units = units_in if units_in else []

            names_out = [names_out[0]] if names_out else []
            expected_y_names = [names_out[0]] if names_out else []
            expected_y_units = [units_out[0]] if units_out else []

        if fixture == "MIMO":
            expected_u_names = names_in if names_in else []
            expected_u_units = units_in if units_in else []

            expected_y_names = names_out if names_out else []
            expected_y_units = units_out if units_out else []

        test_signal_names = names_in + names_out

        # Acual values
        (
            u_dict,
            y_dict,
        ) = ds._classify_signals(*test_signal_names)

        actual_u_names = list(u_dict.keys())
        actual_y_names = list(y_dict.keys())

        actual_u_units = list(u_dict.values())
        actual_y_units = list(y_dict.values())

        # Assert: if no arguments is passed it returns the whole list of
        # Assert
        assert sorted(expected_u_names) == sorted(actual_u_names)
        assert sorted(expected_u_units) == sorted(actual_u_units)

        assert sorted(expected_y_names) == sorted(actual_y_names)
        assert sorted(expected_y_units) == sorted(actual_y_units)

    def test__validate_name_value_tuples(self) -> None:
        # Add a test only if you really need it.
        # This function is called so many times so it is implicitly tested
        # and writing tricky tests won't help
        # Check also the coverage to be convinced of this.
        pass

    @pytest.mark.parametrize(
        "kind",
        [
            "INPUT",
            "OUTPUT",
        ],
    )
    def test_add_signal(
        self, kind: Signal_type, sine_dataframe: pd.DataFrame
    ) -> None:
        df, u_names, y_names, u_units, y_units, fixture = sine_dataframe
        # Test both add an input or an output

        # Expected value.
        # If you remove a mean from a signal, then the mean of the reminder
        # signal must be zero.
        N = 101  # as per fixture

        # Arrange
        name_ds = "my_dataset"
        ds = dmv.dataset.Dataset(
            name_ds, df, u_names, y_names, full_time_interval=True
        )
        # You should get a dataframe with zero mean.
        # Stored dataframe shall be left unchanged.

        test_signal1: dmv.Signal = {
            "name": "test1",
            "values": np.random.rand(120),
            "signal_unit": "m",
            "sampling_period": 0.1,
            "time_unit": "s",
        }

        nans = np.empty(50)
        nans[:] = np.nan

        test_signal2: dmv.Signal = {
            "name": "test2",
            "values": np.concatenate((np.random.rand(110), nans)),
            "signal_unit": "s",
            "sampling_period": 0.1,
            "time_unit": "s",
        }
        expected_values = np.hstack(
            (
                test_signal1["values"][:N].reshape(N, 1).round(NUM_DECIMALS),
                test_signal2["values"][:N].reshape(N, 1).round(NUM_DECIMALS),
                ds.dataset.loc[:, kind].head(N).to_numpy(),
            ),
        )

        # NaNs intervals.
        # We test only the keys but the values because they are hard to test
        # There are no nan intervals in the signals, hence we have an empty Index
        # TODO: test the values
        test_signal1_nan_interval = {
            test_signal1["name"]: pd.Index(data=[], dtype=float)
        }
        test_signal2_nan_interval = {
            test_signal2["name"]: pd.Index(data=[], dtype=float)
        }
        # Concatenate dict:s. OBS! dict update() is update(inplace = True)!!!
        # Big mismatch with pandas!
        expected_nans = ds._nan_intervals
        expected_nans.update(test_signal1_nan_interval)
        expected_nans.update(test_signal2_nan_interval)
        # expected_nans["test_signals2"] = np.linspace(0.5, 1.0, 6)

        # Overall list of signal names
        expected_labels = list(ds.dataset.loc[:, kind].columns) + [
            (test_signal1["name"], test_signal1["signal_unit"]),
            (test_signal2["name"], test_signal2["signal_unit"]),
        ]

        # Act
        if kind == "INPUT":
            ds = ds.add_input(test_signal1, test_signal2)
        elif kind == "OUTPUT":
            ds = ds.add_output(test_signal1, test_signal2)

        # Assert labels
        assert sorted(list(ds.dataset.loc[:, kind].columns)) == sorted(
            expected_labels
        )

        # Assert NaNs intevals.
        assert ds._nan_intervals.keys() == expected_nans.keys()

        # Testing the NaN:s values is a mess... we skip it.
        # Do it manually.
        #  for k, subset in ds._nan_intervals.items():
        #      for ii, _ in enumerate(subset):
        #          assert np.allclose(subset[ii], expected_nans[k][ii])

        # Assert values
        assert np.allclose(
            ds.dataset.loc[:, kind].to_numpy(),
            expected_values,
            atol=ATOL,
        )

        # Signal already exist
        with pytest.raises(KeyError):
            if kind == "INPUT":
                ds.add_input(test_signal1)
            elif kind == "OUTPUT":
                ds.add_output(test_signal1)

        # Signal that cannot be added
        # due to bad sampling period
        test_bad_signal: dmv.Signal = {
            "name": "bad_signal",
            "values": np.random.rand(120),
            "signal_unit": "m",
            "sampling_period": np.pi,
            "time_unit": "s",
        }

        with pytest.raises(Warning):
            if kind == "INPUT":
                ds.add_input(test_bad_signal)
            elif kind == "OUTPUT":
                ds.add_output(test_bad_signal)

        # TODO : You can either test Warning OR that the signal is not added.
        # ds = ds.add_input(test_bad_signal)
        # assert test_bad_signal["name"] in ds.excluded_signals

    def test_remove_signals(self, sine_dataframe: pd.DataFrame) -> None:
        df, u_names, y_names, u_units, y_units, fixture = sine_dataframe

        # Expected value.
        # If you remove a mean from a signal, then the mean of the reminder
        # signal must be zero.

        # Actual value
        name_ds = "my_dataset"
        ds = dmv.dataset.Dataset(
            name_ds, df, u_names, y_names, full_time_interval=True
        )
        # You should get a dataframe with removed signals
        # Stored dataframe shall be left unchanged.

        if fixture == "MIMO":
            ds_test = ds.remove_signals("u1", "y1")
            assert "u1" not in list(
                ds_test.dataset.droplevel(
                    level=["kind", "names"], axis=1
                ).columns
            )
            assert "y1" not in list(
                ds_test.dataset.droplevel(
                    level=["kind", "names"], axis=1
                ).columns
            )
        else:
            # Try to remove an input or an output to SISO, SIMO or MISO dataset
            with pytest.raises(KeyError):
                ds.remove_signals("u1", "y1")

        # Signal does not exist
        with pytest.raises(KeyError):
            ds.remove_signals("potato")

    def test_dump_to_signals(
        self, good_signals_no_nans: list[Signal], tmp_path: str
    ) -> None:

        # You should just get a plot.
        (
            signal_list,
            u_names,
            y_names,
            u_units,
            y_units,
            fixture,
        ) = good_signals_no_nans

        # Expected values
        expected_inputs = [s for s in signal_list if s["name"] in u_names]
        expected_outputs = [s for s in signal_list if s["name"] in y_names]

        # Actual value
        ds = dmv.dataset.Dataset(
            "my_dataset",
            signal_list,
            u_names,
            y_names,
            overlap=True,
            full_time_interval=True,
            verbosity=1,
        )

        # Act
        dumped_inputs = ds.dump_to_signals()["INPUT"]
        dumped_outputs = ds.dump_to_signals()["OUTPUT"]

        # Assert input
        for ii, val in enumerate(expected_inputs):
            assert expected_inputs[ii]["name"] == dumped_inputs[ii]["name"]
            assert np.allclose(
                expected_inputs[ii]["values"],
                dumped_inputs[ii]["values"],
                atol=ATOL,
            )
            assert (
                expected_inputs[ii]["signal_unit"]
                == dumped_inputs[ii]["signal_unit"]
            )
            assert np.isclose(
                expected_inputs[ii]["sampling_period"],
                dumped_inputs[ii]["sampling_period"],
                atol=ATOL,
            )
            assert (
                expected_inputs[ii]["time_unit"]
                == dumped_inputs[ii]["time_unit"]
            )

        # Assert output
        for ii, val in enumerate(expected_outputs):
            assert expected_outputs[ii]["name"] == dumped_outputs[ii]["name"]
            assert np.allclose(
                expected_outputs[ii]["values"],
                dumped_outputs[ii]["values"],
                atol=ATOL,
            )
            assert (
                expected_outputs[ii]["signal_unit"]
                == dumped_outputs[ii]["signal_unit"]
            )
            assert np.isclose(
                expected_outputs[ii]["sampling_period"],
                dumped_outputs[ii]["sampling_period"],
                atol=ATOL,
            )
            assert (
                expected_outputs[ii]["time_unit"]
                == dumped_outputs[ii]["time_unit"]
            )

    def test_remove_means(self, sine_dataframe: pd.DataFrame) -> None:
        df, u_names, y_names, _, _, fixture = sine_dataframe

        # Expected value.
        # If you remove a mean from a signal, then the mean of the reminder
        # signal must be zero.

        # Actual value
        name_ds = "my_dataset"
        ds = dmv.dataset.Dataset(
            name_ds, df, u_names, y_names, full_time_interval=True
        )
        # You should get a dataframe with zero mean.
        # Stored dataframe shall be left unchanged.
        ds_expected = ds.remove_means()

        # Lets see if it is true (the mean of a signal with removed mean is 0.0)
        assert np.allclose(ds_expected.dataset.mean(), 0.0, atol=ATOL)

        # Lets check that the stored DataFrame has not been changed.
        df_actual = ds.dataset
        assert np.allclose(df_actual.to_numpy(), df.to_numpy(), atol=ATOL)

        # Test with passed arguments
        ds_expected = ds.remove_means("u1", "y1")

        # Lets see if it is true (the mean of a signal with removed mean is 0.0)
        assert np.allclose(
            ds_expected.dataset.droplevel(level=["kind", "units" ""], axis=1)
            .loc[:, ["u1", "y1"]]
            .mean(),
            0.0,
            atol=ATOL,
        )

        # Lets check that the stored DataFrame has not been changed.
        df_actual = ds.dataset
        assert np.allclose(df_actual.to_numpy(), df.to_numpy(), atol=ATOL)

    def test_remove_offset(self, constant_ones_dataframe: pd.DataFrame) -> None:
        df, u_names, y_names, _, _, fixture = constant_ones_dataframe

        # Test values, i.e. offset to be removed from the specified signal.
        # The first value of the tuple indicates the signal name, whereas
        # the second value indicated the offset to be removed.
        # OBS! constant_ones_dataframe has 3 input and 3 output.

        # Arrange
        name_ds = "my_dataset"
        ds = dmv.dataset.Dataset(
            name_ds, df, u_names, y_names, full_time_interval=True
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
                index=idx, columns=[u_names, y_names], data=values
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
                index=idx, columns=[u_names, *y_names], data=values
            )
            df_expected.index.name = "Time (s)"
        if fixture == "MISO":
            values = -1.0 * np.ones((N, 4))  # Expected
            df_expected = pd.DataFrame(
                index=idx, columns=[*u_names, y_names], data=values
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
                index=idx, columns=[*u_names, *y_names], data=values
            )
            df_expected.index.name = "Time (s)"

        # Act.
        ds_actual = ds.remove_offset(*test_values[fixture])

        # Assert
        assert np.allclose(ds_actual.dataset, df_expected, atol=ATOL)
        # Assert that the internally stored dataset is not overwritten
        assert np.allclose(ds.dataset, df, atol=ATOL)

    def test_remove_offset_only_input(
        self, constant_ones_dataframe: pd.DataFrame
    ) -> None:
        # It is the same as above.
        df, u_names, y_names, _, _, fixture = constant_ones_dataframe

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
                index=idx, columns=[u_names, y_names], data=values
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
                index=idx, columns=[u_names, *y_names], data=values
            )
            df_expected.index.name = "Time (s)"
        if fixture == "MISO":
            values = np.hstack(
                (-1.0 * np.ones((N, 3)), np.ones((N, 1)))
            )  # Expected
            df_expected = pd.DataFrame(
                index=idx, columns=[*u_names, y_names], data=values
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
                index=idx, columns=[*u_names, *y_names], data=values
            )
            df_expected.index.name = "Time (s)"

        # Actual value
        name_ds = "my_dataset"
        ds = dmv.dataset.Dataset(
            name_ds, df, u_names, y_names, full_time_interval=True
        )
        # Function call
        ds_actual = ds.remove_offset(*u_list[fixture])

        # Assert
        assert np.allclose(ds_actual.dataset, df_expected, atol=ATOL)

        # Assert that the internally stored dataset is not overwritten
        assert np.allclose(ds.dataset, df, atol=ATOL)

    def test_remove_offset_only_output(
        self, constant_ones_dataframe: pd.DataFrame
    ) -> None:
        # It is the same as above.
        df, u_names, y_names, _, _, fixture = constant_ones_dataframe

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
                index=idx, columns=[u_names, y_names], data=values
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
                index=idx, columns=[u_names, *y_names], data=values
            )
            df_expected.index.name = "Time (s)"
        if fixture == "MISO":
            values = np.hstack(
                (np.ones((N, 3)), -1.0 * np.ones((N, 1)))
            )  # Expected
            df_expected = pd.DataFrame(
                index=idx, columns=[*u_names, y_names], data=values
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
                index=idx, columns=[*u_names, *y_names], data=values
            )
            df_expected.index.name = "Time (s)"

        # Actual value
        name_ds = "my_dataset"
        ds = dmv.dataset.Dataset(
            name_ds, df, u_names, y_names, full_time_interval=True
        )
        # Function call
        ds_actual = ds.remove_offset(*y_list[fixture])

        # Assert that the offsets are removed
        assert np.allclose(ds_actual.dataset, df_expected, atol=ATOL)
        # Assert that the internally stored dataset is not overwritten
        assert np.allclose(ds.dataset, df, atol=ATOL)

    def test_dataset_values(self, sine_dataframe: pd.DataFrame) -> None:
        df, u_names, y_names, u_units, y_units, fixture = sine_dataframe

        # Instantiate dataset
        name_ds = "my_dataset"
        ds = dmv.dataset.Dataset(
            name_ds, df, u_names, y_names, full_time_interval=True
        )

        if fixture == "SISO" or fixture == "SIMO":
            u_names = [u_names]
            u_units = [u_units]
        if fixture == "SISO" or fixture == "MISO":
            y_names = [y_names]
            y_units = [y_units]

        # Expected vectors
        t_expected = df.index.to_numpy().round(NUM_DECIMALS)
        u_expected = (
            df.loc[:, list(zip(u_names, u_units))]
            .to_numpy()
            .round(NUM_DECIMALS)
        )
        y_expected = (
            df.loc[:, list(zip(y_names, y_units))]
            .to_numpy()
            .round(NUM_DECIMALS)
        )

        if fixture == "SISO" or fixture == "SIMO":
            u_expected = u_expected.flatten()
        if fixture == "SISO" or fixture == "MISO":
            y_expected = y_expected.flatten()

        # Actuals
        t_actual, u_actual, y_actual = ds.dataset_values()

        # Assert
        assert np.allclose(t_expected, t_actual, atol=ATOL)
        assert np.allclose(u_expected, u_actual, atol=ATOL)
        assert np.allclose(y_expected, y_actual, atol=ATOL)

    def test_signal_list(self, sine_dataframe: pd.DataFrame) -> None:
        df, u_names, y_names, u_units, y_units, fixture = sine_dataframe

        # Instantiate dataset
        name_ds = "my_dataset"
        ds = dmv.dataset.Dataset(
            name_ds, df, u_names, y_names, full_time_interval=True
        )

        # Expected based on fixture
        expected = {
            "SISO": [("INPUT", "u1", "kPa"), ("OUTPUT", "y1", "deg")],
            "SIMO": [
                ("INPUT", "u1", "kPa"),
                ("OUTPUT", "y1", "deg"),
                ("OUTPUT", "y2", "rad/s"),
                ("OUTPUT", "y3", "V"),
                ("OUTPUT", "y4", "A"),
            ],
            "MISO": [
                ("INPUT", "u1", "kPa"),
                ("INPUT", "u2", "bar"),
                ("INPUT", "u3", "deg"),
                ("OUTPUT", "y1", "deg"),
            ],
            "MIMO": [
                ("INPUT", "u1", "kPa"),
                ("INPUT", "u2", "bar"),
                ("INPUT", "u3", "deg"),
                ("OUTPUT", "y1", "deg"),
                ("OUTPUT", "y2", "rad/s"),
                ("OUTPUT", "y3", "V"),
                ("OUTPUT", "y4", "A"),
            ],
        }

        actual = ds.signal_list()
        assert expected[fixture] == actual

    def test_lowpass_filter(
        self,
        sine_dataframe: pd.DataFrame,
    ) -> None:
        # Check if the passed dataset DataFrame is correctly stored as class attribute.

        # Arrange
        df, u_names, y_names, u_units, y_units, fixture = sine_dataframe
        ds = dmv.dataset.Dataset(
            "my_dataset", df, u_names, y_names, tin=0.0, tout=1.0
        )

        # Filter cutoff frequency
        fc_u = 1
        fc_y = 1.5

        u_expected = np.array(
            [
                2.0,
                2.0,
                2.1949,
                2.2467,
                2.065,
                1.9386,
                2.0398,
                2.1678,
                2.1193,
                2.004,
            ]
        )

        y_expected = np.array(
            [
                2.0,
                2.0,
                2.1615,
                2.1881,
                2.1269,
                1.893,
                1.9972,
                2.0376,
                2.2357,
                1.9855,
            ]
        )

        # Test
        ds = ds.low_pass_filter(("u1", fc_u), ("y1", fc_y))
        (t, u, y) = ds.dataset_values()
        if fixture == "SISO" or fixture == "SIMO":
            u_actual = u
        else:
            u_actual = u[:, 0]

        if fixture == "SISO" or fixture == "MISO":
            y_actual = y
        else:
            y_actual = y[:, 0]

        # Assert if ||y_act-y_exp||**2 < 0.1**2
        # Values computed only for the first 10 samples
        assert np.linalg.norm(u_actual[:10] - u_expected[:10]) ** 2 < 0.1**2
        assert np.linalg.norm(y_actual[:10] - y_expected[:10]) ** 2 < 0.1**2

        with pytest.raises(ValueError):
            ds = ds.low_pass_filter(("u1", -0.3))

        with pytest.raises(ValueError):
            ds = ds.low_pass_filter(("y1", -0.3))

    def test_apply(self, constant_ones_dataframe: pd.DataFrame) -> None:
        df, u_names, y_names, _, _, fixture = constant_ones_dataframe

        # Test values, i.e. offset to be removed from the specified signal.
        # The first value of the tuple indicates the signal name, whereas
        # the second value indicated the offset to be removed.
        # OBS! constant_ones_dataframe has 3 input and 3 output.

        # Arrange
        name_ds = "my_dataset"
        ds = dmv.dataset.Dataset(
            name_ds, df, u_names, y_names, full_time_interval=True
        )

        def plus_one(x: float) -> float:
            return x + 1

        def plus_five(x: float) -> float:
            return x + 5

        # from fixture u_units = ["m", "m/s", "bar"]
        # from fixture y_units = ["deg", "m/s**2", "V"]
        test_values = {
            "SISO": [("u1", plus_one, "a")],
            "SIMO": [
                ("u1", plus_one, "a"),
                ("y1", plus_five, "a"),
                ("y3", plus_five, "c"),
            ],
            "MISO": [
                ("u2", plus_one, "b"),
                ("u3", plus_five, "a"),
                ("y1", plus_one, "a"),
            ],
            "MIMO": [
                ("u1", plus_five, "c"),
                ("u3", plus_one, "m/s"),
                ("y2", plus_five, "a"),
                ("y3", plus_five, "b"),
            ],
        }

        # Expected dataframe.
        N = len(df.index)
        idx = np.linspace(0, df.index[-1], N)
        if fixture == "SISO":
            values = np.vstack(
                (2.0 * np.ones(N), np.ones(N))  # Expected  # Expected
            )
            df_expected = pd.DataFrame(
                index=idx, columns=[u_names, y_names], data=values.T
            )
            df_expected.index.name = "Time (s)"
            expected_units = ["a", "deg"]
        if fixture == "SIMO":
            values = np.vstack(
                (
                    2.0 * np.ones(N),
                    6.0 * np.ones(N),
                    np.ones(N),
                    6.0 * np.ones(N),
                )
            ).transpose()  # Expected
            df_expected = pd.DataFrame(
                index=idx, columns=[u_names, *y_names], data=values
            )
            df_expected.index.name = "Time (s)"
            expected_units = ["a", "a", "m/s**2", "c"]
        if fixture == "MISO":
            values = np.vstack(
                (
                    np.ones(N),
                    2.0 * np.ones(N),
                    6.0 * np.ones(N),
                    2.0 * np.ones(N),
                )
            ).transpose()  # Expected

            df_expected = pd.DataFrame(
                index=idx, columns=[*u_names, y_names], data=values
            )
            df_expected.index.name = "Time (s)"
            expected_units = ["m", "b", "a", "a"]
        if fixture == "MIMO":
            values = np.vstack(
                (
                    6.0 * np.ones(N),  # u1
                    np.ones(N),  # u2
                    2.0 * np.ones(N),  # u3
                    np.ones(N),  # y1
                    6.0 * np.ones(N),  # y2
                    6.0 * np.ones(N),  # y3
                )
            ).transpose()  # Expected
            df_expected = pd.DataFrame(
                index=idx, columns=[*u_names, *y_names], data=values
            )
            df_expected.index.name = "Time (s)"
            expected_units = ["c", "m/s", "m/s", "deg", "a", "b"]

        # Act.
        ds_actual = ds.apply(*test_values[fixture])
        print("actual_units", ds_actual.dataset.columns)
        actual_units = list(ds_actual.dataset.columns.get_level_values("units"))

        # Assert
        assert np.allclose(ds_actual.dataset, df_expected, atol=ATOL)
        assert actual_units == expected_units

        # Assert that the internally stored dataset is not overwritten
        assert np.allclose(ds.dataset, df, atol=ATOL)


class Test_Dataset_raise:
    def test__classify_signals_raise(
        self,
        sine_dataframe: pd.DataFrame,
    ) -> None:
        # Check if the passed dataset DataFrame is correctly stored as class attribute.
        # Nominal data
        df, u_names, y_names, u_units, y_units, fixture = sine_dataframe
        ds = dmv.dataset.Dataset(
            "my_dataset", df, u_names, y_names, full_time_interval=True
        )

        # Test values
        u_name_test = "potato"
        y_name_test = "y1"

        with pytest.raises(KeyError):
            ds._classify_signals(u_name_test, y_name_test)


#     THIS WON'T RAISE ANYTHING ANY LONGER'
#     def test__validate_name_value_tuples_raise(self, good_signals: Any) -> None:
#         signal_list, u_names, y_names, u_units, y_units, fixture = good_signals
#
#         # Actual value
#         ds = dmv.dataset.Dataset(
#             "my_dataset",
#             signal_list,
#             u_names,
#             y_names,
#             overlap=True,
#             full_time_interval=True,
#             verbosity=1,
#         )
#
#         if fixture == "SISO" or "MISO":
#             test_value = [("potato", 0.3)]
#         else:
#             test_value = [("u1", 0.3), ("y1", 2), ("y2", 0.0), ("potato", 1)]
#
#         # Try to remove very weird signals.
#         with pytest.raises(KeyError):
#             ds._validate_name_value_tuples(*test_value)
#


class Test_Dataset_plots:

    # Use a non-interactive backend
    matplotlib.use("Agg")

    @pytest.mark.plots
    def test_plot_nominal(
        self, good_signals: list[Signal], tmp_path: str
    ) -> None:

        # You should just get a plot.
        signal_list, u_names, y_names, u_units, y_units, fixture = good_signals

        # Actual value
        ds = dmv.dataset.Dataset(
            "my_dataset",
            signal_list,
            u_names,
            y_names,
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

            _ = ds.plot(("u1", "u2"), "y1", "y2")
            plt.close("all")

            _ = ds.plot("u1", ("u2", "y1"), "y2")
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
        _ = ds.plot(save_as=filename, layout="tight")
        assert os.path.exists(filename + ".png")
        plt.close("all")
        # =============================
        # plot_coverage
        # =============================
        _ = ds.plot_coverage()
        plt.close("all")

        _ = ds.plot_coverage("u1")
        plt.close("all")

        if fixture == "MIMO":
            _ = ds.plot_coverage("u1", "u2", "y1", "y2")
            plt.close("all")

            # Test duplicated input
            _ = ds.plot_coverage("u1", "u1", "y1", "y2")
            plt.close("all")

            # Test duplicated output
            _ = ds.plot_coverage("u1", "u2", "y1", "y1")
            plt.close("all")

        _ = ds.plot_coverage("y1")
        plt.close("all")

        with pytest.raises(TypeError):
            _ = ds.plot_coverage(("y1", "u1"))

        # save on disk
        tmp_path_str = str(tmp_path)
        filename = tmp_path_str + "/potato"
        _ = ds.plot_coverage(save_as=filename, layout="tight")
        plt.close("all")
        assert os.path.exists(filename + ".png")

    @pytest.mark.plots
    def test_plotxy(
        self,
        good_dataframe: pd.DataFrame,
        good_signals: list[Signal],
        tmp_path: str,
    ) -> None:
        # You should just get a plot.
        df, u_names, y_names, u_units, y_units, fixture = good_dataframe

        # Actua value
        name_ds = "my_dataset"
        ds = dmv.dataset.Dataset(
            name_ds,
            df,
            u_names,
            y_names,
            overlap=True,
            tin=0.0,
        )

        # This is only valid if ds does not contain NaN:s, i.e.
        # it is good_dataframe.
        _ = ds.plotxy(("u1", "y1"))
        plt.close("all")

        if fixture == "MIMO":
            _ = ds.plotxy(("u1", "y1"), ("u2", "y2"))

        with pytest.raises(ValueError):
            _ = ds.plotxy(("potato", "u1"))

        # save on disk
        tmp_path_str = str(tmp_path)
        filename = tmp_path_str + "/potato"
        _ = ds.plotxy(save_as=filename, layout="constrained")
        plt.close("all")
        assert os.path.exists(filename + ".png")

    @pytest.mark.plots
    def test_plot_spectrum(
        self,
        good_dataframe: pd.DataFrame,
        good_signals: list[Signal],
        tmp_path: str,
    ) -> None:
        # You should just get a plot.
        df, u_names, y_names, u_units, y_units, fixture = good_dataframe

        # Actua value
        name_ds = "my_dataset"
        ds = dmv.dataset.Dataset(
            name_ds,
            df,
            u_names,
            y_names,
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
            _ = ds.plot_spectrum("u1", "u2", "y1", "y2")
            plt.close("all")

            _ = ds.plot_spectrum("u1", ("u2", "y1"), "y2")
            plt.close("all")

            _ = ds.plot_spectrum("u1", "u2", "y1", "y2", kind="amplitude")
            plt.close("all")

            _ = ds.plot_spectrum(("u1", "u2"), ("y1", "y2"), kind="psd")
            plt.close("all")

            # Duplicated input
            _ = ds.plot_spectrum("u1", "u1", "y1", "y1")
            plt.close("all")

            # Duplicated input and output
            _ = ds.plot_spectrum("u1", "u1", "y2", "y2")
            plt.close("all")

            # Duplicated input
            _ = ds.plot_spectrum("u1", "u1", "y1", "y2", kind="amplitude")
            plt.close("all")

            # Duplicated input and output
            _ = ds.plot_spectrum("u1", "u1", "y2", "y2", kind="amplitude")
            plt.close("all")

            # Duplicated output
            _ = ds.plot_spectrum("u1", "u1", "y2", "y2")
            plt.close("all")

        if fixture == "SIMO":
            _ = ds.plot_spectrum("u1", "y1", "y2")
            plt.close("all")

        if fixture == "MISO":
            _ = ds.plot_spectrum("u1", "u2", "y1")
            plt.close("all")

        if fixture == "SISO":
            _ = ds.plot_spectrum("u1", "y1")
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
        df, u_names, y_names, u_units, y_units, fixture = good_signals

        # Actua value
        name_ds = "my_dataset"
        ds = dmv.dataset.Dataset(
            name_ds,
            df,
            u_names,
            y_names,
            overlap=True,
            full_time_interval=True,
        )

        with pytest.raises(ValueError):
            _ = ds.plot_spectrum(overlap=True)

        with pytest.raises(ValueError):
            _ = ds.plot_spectrum(kind="potato")

    @pytest.mark.plots
    def test_plot_Signals(self, good_signals: list[Signal]) -> None:
        # You should just get a plot.
        signal_list, u_names, y_names, u_units, y_units, fixture = good_signals
        _ = dmv.plot_signals(*signal_list)
        plt.close("all")

    @pytest.mark.plots
    def test_compare_datasets(
        self,
        good_dataframe: pd.DataFrame,
        tmp_path: str,
    ) -> None:
        # You should just get a plot.
        df, u_names, y_names, u_units, y_units, fixture = good_dataframe

        # Actua value
        name_ds = "my_dataset"
        ds = dmv.dataset.Dataset(
            name_ds,
            df,
            u_names,
            y_names,
            overlap=True,
            tin=0.0,
        )

        ds1 = ds.remove_means()
        ds2 = ds.remove_offset(("u1", 1), ("y1", -0.5))

        # This is only valid if ds does not contain NaN:s, i.e.
        # it is good_dataframe.
        print("ds.dataset = ", ds.dataset)
        _ = dmv.compare_datasets(ds, ds1, ds2)
        plt.close("all")

        _ = dmv.compare_datasets(ds, ds1, ds2, kind="power")
        plt.close("all")

        _ = dmv.compare_datasets(ds, ds1, ds2, kind="amplitude")
        plt.close("all")

        _ = dmv.compare_datasets(ds, ds1, ds2, kind="psd")
        plt.close("all")

        _ = dmv.compare_datasets(ds, ds1, ds2, kind="coverage")
        plt.close("all")

        with pytest.raises(TypeError):
            dmv.compare_datasets(ds, "potato")

        # save on disk
        tmp_path_str = str(tmp_path)
        filename = tmp_path_str + "/potato"
        _ = dmv.compare_datasets(ds, ds1, ds2, save_as=filename, layout="tight")
        assert os.path.exists(filename + ".png")
        plt.close("all")


class Test_Signal_validation:
    def test_key_not_found(self, good_signals: list[Signal]) -> None:
        # Nominal values
        signal_list, _, _, _, _, _ = good_signals

        idx = random.randrange(0, len(signal_list))
        key = "name"
        signal_list[idx].pop(key)
        with pytest.raises(KeyError):
            dmv.validate_signals(*signal_list)

    def test_name_unicity(self, good_signals: list[Signal]) -> None:
        # Nominal values
        signal_list, _, _, _, _, _ = good_signals

        signal_list[1]["name"] = signal_list[0]["name"]
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
        signal_list, _, _, _, _, _ = good_signals

        idx = random.randrange(0, len(signal_list))
        signal_list[idx]["values"] = test_input
        with pytest.raises(expected):
            dmv.validate_signals(*signal_list)

    def test_different_time_units(self, good_signals: list[Signal]) -> None:
        # Nominal values
        signal_list, _, _, _, _, _ = good_signals

        signal_list[0]["time_unit"] = "hourszs"
        with pytest.raises(ValueError):
            dmv.validate_signals(*signal_list)

    def test_wrong_key(self, good_signals: list[Signal]) -> None:
        # Nominal values
        signal_list, _, _, _, _, _ = good_signals

        idx = random.randrange(0, len(signal_list))
        k_new = "potato"
        signal_list[idx][k_new] = signal_list[idx].pop("values")
        with pytest.raises(KeyError):
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
        signal_list, _, _, _, _, _ = good_signals

        idx = random.randrange(0, len(signal_list))
        signal_list[idx]["sampling_period"] = test_input
        with pytest.raises(expected):
            dmv.validate_signals(*signal_list)


class Test_validate_dataframe:
    def test_there_is_at_least_one_in_and_one_out(
        self, good_dataframe: pd.DataFrame
    ) -> None:
        # Check u_names, y_names arguments passed
        df, u_names, y_names, u_units, y_units, _ = good_dataframe
        u_names = []
        with pytest.raises(IndexError):
            dmv.validate_dataframe(df, u_names, y_names)

    def test_name_unicity(self, good_dataframe: pd.DataFrame) -> None:
        # Check u_names, y_names arguments passed
        df, u_names, y_names, u_units, y_units, fixture = good_dataframe
        u_names_test = u_names
        y_names_test = y_names
        if fixture == "SISO":  # If SISO the names are obviously unique.
            u_names = y_names
            u_names_test = y_names
        if fixture == "MISO" or fixture == "MIMO":
            u_names_test[-1] = u_names_test[-2]
        if fixture == "SIMO" or fixture == "MIMO":
            y_names_test[0] = y_names_test[1]
        with pytest.raises(ValueError):
            dmv.validate_dataframe(df, u_names_test, y_names)
        with pytest.raises(ValueError):
            dmv.validate_dataframe(df, u_names, y_names_test)
        with pytest.raises(ValueError):
            dmv.validate_dataframe(df, u_names_test, y_names_test)

    def test_input_or_output_not_found(
        self, good_dataframe: pd.DataFrame
    ) -> None:
        # Check u_names, y_names arguments passed
        df, u_names, y_names, u_units, y_units, fixture = good_dataframe
        u_names_test = u_names
        y_names_test = y_names
        if fixture == "SISO":  # If SISO the names are obviously unique.
            u_names_test = "potato"
        if fixture == "MISO" or fixture == "MIMO":
            u_names_test[0] = "potato"
        if fixture == "SIMO" or fixture == "MIMO":
            y_names_test[0] = "potato"
        with pytest.raises(ValueError):
            dmv.validate_dataframe(df, u_names_test, y_names)

    def test_indices_are_tuples_of_str(
        self, good_dataframe: pd.DataFrame
    ) -> None:
        # Nominal values
        # Check if you have any MultiIndex
        df, u_names, y_names, u_units, y_units, _ = good_dataframe

        # Check if columns name type is tuples
        df_test = df.rename(columns={("u1", "kPa"): "potato"})
        with pytest.raises(TypeError):
            dmv.validate_dataframe(df_test, u_names, y_names)

        # Check if index name type is tuples
        df_test.index.name = "potato"
        with pytest.raises(TypeError):
            dmv.validate_dataframe(df_test, u_names, y_names)

        # Check if tuples elements are str
        df_test = df.rename(columns={("u1", "kPa"): (3, "kPa")})
        with pytest.raises(TypeError):
            dmv.validate_dataframe(df_test, u_names, y_names)

        df_test = df.rename(columns={("u1", "kPa"): ("u1", 9)})
        with pytest.raises(TypeError):
            dmv.validate_dataframe(df_test, u_names, y_names)

        df_test.index.name = "potato"
        with pytest.raises(TypeError):
            dmv.validate_dataframe(df_test, u_names, y_names)

    #       This is not needed because in case of multi-index is automatically
    #       covered by the previous test
    #     def test_dataframe_one_level_indices(
    #         self, good_dataframe: pd.DataFrame
    #     ) -> None:
    #         # Nominal values
    #         # Check if you have any MultiIndex
    #         df, u_names, y_names, u_units, y_units, _ = good_dataframe
    #         df_test = df
    #         df_test.columns = pd.MultiIndex.from_product([["potato"], df.columns])
    #         print("df_test = ", df_test.columns)
    #         with pytest.raises(IndexError):
    #             dmv.validate_dataframe(df_test, u_names, y_names)
    #         df_test = df
    #         df_test.index = pd.MultiIndex.from_product([["potato"], df.index])
    #         with pytest.raises(IndexError):
    #             dmv.validate_dataframe(df_test, u_names, y_names)
    #
    def test_at_least_two_samples_per_signal(
        self, good_dataframe: pd.DataFrame
    ) -> None:
        # Nominal values
        df, u_names, y_names, u_units, y_units, _ = good_dataframe
        df_test = df.head(1)
        with pytest.raises(IndexError):
            dmv.validate_dataframe(df_test, u_names, y_names)

    def test_labels_exist_in_dataframe(
        self, good_dataframe: pd.DataFrame
    ) -> None:
        # Nominal values
        df, u_names, y_names, u_units, y_units, fixture = good_dataframe
        if fixture == "SISO":
            u_names = [u_names]
            y_names = [y_names]
        if fixture == "MISO":
            y_names = [y_names]
        if fixture == "SIMO":
            u_names = [u_names]
        u_names[-1] = "potato"
        with pytest.raises(ValueError):
            dmv.validate_dataframe(df, u_names, y_names)

    def test_index_monotonicity(self, good_dataframe: pd.DataFrame) -> None:
        # Nominal values
        df, u_names, y_names, u_units, y_units, _ = good_dataframe
        df.index.values[0:2] = df.index[0]
        with pytest.raises(ValueError):
            dmv.validate_dataframe(df, u_names, y_names)

    def test_values_are_float(self, good_dataframe: pd.DataFrame) -> None:
        # Nominal values
        df, u_names, y_names, u_units, y_units, _ = good_dataframe
        df.iloc[0:1, 0:1] = "potato"
        with pytest.raises(TypeError):
            dmv.validate_dataframe(df, u_names, y_names)


class Test_fix_sampling_periods:
    def test_excluded_signals_no_args(self, good_signals: list[Signal]) -> None:
        # Nominal values
        (
            signal_list,
            input_signal_names,
            output_signal_names,
            input_signal_units,
            output_signal_units,
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
        actual_resampled = list(
            ds.dataset.droplevel(level=["kind", "units"], axis=1)
        )

        assert sorted(actual_excluded) == sorted(expected_excluded)
        assert sorted(actual_resampled) == sorted(expected_resampled)

    def test_excluded_signals(self, good_signals: list[Signal]) -> None:
        # Nominal values
        (
            signal_list,
            input_signal_names,
            output_signal_names,
            input_signal_units,
            output_signal_units,
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
        actual_resampled = list(
            ds.dataset.droplevel(level=["kind", "units"], axis=1)
        )

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
        signal_list, _, _, _, _, _ = good_signals
        with pytest.raises(expected):
            dmv.validate_signals(signal_list, test_input)

    def test_wrong_sampling_period(self, good_signals: list[Signal]) -> None:
        # Nominal values
        (
            signal_list,
            input_signal_names,
            output_signal_names,
            _,
            _,
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
                full_time_interval=True,
            )
