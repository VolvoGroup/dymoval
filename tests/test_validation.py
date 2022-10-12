import pytest
import dymoval as dmv
from dymoval.validation import XCorrelation
import numpy as np
from matplotlib import pyplot as plt
from .fixture_data import *  # noqa
import scipy.signal as signal
import os


class Test_ClassValidationNominal:
    def test_init(self, good_dataframe: pd.DataFrame) -> None:
        # Nominal data
        df, u_labels, y_labels, fixture = good_dataframe
        name_ds = "my_dataset"
        ds = dmv.dataset.Dataset(
            name_ds, df, u_labels, y_labels, full_time_interval=True
        )
        name_vs = "my_validation"
        vs = dmv.ValidationSession(name_vs, ds)

        # Check that the passed Dataset is correctly stored.
        # Main DataFrame
        assert all(vs.Dataset.dataset == ds.dataset)

        for ii in range(4):  # Size of coverage
            assert all(vs.Dataset.coverage[ii] == ds.coverage[ii])

    def test_random_walk(self, good_dataframe: pd.DataFrame) -> None:
        df, u_labels, y_labels, fixture = good_dataframe
        name_ds = "my_dataset"
        ds = dmv.dataset.Dataset(
            name_ds, df, u_labels, y_labels, full_time_interval=True
        )

        name_vs = "my_validation"
        vs = dmv.ValidationSession(name_vs, ds)

        # ==================================
        # Append simulations test
        # ==================================

        # Add one model
        sim1_name = "Model 1"
        sim1_labels = ["my_y1", "my_y2"]  # The fixture has two outputs
        if fixture == "SISO" or fixture == "MISO":
            sim1_labels = [sim1_labels[0]]
        sim1_values = np.random.rand(
            len(df.iloc[:, 0].values), len(sim1_labels)
        )

        vs.append_simulation(sim1_name, sim1_labels, sim1_values)
        # At least the names are there...
        assert sim1_name in vs.simulations_results.columns
        assert sim1_name in vs.auto_correlation.keys()
        assert sim1_name in vs.cross_correlation.keys()
        assert sim1_name in vs.validation_results.columns

        assert np.allclose(sim1_values, vs.simulations_results[sim1_name])

        # # Add second model
        sim2_name = "Model 2"
        sim2_labels = ["my_y1", "my_y2"]  # The fixture has two outputs
        if fixture == "SISO" or fixture == "MISO":
            # You only have one output
            sim2_labels = [sim1_labels[0]]
        sim2_values = vs.Dataset.dataset["OUTPUT"].values + np.random.rand(
            len(vs.Dataset.dataset["OUTPUT"].values), 1
        )

        vs.append_simulation(sim2_name, sim2_labels, sim2_values)
        # At least the names are there...
        assert sim2_name in vs.simulations_results.columns
        assert sim2_name in vs.auto_correlation.keys()
        assert sim2_name in vs.cross_correlation.keys()
        assert sim2_name in vs.validation_results.columns

        assert np.allclose(sim2_values, vs.simulations_results[sim2_name])

        # ===============================================
        # Test get_simulation_signals_list and list_simulations
        # ============================================
        expected_sims = [sim1_name, sim2_name]

        assert sorted(expected_sims) == sorted(vs.get_simulations_name())

        expected_signals1 = sim1_labels
        expected_signals2 = sim2_labels

        assert sorted(expected_signals1) == sorted(
            vs.get_simulation_signals_list(sim1_name)
        )
        assert sorted(expected_signals2) == sorted(
            vs.get_simulation_signals_list(sim2_name)
        )

        # ==================================
        # drop_simulation sim
        # ==================================
        vs.drop_simulation(sim1_name)
        # At least the names are nt there any longer.
        assert sim1_name not in vs.simulations_results.columns
        assert sim1_name not in vs.auto_correlation.keys()
        assert sim1_name not in vs.cross_correlation.keys()
        assert sim1_name not in vs.validation_results.columns

        # ============================================
        # Re-add sim and then clear.
        # ============================================
        vs.append_simulation(sim1_name, sim1_labels, sim1_values)

        vs.clear()

        assert [] == list(vs.simulations_results.columns)
        assert [] == list(vs.auto_correlation.keys())
        assert [] == list(vs.cross_correlation.keys())
        assert [] == list(vs.validation_results.columns)

    def test_get_sim_signal_list_and_metrics_raise(
        self, good_dataframe: pd.DataFrame
    ) -> None:
        df, u_labels, y_labels, fixture = good_dataframe
        name_ds = "my_dataset"
        ds = dmv.dataset.Dataset(
            name_ds, df, u_labels, y_labels, full_time_interval=True
        )
        name_vs = "my_validation"
        vs = dmv.ValidationSession(name_vs, ds)

        # simulation not in the list
        with pytest.raises(KeyError):
            vs.get_simulation_signals_list("potato")

        # Another test with one model
        sim1_name = "Model 1"
        sim1_labels = ["my_y1", "my_y2"]  # The fixture has two outputs
        if fixture == "SISO" or fixture == "MISO":
            sim1_labels = [sim1_labels[0]]
        sim1_values = np.random.rand(
            len(df.iloc[:, 0].values), len(sim1_labels)
        )

        # Same sim nmane
        vs.append_simulation(sim1_name, sim1_labels, sim1_values)

        # Search for a non-existing simulation
        with pytest.raises(KeyError):
            vs.get_simulation_signals_list("potato")


class Test_ClassValidatioNominal_sim_validation:
    def test_existing_sim_raise(self, good_dataframe: pd.DataFrame) -> None:
        df, u_labels, y_labels, fixture = good_dataframe
        name_ds = "my_dataset"
        ds = dmv.dataset.Dataset(
            name_ds, df, u_labels, y_labels, full_time_interval=True
        )
        name_vs = "my_validation"
        vs = dmv.ValidationSession(name_vs, ds)

        # Add one model
        sim1_name = "Model 1"
        sim1_labels = ["my_y1", "my_y2"]  # The fixture has two outputs
        if fixture == "SISO" or fixture == "MISO":
            sim1_labels = [sim1_labels[0]]
        sim1_values = np.random.rand(
            len(df.iloc[:, 0].values), len(sim1_labels)
        )

        # Same sim nmane
        vs.append_simulation(sim1_name, sim1_labels, sim1_values)
        with pytest.raises(ValueError):
            vs.append_simulation(sim1_name, sim1_labels, sim1_values)

    def test_too_many_signals_raise(self, good_dataframe: pd.DataFrame) -> None:
        df, u_labels, y_labels, fixture = good_dataframe
        name_ds = "my_dataset"
        ds = dmv.dataset.Dataset(
            name_ds, df, u_labels, y_labels, full_time_interval=True
        )

        name_vs = "my_validation"
        vs = dmv.ValidationSession(name_vs, ds)

        # Add one model
        sim1_name = "Model 1"
        sim1_labels = [
            "my_y1",
            "my_y2",
            "potato",
        ]  # The fixture has two outputs
        sim1_values = np.random.rand(
            len(df.iloc[:, 0].values), len(sim1_labels)
        )

        # Same sim nmane
        with pytest.raises(IndexError):
            vs.append_simulation(sim1_name, sim1_labels, sim1_values)

    def test_duplicate_names_raise(self, good_dataframe: pd.DataFrame) -> None:
        df, u_labels, y_labels, fixture = good_dataframe
        name_ds = "my_dataset"
        ds = dmv.dataset.Dataset(
            name_ds, df, u_labels, y_labels, full_time_interval=True
        )

        name_vs = "my_validation"
        vs = dmv.ValidationSession(name_vs, ds)

        # Add one model
        sim1_name = "Model 1"
        sim1_labels = ["my_y1", "my_y1"]  # The fixture has two outputs
        if fixture == "SISO" or fixture == "MISO":
            sim1_labels = [sim1_labels[0]]
        sim1_values = np.random.rand(
            len(df.iloc[:, 0].values), len(sim1_labels)
        )

        # Same sim nmane
        if fixture == "SIMO" or fixture == "MIMO":
            with pytest.raises(ValueError):
                vs.append_simulation(sim1_name, sim1_labels, sim1_values)

    def test_mismatch_labels_values_raise(
        self, good_dataframe: pd.DataFrame
    ) -> None:
        df, u_labels, y_labels, fixture = good_dataframe
        name_ds = "my_dataset"
        ds = dmv.dataset.Dataset(
            name_ds, df, u_labels, y_labels, full_time_interval=True
        )

        name_vs = "my_validation"
        vs = dmv.ValidationSession(name_vs, ds)

        # Add one model
        sim1_name = "Model 1"
        sim1_labels = ["my_y1", "my_y2"]  # The fixture has two outputs
        if fixture == "SISO" or fixture == "MISO":
            sim1_labels = [sim1_labels[0]]
        sim1_values = np.random.rand(
            len(df.iloc[:, 0].values), len(sim1_labels) + 1
        )

        # Same sim nmane
        with pytest.raises(IndexError):
            vs.append_simulation(sim1_name, sim1_labels, sim1_values)

    def test_too_many_values_raise(self, good_dataframe: pd.DataFrame) -> None:
        df, u_labels, y_labels, fixture = good_dataframe
        name_ds = "my_dataset"
        ds = dmv.dataset.Dataset(
            name_ds, df, u_labels, y_labels, full_time_interval=True
        )

        name_vs = "my_validation"
        vs = dmv.ValidationSession(name_vs, ds)

        # Add one model
        sim1_name = "Model 1"
        sim1_labels = ["my_y1", "my_y2"]  # The fixture has two outputs
        if fixture == "SISO" or fixture == "MISO":
            sim1_labels = [sim1_labels[0]]
        sim1_values = np.random.rand(
            len(df.iloc[:, 0].values) + 1, len(sim1_labels) + 1
        )

        # Same sim nmane
        with pytest.raises(IndexError):
            vs.append_simulation(sim1_name, sim1_labels, sim1_values)

    def test_values_not_ndarray_raise(
        self, good_dataframe: pd.DataFrame
    ) -> None:
        df, u_labels, y_labels, fixture = good_dataframe
        name_ds = "my_dataset"
        ds = dmv.dataset.Dataset(
            name_ds, df, u_labels, y_labels, full_time_interval=True
        )

        name_vs = "my_validation"
        vs = dmv.ValidationSession(name_vs, ds)

        # Add one model
        sim1_name = "Model 1"
        sim1_labels = ["my_y1", "my_y2"]  # The fixture has two outputs
        if fixture == "SISO" or fixture == "MISO":
            sim1_labels = [sim1_labels[0]]
        sim1_values = "potato"

        # Same sim nmane
        with pytest.raises(ValueError):
            vs.append_simulation(sim1_name, sim1_labels, sim1_values)

    def test_ydata_too_short_raise(self, good_dataframe: pd.DataFrame) -> None:
        df, u_labels, y_labels, fixture = good_dataframe
        name_ds = "my_dataset"
        ds = dmv.dataset.Dataset(
            name_ds, df, u_labels, y_labels, full_time_interval=True
        )

        name_vs = "my_validation"
        vs = dmv.ValidationSession(name_vs, ds)

        # Add one model
        sim1_name = "Model 1"
        sim1_labels = ["my_y1", "my_y2"]  # The fixture has two outputs
        if fixture == "SISO" or fixture == "MISO":
            sim1_labels = [sim1_labels[0]]

        # Short data
        sim1_values = np.random.rand(2, 1)

        # Same sim nmane
        with pytest.raises(IndexError):
            vs.append_simulation(sim1_name, sim1_labels, sim1_values)

    def test_drop_simulation_raise(self, good_dataframe: pd.DataFrame) -> None:
        df, u_labels, y_labels, fixture = good_dataframe
        name_ds = "my_dataset"
        ds = dmv.dataset.Dataset(
            name_ds, df, u_labels, y_labels, full_time_interval=True
        )

        # Create validation session.
        name_vs = "my_validation"
        vs = dmv.ValidationSession(name_vs, ds)

        # Add one model
        sim1_name = "Model 1"
        sim1_labels = ["my_y1", "my_y2"]  # The fixture has two outputs
        if fixture == "SISO" or fixture == "MISO":
            sim1_labels = [sim1_labels[0]]
        sim1_values = np.random.rand(
            len(df.iloc[:, 0].values), len(sim1_labels)
        )

        vs.append_simulation(sim1_name, sim1_labels, sim1_values)

        with pytest.raises(ValueError):
            vs.drop_simulation("potato")


class Test_Plots:
    @pytest.mark.plot
    def test_plots(self, good_dataframe: pd.DataFrame, tmp_path: str) -> None:
        df, u_labels, y_labels, fixture = good_dataframe
        name_ds = "my_dataset"
        ds = dmv.dataset.Dataset(
            name_ds, df, u_labels, y_labels, full_time_interval=True
        )

        name_vs = "my_validation"
        vs = dmv.ValidationSession(name_vs, ds)

        # Add one model
        sim1_name = "Model 1"
        sim1_labels = ["my_y1", "my_y2"]  # The fixture has two outputs
        if fixture == "SISO" or fixture == "MISO":
            sim1_labels = [sim1_labels[0]]
        sim1_values = np.random.rand(
            len(df.iloc[:, 0].values), len(sim1_labels)
        )
        vs.append_simulation(sim1_name, sim1_labels, sim1_values)

        # Add a second
        sim2_name = "Model 2"
        sim2_labels = ["your_y1", "your_y2"]  # The fixture has two outputs
        if fixture == "SISO" or fixture == "MISO":
            sim2_labels = [sim2_labels[0]]
        sim2_values = vs.Dataset.dataset["OUTPUT"].values + np.random.rand(
            len(vs.Dataset.dataset["OUTPUT"].values), 1
        )
        vs.append_simulation(sim2_name, sim2_labels, sim2_values)

        # =============================
        # plot simulations
        # =============================
        fig, _ = vs.plot_simulations()
        fig.clf()
        plt.close("all")

        fig, _ = vs.plot_simulations(dataset="all")
        fig.clf()
        plt.close("all")

        # Test plot - filtered
        fig, _ = vs.plot_simulations("Model 2", dataset="only_out")
        fig.clf()
        plt.close("all")

        # Test plot - all the options
        fig, _ = vs.plot_simulations(["Model 1", "Model 2"], dataset="all")
        fig.clf()
        plt.close("all")

        # =============================
        # save simulations
        # =============================
        tmp_path_str = str(tmp_path)
        filename = tmp_path_str + "/potato"
        fig, _ = vs.plot_simulations(save_as=filename)
        # fig.clf()
        plt.close("all")
        assert os.path.exists(filename + ".png")
        # =============================
        # plot simulations raises
        # =============================
        # Test plot - filtered wrong
        with pytest.raises(KeyError):
            fig, _ = vs.plot_simulations("potato")
        # Test plot - filtered wrong
        vs.clear()
        with pytest.raises(KeyError):
            fig, _ = vs.plot_simulations()
        with pytest.raises(KeyError):
            fig, _ = vs.plot_simulations("potato")

        # =============================
        # plot residuals
        # =============================
        vs.append_simulation(sim1_name, sim1_labels, sim1_values)
        vs.append_simulation(sim2_name, sim2_labels, sim2_values)

        fig1, _, fig2, _ = vs.plot_residuals()
        fig1.clf()
        fig2.clf()
        plt.close("all")

        fig1, _, fig2, _ = vs.plot_residuals("Model 1")
        fig1.clf()
        fig2.clf()
        plt.close("all")

        fig1, _, fig2, _ = vs.plot_residuals(["Model 1", "Model 2"])
        fig1.clf()
        fig2.clf()
        plt.close("all")

        fig1, _, fig2, _ = vs.plot_residuals(["Model 1", "Model 2"])
        fig1.clf()
        fig2.clf()
        plt.close("all")
        # =============================
        # save residuals
        # =============================
        tmp_path_str = str(tmp_path)
        filename = tmp_path_str + "/potato"
        fig1, _, fig2, _ = vs.plot_residuals(save_as=filename)
        # fig1, _, fig2, _ = vs.plot_residuals()
        fig1.clf()
        fig2.clf()
        plt.close("all")
        # TODO: remove comments
        assert os.path.exists(filename + "_eps_eps.png")
        assert os.path.exists(filename + "_u_eps.png")

        # =============================
        # plot residuals raises
        # =============================
        with pytest.raises(KeyError):
            fig, _, fig2, _ = vs.plot_residuals("potato")

        # Empty simulation list
        vs.clear()
        with pytest.raises(KeyError):
            fig, _, fig2, _ = vs.plot_residuals()


class Test_xcorr:
    @pytest.mark.parametrize(
        "X,Y",
        [
            (np.random.rand(10), np.random.rand(10)),
            # (np.random.rand(8, 3), np.random.rand(10)),
            # (np.random.rand(10), np.random.rand(10, 3)),
            # (np.random.rand(5, 1), np.random.rand(10, 4)),
            # (np.random.rand(8, 3), np.random.rand(4, 4)),
            # (np.random.rand(10, 4), np.random.rand(15, 3)),
        ],
    )
    def test_xcorr(self, X: XCorrelation, Y: XCorrelation) -> None:
        # Just test that it won't run any error
        # Next, remove randoms with known values.

        x1 = np.array([0.1419, 0.4218, 0.9157, 0.7922, 0.9595])
        x2 = np.array([0.6557, 0.0357, 0.8491, 0.9340, 0.6787])
        X = np.array([x1, x2]).T

        y1 = np.array([0.7577, 0.7431, 0.3922, 0.6555, 0.1712])
        y2 = np.array([0.7060, 0.0318, 0.2769, 0.0462, 0.0971])
        Y = np.array([y1, y2]).T

        # Expected values pre-computed with Matlab
        Rx1y1_expected = np.array(
            [
                0.5233,
                0.0763,
                -0.1363,
                -0.2526,
                -0.8181,
                0.0515,
                0.1090,
                0.2606,
                0.1864,
            ]
        )

        Rx1y2_expected = np.array(
            [
                0.1702,
                0.3105,
                -0.0438,
                0.0526,
                -0.6310,
                -0.5316,
                0.2833,
                0.0167,
                0.3730,
            ]
        )

        Rx2y1_expected = np.array(
            [
                -0.0260,
                0.6252,
                -0.4220,
                0.0183,
                -0.3630,
                -0.3462,
                0.2779,
                0.2072,
                0.0286,
            ]
        )

        Rx2y2_expected = np.array(
            [
                -0.0085,
                0.1892,
                0.2061,
                -0.2843,
                0.1957,
                -0.8060,
                0.1135,
                0.3371,
                0.0573,
            ]
        )
        lags_expected = np.arange(-4, 5)

        # Call dymoval function
        # OBS! It works only if NUM_DECIMALS = 4, like in Matlab

        # SISO
        XCorr_actual = dmv.xcorr(x1, y1)
        Rxy_actual = XCorr_actual["values"]
        lags_actual = XCorr_actual["lags"]

        assert np.allclose(Rxy_actual[:, 0, 0], Rx1y1_expected, atol=1e-4)
        assert np.allclose(lags_actual, lags_expected)

        # SIMO
        XCorr_actual = dmv.xcorr(x1, Y)
        Rxy_actual = XCorr_actual["values"]
        lags_actual = XCorr_actual["lags"]

        assert np.allclose(Rxy_actual[:, 0, 0], Rx1y1_expected, atol=1e-4)
        assert np.allclose(Rxy_actual[:, 0, 1], Rx1y2_expected, atol=1e-4)
        assert np.allclose(lags_actual, lags_expected)

        # MISO
        XCorr_actual = dmv.xcorr(X, y1)
        Rxy_actual = XCorr_actual["values"]
        lags_actual = XCorr_actual["lags"]

        assert np.allclose(Rxy_actual[:, 0, 0], Rx1y1_expected, atol=1e-4)
        assert np.allclose(Rxy_actual[:, 1, 0], Rx2y1_expected, atol=1e-4)
        assert np.allclose(lags_actual, lags_expected)

        # MIMO
        XCorr_actual = dmv.xcorr(X, Y)
        Rxy_actual = XCorr_actual["values"]
        lags_actual = XCorr_actual["lags"]

        assert np.allclose(Rxy_actual[:, 0, 0], Rx1y1_expected, atol=1e-4)
        assert np.allclose(Rxy_actual[:, 0, 1], Rx1y2_expected, atol=1e-4)
        assert np.allclose(Rxy_actual[:, 1, 0], Rx2y1_expected, atol=1e-4)
        assert np.allclose(Rxy_actual[:, 1, 1], Rx2y2_expected, atol=1e-4)
        assert np.allclose(lags_actual, lags_expected)


class Test_rsquared:
    def test_rsquared_nominal(self) -> None:
        # Just test that it won't run any error
        # Next, remove randoms with known values.

        y1 = np.array(
            [
                0,
                0.5878,
                0.9511,
                0.9511,
                0.5878,
                0.0000,
                -0.5878,
                -0.9511,
                -0.9511,
                -0.5878,
                -0.0000,
            ]
        )

        y2 = np.array(
            [
                0,
                0.7053,
                1.1413,
                1.1413,
                0.7053,
                0.0000,
                -0.7053,
                -1.1413,
                -1.1413,
                -0.7053,
                -0.0000,
            ]
        )

        y1Calc = np.array(
            [
                0.1403,
                0.8620,
                1.0687,
                1.1633,
                0.9208,
                0.2390,
                -0.4537,
                -0.8314,
                -0.7700,
                -0.4187,
                0.1438,
            ]
        )

        y2Calc = np.array(
            [
                0.2233,
                1.0024,
                1.3110,
                1.3130,
                0.7553,
                0.0098,
                -0.5893,
                -1.0143,
                -0.8798,
                -0.3226,
                0.3743,
            ]
        )

        rsquared_expected_SISO = 91.2775
        rsquared_expected_MIMO = 92.6092

        rsquared_actual_SISO = dmv.rsquared(y1, y1Calc)
        rsquared_actual_MIMO = dmv.rsquared(
            np.array([y1, y2]).T, np.array([y1Calc, y2Calc]).T
        )

        assert np.isclose(
            rsquared_expected_SISO, rsquared_actual_SISO, atol=1e-4
        )

        assert np.isclose(rsquared_expected_MIMO, rsquared_actual_MIMO)

    @pytest.mark.parametrize(
        "y_values,y_sim_values",
        [
            (np.random.rand(10), np.random.rand(5)),
            (np.random.rand(8, 3), np.random.rand(10)),
            (np.random.rand(10), np.random.rand(10, 3)),
            (np.random.rand(5, 1), np.random.rand(10, 4)),
            (np.random.rand(8, 3), np.random.rand(4, 4)),
            (np.random.rand(10, 4), np.random.rand(15, 3)),
        ],
    )
    def test_rsquared_raise(
        self, y_values: np.ndarray, y_sim_values: np.ndarray
    ) -> None:
        # Just test that it won't run any error
        # Next, remove randoms with known values.
        with pytest.raises(IndexError):
            dmv.rsquared(y_values, y_sim_values)


class Test_xcorr_norm:
    def test_xcorr_norm_nominal(self) -> None:
        # Just test that it won't run any error
        # Next, remove randoms with known values.
        # Expected values pre-computed with Matlab
        Rx1y1 = np.array(
            [
                0.5233,
                0.0763,
                -0.1363,
                -0.2526,
                -0.8181,
                0.0515,
                0.1090,
                0.2606,
                0.1864,
            ]
        )

        Rx1y2 = np.array(
            [
                0.1702,
                0.3105,
                -0.0438,
                0.0526,
                -0.6310,
                -0.5316,
                0.2833,
                0.0167,
                0.3730,
            ]
        )

        Rx2y1 = np.array(
            [
                -0.0260,
                0.6252,
                -0.4220,
                0.0183,
                -0.3630,
                -0.3462,
                0.2779,
                0.2072,
                0.0286,
            ]
        )

        Rx2y2 = np.array(
            [
                -0.0085,
                0.1892,
                0.2061,
                -0.2843,
                0.1957,
                -0.8060,
                0.1135,
                0.3371,
                0.0573,
            ]
        )

        lags_test = np.arange(-4, 5)

        norm_expected_SISO = 0.1191
        norm_expected_SIMO = 0.1640
        norm_expected_MISO = 0.1607
        norm_expected_MIMO = 0.2249

        # SISO Adjust test values
        R_test = np.empty((len(lags_test), 1, 1))
        R_test[:, 0, 0] = Rx1y1
        Rxy_test = {"values": R_test, "lags": lags_test}

        # Act
        norm_actual = dmv.xcorr_norm(Rxy_test)

        # Assert
        assert np.isclose(norm_actual, norm_expected_SISO, atol=1e-4)

        # SIMO Adjust test values
        R_test = np.empty((len(lags_test), 1, 2))
        R_test[:, 0, 0] = Rx1y1
        R_test[:, 0, 1] = Rx1y2
        Rxy_test = {"values": R_test, "lags": lags_test}

        # Act
        norm_actual = dmv.xcorr_norm(Rxy_test)

        # Assert
        assert np.isclose(norm_actual, norm_expected_SIMO, atol=1e-4)

        # MISO Adjust test values
        R_test = np.empty((len(lags_test), 2, 1))
        R_test[:, 0, 0] = Rx1y1
        R_test[:, 1, 0] = Rx2y1
        Rxy_test = {"values": R_test, "lags": lags_test}

        # Act
        norm_actual = dmv.xcorr_norm(Rxy_test)

        # Assert
        assert np.isclose(norm_actual, norm_expected_MISO, atol=1e-4)

        # MIMO Adjust test values
        R_test = np.empty((len(lags_test), 2, 2))
        R_test[:, 0, 0] = Rx1y1
        R_test[:, 0, 1] = Rx1y2
        R_test[:, 1, 0] = Rx2y1
        R_test[:, 1, 1] = Rx2y2
        Rxy_test = {"values": R_test, "lags": lags_test}

        # Act
        norm_actual = dmv.xcorr_norm(Rxy_test)

        # Assert
        assert np.isclose(norm_actual, norm_expected_MIMO, atol=1e-4)

    @pytest.mark.parametrize(
        "R",
        [np.random.rand(10, 3, 2, 4), np.random.rand(10, 2, 1, 5, 4)],
    )
    def test_xcorr_norm_raise(self, R: XCorrelation) -> None:
        # Just test that it won't run any error
        # Next, remove randoms with known values.
        Rxy = {"values": R, "lags": signal.correlation_lags(len(R), len(R))}
        with pytest.raises(IndexError):
            dmv.xcorr_norm(Rxy)
