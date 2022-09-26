import pytest
import dymoval as dmv
from dymoval.validation import XCorrelation
import numpy as np
from matplotlib import pyplot as plt
from .fixture_data import *  # noqa
import scipy.signal as signal


class TestClassValidationNominal:
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


class TestClassValidatioNominal_sim_validation:
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


class TestPlots:
    def test_plots(self, good_dataframe: pd.DataFrame) -> None:
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

        # Test plot
        vs.plot_simulations()
        plt.close("all")

        #
        _ = vs.plot_simulations(plot_input=True, return_figure=True)
        plt.close("all")

        # Test plot - conditional
        vs.plot_simulations("Model 2")
        plt.close("all")

        # Test plot - all the options
        vs.plot_simulations(
            ["Model 1", "Model 2"], plot_dataset=True, plot_input=True
        )
        plt.close("all")

        # Test plot - conditional wrong
        with pytest.raises(KeyError):
            vs.plot_simulations("potato")
        # Test plot - conditional wrong
        vs.clear()
        with pytest.raises(KeyError):
            vs.plot_simulations()
        with pytest.raises(KeyError):
            vs.plot_simulations("potato")
        # =============================
        # plot residuals
        # =============================
        vs.append_simulation(sim1_name, sim1_labels, sim1_values)
        vs.append_simulation(sim2_name, sim2_labels, sim2_values)
        vs.plot_residuals()
        plt.close("all")
        vs.plot_residuals("Model 1")
        plt.close("all")
        vs.plot_residuals(["Model 1", "Model 2"])
        plt.close("all")
        _ = vs.plot_residuals(["Model 1", "Model 2"], return_figure=True)
        plt.close("all")

        # =============================
        # plot residuals raises
        # =============================
        with pytest.raises(KeyError):
            vs.plot_residuals("potato")

        # Empty simulation list
        vs.clear()
        with pytest.raises(KeyError):
            vs.plot_residuals()


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
        dmv.xcorr(X, Y)


class Test_rsquared:
    @pytest.mark.parametrize(
        "y_values,y_sim_values",
        [
            (np.random.rand(10), np.random.rand(10)),
            (np.random.rand(5, 3), np.random.rand(5, 3)),
            (np.random.rand(3, 5), np.random.rand(3, 5)),
        ],
    )
    def test_rsquared_nominal(
        self, y_values: np.ndarray, y_sim_values: np.ndarray
    ) -> None:
        # Just test that it won't run any error
        # Next, remove randoms with known values.
        dmv.rsquared(y_values, y_sim_values)

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
    @pytest.mark.parametrize(
        "R",
        [
            np.random.rand(10, 3, 2),
            np.random.rand(10, 2, 1),
            np.random.rand(10, 2),
            np.random.rand(10),
        ],
    )
    def test_xcorr_norm_nominal(self, R: XCorrelation) -> None:
        # Just test that it won't run any error
        # Next, remove randoms with known values.
        Rxy = {"values": R, "lags": signal.correlation_lags(len(R), len(R))}
        dmv.xcorr_norm(Rxy)

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
