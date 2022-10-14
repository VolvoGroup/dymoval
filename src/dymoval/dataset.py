# The following are only for Spyder, otherwise things are written in
# the pyproject.toml
# mypy: show_error_codes
"""Module containing everything related to datasets.
Here are defined special datatypes, classes and auxiliary functions to deal with datasets.
"""
from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import io, fft
from .config import *  # noqa
from .utils import *  # noqa, Type
from typing import TypedDict, Any, Literal, overload
from copy import deepcopy


class Signal(TypedDict):
    """Signals are used to represent real-world signals.
    They are used to instantiate :py:class:`Dataset <dymoval.dataset.Dataset>` class objects.

    It is possible to validate signals through :py:meth:`~dymoval.dataset.validate_signals`.

    Although Signals have compulsory attribues that must be set, there is freedom
    to append additional attributes.



    Example
    -------
    >>> import dymoval as dmv
    >>> my_signal: dmv.Signal = {
    "name": "speed",
    "values": np.random.rand(100),
    "signal_unit": "mps",
    "sampling_period": 0.1,
    "time_unit": "s",
    }
    >>> # We create a time vector key for plotting the Signal
    >>> my_signal["time_vec"] = my_signal["sampling_period"]
    *np.arange(0,len(my_signal["values"]))
    >>> # Let's plot
    >>> plt.plot(my_signal["time_vec"], my_signal["values"])
    >>> plt.show()


    Attributes
    ----------
    values: np.ndarray
        Signal values.
    """

    name: str  #: Signal name.
    values: np.ndarray  # values confuse with values() which is a dict method.
    signal_unit: str  #: Signal unit.
    sampling_period: float  #: Signal sampling period.
    time_unit: str  #: Signal sampling period.


class Dataset:
    """The *Dataset* class stores the signals that you want to use as dataset
    and it provides methods for analyzing and manipulating them.

    A *Signal* list shall be passed to the initializer along with two lists of labels
    defining which signal(s) shall be consiedered
    as input and which signal(s) shall be considered as output.

    The signals list can be either a list
    of dymoval :py:class:`Signals <dymoval.dataset.Signal>` type or a
    pandas DataFrame with a well-defined structure.
    See  :py:meth:`~dymoval.dataset.dataframe_validation` for more
    information.


    Parameters
    ----------
    name:
        Dataset name.
    signal_list :
        Signals to be included in the :py:class:`Dataset <dymoval.dataset.Dataset>`.
        See :py:meth:`~dymoval.dataset.validate_signals` and
        :py:meth:`~dymoval.dataset.dataframe_validation` to figure out how
        the list of :py:class:`Signal <dymoval.dataset.Signal>` or the pandas
        DataFrame representing the dataset signals shall look like.
    u_labels :
        List of input signal names. Each signal name must be unique and must be
        contained in the signal_list.
    y_labels :
        List of input signal names. Each signal name must be unique and must be
        contained in the signal_list.
    target_sampling_period :
        The passed signals will be re-sampled towards this target sampling period.
    plot_raw :
        If *True*, then the :py:class:`Signal <dymoval.dataset.Signal>`
        contained in the *signal_list* will be plotted.
        This parameter only have effect when the *signal_list* parameter is
        a list of :py:class:`Signal <dymoval.dataset.Signal>` objects.
    tin :
        Initial time instant of the Dataset.
    tout :
        Final time instant of the Dataset.
    full_time_interval :
        If *True*, the Dataset time interval will be equal to the longest
        time interval among all of the signals included in the *signal_list*
        parameter.
        This is overriden if the parameters *tin* and *tout* are specified.
    overlap :
        If *True* it will overlap the input and output signals plots during the
        dataset time interval selection.
    """

    def __init__(
        self,
        name: str,
        signal_list: list[Signal] | pd.DataFrame,
        u_labels: str | list[str],
        y_labels: str | list[str],
        target_sampling_period: float | None = None,
        plot_raw: bool = False,
        tin: float | None = None,
        tout: float | None = None,
        full_time_interval: bool = False,
        overlap: bool = False,
        verbosity: int = 0,
    ) -> None:

        if all(isinstance(x, dict) for x in signal_list):
            attr_sign = self._new_dataset_from_signals(
                signal_list,
                u_labels,
                y_labels,
                target_sampling_period,
                plot_raw,
                tin,
                tout,
                full_time_interval,
                overlap,
                verbosity,
            )
            (
                df,
                Ts,
                nan_intervals,
                excluded_signals,
                dataset_coverage,
            ) = attr_sign
        # Initialization by pandas DataFrame
        elif isinstance(signal_list, pd.DataFrame):
            attr_df = self._new_dataset_from_dataframe(
                signal_list,
                u_labels,
                y_labels,
                tin,
                tout,
                full_time_interval,
                overlap,
                verbosity,
            )
            df, Ts, nan_intervals, excluded_signals, dataset_coverage = attr_df
        else:
            raise TypeError(
                "Input must be a Signal list or a pandas DataFrame type.",
            )
        # ================================================
        # Class attributes
        # ================================================
        # NOTE: You have to use the #: to add a doc description
        # in class attributes (see sphinx.ext.autodoc)

        self.name: str = name  #: Dataset name.
        self.dataset: pd.DataFrame = deepcopy(df)  #: Actual dataset.
        self.coverage: pd.DataFrame = deepcopy(
            dataset_coverage
        )  # Docstring below,
        """Coverage statistics in terms of mean value and variance"""
        self.information_level: float = 0.0  #: *Not implemented yet!*
        self._nan_intervals: Any = deepcopy(nan_intervals)
        self.excluded_signals: list[str] = excluded_signals
        """Signals that could not be re-sampled."""
        self.sampling_period = Ts  #: Dataset sampling period

    def __str__(self) -> str:
        return f"Dymoval dataset called '{self.name}'."

    # ============= NOT READY =======================
    # def __eq__(self, other):
    #     if not isinstance(other, Dataset):
    #         # don't attempt to compare against unrelated types
    #         return NotImplemented
    #     return all(self.dataset.columns == other.dataset.columns)
    # and self._nan_intervals == other._nan_intervals
    # ==============================================

    # ==============================================
    #   Class methods
    # ==============================================
    def _shade_input_nans(
        self,
        df: pd.DataFrame,
        NaN_intervals: dict[str, dict[str, list[np.ndarray]]],
        axes: matplotlib.axes.Axes,
        u_labels: list[str],
        color: str = "b",
    ) -> None:

        # u_labels are forwarded from other functions and are already
        # converted into a list[str].
        for ii, ax in enumerate(axes):
            input_name = u_labels[ii]
            for idx, val in enumerate(NaN_intervals["INPUT"][input_name]):
                if not val.size == 0:
                    ax.axvspan(min(val), max(val), color=color, alpha=0.2)

    def _shade_output_nans(
        self,
        df: pd.DataFrame,
        NaN_intervals: dict[str, dict[str, list[np.ndarray]]],
        axes: matplotlib.axes.Axes,
        y_labels: list[str],
        color: str = "g",
    ) -> None:

        # y_labels are forwarded from other functions and are already
        # converted into a list[str].
        for ii, ax in enumerate(axes):
            output_name = y_labels[ii]
            for idx, val in enumerate(NaN_intervals["OUTPUT"][output_name]):
                if not val.size == 0:
                    ax.axvspan(min(val), max(val), color=color, alpha=0.2)

    def _init_dataset_coverage(
        self, df: pd.DataFrame
    ) -> tuple[pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]:

        u_mean = df["INPUT"].mean(axis=0)
        u_cov = df["INPUT"].cov()
        y_mean = df["OUTPUT"].mean(axis=0)
        y_cov = df["OUTPUT"].cov()

        return u_mean, u_cov, y_mean, y_cov

    def _init_nan_intervals(
        self, df: pd.DataFrame
    ) -> dict[str, list[np.ndarray]]:
        # Find index intervals (i.e. time intervals) where columns values
        # are NaN.
        # Run an example in the tutorial to see an example on how they are stored.
        sampling_period = df.index[1] - df.index[0]
        NaN_index = {}
        NaN_intervals = {}
        for x in list(df.columns):
            NaN_index[x] = df.loc[df[x].isnull().to_numpy()].index
            idx = np.where(~np.isclose(np.diff(NaN_index[x]), sampling_period))[
                0
            ]
            NaN_intervals[x] = np.split(NaN_index[x], idx + 1)
        return NaN_intervals

    def _init_dataset_time_interval(
        self,
        df: pd.DataFrame,
        NaN_intervals: dict[str, dict[str, list[np.ndarray]]],
        tin: float | None = None,
        tout: float | None = None,
        overlap: bool = False,
        full_time_interval: bool = False,
        verbosity: int = 0,
    ) -> tuple[pd.DataFrame, dict[str, dict[str, list[np.ndarray]]]]:
        # We have to trim the signals to have a meaningful dataset
        # This can be done both graphically or by passing tin and tout
        # if the user knows them before hand.

        # Number of inputs and outputs
        p = len(df["INPUT"].columns)
        q = len(df["OUTPUT"].columns)
        # Check if there is some argument.
        if tin is not None and tout is not None:
            tin_sel = np.round(tin, NUM_DECIMALS)
            tout_sel = np.round(tout, NUM_DECIMALS)
        elif full_time_interval:
            tin_sel = np.round(df.index[0], NUM_DECIMALS)
            tout_sel = np.round(df.index[-1], NUM_DECIMALS)
        else:  # pragma: no cover
            # OBS! This part cannot be automatically tested because the it require
            # manual action from the user (resize window).
            # Hence, you must test this manually
            # The keyword for skippint the coverage is # pragma: no cover
            #  ===========================================================
            # The following code is needed because not all IDE:s
            # have interactive plot set to ON as default.
            #
            is_interactive = plt.isinteractive()
            plt.ion()
            #  ===========================================================

            if overlap:
                n = max(p, q)
                range_out = np.arange(0, q)
            else:
                n = p + q
                range_out = np.arange(p, p + q)
            nrows, ncols = factorize(n)  # noqa
            _, axes = plt.subplots(nrows, ncols, sharex=True, squeeze=False)
            axes = axes.T.flat
            df["INPUT"].plot(subplots=True, grid=True, color="b", ax=axes[0:p])
            df["OUTPUT"].plot(
                subplots=True, grid=True, color="g", ax=axes[range_out]
            )

            for ii in range(ncols):
                axes[nrows - 1 :: nrows][ii].set_xlabel("Frequency")
            plt.suptitle(
                "Sampling time "
                f"= {np.round(df.index[1]-df.index[0],NUM_DECIMALS)}.\n"
                "Select the dataset time interval by resizing "
                "the picture."
            )

            # Shade NaN areas
            self._shade_input_nans(
                df, NaN_intervals, axes[0:p], list(df["INPUT"].columns)
            )
            self._shade_output_nans(
                df, NaN_intervals, axes[range_out], list(df["OUTPUT"].columns)
            )

            # Figure closure handler
            # It can be better done perhaps.
            def close_event(event):  # type:ignore
                time_interval = np.round(
                    axes[0].get_xlim(), NUM_DECIMALS
                )  # noqa
                close_event.tin, close_event.tout = time_interval
                close_event.tin = max(close_event.tin, 0.0)
                close_event.tout = max(close_event.tout, 0.0)
                if is_interactive:
                    plt.ion()
                else:
                    plt.ioff()

            close_event.tin = 0.0  # type:ignore
            close_event.tout = 0.0  # type:ignore
            fig = axes[-1].get_figure()
            cid = fig.canvas.mpl_connect("close_event", close_event)
            fig.canvas.draw()
            plt.show()

            # =======================================================
            # This is needed for Spyder to block the prompt while
            # the figure is opened.
            # An alternative better solution is welcome!
            try:
                while fig.number in plt.get_fignums():
                    plt.pause(0.1)
            except:  # noqa
                plt.close(fig.number)
                raise
            # =======================================================

            fig.canvas.mpl_disconnect(cid)
            tin_sel = close_event.tin  # type:ignore
            tout_sel = close_event.tout  # type:ignore

            if verbosity != 0:
                print("\ntin = ", tin_sel, "tout =", tout_sel)

        # ===================================================================
        # Trim dataset and NaN intervals based on (tin,tout)
        # ===================================================================
        df = df.loc[tin_sel:tout_sel, :]
        # Trim NaN_intevals
        # TODO: code repetition
        for u_name in NaN_intervals["INPUT"].keys():
            # In the following, nan_chunks are time-interval.
            # Note! For a given signal, you may have many nan_chunks.
            for idx, nan_chunk in enumerate(NaN_intervals["INPUT"][u_name]):
                nan_chunk = np.round(nan_chunk, NUM_DECIMALS)  # noqa
                NaN_intervals["INPUT"][u_name][idx] = nan_chunk[
                    nan_chunk >= tin_sel
                ]
                NaN_intervals["INPUT"][u_name][idx] = nan_chunk[
                    nan_chunk <= tout_sel
                ]
        for y_name in NaN_intervals["OUTPUT"].keys():
            for idx, nan_chunk in enumerate(NaN_intervals["OUTPUT"][y_name]):
                nan_chunk = np.round(nan_chunk, NUM_DECIMALS)  # noqa
                NaN_intervals["OUTPUT"][y_name][idx] = nan_chunk[
                    nan_chunk >= tin_sel
                ]
                NaN_intervals["OUTPUT"][y_name][idx] = nan_chunk[
                    nan_chunk <= tout_sel
                ]
        return df, NaN_intervals

    def _shift_dataset_time_interval(
        self,
        df: pd.DataFrame,
        NaN_intervals: dict[str, dict[str, list[np.ndarray]]],
    ) -> tuple[pd.DataFrame, dict[str, dict[str, list[np.ndarray]]]]:
        # ===================================================================
        # Shift tin to zero.
        # ===================================================================
        tin_sel = df.index[0]
        timeVectorFromZero = df.index - tin_sel
        df.index = pd.Index(
            np.round(timeVectorFromZero, NUM_DECIMALS), name=df.index.name
        )

        # Shift also the NaN_intervals to zero.
        for k in NaN_intervals["INPUT"].keys():
            for idx, nan_chunk in enumerate(NaN_intervals["INPUT"][k]):
                nan_chunk_translated = nan_chunk - tin_sel
                NaN_intervals["INPUT"][k][idx] = np.round(
                    nan_chunk_translated, NUM_DECIMALS  # noqa
                )
                NaN_intervals["INPUT"][k][idx] = nan_chunk_translated[
                    nan_chunk_translated >= 0.0
                ]
        for k in NaN_intervals["OUTPUT"].keys():
            for idx, nan_chunk in enumerate(NaN_intervals["OUTPUT"][k]):
                nan_chunk_translated = nan_chunk - tin_sel
                NaN_intervals["OUTPUT"][k][idx] = np.round(
                    nan_chunk_translated, NUM_DECIMALS  # noqa
                )
                NaN_intervals["OUTPUT"][k][idx] = nan_chunk_translated[
                    nan_chunk_translated >= 0.0
                ]
        # Adjust the DataFrame accordingly
        df = df.round(decimals=NUM_DECIMALS)  # noqa

        return df, NaN_intervals

    def _new_dataset_from_dataframe(
        self,
        df: pd.DataFrame,
        u_labels: str | list[str],
        y_labels: str | list[str],
        tin: float | None = None,
        tout: float | None = None,
        full_time_interval: bool = False,
        overlap: bool = False,
        verbosity: int = 0,
    ) -> tuple[
        pd.DataFrame,
        float,
        dict[str, dict[str, list[np.ndarray]]],
        list[str],
        tuple[pd.Series, pd.DataFrame, pd.Series, pd.DataFrame],
    ]:

        # ==============================================================
        # This is the Dataset initializer when the signals are arranged
        # in a pandas DataFrame
        # ==============================================================
        # Arguments validation
        if tin is None and tout is not None:
            tin = df.index[0]
        # If only tin is passed, then set tout to the last time sample.
        if tin is not None and tout is None:
            tout = df.index[-1]
        if tin and tout and tin > tout:
            raise ValueError(
                f" Value of tin ( ={tin}) shall be smaller than the value of tout ( ={tout})."
            )
        # If the user passes a str cast into a list[str]
        u_labels = str2list(u_labels)  # noqa
        y_labels = str2list(y_labels)  # noqa

        # TODO: Start here for adding units
        dataframe_validation(df, u_labels, y_labels)

        # Add column index level with labels 'INPUT' and 'OUTPUT'
        df = df.loc[:, [*u_labels, *y_labels]]
        u_extended_labels = list(zip(["INPUT"] * len(u_labels), u_labels))
        y_extended_labels = list(zip(["OUTPUT"] * len(y_labels), y_labels))
        df.columns = pd.MultiIndex.from_tuples(
            [*u_extended_labels, *y_extended_labels]
        )

        # Initialize NaN intervals
        NaN_intervals = {
            "INPUT": self._init_nan_intervals(df["INPUT"]),
            "OUTPUT": self._init_nan_intervals(df["OUTPUT"]),
        }
        # Trim dataset time interval
        df, NaN_intervals = self._init_dataset_time_interval(
            df, NaN_intervals, tin, tout, overlap, full_time_interval, verbosity
        )
        # Shift dataset tin to 0.0.
        df, NaN_intervals = self._shift_dataset_time_interval(df, NaN_intervals)

        # Initialize coverage region
        dataset_coverage = self._init_dataset_coverage(df)

        # The DataFrame has been built with signals sampled with the same period,
        # therefore there are no excluded signals due to re-sampling
        excluded_signals: list[str] = []
        # In case user passes a DataFrame we need to compute the sampling period
        Ts = df.index[1] - df.index[0]
        return df, Ts, NaN_intervals, excluded_signals, dataset_coverage

    def _new_dataset_from_signals(
        self,
        signal_list: list[Signal],
        u_labels: str | list[str],
        y_labels: str | list[str],
        target_sampling_period: float | None = None,
        plot_raw: bool = False,
        tin: float | None = None,
        tout: float | None = None,
        full_time_interval: bool = False,
        overlap: bool = False,
        verbosity: int = 0,
    ) -> tuple[
        pd.DataFrame,
        float,
        dict[str, dict[str, list[np.ndarray]]],
        list[str],
        tuple[pd.Series, pd.DataFrame, pd.Series, pd.DataFrame],
    ]:

        # If the user pass a string, we need to convert into list
        u_labels = str2list(u_labels)  # noqa
        y_labels = str2list(y_labels)  # noqa

        # Arguments validation
        validate_signals(signal_list)

        if plot_raw:
            plot_signals(signal_list, u_labels, y_labels)

        # Try to align the sampling periods, whenever possible
        # Note! resampled_signals:list[Signals], whereas
        # excluded_signals: list[str]
        resampled_signals, excluded_signals = self._fix_sampling_periods(
            signal_list, target_sampling_period, verbosity=verbosity
        )

        Ts = list(resampled_signals)[0]["sampling_period"]

        # Drop excluded signals from u_labels and y_labels
        u_labels = [
            x["name"] for x in resampled_signals if x["name"] in u_labels
        ]
        y_labels = [
            x["name"] for x in resampled_signals if x["name"] in y_labels
        ]
        # Trim the signals to have equal length,
        # then build the DataFrame for inizializing the Dataset class.
        nsamples = [len(x["values"]) for x in resampled_signals]
        max_idx = min(nsamples)  # Shortest signal
        n = len(resampled_signals)  # number of signals

        df_data: np.ndarray = np.zeros((max_idx, n))
        for ii, s in enumerate(resampled_signals):
            df_data[:, ii] = s["values"][0:max_idx].round(NUM_DECIMALS)

        # Create actual DataFrame
        df = pd.DataFrame(
            data=df_data,
            columns=[*u_labels, *y_labels],
        )
        df.index = pd.Index(data=np.arange(max_idx) * Ts, name="Time")

        # Add some robustness: we validate the built DataFrame, even if
        # it should be correct by construction.
        dataframe_validation(df, u_labels=u_labels, y_labels=y_labels)

        # Call the initializer from DataFrame to get a Dataset object.
        (
            df,
            _,
            nan_intervals,
            _,
            dataset_coverage,
        ) = self._new_dataset_from_dataframe(
            df,
            u_labels,
            y_labels,
            tin,
            tout,
            full_time_interval,
            overlap,
            verbosity,
        )

        return df, Ts, nan_intervals, excluded_signals, dataset_coverage

    def _validate_signals(
        self,
        u_labels: str | list[str],
        y_labels: str | list[str],
    ) -> tuple[list[str], list[str]]:
        # This function check if the signals (labels) from the user
        # exist in the current dataset.
        df = self.dataset

        # Small check
        # Input labels passed
        if u_labels:
            u_labels = str2list(u_labels)
            input_not_found = difference_lists_of_str(
                u_labels, list(df["INPUT"].columns)
            )

            if input_not_found:
                raise KeyError(
                    f"Signal(s) {input_not_found} not found in the input signals dataset. "
                    "Use 'get_signal_list()' to get the list of all available signals. "
                )
        else:
            u_labels = []

        # Output labels passed
        if y_labels:
            y_labels = str2list(y_labels)
            output_not_found = difference_lists_of_str(
                y_labels, list(df["OUTPUT"].columns)
            )

            if output_not_found:
                raise KeyError(
                    f"Signal(s) {output_not_found} not found in the output signal dataset. "
                    "Use 'get_signal_list()' to get the list of all available signals "
                )
        else:
            y_labels = []

        # Nothing passed => take all.
        if not y_labels and not u_labels:
            u_labels = list(df["INPUT"].columns)
            y_labels = list(df["OUTPUT"].columns)

        return u_labels, y_labels

    @overload
    def _validate_name_value_tuples(
        self,
        *,
        u_list: list[tuple[str, float]] | tuple[str, float] = [],
        y_list: list[tuple[str, float]] | tuple[str, float],
    ) -> tuple[
        list[str],
        list[str],
        list[tuple[str, float]],
        list[tuple[str, float]],
    ]:
        ...  # pragma: no cover

    @overload
    def _validate_name_value_tuples(
        self,
        *,
        u_list: list[tuple[str, float]] | tuple[str, float],
        y_list: list[tuple[str, float]] | tuple[str, float] = [],
    ) -> tuple[
        list[str],
        list[str],
        list[tuple[str, float]],
        list[tuple[str, float]],
    ]:
        ...  # pragma: no cover

    def _validate_name_value_tuples(
        self,
        *,
        u_list: list[tuple[str, float]] | tuple[str, float] = [],
        y_list: list[tuple[str, float]] | tuple[str, float] = [],
    ) -> tuple[
        list[str],
        list[str],
        list[tuple[str, float]],
        list[tuple[str, float]],
    ]:
        # This function is needed to validate inputs like [("u1",3.2), ("y1", 0.5)]
        # Think for example to the "remove_offset" function.
        # Return both the list of input and output names and the validated tuples.

        if u_list:
            if not isinstance(u_list, list):
                u_list = [u_list]
            u_labels = [u[0] for u in u_list]
            u_labels, y_labels = self._validate_signals(u_labels, [])
        if y_list:
            if not isinstance(y_list, list):
                y_list = [y_list]
            y_labels = [y[0] for y in y_list]
            u_labels, y_labels = self._validate_signals([], y_labels)
        if u_list and y_list:
            u_labels = [u[0] for u in u_list]
            y_labels = [y[0] for y in y_list]
            u_labels, y_labels = self._validate_signals(u_labels, y_labels)
        if not u_list and not y_list:
            raise TypeError(
                "At least one input or output list must be provided."
            )

        return u_labels, y_labels, u_list, y_list

    def _fix_sampling_periods(
        self,
        signal_list: list[Signal],
        target_sampling_period: float | None = None,
        verbosity: int = 0,
    ) -> tuple[list[Signal], list[str]]:
        # TODO: Implement some other re-sampling mechanism.
        # """
        # Resample the :py:class:`Signals <dymoval.dataset.Signal>` in the *signal_list*.

        # The signals are resampled either with the slowest sampling period as target,
        # or towards the sampling period specified by the *target_sampling_period*
        # parameter, if specified.

        # Nevertheless, signals whose sampling period is not a divisor of the
        # the target sampling period will not be resampled and a list with the names
        # of such signals is returned.

        # Parameters
        # ----------
        # signal_list :
        #     List of :py:class:`Signals <dymoval.dataset.Signal>` to be resampled.
        # target_sampling_period :
        #     Target sampling period.

        # Raises
        # ------
        # ValueError
        #     If the *target_sampling_period* value is not positive.

        # Returns
        # -------
        # resampled_signals:
        #     List of :py:class:`Signal <dymoval.dataset.Signal>` with adjusted
        #     sampling period.
        # excluded_signals:
        #     List of signal names that could not be resampled.
        # """
        # ===========================================================
        # arguments Validation
        #
        if target_sampling_period is not None:
            if (
                not isinstance(target_sampling_period, float)
                or target_sampling_period < 0
            ):
                raise ValueError("'target_sampling_period' must be positive.")
        # ==========================================================

        # Initialization
        excluded_signals: list[str] = []

        # Downsample to the slowest period if target_sampling_period
        # is not given.
        if target_sampling_period is None:
            sampling_periods = [s["sampling_period"] for s in signal_list]
            target_sampling_period = max(sampling_periods)
            if verbosity != 0:
                print(f"target_sampling_period = {target_sampling_period}")

        for s in signal_list:
            N = target_sampling_period / s["sampling_period"]
            # Check if N is integer
            if np.isclose(N, round(N)):
                s["values"] = s["values"][:: int(N)]
                s["sampling_period"] = target_sampling_period
            else:
                excluded_signals.append(s["name"])

        resampled_signals = [
            s for s in list(signal_list) if not (s["name"] in excluded_signals)
        ]
        if verbosity != 0:
            print(
                "\nre-sampled signals =",
                f"{[s['name'] for s in resampled_signals]}",
            )
            print(
                "excluded signals from dataset ="
                f"{[s for s in excluded_signals]}"
            )
            print(f"actual sampling period = {target_sampling_period}")

        return resampled_signals, excluded_signals

    def get_dataset_values(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return the dataset values as a tuple of *numpy ndarrays* corresponding
        to the tuple *(t,u,y)*.

        Returns
        -------
        t:
            The dataset time interval.
        u:
            The values of the input signal.
        y:
            The values of the output signal.
        """

        # If there is a scalar input or output to avoid returning a column vector.
        if len(self.dataset["INPUT"].columns) == 1:
            u_values = (
                self.dataset["INPUT"].to_numpy().round(NUM_DECIMALS)[:, 0]
            )
        else:
            u_values = self.dataset["INPUT"].to_numpy().round(NUM_DECIMALS)

        if len(self.dataset["OUTPUT"].columns) == 1:
            y_values = (
                self.dataset["OUTPUT"].to_numpy().round(NUM_DECIMALS)[:, 0]
            )
        else:
            y_values = self.dataset["OUTPUT"].to_numpy().round(NUM_DECIMALS)

        return (
            self.dataset.index.to_numpy().round(NUM_DECIMALS),
            u_values,
            y_values,
        )

    def export_to_mat(self, filename: str) -> None:  # pragma: no cover
        # This function just uses scipy.io.savemat, not so much to test here
        """
        Write the dataset in a *.mat* file.

        Parameters
        ----------
        filename:
            Target filename. The extension *.mat* is automatically appended.
        """

        (t, u, y) = self.get_dataset_values()
        u_labels = list(self.dataset["INPUT"].columns)
        y_labels = list(self.dataset["OUTPUT"].columns)

        u_dict = {
            "INPUT": {
                s: self.dataset["INPUT"].loc[:, s].to_numpy() for s in u_labels
            }
        }
        y_dict = {
            "OUTPUT": {
                s: self.dataset["OUTPUT"].loc[:, s].to_numpy() for s in y_labels
            }
        }
        time = {"TIME": t}
        dsdict = time | u_dict | y_dict

        io.savemat(filename, dsdict, oned_as="column", appendmat=True)

    def get_signal_list(self) -> list[tuple[str, str]]:
        """Return the list of signal names of the dataset."""
        return list(self.dataset.columns)

    #     def rename_signals(self, name_map: dic[str, str]) -> Dataset:
    #
    #         # name_map = {name[1]:name[1]+"_pi" for name in ds.get_signal_list()}
    #         ds = deepcopy(self)
    #         ds.dataset = ds.dataset.rename(
    #             columns=name_map, level=1, errors="raise"
    #         )
    #
    #         # Update other references for signals name
    #         for key in ds._nan_intervals["INPUT"].keys():
    #             ds._nan_intervals["INPUT"][name_map[key]] = ds._nan_intervals[
    #                 "INPUT"
    #             ].pop(key)
    #
    #         for key in ds._nan_intervals["OUTPUT"].keys():
    #             ds._nan_intervals["OUTPUT"][name_map[key]] = ds._nan_intervals[
    #                 "OUTPUT"
    #             ].pop(key)
    #
    #         return ds

    def plot(
        self,
        # Only positional arguments
        /,
        # Positional or key-word arguments
        overlap: bool = False,
        line_color_input: str = "b",
        linestyle_input: str = "-",
        alpha_input: float = 1.0,
        line_color_output: str = "g",
        linestyle_output: str = "-",
        alpha_output: float = 1.0,
        ax: matplotlib.axes.Axes | None = None,
        *,
        # Only key-word arguments
        u_labels: str | list[str] = [],
        y_labels: str | list[str] = [],
        save_as: str | None = None,
    ) -> matplotlib.axes.Axes:
        # -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        """Plot the Dataset.

        Possible values for the parameters describing the line used in the plot
        (e.g. *line_color_input* , *alpha_output*. etc).
        are the same for the corresponding plot function in matplotlib.

        Parameters
        ----------
        u_labels:
            List of input signals.
        y_labels:
            List of output signals.
        overlap:
            If true *True* overlaps the input and the output signals plots.
        line_color_input:
            Line color for the input signals.
        linestyle_input:
            Line style for the input signals.
        alpha_input:
            Alpha channel value for the input signals.
        line_color_output:
            Line color for the output signals.
        linestyle_output:
            Line style for the output signals.
        alpha_output:
            Alpha channel value for the output signals.
        ax:
            Matplotlib Axes where to place the plot.
        save_as:
            Save the figure with a specified name.
            The figure is automatically resized with a 16:9 aspect ratio.
            You must specify the complete *filename*, including the path.
        """
        # Validation
        u_labels, y_labels = self._validate_signals(u_labels, y_labels)

        # df points to self.dataset.
        df = self.dataset

        # Input-output length
        p = len(u_labels)
        q = len(y_labels)

        if overlap:
            n = max(p, q)
            range_out = np.arange(0, q)

            # Adjust titles
            m = min(p, q)
            titles = ["IN/OUT #" + str(ii + 1) for ii in range(m)]
            u_titles_trail = ["INPUT #" + str(ii + 1) for ii in range(m, p)]
            y_titles_trail = ["OUTPUT #" + str(ii + 1) for ii in range(m, q)]
        else:
            n = p + q
            range_out = np.arange(p, p + q)

        # Find nrwos and ncols for the plot
        nrows, ncols = factorize(n)  # noqa

        # Overwrite axes if you want the plot to be on the figure instantiated by the caller.
        if ax is None:  # Create new figure and axes
            fig, axes = plt.subplots(nrows, ncols, sharex=True, squeeze=False)
        else:  # Otherwise use what is passed
            axes = np.asarray(ax)

        # Flatten array for more readable code
        axes = axes.T.flat

        if u_labels:
            df["INPUT"].loc[:, u_labels].plot(
                subplots=True,
                grid=True,
                color=line_color_input,
                linestyle=linestyle_input,
                alpha=alpha_input,
                title=titles + u_titles_trail,
                ax=axes[0:p],
            )

            self._shade_input_nans(
                self.dataset,
                self._nan_intervals,
                axes[0:p],
                u_labels,
                color=line_color_input,
            )

        if y_labels:
            df["OUTPUT"].loc[:, y_labels].plot(
                subplots=True,
                grid=True,
                color=line_color_output,
                linestyle=linestyle_output,
                alpha=alpha_output,
                title=titles + y_titles_trail,
                ax=axes[range_out],
            )

            self._shade_output_nans(
                self.dataset,
                self._nan_intervals,
                axes[range_out],
                y_labels,
                color=line_color_output,
            )

        for ii in range(ncols):
            axes[nrows - 1 :: nrows][ii].set_xlabel(df.index.name)
        plt.suptitle(f"Dataset {self.name}. ")

        # Eventually save and return figures.
        if save_as is not None and ax is None:
            # Keep 16:9 ratio
            height = 2.5
            width = 1.778 * height
            fig.set_size_inches(ncols * width, nrows * height)
            save_plot_as(fig, save_as)  # noqa

        # return stuff
        #  if ax is None:
        #      return fig
        #  else:
        #      return None
        return axes

    def plot_coverage(
        self,
        nbins: int = 100,
        line_color_input: str = "b",
        alpha_input: float = 1.0,
        line_color_output: str = "g",
        alpha_output: float = 1.0,
        *,
        u_labels: str | list[str] = [],
        y_labels: str | list[str] = [],
        save_as: str | None = None,
    ) -> tuple[
        matplotlib.figure.Figure,
        matplotlib.axes.Axes,
        matplotlib.figure.Figure | None,
        matplotlib.axes.Axes | None,
    ]:
        """
        Plot the dataset coverage as histograms.


        Parameters
        ----------
        u_labels:
            List of input signals.
        y_labels:
            List of output signals.
        nbins:
            The number of bins.
        line_color_input:
            Line color for the input signals.
        alpha_input:
            Alpha channel value for the input signals.
        line_color_output:
            Line color for the output signals.
        alpha_output:
            Alpha channel value for the output signals.
        save as:
            Save the figures with a specified name.
            It appends the suffix *_in* and *_out* to the input and output figure,
            respectively.
            The *filename* shall include the path.
        """
        # Extract dataset
        df = self.dataset

        u_labels, y_labels = self._validate_signals(u_labels, y_labels)

        if u_labels:
            p = len(u_labels)
            titles = ["INPUT #" + ii for ii in range(p)]
            nrows_in, ncols_in = factorize(p)  # noqa
            fig_in, axes_in = plt.subplots(nrows_in, ncols_in, squeeze=False)
            axes_in = axes_in.flat
            df.loc[:, ("INPUT", df["INPUT"].columns)].hist(
                grid=True,
                bins=nbins,
                color=line_color_input,
                alpha=alpha_input,
                title=titles,
                ax=axes_in[0:p],
            )

            for ii in range(p):
                axes_in[ii].set_xlabel(u_labels[ii][1])
            plt.suptitle("Coverage region.")

        if y_labels:
            q = len(y_labels)
            nrows_out, ncols_out = factorize(q)  # noqa
            fig_out, axes_out = plt.subplots(
                nrows_out, ncols_out, squeeze=False
            )
            axes_out = axes_out.flat
            df.loc[:, ("OUTPUT", df["OUTPUT"].columns)].hist(
                grid=True,
                bins=nbins,
                color=line_color_output,
                alpha=alpha_output,
                ax=axes_out[0:q],
            )

            for ii in range(q):
                axes_out[ii].set_xlabel(y_labels[ii][1])
            plt.suptitle("Coverage region.")

        if save_as is not None:
            # Keep 16:9 ratio
            # TODO Move height to the config file
            height = 2.5
            width = 1.778 * height

            fig_in.set_size_inches(ncols_in * width, nrows_in * height)
            save_plot_as(fig_in, save_as + "_in")  # noqa

            fig_out.set_size_inches(ncols_out * width, nrows_out * height)
            save_plot_as(fig_out, save_as + "_out")  # noqa

        # Return
        if u_labels and y_labels:
            return fig_in, axes_in, fig_out, axes_out

        elif u_labels and not y_labels:
            return fig_in, axes_in, None, None
        else:  # The only option left is not u_labels and y_labels
            return None, None, fig_out, axes_out

    def fft(
        self,
        *,
        u_labels: str | list[str] = [],
        y_labels: str | list[str] = [],
    ) -> pd.DataFrame:
        """Return the FFT of the dataset as pandas DataFrame.

        It only works with real-valued signals.

        Parameters
        ----------
        u_labels:
            List of input signal included in the FFT transform.
        y_labels:
            List of output signal included in the FFT transform.


        Raises
        ------
        ValueError
            If the dataset contains *NaN*:s
        """
        # Validation
        u_labels, y_labels = self._validate_signals(u_labels, y_labels)
        # Remove 'INPUT' 'OUTPUT' columns level from dataframe
        df_temp = self.dataset.droplevel(level=0, axis=1)

        # Check if there are any NaN:s
        if df_temp.isna().any(axis=None):
            raise ValueError(
                f"Dataset '{self.name}' contains NaN:s. I Cannot compute the FFT."
            )

        # Compute FFT. All the input signals are real (dataset only contains float)
        # We normalize the fft with N to secure energy balance (Parseval's Theorem),
        # namely it must hold "int_T x(t) = int_F X(f)".
        # See https://stackoverflow.com/questions/20165193/fft-normalization
        N = len(df_temp.index)

        vals = fft.rfftn(df_temp.loc[:, u_labels + y_labels], axes=0) / N

        # Create a new Dataframe
        u_extended_labels = list(zip(["INPUT"] * len(u_labels), u_labels))
        y_extended_labels = list(zip(["OUTPUT"] * len(y_labels), y_labels))
        cols = pd.MultiIndex.from_tuples(
            [*u_extended_labels, *y_extended_labels]
        )
        df_freq = pd.DataFrame(data=vals, columns=cols)
        df_freq.index.name = "Frequency"

        return df_freq

    def plot_spectrum(
        self,
        overlap: bool = False,
        line_color_input: str = "b",
        linestyle_input: str = "-",
        alpha_input: float = 1.0,
        line_color_output: str = "g",
        linestyle_output: str = "-",
        alpha_output: float = 1.0,
        *,
        kind: Literal["amplitude", "power", "psd"] = "power",
        u_labels: str | list[str] = [],
        y_labels: str | list[str] = [],
        save_as: str | None = None,
    ) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        """
        Plot the spectrum of the specified signals in the dataset in different format.

        If some signals have *NaN* values, then the FFT cannot be computed and
        an error is raised.


        Parameters
        ----------
        u_labels:
            List of input signals.
            If not specified, the FFT is performed over all the input signals.
        y_labels:
            List of output signals.
            If not specified, the FFT is performed over all the output signals.
        overlap:
            If true it overlaps the input and the output signals plots.
        kind:

            - *amplitude* plot both the amplitude and phase spectrum.
              If the signal has unit V, then the amplitude has unit *V*.
              Phase is in radians.
            - *power* plot the autopower spectrum.
              If the signal has unit V, then the amplitude has unit *V^2*.
            - *psd* plot the power density spectrum.
              If the signal has unit V and the time is *s*, then the amplitude has unit *V^2/Hz*.

        line_color_input:
            Line color for the input signals.
        linestyle_input:
            Line style for the input signals.
        alpha_input:
            Alpha channel value for the input signals.
        line_color_output:
            Line color for the output signals.
        linestyle_output:
            Line style for the output signals.
        alpha_output:
            Alpha channel value for the output signals.
        save_as:
            Save the figure with a specified name.
            The figure is automatically resized with a 16:9 aspect ratio.
            You must specify the complete *filename*, including the path.


        Raises
        ------
        ValueError
            If *kind* doen not match any possible values.
        """
        # validation
        u_labels, y_labels = self._validate_signals(u_labels, y_labels)

        allowed_kind = ["amplitude", "power", "psd"]
        if kind not in allowed_kind:
            raise ValueError(f"kind must be one of {allowed_kind}")

        # Input-output lengths.
        # If we want to plot abs and phase, then we need to double
        # the number of axes in the figure
        if u_labels:
            p = 2 * len(u_labels) if kind == "amplitude" else len(u_labels)
        else:
            p = 1

        if y_labels:
            q = 2 * len(y_labels) if kind == "amplitude" else len(y_labels)
        else:
            q = 1
        # Compute FFT.
        # For real signals, the spectrum is Hermitian anti-simmetric, i.e.
        # the amplitude is symmetric wrt f=0 and the phase is antisymmetric wrt f=0.
        # See e.g. https://ccrma.stanford.edu/~jos/ReviewFourier/Symmetries_Real_Signals.html
        df_freq = self.fft(u_labels=u_labels, y_labels=y_labels)
        # Frequency and number of samples
        Ts = self.sampling_period
        N = len(self.dataset.index)  # number of samples
        # Compute frequency bins
        f_bins = fft.rfftfreq(N, Ts)
        # Update DataFame index.
        # NOTE: I had to use pd.Index to preserve the name and being able to
        # replace the values
        df_freq.index = pd.Index(f_bins, name=df_freq.index.name)

        # Switch between the kind
        if kind == "amplitude":
            # Add another level to specify abs and phase
            df_freq = df_freq.agg([np.abs, np.angle])

        elif kind == "power":
            df_freq = df_freq.abs() ** 2
            # We take half spectrum, so for conserving the energy we must consider
            # also the negative frequencies with the exception of the DC compontent
            # because that is not mirrored. This is why we multiply by 2.
            # The same happens for the psd.
            df_freq[1:-1] = 2 * df_freq[1:-1]
        elif kind == "psd":
            Delta_f = 1 / (Ts * N)  # Size of each frequency bin
            df_freq = df_freq.abs() ** 2 / Delta_f
            df_freq[1:-1] = 2 * df_freq[1:-1]

        # Start plot ritual
        if overlap:
            n = max(p, q)
            range_out = np.arange(0, q)
        else:
            n = p + q
            range_out = np.arange(p, p + q)
        nrows, ncols = factorize(n)  # noqa

        # To have the phase plot below the abs, the number of rows must be an
        # even number, otherwise the plot got screwed.
        if kind == "amplitude":
            if np.mod(nrows, 2) != 0:
                nrows -= 1
                ncols += int(np.ceil(nrows / ncols))

        fig, axes = plt.subplots(nrows, ncols, sharex=True, squeeze=False)
        axes = axes.T.flat

        if u_labels:
            df_freq.loc[:, ("INPUT", df_freq["INPUT"].columns)].plot(
                subplots=True,
                grid=True,
                color=line_color_input,
                linestyle=linestyle_input,
                alpha=alpha_input,
                legend=u_labels,
                ax=axes[0:p],
            )

        if y_labels:
            df_freq.loc[:, ("OUTPUT", df_freq["OUTPUT"].columns)].plot(
                subplots=True,
                grid=True,
                color=line_color_output,
                linestyle=linestyle_output,
                alpha=alpha_output,
                legend=y_labels,
                ax=axes[range_out],
            )

        for ii in range(ncols):
            axes[nrows - 1 :: nrows][ii].set_xlabel(df_freq.index.name)

        plt.suptitle(f"{kind.capitalize()} spectrum.")

        # Save and return
        if save_as is not None:
            # Keep 16:9 ratio
            height = 2.5
            width = 1.778 * height

            fig.set_size_inches(ncols * width, nrows * height)
            save_plot_as(fig, save_as)

        return fig, axes

    def remove_means(
        self,
        *,
        u_labels: str | list[str] = [],
        y_labels: str | list[str] = [],
    ) -> Dataset:
        """
        Remove the mean value to the specified signals.

        Parameters
        ----------
        u_labels :
            List of the input signal names.
            If not specified, then the mean value is removed to all
            the input signals in the dataset.

        y_labels :
            List of the output signal names.
            If not specified, then the mean value is removed to all
            the output signals in the dataset.
        """
        # Arguments validation
        u_labels, y_labels = self._validate_signals(u_labels, y_labels)

        # Safe copy
        ds_temp = deepcopy(self)
        df_temp = ds_temp.dataset

        # Remove means from input signals
        cols = list(zip(len(u_labels) * ["INPUT"], u_labels))
        df_temp.loc[:, cols] = (
            self.dataset.loc[:, cols] - self.dataset.loc[:, cols].mean()
        )

        # Remove means from output signals
        cols = list(zip(len(y_labels) * ["OUTPUT"], y_labels))
        df_temp.loc[:, cols] = (
            self.dataset.loc[:, cols] - self.dataset.loc[:, cols].mean()
        )

        # round result
        df_temp.round(decimals=NUM_DECIMALS)

        return ds_temp

    @overload
    def remove_offset(
        self,
        *,
        u_list: list[tuple[str, float]] | tuple[str, float],
        y_list: list[tuple[str, float]] | tuple[str, float] = [],
    ) -> Dataset:
        ...  # pragma: no cover

    @overload
    def remove_offset(
        self,
        *,
        u_list: list[tuple[str, float]] | tuple[str, float] = [],
        y_list: list[tuple[str, float]] | tuple[str, float],
    ) -> Dataset:
        ...  # pragma: no cover

    def remove_offset(
        self,
        *,
        u_list: list[tuple[str, float]] | tuple[str, float] = [],
        y_list: list[tuple[str, float]] | tuple[str, float] = [],
    ) -> Dataset:
        # At least one argument shall be passed.
        # This is the reason why the @overload is added in that way
        """
        Remove a specified offset to the list of specified signals.

        For each target signal a tuple of the form *(name,value)*
        shall be passed.
        The value specified in the *offset* parameter is removed from
        the signal with name *name*.

        For multiple signals, the tuples shall be arranged in a list.

        Parameters
        ----------
        u_list:
            List of tuples of the form *(name, offset)*.
            The *name* parameter must match the name of any input signal stored
            in the dataset.
        y_list:
            List of tuples of the form *(name, offset)*.
            The *name* parameter must match the name of any output signal stored
            in the dataset.

        Raises
        ------
        TypeError
            If no arguments are passed.
        """

        # Safe copy
        ds_temp = deepcopy(self)
        df_temp = ds_temp.dataset

        # Validate passed arguments
        (
            u_labels,
            y_labels,
            u_list,
            y_list,
        ) = self._validate_name_value_tuples(u_list=u_list, y_list=y_list)

        # First adjust the input columns
        if u_list:
            u_offset = [u[1] for u in u_list]
            cols = list(zip(len(u_labels) * ["INPUT"], u_labels))

            df_temp.loc[:, cols] = self.dataset.loc[:, cols].apply(
                lambda x: x.subtract(u_offset), axis=1
            )

        # Then adjust the output columns
        if y_list:
            y_offset = [y[1] for y in y_list]
            cols = list(zip(len(y_labels) * ["OUTPUT"], y_labels))
            df_temp.loc[:, cols] = self.dataset.loc[:, cols].apply(
                lambda x: x.subtract(y_offset), axis=1
            )

        df_temp.round(NUM_DECIMALS)

        return ds_temp

    @overload
    def low_pass_filter(
        self,
        *,
        u_list: list[tuple[str, float]] | tuple[str, float],
        y_list: list[tuple[str, float]] | tuple[str, float] = [],
    ) -> Dataset:
        ...  # pragma: no cover

    @overload
    def low_pass_filter(
        self,
        *,
        u_list: list[tuple[str, float]] | tuple[str, float] = [],
        y_list: list[tuple[str, float]] | tuple[str, float],
    ) -> Dataset:
        ...  # pragma: no cover

    def low_pass_filter(
        self,
        *,
        u_list: list[tuple[str, float]] | tuple[str, float] = [],
        y_list: list[tuple[str, float]] | tuple[str, float] = [],
    ) -> Dataset:
        """
        Low-pass filter a list of specified signals.

        For each target signal a tuple of the form *(name, cutoff_frequency)*
        shall be passed as parameter.
        For multiple signals, the tuples shall be arranged in a list.

        The low-pass filter is first-order IIR filter.

        Parameters
        ----------
        u_list:
            List of tuples of the form *(name, cutoff_frequency)*.
            The *name* parameter must match the name of any input signal stored
            in the dataset.
            Such a signal is filtered with a low-pass
            filter whose cutoff frequency is specified by the *cutoff_frequency*
            parameter.
        y_list:
            List of tuples of the form *(name, cutoff_frequency)*.
            The *name* parameter must match the name of any output signal stored
            in the dataset.
            Such a signal is filtered with a low-pass
            filter whose cutoff frequency is specified by the *cutoff_frequency*
            parameter.

        Raises
        ------
        TypeError
            If no arguments are passed.
        """
        # Safe copy
        ds_temp = deepcopy(self)
        df_temp = ds_temp.dataset

        # Validate passed arguments
        (
            u_labels,
            y_labels,
            u_list,
            y_list,
        ) = self._validate_name_value_tuples(u_list=u_list, y_list=y_list)

        # Sampling frequency
        fs = 1 / (self.dataset.index[1] - self.dataset.index[0])
        N = len(self.dataset.index)

        # List of all the requested input cutoff frequencies
        # INPUT
        if u_list:
            u_fc = [u[1] for u in u_list]
            for ii, u in enumerate(u_labels):
                # Low-pass filter implementatiom
                fc = u_fc[ii]
                u_filt = df_temp[("INPUT", u)].to_numpy()
                y_filt = np.zeros(N)
                y_filt[0]
                for kk in range(0, N - 1):
                    y_filt[kk + 1] = (1.0 - fc / fs) * y_filt[kk] + (
                        fc / fs
                    ) * u_filt[kk]
                df_temp.loc[:, ("INPUT", u)] = y_filt
        # OUTPUT
        # List of all the requested input cutoff frequencies
        if y_list:
            y_fc = [y[1] for y in y_list]
            for ii, y in enumerate(y_labels):
                fc = y_fc[ii]  # cutoff frequency
                # Low-pass filter implementatiom
                u_filt = df_temp[("OUTPUT", y)].to_numpy()
                y_filt = np.zeros(N)
                y_filt[0]
                for kk in range(0, N - 1):
                    y_filt[kk + 1] = (1.0 - fc / fs) * y_filt[kk] + (
                        fc / fs
                    ) * u_filt[kk]
                df_temp.loc[:, ("OUTPUT", y)] = y_filt
        # Round value
        df_temp = np.round(self.dataset, NUM_DECIMALS)  # noqa

        return ds_temp

    # def filter(self) -> Dataset:
    #     """To be implemented!"""
    #     print("Not implemented yet!")

    def replace_NaNs(
        self,
        method: Literal["interpolate", "fillna"] = "interpolate",
        fill_value: float = 0.0,
    ) -> Dataset:
        """Replace NaN:s in the dataset.


        Parameters
        ----------
        method :
            Interpolation method.
        fill_value :
            When *method* = "fillna", then the *NaN*:s vales are filled with this value.

        Raises
        ------
        ValueError
            If the passed method is not 'interpolate' or 'fillna'.
        """

        # Safe copy
        ds_temp = deepcopy(self)

        if method == "interpolate":
            ds_temp.dataset = ds_temp.dataset.interpolate()
        elif method == "fillna":
            ds_temp.dataset = ds_temp.dataset.fillna(fill_value)
        else:
            raise ValueError(
                "Unknown method. Choose between 'interpolate' or 'fillna'"
            )

        return ds_temp


# ====================================================
# Useful functions
# ====================================================


def validate_signals(signal_list: list[Signal]) -> None:
    """
    Perform a number of checks to verify that the passed
    list of :py:class:`Signals <dymoval.dataset.Dataset>`
    can be used to create a Dataset.

    Every :py:class:`Signal <dymoval.dataset.Signal>` in the *signal_list*
    parameter must have all the attributes adequately set.

    To figure how the attributes shall be set, look at the *RAISES* section below.

    Parameters
    ----------
    signal_list :
        List of signal to be checked.

    Raises
    ------
    ValueError
        If signal names are not unique, or if values is not a *1-D numpy ndarray*,
        or if sampling period must positive.
    KeyError
        If signal attributes are not found or not allowed.
    IndexError
        If signal have less than two samples.
    TypeError
        If values is not a *1-D numpy ndarray*, or if sampling period is not a *float*.
    """

    # Name unicity
    signal_names = [s["name"] for s in signal_list]
    if len(signal_names) > len(set(signal_names)):
        raise ValueError("Signal names are not unique")
    #
    ALLOWED_KEYS = Signal.__required_keys__
    for s in signal_list:
        keys = s.keys()
        #
        # Existence
        not_found_keys = difference_lists_of_str(
            list(ALLOWED_KEYS), list(keys)
        )  # noqa
        if not_found_keys:
            raise KeyError(
                f"Key {not_found_keys} not found in signal {s['name']}."
            )
        for key in keys:
            if key == "values":
                cond = (
                    not isinstance(s[key], np.ndarray) or s[key].ndim != 1  # type: ignore
                )
                if cond:
                    raise TypeError("Key {key} must be 1-D numpy array'.")
                if s[key].size < 2:  # type: ignore
                    raise IndexError(
                        "Signal {s[name']} has only one sample.",
                        "A signal must have at least two samples.",
                    )
            if key == "sampling_period":
                if not isinstance(s[key], float):  # type: ignore
                    raise TypeError(
                        "Key 'sampling_period' must be a positive float."
                    )
                if s[key] < 0.0 or np.isclose(s[key], 0.0):  # type: ignore
                    raise ValueError(
                        "Key 'sampling_period' must be a positive float."
                    )


def dataframe_validation(
    df: pd.DataFrame,
    u_labels: str | list[str],
    y_labels: str | list[str],
) -> None:
    """
    Check if a pandas Dataframe is suitable for instantiating
    a :py:class:`Dataset <dymoval.dataset.Dataset>` object.

    When the signals are sampled with the same period, then they can be arranged
    in a pandas DataFrame, where the index represents the common time vector
    and each column represent a signal.

    Once the signals are arranged in a DataFrame,
    it must be specified which signal(s) are the input through the *u_labels* and
    which signal(s) is the output through the  *y_labels* parameters.

    Furthermore, the candidate DataFrame shall meet the following requirements

    - only one index and columns levels are allowed (no *MultiIndex*),
    - each column shall correspond to one signal,
    - the column names must be unique,
    - each signal must have at least two samples (i.e. the DataFrame has at least two rows),
    - both the index values and the column values must be *float*,
    - each signal in the *u_labels* and *y_labels* must be equal to exactly, one column name of the input DataFrame

    Parameters
    ----------
    df :
        DataFrame to be validated.
    u_labels :
        List of input signal names.
    y_labels :
        List of output signal names.

    Raises
    ------
    IndexError
        If there is not at least one input and output signal,
        or if there is any signal that don't have at least two samples,
        or if there is any *MultiIndex* index.
    ValueError
        If the signal names are not unique,
        or if the DataFrame *Index* (time vector) values are not monotonically increasing,
        or if some input or output signal names don't correspond to any
        DataFrame column name.
    TypeError
        If the DataFrame values are not all *float*.
    """
    # ===================================================================
    # Checks performed:
    # 1. There is at least one input and one output,
    # 2. Signals names are unique,
    # 3. The DataFrame must have only one index and columns levels,
    # 4. Signals shall have at least two samples.
    # 5. Input and output names exist in the passed DataFrame,
    # 6. The index is made of monotonically increasing floats,
    # 7. The datatype of the DataFrame's elements are float.
    # ===================================================================

    u_labels = str2list(u_labels)  # noqa
    y_labels = str2list(y_labels)  # noqa

    # 1. Check that you have at least one input and one output
    if not u_labels or not y_labels:
        raise IndexError(
            "You need at least one input and one output signal."
            "Check 'u_labels' and 'y_labels'."
        )

    # 2. Check unicity of signal names
    if (
        len(u_labels) != len(set(u_labels))
        or len(y_labels) != len(set(y_labels))
        or (set(u_labels) & set(y_labels))  # Non empty intersection
    ):
        raise ValueError(
            "Signal names must be unique." "Check 'u_labels' and 'y_labels'."
        )

    # 3. The DataFrame must have only one index and columns levels
    if df.columns.nlevels > 1 or df.index.nlevels > 1:
        raise IndexError(
            "The number index levels must be one for both the index and the columns.",
            "The index shall represent a time vector of equi-distant time instants",
            "and each column shall correspond to one signal values.",
        )

    # 4. At least two samples
    if df.index.size < 2:
        raise IndexError("A signal needs at least two samples.")

    # 5. Check if u_labels and y_labels exist in the passed DataFrame
    input_not_found = difference_lists_of_str(
        u_labels, list(df.columns)
    )  # noqa
    if input_not_found:
        raise ValueError(f"Input(s) {input_not_found} not found.")
    output_not_found = difference_lists_of_str(
        y_labels, list(df.columns)
    )  # noqa
    # Check output
    if output_not_found:
        raise ValueError(f"Output(s) {output_not_found} not found.")

    # 6. The index is a 1D vector of monotonically increasing floats.
    # OBS! Builtin methds df.index.is_monotonic_increasing combined with
    # df.index.is_unique won't work due to floats.
    sampling_period = df.index[1] - df.index[0]
    if not np.all(np.isclose(np.diff(df.index), sampling_period)):
        raise ValueError(
            "Index must be a 1D vector of monotonically",
            "equi-spaced, increasing floats.",
        )

    # 7. The dataframe elements are all floats
    if not all(df.dtypes == float):
        raise TypeError("Elements of the DataFrame must be float.")


def plot_signals(
    signal_list: list[Signal],
    u_labels: str | list[str] = [],
    y_labels: str | list[str] = [],
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """
    Plot the :py:class:`Signals <dymoval.dataset.Signal>` of a signal list.

    Parameters
    ----------
    signal_list :
        List of :py:class:`Dataset <dymoval.dataset.Dataset>`.
    u_labels :
        Used for specifying which signals shall be considered as input.
    y_labels :
        Used for specifying which signals shall be considered as output.

    Raises
    ------
    KeyError
        If there is any label specified in *u_labels* or *y_labels* that does not
        correspond to any :py:class:`Dataset <dymoval.dataset.Dataset>` name.
    """
    # Validate signals first
    validate_signals(signal_list)
    signal_names = [s["name"] for s in signal_list]
    if u_labels:
        u_labels = str2list(u_labels)  # noqa
        inputs_not_found = difference_lists_of_str(
            u_labels, signal_names
        )  # noqa
        if inputs_not_found:
            raise KeyError(
                f"Signal(s) {inputs_not_found} not in the signals list. "
                f"Available signals are {signal_names}."
            )
    if y_labels:
        y_labels = str2list(y_labels)  # noqa
        outputs_not_found = difference_lists_of_str(
            y_labels, signal_names
        )  # noqa
        if outputs_not_found:
            raise KeyError(
                f"Signal(s) {outputs_not_found} not in the signals list. "
                f"Available signals are {signal_names}."
            )
    # Plot raw signals
    n = len(signal_list)
    nrows, ncols = factorize(n)  # noqa
    fig, ax = plt.subplots(nrows, ncols, squeeze=False, sharex=True)
    ax = ax.T.flat
    for ii, s in enumerate(signal_list):
        timeline = np.linspace(
            0.0, len(s["values"]) * s["sampling_period"], len(s["values"])
        )
        if u_labels and s["name"] in u_labels:
            line_color = "blue"
            sig_label = (s["name"], "input")
        elif y_labels and s["name"] in y_labels:
            line_color = "green"
            sig_label = (s["name"], "output")
        else:
            line_color = "k"
            sig_label = (s["name"], "")
        ax[ii].plot(timeline, s["values"], label=sig_label, color=line_color)
        ax[ii].text(
            0.8,
            0.8,
            f"$T_s$ = {s['sampling_period']} s",
            bbox=dict(facecolor="yellow", alpha=0.8),
        )
        ax[ii].grid()

        # Write time only in the last row
        if ii >= (nrows - 1) * ncols:
            ax[ii].set_xlabel("Time")
        ax[ii].legend()
    plt.suptitle("Raw signals.")

    return fig, ax


def compare_datasets(*datasets: Dataset, target: str = "all") -> None:
    # arguments validation
    for ds in datasets:
        if not isinstance(ds, Dataset):
            raise TypeError("Input must be a dymoval Dataset type.")

    ALLOWED_TARGETS = ["time", "freq", "cov", "all"]
    if target not in ALLOWED_TARGETS:
        raise ValueError(
            f"Target {target} not valid. Allowed targets are {ALLOWED_TARGETS}"
        )

    # Get size of wider dataset
    n = max(
        [len(ds.dataset.droplevel(level=0, axis=1).columns) for ds in datasets]
    )
    nrows, ncols = factorize(n)

    # Create a unified figure
    fig, ax = plt.subplots(nrows, ncols, sharex=True, squeeze=False)

    cmap = plt.get_cmap(COLORMAP)  # noqa
    # cols = []
    # axes = np.zeros(n).flat
    for ii, ds in enumerate(datasets):
        axes = ds.plot(
            line_color_input=cmap(ii), line_color_output=cmap(ii), ax=ax
        )

    # Adjust legend
    ds_names = [ds.name for ds in datasets]
    for ii, ax in enumerate(axes):
        handles, labels = ax.get_legend_handles_labels()
        # Adjust legend
        if labels:
            new_labels = [
                ds_names[jj] + ", " + labels[jj]
                for jj, _ in enumerate(datasets)
            ]
        ax.legend(handles, new_labels)

    fig.suptitle(f"Dataset comparison")

    return axes
    # Adjust legends
    # print("legend = ", legend)
    # print("legend[0] = ", legend[0])

    #      for ii in range(p):
    #         df_ax[ii].legend([legend[ii]])

    # for ii,ds in enumerate(datasets):
    # df["INPUT"].plot(subplots=True, grid=True, color="b", ax=axes[0:p])


# def analyze_inout_dataset(df):
# -	Remove trends. Cannot do.
# -	(Resample). One line of code with pandas
