# The following are only for Spyder, otherwise things are written in
# the pyproject.toml
# mypy: show_error_codes

""" Module containing everything related to datasets.
Here are defined special datatypes, classes and auxiliary functions to deal with datasets.
"""
from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec
import itertools
from matplotlib import pyplot as plt
from scipy import io, fft
from .config import *  # noqa
from .utils import *  # noqa, Type
from typing import TypedDict, Any, Literal
from copy import deepcopy

# from itertools import product


class Signal(TypedDict):
    """*Signals* are used to represent real-world measurements.

    They are used to instantiate :py:class:`Dataset <dymoval.dataset.Dataset>` objects.
    Before instantiating a :py:class:`Dataset <dymoval.dataset.Dataset>` object, it is good practice
    to validate Signals through the :py:meth:`~dymoval.dataset.validate_signals` function.

    Although Signals have compulsory attribtues, there is freedom
    to append additional ones.


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
    # This is the reason why they are reported in the docstring.
    signal_unit: str  #: Signal unit.
    sampling_period: float  #: Signal sampling period.
    time_unit: str  #: Signal sampling period.


class Dataset:
    """The *Dataset* class stores the candidate signals to be used as a dataset
    and it provides methods for analyzing and manipulating them.

    A *Signal* list shall be passed to the initializer along with two lists of labels
    specifying which signal(s) shall be consiedered
    as input and which signal(s) shall be considered as output.

    The signals list can be either a list
    of dymoval :py:class:`Signals <dymoval.dataset.Signal>` type or a
    pandas DataFrame with a well-defined structure.
    See :py:meth:`~dymoval.dataset.validate_signals` and
    :py:meth:`~dymoval.dataset.validate_dataframe` for more information.


    Parameters
    ----------
    name:
        Dataset name.
    signal_list :
        Signals to be included in the :py:class:`Dataset <dymoval.dataset.Dataset>`.
        See :py:meth:`~dymoval.dataset.validate_signals` and
        :py:meth:`~dymoval.dataset.validate_dataframe` to figure out how
        the list of :py:class:`Signal <dymoval.dataset.Signal>` or the pandas
        DataFrame representing the dataset signals shall look like.
    u_names :
        List of input signal names. Each signal name must be unique and must be
        contained in the signal_list.
    y_names :
        List of input signal names. Each signal name must be unique and must be
        contained in the signal_list.
    target_sampling_period :
        The passed signals will be re-sampled towards this target sampling period.
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
        The units of the outputs are displayed on the secondary y-axis.
    verbosity:
        Display information depending on its level.

    Raises
    -----
    TypeError
        If the signals_list has the wrong datatype.
    """

    def __init__(
        self,
        name: str,
        signal_list: list[Signal] | pd.DataFrame,
        u_names: str | list[str],
        y_names: str | list[str],
        target_sampling_period: float | None = None,
        tin: float | None = None,
        tout: float | None = None,
        full_time_interval: bool = False,
        overlap: bool = False,
        verbosity: int = 0,
    ) -> None:

        # Initialization by Signals.
        # It will call _new_dataset_from_dataframe
        if all(isinstance(x, dict) for x in signal_list):
            self._new_dataset_from_signals(
                name,
                signal_list,
                u_names,
                y_names,
                target_sampling_period,
                tin,
                tout,
                full_time_interval,
                overlap,
                verbosity,
            )

        # Initialization by pandas DataFrame
        # All the class attributes are initialized inside this function
        elif isinstance(signal_list, pd.DataFrame):
            self._new_dataset_from_dataframe(
                name,
                signal_list,
                u_names,
                y_names,
                tin,
                tout,
                full_time_interval,
                overlap,
                verbosity,
            )
        else:
            raise TypeError(
                "Input must be a Signal list or a pandas DataFrame type.",
            )

    def __str__(self) -> str:
        return f"Dymoval dataset called '{self.name}'."

    # ==============================================
    #   Class methods
    # ==============================================

    def _shade_nans(
        self,
        axes: matplotlib.axes.Axes,
    ) -> None:

        # Reference to self._nan_intervals
        NaN_intervals = self._nan_intervals

        # Unify left and right axes in a unique list
        axes_all = axes[0].get_shared_x_axes().get_siblings(axes[0])

        for ii, ax in enumerate(axes_all):
            lines, signal_names = ax.get_legend_handles_labels()
            for line, signal in zip(lines, signal_names):
                color = line.get_color()
                # Every signal may have a disjoint union of NaN intervals
                for _, val in enumerate(NaN_intervals[signal]):
                    # Shade only if there is a non-empty interval
                    if not val.size == 0:
                        ax.axvspan(min(val), max(val), color=color, alpha=0.2)

    def _find_dataset_coverage(
        self,
    ) -> tuple[pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]:

        df = self.dataset
        u_mean = df["INPUT"].mean(axis=0).round(NUM_DECIMALS)
        u_cov = df["INPUT"].cov().round(NUM_DECIMALS)
        y_mean = df["OUTPUT"].mean(axis=0).round(NUM_DECIMALS)
        y_cov = df["OUTPUT"].cov().round(NUM_DECIMALS)

        return u_mean, u_cov, y_mean, y_cov

    def _find_nan_intervals(
        self,
    ) -> dict[str, list[np.ndarray]]:
        # Find index intervals (i.e. time intervals) where columns values
        # are NaN.
        # It requires a dataset with extended columns (MultiIndex))
        df = self.dataset.droplevel(level=["kind", "units"], axis=1)
        sampling_period = self.sampling_period

        NaN_index = {}
        NaN_intervals = {}
        for s in list(df.columns):
            NaN_index[s] = df.loc[df[s].isnull().to_numpy()].index
            idx = np.where(
                ~np.isclose(np.diff(NaN_index[s]), sampling_period, atol=ATOL)
            )[0]
            NaN_intervals[s] = np.split(NaN_index[s], idx + 1)
        return NaN_intervals

    def _shift_dataset_tin_to_zero(
        self,
    ) -> None:
        # ===================================================================
        # Shift tin to zero.
        # ===================================================================
        # The values are already set due to the .loc[tin:tout] done elsewhere
        # You only need to shift the timestamps in all the time-related attributes
        tin = self.dataset.index[0]
        timeVectorFromZero = self.dataset.index - tin
        new_index = pd.Index(
            np.round(timeVectorFromZero, NUM_DECIMALS),
            name=self.dataset.index.name,
        )
        # Update the index
        self.dataset.index = new_index
        self.dataset = self.dataset.round(NUM_DECIMALS)

        # Shift also the NaN_intervals to tin = 0.0.
        # Create a reference to self._nan_intervals
        NaN_intervals = self._nan_intervals

        for k in NaN_intervals.keys():
            for idx, nan_chunk in enumerate(NaN_intervals[k]):
                nan_chunk_translated = nan_chunk - tin
                NaN_intervals[k][idx] = np.round(
                    nan_chunk_translated, NUM_DECIMALS  # noqa
                )
                NaN_intervals[k][idx] = nan_chunk_translated[
                    nan_chunk_translated >= 0.0
                ]

    def _new_dataset_from_dataframe(
        self,
        name: str,
        df: pd.DataFrame,
        u_names: str | list[str],
        y_names: str | list[str],
        tin: float | None = None,
        tout: float | None = None,
        full_time_interval: bool = False,
        overlap: bool = False,
        verbosity: int = 0,
        _excluded_signals: list[str] = [],
    ) -> None:

        # ==============================================================
        # All the class attributes are defined and initialized here
        #
        # self.name
        # self.dataset
        # self.coverage
        # self.information_level
        # self._nan_intervals
        # self.excluded_signals
        # self.sampling_period
        #
        # ==============================================================

        # Arguments validation
        if tin and tout and tin > tout:
            raise ValueError(
                f" Value of tin ( ={tin}) shall be smaller than the value of tout ( ={tout})."
            )
        # CHECK IF THE NAMES ARE IN THE AVAIL NAMES

        # NOTE: You have to use the #: to add a doc description
        # in class attributes (see sphinx.ext.autodoc)
        # Set easy-to-set attributes
        self.name: str = name  #: Dataset name.
        self.information_level: float = 0.0  #: *Not implemented yet!*
        self.sampling_period: float = np.round(
            df.index[1] - df.index[0], NUM_DECIMALS
        )  #: Dataset sampling period.

        # Excluded signals list is either passed by _new_dataset_from_signals
        # or it is empty if a dataframe is passed by the user (all the signals
        # in this case shall be sampled with the same sampling period).
        self.excluded_signals: list[str] = _excluded_signals
        """Signals that could not be re-sampled."""
        # If the user passes a str cast into a list[str]
        u_names = str2list(u_names)
        y_names = str2list(y_names)

        # Keep only the signals specified by the user and respect the order
        # This is helpful in case the df is automatically imported e.g.
        # from a large csv but we only want few signals
        #
        # Filter columns
        u_cols = [u for u in df.columns if u[0] in u_names]
        y_cols = [y for y in df.columns if y[0] in y_names]

        # Order columns according to passed order
        u_cols.sort(key=lambda x: u_names.index(x[0]))
        y_cols.sort(key=lambda x: y_names.index(x[0]))
        df = df.reindex(columns=u_cols + y_cols)

        # Now we can validae the resulting dataframe
        validate_dataframe(df, u_names, y_names)

        # Fix data and index
        data = df.to_numpy()
        index = df.index

        # Adjust column multiIndex
        u_units = [u[1] for u in df.columns if u[0] in u_names]
        u_kind = ["INPUT"] * len(u_names)
        u_multicolumns = list(zip(u_kind, u_names, u_units))

        y_units = [y[1] for y in df.columns if y[0] in y_names]
        y_kind = ["OUTPUT"] * len(y_names)
        y_multicolumns = list(zip(y_kind, y_names, y_units))

        # Create extended DataFrame
        df_ext = pd.DataFrame(data=data, index=index)
        levels_name = ["kind", "names", "units"]
        df_ext.columns = pd.MultiIndex.from_tuples(
            u_multicolumns + y_multicolumns, name=levels_name
        )

        # Take the whole dataframe as dataset before trimming.
        self.dataset: pd.DataFrame = df_ext  #: The actual dataset

        # Initialize NaN intervals, full time interval
        self._nan_intervals: Any = self._find_nan_intervals()

        # Initialize coverage region
        self.coverage: pd.DataFrame = (
            self._find_dataset_coverage()
        )  # Docstring below,
        """Coverage statistics. Mean (vector) and covariance (matrix) of
        both input and output signals."""

        # =============================================
        # Trim the dataset
        # =============================================
        if full_time_interval:
            tin = np.round(self.dataset.index[0], NUM_DECIMALS)
            tout = np.round(self.dataset.index[-1], NUM_DECIMALS)

        # All possible names
        u_names = list(self.dataset["INPUT"].columns.get_level_values("names"))
        y_names = list(self.dataset["OUTPUT"].columns.get_level_values("names"))

        if overlap:
            # OBS! zip cuts the longest list
            p = len(u_names) - len(y_names)
            leftovers = u_names[p + 1 :] if p > 0 else y_names[p + 1 :]
            signals = list(zip(u_names, y_names)) + leftovers
        else:
            signals = []

        # Trim dataset and all the attributes
        tmp = self.trim(
            *signals,
            tin=tin,
            tout=tout,
            verbosity=verbosity,
        )

        # Update the current Dataset instance
        self.dataset = tmp.dataset
        self._nan_intervals = tmp._nan_intervals
        self.coverage = tmp.coverage

    def _new_dataset_from_signals(
        self,
        name: str,
        signal_list: list[Signal],
        u_names: str | list[str],
        y_names: str | list[str],
        target_sampling_period: float | None = None,
        tin: float | None = None,
        tout: float | None = None,
        full_time_interval: bool = False,
        overlap: bool = False,
        verbosity: int = 0,
    ) -> None:

        # Do not initialize any class attribute here!
        # All attributes are initialized in the _new_dataset_from_dataframe method

        # Arguments validation
        validate_signals(*signal_list)

        # If the user pass a single signal as astring,
        # then we need to convert into a list
        u_names = str2list(u_names)
        y_names = str2list(y_names)

        # Try to align the sampling periods, whenever possible
        # Note! resampled_signals:list[Signals], whereas
        # excluded_signals: list[str]
        resampled_signals, excluded_signals = self._fix_sampling_periods(
            signal_list, target_sampling_period, verbosity=verbosity
        )

        Ts = list(resampled_signals)[0]["sampling_period"]

        # Drop excluded signals from u_names and y_names lists
        u_names = [x["name"] for x in resampled_signals if x["name"] in u_names]
        y_names = [x["name"] for x in resampled_signals if x["name"] in y_names]

        u_units = [
            x["signal_unit"] for x in resampled_signals if x["name"] in u_names
        ]
        y_units = [
            x["signal_unit"] for x in resampled_signals if x["name"] in y_names
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
        # Single column level but the cols name is a tuple (name,unit)
        index = pd.Index(
            data=np.arange(max_idx) * Ts,
            name=("Time", f"{resampled_signals[0]['time_unit']}"),
        )
        columns_tuples = list(zip(u_names + y_names, u_units + y_units))
        df = pd.DataFrame(
            index=index,
            data=df_data,
            columns=columns_tuples,
        )

        # Add some robustness: we validate the built DataFrame, even if
        # it should be correct by construction.
        validate_dataframe(df, u_names=u_names, y_names=y_names)

        # Call the initializer from DataFrame to get a Dataset object.
        self._new_dataset_from_dataframe(
            name,
            df,
            u_names,
            y_names,
            tin,
            tout,
            full_time_interval,
            overlap,
            verbosity,
            _excluded_signals=excluded_signals,
        )

    def _classify_signals(
        self,
        *signals: str,
    ) -> tuple[dict[str, str], dict[str, str], list[int], list[int]]:
        # You pass a list of signal names and the function recognizes who is input
        # and who is output. The dicts are name:unit
        # Is no argument is passed, then it takes the whole for u_names and y_names
        # If only input or output signals are passed, then it return an empty list
        # for the non-returned labels.
        df = self.dataset

        # Separate in from out.
        # By default, u_names and y_names are all the possible names.
        # If the user passes some signals, then the full list is.
        u_names = list(df["INPUT"].columns.get_level_values("names"))
        y_names = list(df["OUTPUT"].columns.get_level_values("names"))

        # Small check. Not very pythonic but still...
        signals_not_found = difference_lists_of_str(
            list(signals), u_names + y_names
        )
        if signals_not_found:
            raise KeyError(
                f"Signal(s) {signals_not_found} not found in the dataset. "
                "Use 'signal_list()' to get the list of all available signals. "
            )
        # ========================================================

        # If the signals are passed, then classify in IN and OUT.
        if signals:
            u_names = [
                s
                for s in signals
                if s in df["INPUT"].columns.get_level_values("names")
            ]
            y_names = [
                s
                for s in signals
                if s in df["OUTPUT"].columns.get_level_values("names")
            ]

        # Compute indices
        u_names_idx = [
            df["INPUT"].droplevel(level="units", axis=1).columns.get_loc(u)
            for u in u_names
        ]
        y_names_idx = [
            df["OUTPUT"].droplevel(level="units", axis=1).columns.get_loc(y)
            for y in y_names
        ]

        # Use the indices to locate the units
        u_units = list(
            df["INPUT"].iloc[:, u_names_idx].columns.get_level_values("units")
        )
        y_units = list(
            df["OUTPUT"].iloc[:, y_names_idx].columns.get_level_values("units")
        )

        # Collect in dicts as it is cleaner
        u_dict = dict(zip(u_names, u_units))
        y_dict = dict(zip(y_names, y_units))

        return u_dict, y_dict, u_names_idx, y_names_idx

    def _validate_name_value_tuples(
        self,
        *signals_values: tuple[str, float],
    ) -> tuple[
        dict[str, str],
        dict[str, str],
        list[tuple[str, float]],
        list[tuple[str, float]],
    ]:
        # This function is needed to validate inputs like [("u1",3.2), ("y1", 0.5)]
        # Think for example to the "remove_offset" function.
        # Return both the list of input and output names along their units
        # and the validated tuples.

        signals = [s[0] for s in signals_values]
        u_dict, y_dict, _, _ = self._classify_signals(*signals)

        u_names = list(u_dict.keys())
        y_names = list(y_dict.keys())

        u_list = [(s[0], s[1]) for s in signals_values if s[0] in u_names]
        y_list = [(s[0], s[1]) for s in signals_values if s[0] in y_names]

        return u_dict, y_dict, u_list, y_list

    # def _get_plot_params(
    #     self,
    #     p: int,
    #     q: int,
    #     u_names_idx: list[int],
    #     y_names_idx: list[int],
    #     overlap: bool,
    # ) -> tuple[int, np.ndarray, np.ndarray, list[str], list[str]]:
    #     # Return n, range_in, range_out, u_titles, y_titles.

    #     # Input range is always [0:p]
    #     range_in = np.arange(0, p)
    #     if overlap:
    #         n = max(p, q)
    #         range_out = np.arange(0, q)

    #         # Adjust subplot titles
    #         m = min(p, q)
    #         # Common titles
    #         titles_a = ["IN #" + str(ii + 1) for ii in u_names_idx]
    #         titles_b = [" - OUT #" + str(ii + 1) for ii in y_names_idx]
    #         common_titles = [s1 + s2 for s1, s2 in zip(titles_a, titles_b)]

    #         if p > q:
    #             trail = ["INPUT #" + str(ii + 1) for ii in u_names_idx[m:]]
    #             u_titles = common_titles + trail
    #             y_titles = []
    #         else:
    #             trail = ["OUTPUT #" + str(ii + 1) for ii in y_names_idx[m:]]
    #             u_titles = []
    #             y_titles = common_titles + trail
    #     else:
    #         n = p + q
    #         range_out = np.arange(p, p + q)

    #         # Adjust titles
    #         u_titles = ["INPUT #" + str(ii + 1) for ii in u_names_idx]
    #         y_titles = ["OUTPUT #" + str(ii + 1) for ii in y_names_idx]

    #     return n, range_in, range_out, u_titles, y_titles

    def _fix_sampling_periods(
        self,
        signal_list: list[Signal],
        target_sampling_period: float | None = None,
        verbosity: int = 0,
    ) -> tuple[list[Signal], list[str]]:
        # The signals are resampled either with the slowest sampling period as target,
        # or towards the sampling period specified by the *target_sampling_period*
        # parameter, if specified.

        # arguments Validation
        if target_sampling_period is not None:
            if (
                not isinstance(target_sampling_period, float)
                or target_sampling_period < 0
            ):
                raise ValueError("'target_sampling_period' must be positive.")

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
            if np.isclose(N, round(N), atol=ATOL):
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

    def _add_signals(self, kind: Signal_type, *signals: Signal) -> Any:

        # Validate the dymoval Signals
        validate_signals(*signals)

        # Target dataset
        ds = deepcopy(self)

        # Check if the sampling period is OK
        signals_ok = [
            s
            for s in signals
            if np.isclose(s["sampling_period"], ds.sampling_period, atol=ATOL)
        ]
        signals_name = [s["name"] for s in signals]
        signals_unit = [s["signal_unit"] for s in signals]

        # Update excluded_signals attribute
        excluded_signals = [s["name"] for s in signals if s not in signals_ok]
        ds.excluded_signals = ds.excluded_signals + excluded_signals
        if excluded_signals:
            raise Warning(f"Signals {excluded_signals} cannot be added.")

        # Check if the signal name(s) already exist in the current Dataset
        _, signal_names, _ = zip(*self.signal_list())
        name_found = [
            s["name"] for s in signals_ok if s["name"] in signal_names
        ]
        if name_found:
            raise KeyError(f"Signal(s) {name_found} already exist.")

        # Adjust the signals length of the new signals
        ds_length = len(self.dataset.index)
        for s in signals_ok:
            if s["values"].size >= ds_length:
                s["values"] = s["values"][:ds_length]
            else:
                nan_vec = np.empty(ds_length - s["values"].size)
                nan_vec[:] = np.NaN
                s["values"] = np.concatenate((s["values"], nan_vec))

        # Create DataFrame from signals to be appended (concatenated) to
        # the current dataset
        data = np.stack([s["values"] for s in signals_ok]).round(NUM_DECIMALS).T
        df_temp = pd.DataFrame(data=data, index=ds.dataset.index)
        df_temp.columns = pd.MultiIndex.from_tuples(
            zip(str2list(kind) * len(signals_name), signals_name, signals_unit),
            name=["kind", "names", "units"],
        )

        # concatenate new DataFrame containing the added signals
        ds.dataset = pd.concat([df_temp, ds.dataset], axis=1).sort_index(
            level="kind", axis=1, sort_remaining=False
        )

        # Update NaN intervals
        NaN_intervals = self._find_nan_intervals(df_temp)
        ds._nan_intervals.update(
            NaN_intervals
        )  # Join two dictionaries through update()

        ds.dataset = ds.dataset.round(NUM_DECIMALS)
        return ds

    def trim(
        self,
        *signals: str | tuple[str, str] | None,
        tin: float | None = None,
        tout: float | None = None,
        verbosity: int = 0,
    ) -> Dataset:
        """
        MISSING DOCSTRUNG!!!!

        Parameters
        ----------
        *signals : str | tuple[str, str] | None
            DESCRIPTION.
        tin : float | None, optional
            DESCRIPTION. The default is None.
        tout : float | None, optional
            DESCRIPTION. The default is None.
        verbosity : int, optional
            DESCRIPTION. The default is 0.



        """
        # We have to trim the signals to have a meaningful dataset
        # This can be done both graphically or by passing tin and tout
        # if the user knows them before hand or by setting full_time_interval = True.
        # Once done, the dataset shall be automatically shifted to the point tin = 0.0.

        def _graph_selection(
            ds: Dataset,
            *signals: str | tuple[str, str] | None,
        ) -> tuple[float, float]:  # pragma: no cover
            # Select the time interval graphically
            # OBS! This part cannot be automatically tested because the it require
            # manual action from the user (resize window).
            # Hence, you must test this manually.

            # The following code is needed because not all IDE:s
            # have interactive plot set to ON as default.
            is_interactive = plt.isinteractive()
            plt.ion()

            # Get axes from the plot and use them to extract tin and tout
            axes = ds.plot(*signals)
            axes = axes.T.flat

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
            fig = axes[0].get_figure()
            cid = fig.canvas.mpl_connect("close_event", close_event)
            fig.canvas.draw()
            plt.show()

            fig.suptitle(
                "Sampling time "
                f"= {np.round(self.dataset.index[1]-self.dataset.index[0],NUM_DECIMALS)} {self.dataset.index.name[1]}.\n"
                "Select the dataset time interval by resizing "
                "the picture."
            )

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

            return np.round(tin_sel, NUM_DECIMALS), np.round(
                tout_sel, NUM_DECIMALS
            )

        # =============================================
        # Trim Dataset main function
        # The user can either pass the pair (tin,tout) or
        # he/she can select it graphically if nothing has passed
        # =============================================

        ds = deepcopy(self)
        # Check if info on (tin,tout) is passed
        if tin is None and tout is not None:
            tin_sel = ds.dataset.index[0]
            tout_sel = tout
        # If only tin is passed, then set tout to the last time sample.
        elif tin is not None and tout is None:
            tin_sel = tin
            tout_sel = ds.dataset.index[-1]
        elif tin is not None and tout is not None:
            tin_sel = np.round(tin, NUM_DECIMALS)
            tout_sel = np.round(tout, NUM_DECIMALS)
        else:  # pragma: no cover
            tin_sel, tout_sel = _graph_selection(ds, *signals)

        if verbosity != 0:
            print(
                f"\n tin = {tin_sel}{ds.dataset.index.name[1]}, tout = {tout_sel}{ds.dataset.index.name[1]}"
            )

        # Now you can trim the dataset and update all the
        # other time-related attributes
        ds.dataset = ds.dataset.loc[tin_sel:tout_sel, :]  # type:ignore
        ds._nan_intervals = ds._find_nan_intervals()
        ds.coverage = ds._find_dataset_coverage()

        # ... and shift it such that tin = 0.0
        ds._shift_dataset_tin_to_zero()
        ds.dataset = ds.dataset.round(NUM_DECIMALS)

        return ds

    def dataset_values(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        if len(self.dataset["INPUT"].columns.get_level_values("names")) == 1:
            u_values = (
                self.dataset["INPUT"].to_numpy().round(NUM_DECIMALS)[:, 0]
            )
        else:
            u_values = self.dataset["INPUT"].to_numpy().round(NUM_DECIMALS)

        if len(self.dataset["OUTPUT"].columns.get_level_values("names")) == 1:
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
            Target filename.
        """

        signal_list = self.dump_to_signals()

        # We start storing the timeline
        ds_dict = {"TIME": self.dataset.index.to_numpy()}

        for kind in SIGNAL_KIND:
            # Yet another issue with mypy and TypedDict. https://github.com/python/mypy/issues/4976
            dict_tmp = {kind: {s.pop("name"): s for s in signal_list[kind]}}  # type: ignore
            ds_dict = ds_dict | dict_tmp

        io.savemat(filename, ds_dict, oned_as="column")

    def dump_to_signals(self) -> dict[Signal_type, list[Signal]]:
        """Dump a *Dataset* object into *Signal* objects.


        Warning
        -------
        Additional information contained in the *Dataset*, such as *NaNs*
        intervals, coverage region, etc. are lost.

        """

        signal_list = {}
        for kind in SIGNAL_KIND:
            temp_lst = []
            df = self.dataset[kind]
            cols = df.columns
            names = cols.get_level_values("names")
            signal_units = cols.get_level_values("units")
            time_unit = df.index.name[1]
            sampling_period = self.sampling_period

            for ii, val in enumerate(cols):
                # This is the syntax for defining a dymoval signal
                temp: dmv.Signal = {
                    "name": names[ii],
                    "values": df.loc[:, val].to_numpy().round(NUM_DECIMALS),
                    "signal_unit": signal_units[ii],
                    "sampling_period": sampling_period,
                    "time_unit": time_unit,
                }
                temp_lst.append(deepcopy(temp))
            signal_list[kind] = temp_lst

        return signal_list

    def signal_list(self) -> list[tuple[str, str, str]]:
        """Plot signal y against signal x.

        It is possible to pass an arbitrary number of tuples of the
        form *(signal_name_x,signal_name_y).*


        Raises
        ------
        ValueError:
            If any passed signal does not exist in the dataset.
        """
        return list(self.dataset.columns)

    def plotxy(
        self,
        # Only positional arguments
        /,
        *signal_pairs: tuple[str, str],
        save_as: str | None = None,
    ) -> matplotlib.axes.Axes:
        """You must specify the complete *filename*, including the path."""
        # df points to self.dataset.
        df = self.dataset

        # Check if the passed names exist
        available_names = list(df.columns.get_level_values("names"))
        a, b = zip(*signal_pairs)
        passed_names = list(a + b)
        names_not_found = difference_lists_of_str(
            passed_names, available_names
        )  # noqa
        if names_not_found:
            raise ValueError(f"Signal(s) {names_not_found} not found.")

        # signal_name:signal_unit dict for the axes labels
        signal_unit = dict(df.droplevel(level="kind", axis=1).columns)

        # Start the plot ritual
        n = len(signal_pairs)
        nrows, ncols = factorize(n)  # noqa
        fig, axes = plt.subplots(nrows, ncols, squeeze=False)
        # Flatten array for more readable code
        axes = axes.T.flat

        for ii, val in enumerate(signal_pairs):
            df.droplevel(level=["kind", "units"], axis=1).plot(
                x=val[0],
                y=val[1],
                ax=axes[ii],
                legend=None,
                xlabel=f"{val[0]}, ({ signal_unit[val[0]]})",
                ylabel=f"{val[1]}, ({ signal_unit[val[1]]})",
                grid=True,
            )

        # Eventually save and return figures.
        if save_as is not None and ax is None:
            save_plot_as(fig, axes, save_as)  # noqa

        return axes.base

    def plot(
        self,
        # Only positional arguments
        /,
        *signals: str | tuple[str, str] | None,
        # Key-word arguments
        overlap: bool = False,
        line_color_input: str = "blue",
        linestyle_fg: str = "-",
        alpha_fg: float = 1.0,
        line_color_output: str = "green",
        linestyle_bg: str = "--",
        alpha_bg: float = 1.0,
        ax: matplotlib.axes.Axes | None = None,
        p_max: int = 0,
        save_as: str | None = None,
    ) -> matplotlib.axes.Axes:
        """Plot the Dataset.

        For each signal pair passed as a *tuple*, the signals will be placed in
        the same subplot.
        Signals passed as a string will be plotted in a separate subplot.

        For example, if *ds* is a :py:class:`Dataset <dymoval.dataset.Dataset>`
        with signals *s1, s2, ... sn*, then
        *ds.plot(("s1","s2"), "s3", "s4")* will plot *s1* and *s2* on the same subplot
        and it will plot *s3* and "s4" on separate subplots, thus displaying the
        total of 3 subplots.

        It is possible to overlap at most two signals (this to avoid adding too many
        y-axes).

        Possible values for the parameters describing the line used in the plot
        (e.g. *line_color_input* , *alpha_output*. etc).
        are the same for the corresponding plot function in matplotlib.

        TODO TIGHT FIGURE HANDLING

        Parameters
        ----------
        *signals:
            Signals to be plotted.
        overlap:
            If *True* overlap input the output signals plots
            pairwise.
            Eventual passed *signals will be discarded.
            The units of the outputs are displayed on the secondary y-axis.
        line_color_input:
            Line color of the input signals.
        linestyle_fg:
            Line style of the signals passed as strings or line style of the first
            element if two signals are passed as a tuple.
        alpha_fg:
            Alpha channel value of the signals passed as strings or line style
            of the first element if two signals are passed as a tuple.
        line_color_output:
            Line color of the output signals.
        linestyle_bg:
            Line style of the second output signals.
        alpha_bg:
            Alpha channel value for the output signals.
        ax:
            Matplotlib Axes where to place the plot.
        p_max:
            Maximum number of inputs. It is a parameters used internally so
            it can be left alone.
        save_as:
            Save the figure with a specified name.
            The figure is automatically resized to try to keep a 16:9 aspect ratio.
            You must specify the complete *filename*, including the path.
        """

        # df points to self.dataset.
        df = self.dataset

        # All possible names
        u_names = list(df["INPUT"].columns.get_level_values("names"))
        y_names = list(df["OUTPUT"].columns.get_level_values("names"))

        if overlap:
            # OBS! zip cuts the longest list
            p = len(u_names) - len(y_names)
            leftovers = u_names[p + 1 :] if p > 0 else y_names[p + 1 :]
            signals = list(zip(u_names, y_names)) + leftovers
        # Convert passed tuple to list
        elif signals:
            signals = [s for s in signals]
        else:
            signals = u_names + y_names

        # Make all tuples like [('u0', 'u1'), ('y0',), ('u1', 'y1', 'u0')]
        for ii, s in enumerate(signals):
            if isinstance(s, str):
                signals[ii] = (s,)

        # signals_lst_plain, e.g. ['u0', 'u1', 'y0', 'u1', 'y1', 'u0']
        signals_lst_plain = [item for t in signals for item in t]

        # Arguments validation
        (
            u_dict,
            y_dict,
            u_names_idx,
            y_names_idx,
        ) = self._classify_signals(*signals_lst_plain)

        # ===================================================
        # Fix the parameters to pass to the plot method
        # ===================================================

        # All the lists are plain (i.e. no list of tuples)
        # colors
        colors = [
            line_color_input if s in u_dict.keys() else line_color_output
            for s in signals_lst_plain
        ]
        colors_tpl = _list_to_structured_list_of_tuple(signals, colors)

        # units
        s_dict = deepcopy(u_dict)
        s_dict.update(y_dict)
        units = [s_dict[s] for s in signals_lst_plain]
        units_tpl = _list_to_structured_list_of_tuple(signals, units)

        # Linestyles
        linestyles_tpl = []
        for val in colors_tpl:
            if len(val) == 2:
                if val[0] == val[1]:
                    linestyles_tpl.append((linestyle_fg, linestyle_bg))
                else:
                    linestyles_tpl.append((linestyle_fg, linestyle_fg))
            else:
                linestyles_tpl.append((linestyle_fg,))

        # Figure and axes
        n = len(signals)
        nrows, ncols = factorize(n)  # noqa
        if ax is None:  # Create new figure and axes
            fig, axes = plt.subplots(nrows, ncols, sharex=True, squeeze=False)
        else:  # Otherwise use what is passed
            axes = np.asarray(ax)
        axes = axes.T.flat

        for ax in axes[n:]:
            ax.remove()

        for ax in axes:
            # ax.spines.set(visible=True)
            ax.tick_params(axis="x")
        # Title
        plt.suptitle(
            f"Dataset {self.name}. \n {line_color_input} lines are inputs and {line_color_output} lines are outputs."
        )

        for ii, s in enumerate(signals):
            df.droplevel(level=["kind", "units"], axis=1).loc[:, s[0]].plot(
                subplots=True,
                grid=True,
                legend=True,
                color=colors_tpl[ii][0],
                linestyle=linestyles_tpl[ii][0],
                alpha=alpha_fg,
                ylabel=units_tpl[ii][0],
                ax=axes[ii],
            )

            # In case the user wants to overlap plots...
            # If the overlapped plots have the same units, then there is
            # no point in using a secondary_y
            if len(s) == 2:
                if units_tpl[ii][0] == units_tpl[ii][1]:
                    ylabel = None
                    secondary_y = False
                else:
                    ylabel = units_tpl[ii][1]
                    secondary_y = True

                # Now you can plot
                df.droplevel(level=["kind", "units"], axis=1).loc[:, s[1]].plot(
                    subplots=True,
                    grid=True,
                    legend=True,
                    color=colors_tpl[ii][1],
                    linestyle=linestyles_tpl[ii][1],
                    alpha=alpha_bg,
                    secondary_y=secondary_y,
                    ylabel=ylabel,
                    ax=axes[ii],
                )

        self._shade_nans(
            axes,
        )

        # Eventually save and return figures.
        if save_as is not None and ax is None:
            save_plot_as(fig, axes, save_as)  # noqa

        return axes.base

    def plot_coverage(
        self,
        *signals: str,
        nbins: int = 100,
        line_color_input: str = "b",
        alpha_input: float = 1.0,
        line_color_output: str = "g",
        alpha_output: float = 1.0,
        ax_in: matplotlib.axes.Axes | None = None,
        ax_out: matplotlib.axes.Axes | None = None,
        save_as: str | None = None,
    ) -> tuple[matplotlib.axes.Axes | None, matplotlib.axes.Axes | None]:
        """
        Plot the dataset coverage as histograms.


        Parameters
        ----------
        *signals:
            The coverage of these signals will be plotted.
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
        as_in:
            Axes where the input signal coverage plot shall be placed.
        as_out:
            Axes where the output signal coverage plot shall be placed.
        save as:
            Save the figures with a specified name.
            It appends the suffix *_in* and *_out* to the input and output figure,
            respectively.
            The *filename* shall include the path.
        """
        # Extract dataset
        df = self.dataset

        (
            u_dict,
            y_dict,
            u_names_idx,
            y_names_idx,
        ) = self._classify_signals(*signals)

        # Remove duplicated labels as they are not needed for coverage
        # This because you don't need to overlap e.g. (u1,y1), (u1,y2), etc
        # The following is the way from Python 3.7 you remove items while
        # preserving the order
        u_names = list(u_dict.keys())
        u_units = list(u_dict.values())
        y_names = list(y_dict.keys())
        y_units = list(y_dict.values())

        # Input-output length and indices
        p = len(u_names)
        q = len(y_names)

        # get some plot parameters based on user preferences
        # _, _, _, u_titles, y_titles = self._get_plot_params(
        #     p, q, u_names_idx, y_names_idx, overlap=False
        # )

        # This if is needed in case the user wants to plot only one signal
        if u_names:
            nrows_in, ncols_in = factorize(p)  # noqa

            if ax_in is None:  # Create new figure and axes
                fig_in, axes_in = plt.subplots(
                    nrows_in, ncols_in, squeeze=False
                )
            else:  # Otherwise use what is passed
                axes_in = np.asarray(ax_in)

            axes_in = axes_in.T.flat
            df["INPUT"].droplevel(level="units", axis=1).loc[:, u_names].hist(
                grid=True,
                bins=nbins,
                color=line_color_input,
                alpha=alpha_input,
                legend=u_names,
                ax=axes_in[0:p],
            )

            # Set xlabel
            for ii, unit in enumerate(u_units):
                # axes_in[ii].set_title(f"INPUT #{u_names_idx[ii]+1}")
                axes_in[ii].set_xlabel(f"({unit})")
            plt.suptitle("Coverage region (INPUT).")

            # Tight figure if no ax is passed (e.g. from compare_datasets())
            # if ax_in is None:
            #    fig_in.tight_layout()

        if y_names:
            nrows_out, ncols_out = factorize(q)  # noqa

            if ax_out is None:  # Create new figure and axes
                fig_out, axes_out = plt.subplots(
                    nrows_out, ncols_out, squeeze=False
                )
            else:  # Otherwise use what is passed
                axes_out = np.asarray(ax_out)

            axes_out = axes_out.T.flat
            df["OUTPUT"].droplevel(level="units", axis=1).loc[:, y_names].hist(
                grid=True,
                bins=nbins,
                color=line_color_output,
                alpha=alpha_output,
                legend=y_names,
                ax=axes_out[0:q],
            )

            # Set xlabel, ylabels
            for ii, unit in enumerate(y_units):
                # axes_out[ii].set_title(f"OUTPUT #{y_names_idx[ii]+1}")
                axes_out[ii].set_xlabel(f"({unit})")
            plt.suptitle("Coverage region (OUTPUT).")

            # tight figure if no ax is passed (e.g. from compare_datasets())
            # if ax_out is None:
            #    fig_out.tight_layout()

        if save_as is not None:
            save_plot_as(fig_in, axes_in, save_as + "_in")  # noqa
            save_plot_as(fig_out, axes_out, save_as + "_out")  # noqa

        # Return
        if u_names and y_names:
            return axes_in.base, axes_out.base

        elif u_names and not y_names:
            return axes_in.base, None
        else:  # The only option left is not u_names and y_names
            return None, axes_out.base

    def fft(
        self,
        *signals: str,
    ) -> pd.DataFrame:
        """Return the FFT of the dataset as pandas DataFrame.

        It only works with real-valued signals.

        Parameters
        ----------
        signals:
            The FFT is computed for these signals.

        Raises
        ------
        ValueError
            If the dataset contains *NaN*:s
        """
        # Validation
        u_names, y_names, _, _, _, _ = self._classify_signals(*signals)
        # Remove 'INPUT' 'OUTPUT' columns level from dataframe
        df_temp = self.dataset.droplevel(level=["kind", "units"], axis=1)

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

        vals = fft.rfftn(df_temp.loc[:, u_names + y_names], axes=0) / N
        vals = vals.round(NUM_DECIMALS)

        # Create a new Dataframe
        u_extended_labels = list(zip(["INPUT"] * len(u_names), u_names))
        y_extended_labels = list(zip(["OUTPUT"] * len(y_names), y_names))
        cols = pd.MultiIndex.from_tuples(
            [*u_extended_labels, *y_extended_labels]
        )
        df_freq = pd.DataFrame(data=vals.round(NUM_DECIMALS), columns=cols)
        df_freq = df_freq.T.drop_duplicates().T  # Drop duplicated columns
        time2freq_units = {
            "s": "Hz",
            "ms": "kHz",
            "us": "MHz",
            "ns": "GHz",
            "ps": "THz",
        }
        df_freq.index.name = (
            f"Frequency ({time2freq_units[df_temp.index.name[1]]})"
        )

        return df_freq.round(NUM_DECIMALS)

    def plot_spectrum(
        self,
        *signals: str,
        kind: Literal["amplitude", "power", "psd"] = "power",
        overlap: bool = False,
        line_color_input: str = "b",
        linestyle_input: str = "-",
        alpha_input: float = 1.0,
        line_color_output: str = "g",
        linestyle_output: str = "-",
        alpha_output: float = 1.0,
        ax: matplotlib.axes.Axes | None = None,
        save_as: str | None = None,
    ) -> matplotlib.axes.Axes:
        """
        Plot the spectrum of the specified signals in the dataset in different format.

        If some signals have *NaN* values, then the FFT cannot be computed and
        an error is raised.


        Parameters
        ----------
        *signals:
            The spectrum of these signals will be plotted.
        kind:

            - *amplitude* plot both the amplitude and phase spectrum.
              If the signal has unit V, then the amplitude has unit *V*.
              Phase is in radians.
            - *power* plot the autopower spectrum.
              If the signal has unit V, then the amplitude has unit *V^2*.
            - *psd* plot the power density spectrum.
              If the signal has unit V and the time is *s*, then the amplitude has unit *V^2/Hz*.
        overlap:
            If true it overlaps the input and the output signals plots.
            The units of the outputs are displayed on the secondary y-axis.
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
            Axes where the spectrum plot will be placed.
        save_as:
            Save the figure with a specified name.
            The figure is automatically resized with a 16:9 aspect ratio.
            You must specify the complete *filename*, including the path.


        Raises
        ------
        ValueError
            If *kind* doen not match any allowed values.
        """
        # validation
        (
            u_names,
            y_names,
            u_units,
            y_units,
            u_names_idx,
            y_names_idx,
        ) = self._classify_signals(*signals)

        if kind not in SPECTRUM_KIND:
            raise ValueError(f"kind must be one of {SPECTRUM_KIND}")

        # ====================================
        # Compute Spectrums
        # ===================================
        # For real signals, the spectrum is Hermitian anti-simmetric, i.e.
        # the amplitude is symmetric wrt f=0 and the phase is antisymmetric wrt f=0.
        # See e.g. https://ccrma.stanford.edu/~jos/ReviewFourier/Symmetries_Real_Signals.html
        df_freq = self.fft(*signals)
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

        # ====================================
        # Plot Spectrums
        # ===================================
        # Adjust input-output lengths and labels for plots.

        p = len(u_names)
        q = len(y_names)

        # If the spectrum type is amplitude, then we need to plot
        # abs and phase, so we need to double a number of things
        if kind == "amplitude":
            # Adjust (p,q): numbers of input and outputs are doubled
            # because they contain (abs,angle)
            p = 2 * p
            q = 2 * q
            # Adjust, labels idx, i.e. each entry appears two times
            u_names_idx = [u for u in u_names_idx for ii in range(2)]
            y_names_idx = [y for y in y_names_idx for ii in range(2)]

        # get some plot parameters based on user preferences
        n, range_in, range_out, u_titles, y_titles = self._get_plot_params(
            p, q, u_names_idx, y_names_idx, overlap
        )

        # Find nrwos and ncols for the plot
        nrows, ncols = factorize(n)

        if overlap:
            secondary_y = True
        else:
            secondary_y = False

        if kind == "amplitude":
            # To have the phase plot below the abs plot, then the number
            # of rows must be an even number, otherwise the plot got screwed.
            # OBS: the same code is in compare_datasets()
            if np.mod(nrows, 2) != 0:
                nrows -= 1
                ncols += int(np.ceil(ncols / nrows))

        # Overwrite axes if you want the plot to be on the figure instantiated by the caller.
        if ax is None:  # Create new figure and axes
            fig, axes = plt.subplots(nrows, ncols, sharex=True, squeeze=False)
        else:  # Otherwise use what is passed
            axes = np.asarray(ax)

        # Flatten array for more readable code
        axes = axes.T.flat

        if u_names:
            df_freq["INPUT"].loc[:, u_names].plot(
                subplots=True,
                grid=True,
                color=line_color_input,
                linestyle=linestyle_input,
                alpha=alpha_input,
                legend=u_names,
                title=u_titles,
                ax=axes[range_in],
            )

        if y_names:
            df_freq["OUTPUT"].loc[:, y_names].plot(
                subplots=True,
                grid=True,
                color=line_color_output,
                linestyle=linestyle_output,
                alpha=alpha_output,
                secondary_y=secondary_y,
                legend=y_names,
                title=y_titles,
                ax=axes[range_out],
            )

        # Set ylabels
        # input
        for ii, unit in enumerate(u_units):
            if kind == "power":
                axes[ii].set_ylabel(f"({unit}**2)")
            elif kind == "psd":
                axes[ii].set_ylabel(f"({unit}**2/Hz)")
            elif kind == "amplitude":
                axes[2 * ii].set_ylabel(f"({unit})")
                axes[2 * ii + 1].set_ylabel("(rad)")

        # Output
        if not sorted(u_units) == sorted(y_units):
            for jj, unit in enumerate(y_units):
                if overlap:
                    if kind == "power":
                        axes[jj].right_ax.set_ylabel(f"({unit}**2)")
                        axes[jj].right_ax.grid(None, axis="y")
                    elif kind == "psd":
                        axes[jj].right_ax.set_ylabel(f"({unit}**2/Hz)")
                        axes[jj].right_ax.grid(None, axis="y")
                    elif kind == "amplitude":
                        axes[2 * jj].right_ax.set_ylabel(f"({unit})")
                        axes[2 * jj].right_ax.grid(None, axis="y")
                        axes[2 * jj + 1].set_ylabel("(rad)")
                        axes[2 * jj + 1].right_ax.grid(None, axis="y")

                else:
                    if kind == "power":
                        axes[p + jj].set_ylabel(f"({unit}**2)")
                    elif kind == "psd":
                        axes[p + jj].set_ylabel(f"({unit}**2/Hz)")
                    elif kind == "amplitude":
                        axes[p + 2 * jj].set_ylabel(f"({unit})")
                        axes[p + 2 * jj + 1].set_ylabel("(rad)")

        # Set xlabel
        for ii in range(ncols):
            axes[nrows - 1 :: nrows][ii].set_xlabel(df_freq.index.name)

        plt.suptitle(f"{kind.capitalize()} spectrum.")

        # Tight figure if no ax is passed (e.g. from compare_datasets())
        # if ax is None:
        #    fig.tight_layout()

        # Save and return
        if save_as is not None:
            save_plot_as(fig, axes, save_as)

        return axes.base

    def remove_means(
        self,
        *signals: str,
    ) -> Dataset:
        """
        Remove the mean value to the specified signals.

        Parameters
        ----------
        *signals:
            Remove means to the specified signals.
            If not specified, then the mean value is removed to all
            the input signals in the dataset.
        """
        # Arguments validation
        u_names, y_names, u_units, y_units, _, _ = self._classify_signals(
            *signals
        )

        # Safe copy
        ds_temp = deepcopy(self)
        df_temp = ds_temp.dataset

        # Remove means from input signals
        cols = list(
            zip(["INPUT"] * len(u_names), u_names, u_units),
        )

        df_temp.loc[:, cols] = (
            df_temp.loc[:, cols] - df_temp.loc[:, cols].mean()
        )

        # Remove means from output signals
        cols = list(
            zip(["OUTPUT"] * len(y_names), y_names, y_units),
        )
        df_temp.loc[:, cols] = (
            df_temp.loc[:, cols] - df_temp.loc[:, cols].mean()
        )

        # round result
        ds_temp.dataset = df_temp.round(NUM_DECIMALS)

        return ds_temp

    def remove_offset(
        self,
        *signals_values: tuple[str, float],
    ) -> Dataset:
        # At least one argument shall be passed.
        # This is the reason why the @overload is added in that way
        """
        Remove specified offsets to the specified signals.

        For each target signal a tuple of the form *(name,value)*
        shall be passed.
        The value specified in the *offset* parameter is removed from
        the signal with name *name*.

        Parameters
        ----------
        *signals:
            Tuples of the form *(name, offset)*.
            The *name* parameter must match the name of a signal stored
            in the dataset.
            The *offset* parameter is the value to remove to the *name* signal.
        """

        # Safe copy
        ds_temp = deepcopy(self)
        df_temp = ds_temp.dataset

        # Validate passed arguments
        (
            u_names,
            y_names,
            u_units,
            y_units,
            u_list,
            y_list,
        ) = self._validate_name_value_tuples(*signals_values)

        # First adjust the input columns
        if u_list:
            u_offset = [u[1] for u in u_list]
            cols = list(
                zip(["INPUT"] * len(u_names), u_names, u_units),
            )

            df_temp.loc[:, cols] = df_temp.loc[:, cols].apply(
                lambda x: x.subtract(u_offset), axis=1
            )

        # Then adjust the output columns
        if y_list:
            y_offset = [y[1] for y in y_list]
            cols = list(
                zip(["OUTPUT"] * len(y_names), y_names, y_units),
            )
            df_temp.loc[:, cols] = df_temp.loc[:, cols].apply(
                lambda x: x.subtract(y_offset), axis=1
            )

        df_temp.round(NUM_DECIMALS)
        # ds_temp.dataset.loc[:, :] = df_temp.to_numpy().round(NUM_DECIMALS)

        return ds_temp

    def low_pass_filter(
        self,
        *signals_values: tuple[str, float],
    ) -> Dataset:
        """
        Low-pass filter a list of specified signals.

        For each target signal a tuple of the form *(name, cutoff_frequency)*
        shall be passed as parameter.
        For multiple signals, the tuples shall be arranged in a list.

        The low-pass filter is first-order IIR filter.

        Parameters
        ----------
        *signals:
            Tuples of the form *(name, cutoff_frequency)*.
            The *name* parameter must match the name of any signal stored
            in the dataset.
            Such a signal is filtered with a low-pass
            filter whose cutoff frequency is specified by the *cutoff_frequency*
            parameter.


        Raises
        ------
        TypeError
            If no arguments are passed.
        ValueError
            If any of the passed cut-off frequencies is negative.
        """
        # Safe copy
        ds_temp = deepcopy(self)
        df_temp = ds_temp.dataset

        # Validate passed arguments
        (
            u_names,
            y_names,
            u_units,
            y_units,
            u_list,
            y_list,
        ) = self._validate_name_value_tuples(*signals_values)

        # Sampling frequency
        fs = 1 / (ds_temp.dataset.index[1] - ds_temp.dataset.index[0])
        N = len(ds_temp.dataset.index)

        # INPUT LPF
        if u_list:
            u_fc = [u[1] for u in u_list]
            if any(u_val < 0 for u_val in u_fc):
                raise ValueError("Cut-off frequencies must be positive.")
            for ii, u in enumerate(u_names):
                # Low-pass filter implementatiom
                fc = u_fc[ii]
                u_filt = df_temp.loc[:, ("INPUT", u, u_units[ii])].to_numpy()
                y_filt = np.zeros(N)
                y_filt[0] = u_filt[0]
                for kk in range(0, N - 1):
                    y_filt[kk + 1] = (1.0 - fc / fs) * y_filt[kk] + (
                        fc / fs
                    ) * u_filt[kk]
                df_temp.loc[:, ("INPUT", u, u_units[ii])] = y_filt

        # OUTPUT
        # List of all the requested input cutoff frequencies
        if y_list:
            y_fc = [y[1] for y in y_list]
            if any(y_val < 0 for y_val in y_fc):
                raise ValueError("Cut-off frequencies must be positive.")
            for ii, y in enumerate(y_names):
                fc = y_fc[ii]  # cutoff frequency
                # Low-pass filter implementatiom
                u_filt = df_temp.loc[:, ("OUTPUT", y, y_units[ii])].to_numpy()
                y_filt = np.zeros(N)
                y_filt[0] = u_filt[0]
                for kk in range(0, N - 1):
                    y_filt[kk + 1] = (1.0 - fc / fs) * y_filt[kk] + (
                        fc / fs
                    ) * u_filt[kk]
                df_temp.loc[:, ("OUTPUT", y, y_units[ii])] = y_filt
        # Round value
        ds_temp.dataset = df_temp.round(NUM_DECIMALS)  # noqa

        return ds_temp

    # def filter(self) -> Dataset:
    #     """To be implemented!"""
    #     print("Not implemented yet!")

    def apply(
        self,
        *signal_function: tuple[str, Any, str],
        **kwargs: Any,
    ) -> Dataset:
        """Apply a function to specified signals and change their unit.


        Note
        ----
        If you need to heavily manipulate your signals, it is suggested
        to dump the Dataset into Signals, manipulate them,
        and then create a brand new Dataset.


        Warning
        -------
        This function may be removed.


        Parameters
        ----------
        signal_function:
            Signals where to apply a function.
            This argument shall have the form *(name, func, new_unit)*
        **kwargs:
            Additional keyword arguments to pass as keywords arguments to
            the underlying pandas DataFrame *apply* method.


        Raises
        ------
        ValueError:
            If the passed signal name does not exist in the current Dataset.
        """

        # Safe copy
        ds_temp = deepcopy(self)
        level_names = ds_temp.dataset.columns.names

        # Check if passed signals exist
        available_names = list(
            ds_temp.dataset.columns.get_level_values("names")
        )
        a, b, c = zip(*signal_function)
        passed_names = list(a)
        names_not_found = difference_lists_of_str(
            passed_names, available_names
        )  # noqa
        if names_not_found:
            raise ValueError(f"Signal(s) {names_not_found} not found.")

        # signal:function dictionary
        sig_func = dict(zip(a, b))
        sig_unit = dict(zip(a, c))

        # Selec the whole columns names based on passed signal names
        cols_idx = [
            col for col in ds_temp.dataset.columns if col[1] in passed_names
        ]
        sig_cols = dict(zip(a, list(cols_idx)))

        # Start to apply functions
        for s in passed_names:
            ds_temp.dataset.loc[:, sig_cols[s]] = (
                ds_temp.dataset.loc[:, sig_cols[s]]
                .apply(sig_func[s])
                .round(NUM_DECIMALS)
            )

            # Update units.
            # You must rewrite the whole columns MultiIndex
            new_col_name = list(sig_cols[s])
            new_col_name[2] = sig_unit[s]
            ds_temp.dataset.columns = ds_temp.dataset.columns.to_flat_index()
            ds_temp.dataset = ds_temp.dataset.rename(
                columns={sig_cols[s]: tuple(new_col_name)}
            )

            new_idx = pd.MultiIndex.from_tuples(
                ds_temp.dataset.columns, names=level_names
            )
            ds_temp.dataset.columns = new_idx

        # Update the coverage
        ds_temp.coverage = self._init_dataset_coverage(ds_temp.dataset)

        return ds_temp

    def remove_NaNs(
        self,
        **kwargs: Any,
    ) -> Dataset:  # pragma: no cover
        """Replace NaN:s in the dataset.

        It uses pandas *DataFrame.interpolate()* method, so the **kwargs are directly
        routed to such a method.


        Parameters
        ----------
        **kwargs :
           Keyword arguments to pass on to the interpolating function.
        """

        # Safe copy
        ds_temp = deepcopy(self)
        ds_temp.dataset = ds_temp.dataset.interpolate(**kwargs)
        ds_temp.dataset = ds_temp.dataset.round(NUM_DECIMALS)

        return ds_temp

    def add_input(self, *signals: Signal) -> Any:
        """Add input signals to the dataset.

        Signals will be trimmed to the length of the Dataset.
        Signals who are shorter, will be padded with 'NaN:s'

        Excluded signals are stored in the *excluded_signals* attibute
        of the calling Dataset object.

        Parameters
        ----------
        *signals:
            Signals to be added in form *("INPUT"|"OUTPUT", Signal)*.
        """
        kind: Signal_type = "INPUT"
        return self._add_signals(kind, *signals)

    def add_output(self, *signals: Signal) -> Any:
        """Add output signals to the dataset.

        Signals will be trimmed to the length of the Dataset.
        Signals who are shorter, will be padded with 'NaN:s'

        Excluded signals are stored in the *excluded_signals* attibute
        of the calling  Dataset.

        Parameters
        ----------
        *signals:
            Signals to be added in form *("INPUT"|"OUTPUT", Signal)*.
        """
        kind: Signal_type = "OUTPUT"
        return self._add_signals(kind, *signals)

    def remove_signals(self, *signals: str) -> Dataset:
        """Remove signals from dataset.


        Raises
        ------
        KeyError:
            If signal(s) not found in the *Dataset* object.
        KeyError:
            If the reminder Dataset object has less than one input or one output.
        """

        ds = deepcopy(self)

        available_signals = [s[1] for s in self.signal_list()]
        signals_not_found = [s for s in signals if s not in available_signals]
        if signals_not_found:
            raise KeyError(
                f"Signal(s)) {signals_not_found} not found "
                f"in Dataset '{self.name}'."
            )

        for s in signals:
            # Input detected
            cond1 = (
                s in list(ds.dataset["INPUT"].columns.get_level_values("names"))
                and len(
                    list(ds.dataset["INPUT"].columns.get_level_values("names"))
                )
                > 1
            )

            # Output detected
            cond2 = (
                s
                in list(ds.dataset["OUTPUT"].columns.get_level_values("names"))
                and len(
                    list(ds.dataset["OUTPUT"].columns.get_level_values("names"))
                )
                > 1
            )

            # Remove Signal
            if cond1 or cond2:
                ds.dataset = ds.dataset.drop(s, axis=1, level="names")
                del ds._nan_intervals[s]
            else:
                raise KeyError(
                    f"Cannot remove {s}. The dataset must have at least "
                    "one input and one output."
                )

        return ds


# ====================================================
# Useful functions
# ====================================================


def _list_to_structured_list_of_tuple(
    tpl: tuple[Any], lst: list[str]
) -> list[tuple[Any]]:
    """
    MISSING DOCSTRUNG

    Parameters
    ----------
    tpl : tuple[Any]
        DESCRIPTION.
    lst : list[str]
        DESCRIPTION.

    Returns
    -------
    list[tuple[Any]]
        DESCRIPTION.

    """
    # Convert a plain list to a list of tuple of a given structure, i.e.
    # Given tpl = [("a0","a1"),("b0",),("b1","a1","b0"),("a0","a1"),("b0",)]
    # and lst = ["u0", "u1", "u2", "u3", "u4", "u5", "u6", "u7" , "u8"]
    # it returns [("u0", "u1"), ("u2",), ("u3", "u4", "u5"), ("u6", "u7") , ("u8",)]
    R = []
    idx = 0
    for ii in [len(jj) for jj in tpl]:
        R.append(tuple(lst[idx : idx + ii]))
        idx += ii
    return R
    # it = iter(lst)
    # return [tuple(itertools.islice(it, len(t))) for t in tpl]


def change_axes_layout(
    fig: matplotlib.axes.Figure,
    axes: matplotlib.axes.Axes,
    nrows: int,
    ncols: int,
) -> tuple[matplotlib.axes.Figure, matplotlib.axes.Axes]:
    """Change *Axes* layout of an existing Matplotlib *Figure*."""

    old_nrows: int = axes.shape[0]
    old_ncols: int = axes.shape[1]
    # Remove all old axes
    for ii in range(old_nrows):
        for jj in range(old_ncols):
            axes[ii, jj].remove()
    # New number of rows and columns
    # Set gridspec according to new new_nrows and new_ncols
    gs = gridspec.GridSpec(nrows, ncols, figure=fig)
    axes = np.ndarray((nrows, ncols), dtype=matplotlib.axes.SubplotBase)
    # Add new axes
    for ii in range(nrows):
        for jj in range(ncols):
            axes[ii, jj] = fig.add_subplot(gs[ii, jj], sharex=axes[0, 0])
    return fig, axes


def validate_signals(*signals: Signal) -> None:
    """
    Perform a number of checks to verify that the passed
    list of :py:class:`Signals <dymoval.dataset.Dataset>`
    can be used to create a Dataset.

    Every :py:class:`Signal <dymoval.dataset.Signal>` in the *signal_list*
    parameter must have all the attributes adequately set.

    To figure how the attributes shall be set, look at the *RAISES* section below.

    Parameters
    ----------
    *signal :
        Signal to be validated.

    Raises
    ------
    ValueError
        If values is not a *1-D numpy ndarray*,
        or if sampling period must positive or the signals *time_unit* keys are
        not the same.
    KeyError
        If signal attributes are not found or not allowed or if signal names are
        not unique,
    IndexError
        If signal have less than two samples.
    TypeError
        If values is not a *1-D numpy ndarray*, or if sampling period is not a *float*.
    """

    # Name unicity
    signal_names = [s["name"] for s in signals]
    if len(signal_names) > len(set(signal_names)):
        raise KeyError("Signal names are not unique")
    #
    for s in signals:
        # Check that the user wrote the necessary keys
        not_found_keys = difference_lists_of_str(
            list(SIGNAL_KEYS), list(s.keys())
        )
        if not_found_keys:
            raise KeyError(
                f"Key {not_found_keys} not found in signal {s['name']}."
            )

        # Check that "values" makes sense
        cond = not isinstance(s["values"], np.ndarray) or s["values"].ndim != 1
        if cond:
            raise TypeError("Key {key} must be 1-D numpy array'.")
        if s["values"].size < 2:
            raise IndexError(
                "Signal {s[name']} has only one sample.",
                "A signal must have at least two samples.",
            )

        # sampling period check
        if not isinstance(s["sampling_period"], float):
            raise TypeError("Key 'sampling_period' must be a positive float.")
        if s["sampling_period"] < 0.0 or np.isclose(
            s["sampling_period"], 0.0, atol=ATOL
        ):
            raise ValueError("Key 'sampling_period' must be a positive float.")
        # Check that all signals have been sampled with the same time_unit

    time_units = [s["time_unit"] for s in signals]
    if len(set(time_units)) > 1:
        raise ValueError("All the signals time units shall be the same.")


def validate_dataframe(
    df: pd.DataFrame,
    u_names: str | list[str],
    y_names: str | list[str],
) -> None:
    """
    Check if a pandas DataFrame is suitable for instantiating
    a :py:class:`Dataset <dymoval.dataset.Dataset>` object.

    The index of the DataFrame shall represent the common timeline for all the
    signals, whereas the *j-th* column shall represents the realizations
    of the *j-th* signal.

    It must be specified which signal(s) are the input through the *u_names* and
    which signal(s) is the output through the  *y_names* parameters.

    The candidate DataFrame shall meet the following requirements

    - Columns names shall be unique,
    - Columns name shall be a tuple of *str* of the form *(name,unit)*,
    - The index shall represent the common timeline for all  and its
      name shall be *'(Time, s)'*
    - Each signal must have at least two samples (i.e. the DataFrame has at least two rows),
    - Only one index and columns levels are allowed (no *MultiIndex*),
    - There shall be at least two signals representing one input and one output,
    - Each signal in the *u_names* and *y_names* must be equal to exactly,
      one column name of the input DataFrame.
    - Both the index values and the column values must be *float* and the index values
      must be a a 1D vector of monotonically, equi-spaced, increasing *floats*.

    Parameters
    ----------
    df :
        DataFrame to be validated.
    u_names :
        List of input signal names.
    y_names :
        List of output signal names.

    Raises
    ------
    Error:
        Depending on the issue found an appropriate error will be raised.
    """
    # ===================================================================
    # Checks performed: Read below
    # ===================================================================

    u_names = str2list(u_names)  # noqa
    y_names = str2list(y_names)  # noqa

    # ==========================================
    # u_names and y_names arguments check
    # ========================================

    # Check that you have at least one input and one output
    if not u_names or not y_names:
        raise IndexError(
            "You need at least one input and one output signal. "
            "Check 'u_names' and 'y_names'."
        )

    # Check unicity of signal names
    if (
        len(u_names) != len(set(u_names))
        or len(y_names) != len(set(y_names))
        or (set(u_names) & set(y_names))  # Non empty intersection
    ):
        raise ValueError(
            "Signal names must be unique. Check 'u_names' and 'y_names'."
        )

    # ==========================================
    # DataFrame check
    # ========================================

    # Check that all elements in columns are tuples
    cond1 = all(isinstance(sig, tuple) for sig in df.columns)
    # Check that each tuple has len == 2
    cond2 = all(len(sig) == 2 for sig in df.columns)
    if not cond1 or not cond2:
        raise TypeError(
            "Each column name shall be of the form (name,unit), where 'name' and 'unit' are strings."
        )

    # check if the index is a tuple
    if not isinstance(df.index.name, tuple):
        raise TypeError("Index name must be a tuple ('Time',unit).")

    # Check that each component of the tuple is a string
    available_names, available_units = list(zip(*df.columns))
    available_names = list(available_names)
    available_units = list(available_units)

    cond1 = all(isinstance(name, str) for name in available_names)
    cond2 = all(isinstance(unit, str) for unit in available_units)

    if not cond1 or not cond2:
        raise TypeError(
            "Each column name shall be of the form (name,unit), where 'name' and 'unit' are strings."
        )

    # Check that the DataFrame must have only one index and columns levels
    #     if df.columns.nlevels > 1 or df.index.nlevels > 1:
    #         raise IndexError(
    #             "The number index levels must be one for both the index and the columns.",
    #             "The index shall represent a time vector of equi-distant time instants",
    #             "and each column shall correspond to one signal values.",
    #         )
    #
    # At least two samples
    if df.index.size < 2:
        raise IndexError("A signal needs at least two samples.")

    # Check if u_names and y_names exist in the passed DataFrame
    input_not_found = difference_lists_of_str(u_names, available_names)  # noqa
    if input_not_found:
        raise ValueError(f"Input(s) {input_not_found} not found.")
    # Check output
    output_not_found = difference_lists_of_str(y_names, available_names)  # noqa
    if output_not_found:
        raise ValueError(f"Output(s) {output_not_found} not found.")

    # The index is a 1D vector of monotonically increasing floats.
    # OBS! Builtin methds df.index.is_monotonic_increasing combined with
    # df.index.is_unique won't work due to floats.
    sampling_period = df.index[1] - df.index[0]
    if not np.all(np.isclose(np.diff(df.index), sampling_period, atol=ATOL)):
        raise ValueError(
            "Index must be a 1D vector of monotonically",
            "equi-spaced, increasing floats.",
        )

    # The dataframe elements are all floats
    if not all(df.dtypes == float):
        raise TypeError("Elements of the DataFrame must be float.")


def plot_signals(
    *signals: Signal,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """
    Plot :py:class:`Signals <dymoval.dataset.Signal>`.

    Parameters
    ----------
    *signals :
        :py:class:`Signals <dymoval.dataset.Signal>` to be plotted.
    """
    # Validate signals first
    validate_signals(*signals)

    # Plot raw signals
    signal_names = [s["name"] for s in signals]
    n = len(signals)
    nrows, ncols = factorize(n)  # noqa
    fig, ax = plt.subplots(nrows, ncols, squeeze=False, sharex=True)
    ax = ax.T.flat
    for ii, s in enumerate(signals):
        timeline = np.linspace(
            0.0, len(s["values"]) * s["sampling_period"], len(s["values"])
        )
        ax[ii].plot(timeline, s["values"], label=signal_names[ii])
        ax[ii].text(
            0.8,
            0.8,
            f"$T_s$ = {s['sampling_period']} s",
            bbox=dict(facecolor="yellow", alpha=0.8),
        )
        ax[ii].grid()
        ax[ii].legend()
        ax[ii].set_xlabel(f"Time ({s['time_unit']})")
        ax[ii].set_ylabel(f"({s['signal_unit']})")

    fig.suptitle("Raw signals.")
    # fig.tight_layout()

    return fig, ax.base


def compare_datasets(
    *datasets: Dataset,
    kind: Literal["time", "coverage"] | Spectrum_type = "time",
) -> None:
    """
    Compare different :py:class:`Datasets <dymoval.dataset.Dignal>` graphically
    by overlapping them.


    Parameters
    ----------
    *datasets :
        :py:class:`Datasets <dymoval.dataset.Dataset>` to be compared.
    kind:
        Kind of graph to be plotted.
    """

    # Utility function to avoid too much code repetition
    def _adjust_legend(ds_names: list[str], axes: matplotlib.axes.Axes) -> None:
        # Based on the pair (handles, labels), where handles are e.g. Line2D,
        # or other matplotlib Artist (an Artist is everything you can draw
        # on a figure)

        axes = axes.T.flat
        for ii, ax in enumerate(axes):
            handles, labels = ax.get_legend_handles_labels()
            # print(handles)
            # print(labels)
            # Be sure that your plot show legends!
            if labels:
                new_labels = [
                    ds_names[jj] + ", " + labels[jj]
                    for jj, _ in enumerate(ds_names)
                    if jj < len(labels)
                ]
            ax.legend(handles, new_labels)

    def _arrange_fig_axes(
        *dfs: pd.DataFrame,
    ) -> tuple[matplotlib.axes.Figure, matplotlib.axes.Axes, int, int]:
        # When performing many plots on the same figure,
        # it find the largest number of axes needed

        # Find the larger dataset
        p_max = max([len(df["INPUT"].columns) for df in dfs])
        q_max = max([len(df["OUTPUT"].columns) for df in dfs])
        n = p_max + q_max

        # Set nrows and ncols
        nrows, ncols = factorize(n)

        # Create a unified figure
        fig, ax = plt.subplots(nrows, ncols, sharex=True, squeeze=False)
        return fig, ax, p_max, q_max

    # ========================================
    #    MAIN IMPLEMENTATION
    # ========================================
    # arguments validation
    for ds in datasets:
        if not isinstance(ds, Dataset):
            raise TypeError("Input must be a dymoval Dataset type.")

    # ========================================
    # time comparison
    # ========================================
    if kind == "time" or kind == "all":

        # Arrange figure
        # Accumulate all the dataframes at signal_name level
        dfs = [ds.dataset.droplevel(level="units", axis=1) for ds in datasets]
        fig_time, axes_time, p_max, _ = _arrange_fig_axes(*dfs)

        # Revert axes.ndarray to flatirr
        # axes_time = axes_time.T.flat

        # All the plots made on the same axis
        cmap = plt.get_cmap(COLORMAP)
        for ii, ds in enumerate(datasets):
            axes_time = ds.plot(
                line_color_input=cmap(ii),
                line_color_output=cmap(ii),
                ax=axes_time,
                p_max=p_max,
            )

        # Adjust legend
        ds_names = [ds.name for ds in datasets]
        _adjust_legend(ds_names, axes_time)
        fig_time.suptitle("Dataset comparison")
        # fig_time.tight_layout()

    # ========================================
    # coverage comparison
    # ========================================
    if kind == "coverage" or kind == "all":

        dfs = [ds.dataset.droplevel(level="units", axis=1) for ds in datasets]

        # The following is a bit of code repetition but I could not come
        # with a better idea.

        # INPUT
        p_max = max([len(df["INPUT"].columns) for df in dfs])
        nrows, ncols = factorize(p_max)
        fig_cov_in, ax_cov_in = plt.subplots(nrows, ncols, squeeze=False)
        ax_cov_in = ax_cov_in.T.flat

        # OUTPUT
        q_max = max([len(df["OUTPUT"].columns) for df in dfs])
        nrows, ncols = factorize(q_max)
        fig_cov_out, ax_cov_out = plt.subplots(nrows, ncols, squeeze=False)

        # Actual plot
        cmap = plt.get_cmap(COLORMAP)  # noqa
        for ii, ds in enumerate(datasets):
            axes_cov_in, axes_cov_out = ds.plot_coverage(
                line_color_input=cmap(ii),
                line_color_output=cmap(ii),
                ax_in=ax_cov_in,
                ax_out=ax_cov_out,
            )

        # Adjust input legend
        ds_names = [ds.name for ds in datasets]
        _adjust_legend(ds_names, axes_cov_in)
        _adjust_legend(ds_names, axes_cov_out)

        fig_cov_in.suptitle("Dataset comparison")
        fig_cov_out.suptitle("Dataset comparison")

        # fig_cov_in.tight_layout()
        # fig_cov_out.tight_layout()

    # ========================================
    # frequency comparison
    # ========================================
    if kind in SPECTRUM_KIND or kind == "all":
        # plot_spectrum won't accept kind = "all"
        if kind == "all":
            kind = "power"

        # Arrange figure
        # Accumulate all the dataframes at signal_name level
        dfs = [ds.dataset.droplevel(level="units", axis=1) for ds in datasets]
        fig_freq, axes_freq, p_max, _ = _arrange_fig_axes(*dfs)
        # axes_freq = axes_freq.T.flat

        if kind == "amplitude":
            nrows_old: int = axes_freq.shape[0]
            ncols_old: int = axes_freq.shape[1]

            # Due to (abs,angle) we need to double the number of axes
            nrows, ncols = factorize(2 * nrows_old * ncols_old)

            # To have the phase plot below the abs plot, then the number
            # of rows must be an even number, otherwise the plot got screwed.
            # OBS: the same code is in plot_spectrum()
            if np.mod(nrows, 2) != 0:
                nrows -= 1
                ncols += int(np.ceil(ncols / nrows))

            fig_freq, axes_freq = change_axes_layout(
                fig_freq, axes_freq, nrows, ncols
            )
            # axes_freq = axes_freq.T.flat

        # All datasets plots on the same axes
        cmap = plt.get_cmap(COLORMAP)  # noqa
        for ii, ds in enumerate(datasets):
            axes_freq = ds.plot_spectrum(
                line_color_input=cmap(ii),
                line_color_output=cmap(ii),
                ax=axes_freq,
                # p_max=p_max,
                kind=kind,  # type:ignore
            )

        # Adjust legend
        ds_names = [ds.name for ds in datasets]
        _adjust_legend(ds_names, axes_freq)
        fig_freq.suptitle("Dataset comparison")
        # fig_freq.tight_layout()


# def analyze_inout_dataset(df):
# -	Remove trends. Cannot do.
# -	(Resample). One line of code with pandas
#
