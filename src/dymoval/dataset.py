# The following are only for Spyder, otherwise things are written in
# the pyproject.toml
# mypy: show_error_codes

"""Module containing everything related to datasets.
Here are defined special datatypes, classes and auxiliary functions.
"""
from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt
from scipy import io, fft
from .config import *  # noqa
from .utils import *  # noqa, Type
from typing import TypedDict, Any, Literal
from copy import deepcopy

# %%
a = 4
b = 3


# %%
c = 9
d = 69

# %%


class Signal(TypedDict):
    """
    :py:class:`Signals <dymoval.dataset.Signal>` are used to represent real-world measurements.


    They are used to instantiate :py:class:`Dataset <dymoval.dataset.Dataset>` objects.
    Before instantiating a :py:class:`Dataset <dymoval.dataset.Dataset>` object, it is good practice
    to validate
    :py:class:`Signals <dymoval.dataset.Signal>`
    through the :py:meth:`~dymoval.dataset.validate_signals` function.

    Although
    :py:class:`Signals <dymoval.dataset.Signal>`
    have compulsory attribtues, there is freedom
    to append additional ones.


    Example
    -------
    >>> # How to create a simple dymoval Signal
    >>> import dymoval as dmv
    >>>
    >>> my_signal: dmv.Signal = {
    "name": "speed",
    "values": np.random.rand(100),
    "signal_unit": "mps",
    "sampling_period": 0.1,
    "time_unit": "s",
    }
    >>> dmv.plot_signals(my_signal)


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
    """The :py:class:`Dataset <dymoval.dataset.Dataset>` class stores
    the candidate signals to be used as a dataset
    and it provides methods for analyzing and manipulating them.

    A :py:class:`Signal <dymoval.dataset.Signal>` list shall be passed
    to the initializer along with:

    1. the list of signal names that shall be considered as **input**
    2. the list of signal names that shall be considered as **output**

    The initializer will attempt to resample all the signals to have
    the same sampling period.
    Signals that cannot be resampled will be excluded from the
    :py:class:`Dataset <dymoval.dataset.Dataset>`
    and will be stored in the
    :py:attr:`<~dymoval.dataset.Dataset.excluded_signals>`
    attribute.

    Furthermore, all the signals will be trimmed to have the
    same length.

    If none of **tin, tout** and **full_time_interval** arguments
    are passed, then the
    :py:class:`Dataset <dymoval.dataset.Dataset>`
    time-interval selection is done graphically.


    Example
    -------
    >>> # How to create a Dataset object from a list of Signals
    >>>
    >>> import numpy as np
    >>> import dymoval as dmv
    >>>
    >>> signal_names = [
    >>>     "SpeedRequest",
    >>>     "AccelPedalPos",
    >>>     "OilTemp",
    >>>     "ActualSpeed",
    >>>     "SteeringAngleRequest",
    >>> ]
    >>> signal_values = np.random.rand(5, 100)
    >>> signal_units = ["m/s", "%", "°C", "m/s", "deg"]
    >>> sampling_periods = [0.1, 0.1, 0.1, 0.1, 0.1]
    >>> # Create dymoval signals
    >>> signals = []
    >>> for ii, val in enumerate(signal_names):
    >>>     tmp: dmv.Signal = {
    >>>         "name": val,
    >>>         "values": signal_values[ii],
    >>>         "signal_unit": signal_units[ii],
    >>>         "sampling_period": sampling_periods[ii],
    >>>         "time_unit": "s",
    >>>     }
    >>>     signals.append(tmp)
    >>> # Validate signals
    >>> dmv.validate_signals(*signals)
    >>> # Specify which signals are inputs and which signals are output
    >>> input_labels = ["SpeedRequest", "AccelPedalPos", "SteeringAngleRequest"]
    >>> output_labels = ["ActualSpeed", "OilTemp"]
    >>> # Create dymoval Dataset objects
    >>> ds = dmv.Dataset("my_dataset", signals, input_labels, output_labels)
    >>>
    >>> # At this point you can plot, trim, manipulate, analyze, etc you dataset
    >>> # through the Dataset class methods.

    You can also create :py:class:`Dataset <dymoval.dataset.Dataset>` objects
    from  *pandas DataFrames* if the DataFrames have a certain structure.
    Look at :py:meth:`~dymoval.dataset.validate_dataframe`
    for more information about this option.


    Example
    -------
    >>> # How to create a Dataset object from a pandas DataFrame
    >>>
    >>> import dymoval as dmv
    >>> import pandas as pd
    >>>
    >>> # Signals names, units and values
    >>> signal_names = [
    >>>     "SpeedRequest",
    >>>     "AccelPedalPos",
    >>>     "OilTemp",
    >>>     "ActualSpeed",
    >>>     "SteeringAngleRequest",
    >>> ]
    >>> signal_units = ["m/s", "%", "°C", "m/s", "deg"]
    >>> signal_values = np.random.rand(100, 5)
    >>>
    >>> # time axis
    >>> sampling_period = 0.1
    >>> timestamps = np.arange(0, 10, sampling_period)
    >>>
    >>> # Build a candidate DataFrame
    >>> cols = list(zip(signal_names, signal_units))
    >>> index_name = ("Time", "s")
    >>>
    >>> # Create dymoval signals
    >>> df = pd.DataFrame(data=signal_values, columns=cols)
    >>> df.index = pd.Index(data=timestamps, name=index_name)
    >>>
    >>> # Check if the dataframe is suitable for a dymoval Dataset
    >>> dmv.validate_dataframe(df)
    >>>
    >>> # Specify input and output signals
    >>> input_labels = ["SpeedRequest", "AccelPedalPos", "SteeringAngleRequest"]
    >>> output_labels = ["ActualSpeed", "OilTemp"]
    >>>
    >>> # Create dymoval Dataset objects
    >>> ds = dmv.Dataset("my_dataset", df, input_labels, output_labels)



    Parameters
    ----------
    name:
        Dataset name.
    signal_list :
        Signals to be included in the :py:class:`Dataset <dymoval.dataset.Dataset>` object.
    u_names :
        List of input signal names. Each signal name must be unique and must be
        contained in the passed **signal_list** names.
    y_names :
        List of input signal names. Each signal name must be unique and must be
        contained in the passed **signal_list** names.
    target_sampling_period :
        The passed signals will be re-sampled at this sampling period.
        If some signal could not be resampled, then its name will be added in the
        :py:attr:`~dymoval.dataset.Dataset.excluded_signals` attr.
    tin :
        Initial time instant of the :py:class:`Dataset <dymoval.dataset.Dataset>`.
    tout :
        Final time instant of the :py:class:`Dataset <dymoval.dataset.Dataset>`.
    full_time_interval :
        If *True*, the :py:class:`Dataset <dymoval.dataset.Dataset>`
        time interval will be equal to the longest
        time interval among all of the signals included in the **signal_list**
        parameter.
        This is overriden if the parameters **tin** and **tout** are specified.
    overlap :
        If *True* it will overlap the input and output signals plots
        in the
        :py:class:`Dataset <dymoval.dataset.Dataset>`
        time interval graphical selection.
        The units of the outputs are displayed on the secondary y-axis.
    verbosity:
        Display information depending on its level.
        Higher numbers correspond to higher verbosity.
    """

    #  Raises
    #  -----
    #  TypeError
    #      If the *signals_list* has the wrong datatype.

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
        # ==============================
        # Instance attributes
        # ==============================
        # Here are declared but they will be initialized
        # in the method _new_dataset_from_dataframe()
        self.name: str = "foo"
        self.dataset: pd.DataFrame = pd.DataFrame()
        self.coverage: pd.DataFrame = pd.DataFrame()
        self.information_level: float = 0.0
        self._nan_intervals: dict[str, list[np.ndarray]] = {}
        self.excluded_signals: list[str] = []
        self.sampling_period: float = 1.0

        # ==============================
        # Initialization functions
        # ==============================
        # Initialization by Signals.
        # It will call _new_dataset_from_dataframe()
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
        # Function for cfiltering out only the signals present in the dataset
        # Note that only signals in the dataset can have NaN:s
        def filter_signals(
            avail_signals: list[str], ax: matplotlib.axes.Axes
        ) -> list[tuple[matplotlib.lines.Line2D, str]]:
            # Implementation
            lines, labels = ax.get_legend_handles_labels()
            lines_labels_all = list(zip(lines, labels))
            lines_labels_filt = []
            for line_label in lines_labels_all:
                for s in available_signals:
                    if s in line_label[1]:  # check if s is in the legend
                        # if so, we save the associated line (to get the color)
                        lines_labels_filt.append((line_label[0], s))

            return lines_labels_filt

        # ==============================================
        # Main function
        # ==============================================
        # Reference to self._nan_intervals
        NaN_intervals = self._nan_intervals
        # All the signals present in the dataset
        available_signals = list(self.dataset.columns.get_level_values("names"))

        for ii, ax in enumerate(axes):
            lines_labels = filter_signals(available_signals, ax)
            # if lines_labels:  # TODO: maybe this is always verified
            for line, signal in lines_labels:
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
        tin: float = self.dataset.index[0]
        timeVectorFromZero: np.ndarray = self.dataset.index - tin
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
                    nan_chunk_translated, NUM_DECIMALS
                )  # noqa
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
        #
        # =============== Arguments validation ========================
        validate_dataframe(df)

        if tin and tout and tin > tout:
            raise ValueError(
                f" Value of tin ( ={tin}) shall be smaller than the value of tout ( ={tout})."
            )

        # If the user passes a str cast into a list[str]
        u_names = str2list(u_names)
        y_names = str2list(y_names)
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

        # Check if u_names and y_names exist in the passed DataFrame
        available_names, available_units = list(zip(*df.columns))
        available_names = list(available_names)
        input_not_found = difference_lists_of_str(
            u_names, available_names
        )  # noqa
        if input_not_found:
            raise ValueError(f"Input(s) {input_not_found} not found.")
        # Check output
        output_not_found = difference_lists_of_str(
            y_names, available_names
        )  # noqa
        if output_not_found:
            raise ValueError(f"Output(s) {output_not_found} not found.")
        # ================== End of validation =======================

        # ================= Start to fill in attributes =========
        # NOTE: You have to use the #: to add a doc description
        # in class attributes (see sphinx.ext.autodoc)
        # Set easy-to-set attributes
        self.name = name  #: Dataset name.
        self.information_level = 0.0  #: *Not implemented yet!*
        self.sampling_period = np.round(
            df.index[1] - df.index[0], NUM_DECIMALS
        )  #: Dataset sampling period.

        # Excluded signals list is either passed by _new_dataset_from_signals()
        # or it is empty if a dataframe is passed by the user (all the signals
        # in this case shall be sampled with the same sampling period).
        self.excluded_signals = _excluded_signals
        """Signals that could not be re-sampled."""
        # ==================================================

        # Keep only the signals specified by the user and respect the order
        # This is helpful in case the df is automatically imported e.g.
        # from a large csv but we only want few signals
        # Filter columns
        u_cols = [u for u in df.columns if u[0] in u_names]
        y_cols = [y for y in df.columns if y[0] in y_names]

        # Order columns according to passed order
        u_cols.sort(key=lambda x: u_names.index(x[0]))
        y_cols.sort(key=lambda x: y_names.index(x[0]))
        df = df.reindex(columns=u_cols + y_cols)

        # Fix data and index
        data = df.to_numpy()
        index = df.index

        # Adjust column MultiIndex
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
        self.dataset = df_ext  #: The actual dataset

        # Initialize NaN intervals, full time interval
        self._nan_intervals = self._find_nan_intervals()

        # Initialize coverage region
        self.coverage = self._find_dataset_coverage()  # Docstring below,
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
        plot_kwargs = {
            "overlap": overlap,
            "linecolor_input": "b",
            "linestyle_fg": "-",
            "alpha_fg": 1.0,
            "linecolor_output": "green",
            "linestyle_bg": "--",
            "alpha_bg": 1.0,
            "_grid": None,
            "layout": None,
            "ax_height": 1.8,
            "ax_width": 4.445,
        }
        tmp = self.trim(
            *signals,
            tin=tin,
            tout=tout,
            verbosity=verbosity,
            # overlap=overlap,
            **plot_kwargs,
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
        # All attributes are initialized in the _new_dataset_from_dataframe() method

        # Arguments validation
        validate_signals(*signal_list)

        # If the user pass a single signal as a str,
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
            name=("Time", resampled_signals[0]["time_unit"]),
        )
        columns_tuples = list(zip(u_names + y_names, u_units + y_units))
        df = pd.DataFrame(
            index=index,
            data=df_data,
            columns=columns_tuples,
        )

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
    ) -> tuple[dict[str, str], dict[str, str]]:
        # You pass a list of signal names and the function recognizes who is input
        # and who is output. The dicts are name:unit
        # Is no argument is passed, then the whole for u_names and y_names is taken
        # If only input or output signals are passed, then an empty list
        # for the non-returned labels is returned.
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

        # Classify in IN and OUT in case signals are passed.
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

        return u_dict, y_dict

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
        u_dict, y_dict = self._classify_signals(*signals)

        u_names = list(u_dict.keys())
        y_names = list(y_dict.keys())

        u_list = [(s[0], s[1]) for s in signals_values if s[0] in u_names]
        y_list = [(s[0], s[1]) for s in signals_values if s[0] in y_names]

        return u_dict, y_dict, u_list, y_list

    def _select_signals_to_plot(
        self, *signals: str | tuple[str, str] | None, overlap: bool
    ) -> tuple[
        dict[str, str], dict[str, str], list[tuple[str, ...]], list[str]
    ]:
        # ===================================================
        # Selection of signals to plot
        # ===================================================
        # Returns the signals passed by the user.
        # The signals passed by the user can be a mixed of tuples and strings
        # If None has passed, it takes all the signals in the Dataset instance.
        # overlap parameter override eventual signals passed by the user

        # df points to self.dataset
        df = self.dataset

        # All possible names
        u_names = list(df["INPUT"].columns.get_level_values("names"))
        y_names = list(df["OUTPUT"].columns.get_level_values("names"))

        if overlap:
            # OBS! zip cuts the longest list, so we have to manually add
            # the lefotovers
            p = len(u_names) - len(y_names)
            leftovers = u_names[p + 1 :] if p > 0 else y_names[p + 1 :]
            signals_from_user = list(zip(u_names, y_names)) + leftovers
        # Convert passed tuple[tuple|str] to list[tuple|str]
        elif signals:
            signals_from_user = list(signals)
        else:
            signals_from_user = u_names + y_names

        # Make all tuples like [('u0', 'u1'), ('y0',), ('u1', 'y1', 'u0')]
        # for easier indexining
        for ii, s in enumerate(signals_from_user):
            if isinstance(s, str):
                signals_from_user[ii] = (s,)

        # signals_lst_plain, e.g. ['u0', 'u1', 'y0', 'u1', 'y1', 'u0']
        signals_lst_plain = [item for t in signals_from_user for item in t]

        # validation
        (
            u_dict,
            y_dict,
        ) = self._classify_signals(*signals_lst_plain)

        return u_dict, y_dict, signals_from_user, signals_lst_plain

    def _create_plot_args(
        self,
        kind: Literal["time", "coverage"] | Spectrum_type,
        u_dict: dict[str, str],
        y_dict: dict[str, str],
        signals_lst_plain: list[str],
        signals_tpl: list[tuple[str, ...]],
        linecolor_input: str,
        linecolor_output: str,
        linestyle_fg: str,
        linestyle_bg: str,
    ) -> tuple[
        list[tuple[str, ...]],
        list[tuple[str, ...]],
        list[tuple[str, ...]],
        list[tuple[str, ...]],
    ]:
        # signals_tpl is passed only to be a reference for casting
        # signals_lst_plain, which is e.g.
        # signals_lst_plain = ["u1","u2","u3", "y1","y4",...]
        # to a structure that resemble signals_tpl, e.g.
        # signals_tpl[("u1","u2"),("u3",), ("y1","y4")...]

        # linecolors
        linecolors = [
            linecolor_input if s in u_dict.keys() else linecolor_output
            for s in signals_lst_plain
        ]
        linecolors_tpl = _list_to_structured_list_of_tuple(
            signals_tpl, linecolors
        )

        # ylabels (units)
        s_dict = deepcopy(u_dict)
        s_dict.update(y_dict)
        if kind in ["time", "coverage", "amplitude"]:
            ylabels = [f"({s_dict[s]})" for s in signals_lst_plain]
        if kind == "power":
            ylabels = [f"(({s_dict[s]})^2)" for s in signals_lst_plain]
        elif kind == "psd":
            ylabels = [f"(({s_dict[s]})^2/Hz)" for s in signals_lst_plain]

        # Create a structure with all tuples e.g. [("u1",),("y1","u3"),("y4",)]
        ylabels_tpl = _list_to_structured_list_of_tuple(signals_tpl, ylabels)

        # labels (=legend entries)
        # OBS: each Artist has a label. This is what is used
        # in the legend. So, for each Artist (e.g. Line2D) you should
        # set the label with Artist.set_label().
        # Note that ax.legend(handles,labels) will only display
        # with what you want, but it won't change the Artist label.
        if kind != "amplitude":
            labels = signals_lst_plain
        else:
            labels = signals_lst_plain

        labels_tpl = _list_to_structured_list_of_tuple(signals_tpl, labels)

        # Linestyles
        linestyles_tpl: list[tuple[str, ...]] = []
        for val in linecolors_tpl:
            if len(val) == 2:
                if val[0] == val[1]:
                    linestyles_tpl.append((linestyle_bg, linestyle_fg))
                else:
                    linestyles_tpl.append((linestyle_fg, linestyle_fg))
            else:
                linestyles_tpl.append((linestyle_fg,))

        return linecolors_tpl, ylabels_tpl, linestyles_tpl, labels_tpl

    def _plot_actual(
        self,
        df: pd.DataFrame,
        signals_tpl: list[tuple[str, ...]],
        grid: matplotlib.gridspec.GridSpec,
        linecolors_tpl: list[tuple[str, ...]],
        ylabels_tpl: list[tuple[str, ...]],
        labels_tpl: list[tuple[str, ...]],
        linestyles_tpl: list[tuple[str, ...]],
        alpha_fg: float,
        alpha_bg: float,
    ) -> matplotlib.figure.Figure:
        # This is the function who makes the actual plot once all the
        # parameters are set and passed
        # Adjust passed dataframe.

        # Initialize iteration
        fig = grid.figure
        if fig.get_axes():
            # TODO This indexing shall be improved
            axes_tpl = _list_to_structured_list_of_tuple(
                signals_tpl, fig.get_axes()
            )
        else:
            axes_tpl = []
            axes = fig.add_subplot(grid[0])

        # Iterations
        for ii, s in enumerate(signals_tpl):
            # at each iteration a new axes who sharex with the
            # previously created axes, is created
            # However, a new axes is created only if it does not
            # exists in the givem grid position.
            if len(axes_tpl) > 0:
                # Reuse existing axes if exist
                axes = axes_tpl[ii][0]
            else:
                # Otherwise create a new axes
                axes = fig.add_subplot(grid[ii], sharex=axes)

            df.droplevel(level=["kind", "units"], axis=1).loc[:, s[0]].plot(
                subplots=True,
                grid=True,
                color=linecolors_tpl[ii][0],
                linestyle=linestyles_tpl[ii][0],
                alpha=alpha_fg,
                ylabel=ylabels_tpl[ii][0],
                ax=axes,
            )
            # Grt handle
            line_l, _ = axes.get_legend_handles_labels()
            # Update label
            label_l = [labels_tpl[ii][0]]
            line_l[0].set_label(*label_l)
            # In case the user wants to overlap plots...
            # If the overlapped plots have the same units, then there is
            # no point in using a secondary_y
            line_r = []
            label_r = []
            if len(s) == 2:  # tuple like ("u1","u2")
                if len(axes_tpl) > 0:  # TODO: this may be alwaye true.
                    axes_right = axes_tpl[ii][1]
                else:
                    # Otherwise create a new axes
                    axes_right = axes.twinx()

                # If the two signals have the same unit, then show the unit only once
                if ylabels_tpl[ii][0] == ylabels_tpl[ii][1]:
                    ylabel = None
                else:
                    ylabel = ylabels_tpl[ii][1]

                df.droplevel(level=["kind", "units"], axis=1).loc[:, s[1]].plot(
                    subplots=True,
                    grid=False,
                    legend=False,
                    color=linecolors_tpl[ii][1],
                    linestyle=linestyles_tpl[ii][1],
                    alpha=alpha_bg,
                    ylabel=ylabel,
                    ax=axes_right,
                )

                # Get handle (and ignore label)
                # (in this case only one Line2D and one str)
                (
                    line_r,
                    _,
                ) = axes_right.get_legend_handles_labels()  # type:ignore
                # Update label
                label_r = [labels_tpl[ii][1]]
                line_r[0].set_label(*label_r)

            # Set nice legend
            axes.legend(line_l + line_r, label_l + label_r)

        # Remove dummy axes created at the beginning
        axes_all = fig.get_axes()
        for ax in axes_all:
            if len(ax.get_lines()) == 0:
                ax.remove()

        return fig

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

    def _add_signals(self, kind: Signal_type, *signals: Signal) -> Dataset:
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
        NaN_intervals = self._find_nan_intervals()
        ds._nan_intervals.update(
            NaN_intervals
        )  # Join two dictionaries through update()

        ds.dataset = ds.dataset.round(NUM_DECIMALS)
        return ds

    def trim(
        self: Dataset,
        *signals: str | tuple[str, str] | None,
        tin: float | None = None,
        tout: float | None = None,
        verbosity: int = 0,
        **kwargs: Any,
    ) -> Dataset:
        """
        Trim the Dataset
        :py:class:`Dataset <dymoval.dataset.Dataset>` object.

        If not *tin* or *tout* are passed, then the selection is
        made graphically.

        Parameters
        ----------
        *signals :
            Signals to be plotted in case of trimming from a plot.
        tin :
            Initial time of the desired time interval
        tout :
            Final time of the desired time interval.
        verbosity :
            Depending on its level, more or less info is displayed.
            The higher the value, the higher is the verbosity.
        **kwargs:
            kwargs to be passed to the
            :py:meth:`Dataset <dymoval.dataset.Dataset.plot>` method.

        """
        # We have to trim the signals to have a meaningful dataset
        # This can be done both graphically or by passing tin and tout
        # if the user knows them before hand or by setting full_time_interval = True.
        # Once done, the dataset shall be automatically shifted to the point tin = 0.0.

        def _graph_selection(
            ds: Dataset,
            *signals: str | tuple[str, str] | None,
            **kwargs: Any,
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
            figure = ds.plot(*signals, **kwargs)
            axes = figure.get_axes()

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
            tin_sel, tout_sel = _graph_selection(self, *signals, **kwargs)

        if verbosity != 0:
            print(
                f"\n tin = {tin_sel}{ds.dataset.index.name[1]}, tout = {tout_sel}{ds.dataset.index.name[1]}"
            )

        # Now you can trim the dataset and update all the
        # other time-related attributes
        ds.dataset = ds.dataset.loc[tin_sel:tout_sel, :]  # type:ignore
        ds._nan_intervals = ds._find_nan_intervals()
        ds.coverage = ds._find_dataset_coverage()

        # ... and shift everything such that tin = 0.0
        ds._shift_dataset_tin_to_zero()
        ds.dataset = ds.dataset.round(NUM_DECIMALS)

        return ds

    def dataset_values(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return the dataset values as a tuple *(t,u,y)* of *numpy ndarrays*.

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
        """Dump a :py:class:`Dataset <dymoval.dataset.Dataset>` object
        into a list of :py:class:`Signals <dymoval.dataset.Signal>` objects.


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
                # The following is the syntax for defining a dymoval signal
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
        """Return the list of signals in form *(["INPUT" | "OUTPUT"], name, unit)*"""
        return list(self.dataset.columns)

    def plotxy(
        self,
        # Only positional arguments
        /,
        *signal_pairs: tuple[str, str],
        layout: Literal["constrained", "compressed", "tight", "none"] = "tight",
        ax_height: float = 1.8,
        ax_width: float = 4.445,
    ) -> matplotlib.figure.Figure:
        """Plot a signal against another signal in a plane (XY-plot).

        The *signal_pairs* shall be passed as tuples.
        If no *signal_pairs* is passed then the function will *zip* the input
        and output signals.

        The method return a `matplotlib.figure.Figure`, so you can perform further
        plot manipulations by resorting to the *matplotlib* API.


        Example
        -------
        >>> fig = ds.plotxy() # ds is a dymoval Dataset
        >>> fig = ds.plotxy(("u1","y3"),("u2","y1"))
        # The following are methods of the class `matplotlib.figure.Figure`
        >>> fig.set_size_inches(10,5)
        >>> fig.set_layout_engine("constrained")
        >>> fig.savefig("my_plot.svg")


        Parameters
        ----------
        signals_pairs:
            Pairs of signals to plot in a XY-diagram.
        layout:
            Figure layout.
        ax_height:
            Approximative height (inches) of each subplot.
        ax_width:
            Approximative width (inches) of each subplot.
        """

        # df points to self.dataset.
        df = self.dataset

        if not signal_pairs:
            u_names = list(df["INPUT"].columns.get_level_values("names"))
            y_names = list(df["OUTPUT"].columns.get_level_values("names"))
            signal_pairs = tuple(zip(u_names, y_names))

        # Check if the passed names exist
        available_names = list(df.columns.get_level_values("names"))
        a, b = zip(*signal_pairs)
        passed_names = list(a + b)
        names_not_found = difference_lists_of_str(
            passed_names, available_names
        )  # noqa
        if names_not_found:
            raise ValueError(f"Signal(s) {names_not_found} not found.")

        # signal_name:signals_units dict for the axes labels
        signals_units = dict(df.droplevel(level="kind", axis=1).columns)

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
                xlabel=f"{val[0]}, ({ signals_units[val[0]]})",
                ylabel=f"{val[1]}, ({ signals_units[val[1]]})",
                grid=True,
            )

        fig.suptitle("XY-plot.")

        # Adjust fig size and layout
        nrows = fig.get_axes()[0].get_gridspec().get_geometry()[0]
        ncols = fig.get_axes()[0].get_gridspec().get_geometry()[1]
        fig.set_size_inches(ncols * ax_width, nrows * ax_height + 1.25)
        fig.set_layout_engine(layout)

        return fig

    def plot(
        self,
        /,
        # Only positional arguments
        *signals: str | tuple[str, str] | None,
        # Key-word arguments
        overlap: bool = False,
        linecolor_input: str = "blue",
        linestyle_fg: str = "-",
        alpha_fg: float = 1.0,
        linecolor_output: str = "green",
        linestyle_bg: str = "--",
        alpha_bg: float = 1.0,
        _grid: matplotlib.gridspec.GridSpec | None = None,
        layout: Literal["constrained", "compressed", "tight", "none"] = "tight",
        ax_height: float = 1.8,
        ax_width: float = 4.445,
    ) -> matplotlib.figure.Figure:
        """Plot the
        :py:class:`Dataset <dymoval.dataset.Dataset>`.

        If two signals are passed as a  *tuple*, then they will be placed in the
        same subplot.
        For example, if *ds* is a :py:class:`Dataset <dymoval.dataset.Dataset>` object
        with signals *s1, s2, ... sn*, then
        **ds.plot(("s1", "s2"), "s3", "s4")** will plot *s1* and *s2* on the same subplot
        and it will plot *s3* and *s4* on separate subplots, thus displaying the
        total of three subplots.


        Possible values for the parameters describing the line used in the plot
        (e.g. *linecolor_input* , *alpha_output*. etc).
        are the same for the corresponding plot function in matplotlib.

        The method return a `matplotlib.figure.Figure`, so you can perform further
        plot manipulations by resorting to the *matplotlib* API.


        Note
        ----
        It is possible to overlap at most two signals (this to avoid adding too many
        y-axes in the same subplot).


        Example
        -------
        >>> fig = ds.plot() # ds is a dymoval Dataset
        >>> fig = ds.plot(("u1","y3"),"y1") # Signals u1 and y3 will be placed
        # in the same subplot whereas y1 will be placed in another subplot
        # The following are methods of the class `matplotlib.figure.Figure`
        >>> fig.set_size_inches(10,5)
        >>> fig.set_layout_engine("constrained")
        >>> fig.savefig("my_plot.svg")


        Parameters
        ----------
        *signals:
            Signals to be plotted.
        overlap:
            If *True* overlap input the output signals plots
            pairwise.
            Eventual signals passed as argument will be discarded.
            The units of the outputs are displayed on the secondary y-axis.
        linecolor_input:
            Line color of the input signals.
        linestyle_fg:
            Line style of the first signal of the tuple
            if two signals are passed as a tuple.
        alpha_fg:
            Transparency value of the first signal of the tuple
            if two signals are passed as a tuple.
        linecolor_output:
            Line color of the output signals.
        linestyle_bg:
            Line style of the second signal of the tuple
            if two signals are passed as a tuple.
        alpha_bg:
            Transparency value of the second signal of the tuple
            if two signals are passed as a tuple.
        _grid:
            Grid where the spectrum ploat will be placed *(Used only internally.)*
        layout:
            Figure layout.
        ax_height:
            Approximative height (inches) of each subplot.
        ax_width:
            Approximative width (inches) of each subplot.
        """

        # df points to self.dataset.
        df = self.dataset

        # ===================================================
        # Selection of signals to plot
        # ===================================================
        (
            u_dict,
            y_dict,
            signals_tpl,
            signals_lst_plain,
        ) = self._select_signals_to_plot(*signals, overlap=overlap)

        # ===================================================
        # Arrange colors, ylabels (=units) and linestyles
        # ===================================================
        (
            linecolors_tpl,
            ylabels_tpl,
            linestyles_tpl,
            labels_tpl,
        ) = self._create_plot_args(
            "time",
            u_dict,
            y_dict,
            signals_lst_plain,
            signals_tpl,
            linecolor_input,
            linecolor_output,
            linestyle_fg,
            linestyle_bg,
        )

        # ===================================================
        # Main function
        # ===================================================
        # create a figure and a grid first.
        # Subplots will be dynamically created
        if not _grid:
            fig = plt.figure()
            n = len(signals_tpl)
            nrows, ncols = factorize(n)  # noqa
            grid = fig.add_gridspec(nrows, ncols)
        else:
            grid = _grid

        # ===================================================
        # Actual plot
        # ===================================================

        fig = self._plot_actual(
            df,
            signals_tpl,
            grid,
            linecolors_tpl,
            ylabels_tpl,
            labels_tpl,
            linestyles_tpl,
            alpha_fg,
            alpha_bg,
        )

        # Title
        plt.suptitle(
            f"Dataset '{self.name}'. \n {linecolor_input} lines are inputs and {linecolor_output} lines are outputs."
        )

        # Shade NaN:s areas
        self._shade_nans(fig.get_axes())

        # Adjust fig size and layout
        nrows = fig.get_axes()[0].get_gridspec().get_geometry()[0]
        ncols = fig.get_axes()[0].get_gridspec().get_geometry()[1]
        fig.set_size_inches(ncols * ax_width, nrows * ax_height + 1.25)
        fig.set_layout_engine(layout)

        return fig

    def plot_coverage(
        self,
        *signals: str,
        nbins: int = 100,
        linecolor_input: str = "b",
        linecolor_output: str = "g",
        alpha: float = 1.0,
        histtype: Literal["bar", "barstacked", "step", "stepfilled"] = "bar",
        _grid: matplotlib.gridspec.GridSpec | None = None,
        layout: Literal["constrained", "compressed", "tight", "none"] = "tight",
        ax_height: float = 1.8,
        ax_width: float = 4.445,
    ) -> matplotlib.figure.Figure:
        """
        Plot the dataset
        :py:class:`Dataset <dymoval.dataset.Dataset>` coverage in histograms.

        The method return a `matplotlib.figure.Figure`, so you can perform further
        plot manipulations by resorting to the *matplotlib* API.


        Example
        -------
        >>> fig = ds.plot_coverage() # ds is a dymoval Dataset
        >>> fig = ds.plot_coverage("u1","y3","y1")
        # The following are methods of the class `matplotlib.figure.Figure`
        >>> fig.set_size_inches(10,5)
        >>> fig.set_layout_engine("constrained")
        >>> fig.savefig("my_plot.svg")


        Parameters
        ----------
        *signals:
            The coverage of these signals will be plotted.
        nbins:
            The number of bins in the x-axis.
        linecolor_input:
            Line color for the input signals.
        linecolor_output:
            Line color for the output signals.
        alpha:
            Transparency value for the plots.
        histtype:
            Histogram aesthetic.
        _grid:
            Grid where the spectrum ploat will be placed *(Used only internally.)*
        layout:
            Figure layout.
        ax_height:
            Approximative height (inches) of each subplot.
        ax_width:
            Approximative width (inches) of each subplot.
        """
        # df points to self.dataset.
        df = self.dataset

        sigs = [s for s in signals if not isinstance(s, str)]
        if len(sigs) > 0:
            raise TypeError(
                f"It seems that you are trying to overlap {sigs}. Coverage plots cannot be overlapped."
            )
        # ===================================================
        # Selection of signals to plot
        # ===================================================
        (
            u_dict,
            y_dict,
            signals_lst,
            signals_lst_plain,
        ) = self._select_signals_to_plot(*signals, overlap=False)

        # ===================================================
        # Arrange colors, ylabels (=units) and linestyles
        # ===================================================
        (linecolors_tpl, xlabels_tpl, histtype_tpl, _) = self._create_plot_args(
            "coverage",
            u_dict,
            y_dict,
            signals_lst_plain,
            signals_lst,
            linecolor_input,
            linecolor_output,
            histtype,
            histtype,
        )

        # ===================================================
        # Main function
        # ===================================================
        # create a figure and a grid first.
        # Subplots will be dynamically created
        if not _grid:
            fig = plt.figure()
            n = len(signals_lst)
            nrows, ncols = factorize(n)  # noqa
            grid = fig.add_gridspec(nrows, ncols)
        else:
            grid = _grid

        # ===================================================
        # Actual plot
        # ===================================================

        # Initialize iteration
        fig = grid.figure
        if fig.get_axes():
            pass
        else:
            axes = fig.add_subplot(grid[0])
        # Iteration
        for ii, s in enumerate(signals_lst):
            # at each iteration a new axes who sharex with the
            # previously created axes, is created
            if len(fig.get_axes()) > ii:
                # Reuse existing axes if exists
                axes = fig.get_axes()[ii]
            else:
                # Create a new axes
                axes = fig.add_subplot(grid[ii], sharex=axes)
            # Actual plot
            df.droplevel(level=["kind", "units"], axis=1).loc[:, s[0]].hist(
                grid=True,
                bins=nbins,
                color=linecolors_tpl[ii][0],
                legend=True,
                histtype=histtype_tpl[ii][0],
                alpha=alpha,
                ax=axes,
            )
            axes.set_xlabel(xlabels_tpl[ii][0])

        fig.suptitle("Coverage region.")

        # Adjust fig size and layout
        nrows = fig.get_axes()[0].get_gridspec().get_geometry()[0]
        ncols = fig.get_axes()[0].get_gridspec().get_geometry()[1]
        fig.set_size_inches(ncols * ax_width, nrows * ax_height + 1.25)
        fig.set_layout_engine(layout)

        return fig

    def fft(
        self,
        *signals: str,
    ) -> pd.DataFrame:
        """Return the *FFT* of the dataset as pandas DataFrame.

        It only works with real-valued signals.

        Parameters
        ----------
        signals:
            The FFT is computed for these signals.

        """
        #  Raises
        #  ------
        #  ValueError
        #      If the dataset contains *NaN*:s

        # Validation
        u_dict, y_dict = self._classify_signals(*signals)

        # Pointer to DataFrame
        df_temp = self.dataset
        Ts = self.sampling_period
        N = len(self.dataset.index)  # number of samples

        u_names = list(u_dict.keys())
        y_names = list(y_dict.keys())

        # Check if there are any NaN:s
        if df_temp.isna().any(axis=None):
            raise ValueError(
                f"Dataset '{self.name}' contains NaN:s. I Cannot compute the FFT."
            )

        # Compute FFT. All the input signals are real (dataset only contains float)
        # We normalize the fft with N to secure energy balance (Parseval's Theorem),
        # namely it must hold "int_T x(t) = int_F X(f)".
        # See https://stackoverflow.com/questions/20165193/fft-normalization
        # N = len(df_temp.index)

        vals = (
            fft.rfftn(
                df_temp.droplevel(level=["kind", "units"], axis=1).loc[
                    :, u_names + y_names
                ],
                axes=0,
            )
            / N
        )
        vals = vals.round(NUM_DECIMALS)

        # Create a new Dataframe
        u_cols = [col for col in df_temp.columns for u in u_names if u in col]
        y_cols = [col for col in df_temp.columns for y in y_names if y in col]
        cols = pd.MultiIndex.from_tuples(
            u_cols + y_cols, names=df_temp.columns.names
        )
        df_freq = pd.DataFrame(data=vals, columns=cols)
        df_freq = df_freq.T.drop_duplicates().T  # Drop duplicated columns

        # Adjust index
        # time_units_to_frequency_units conversion
        time2freq_units = {
            "s": "Hz",
            "ms": "kHz",
            "us": "MHz",
            "ns": "GHz",
            "ps": "THz",
        }

        # If the time unit are in time2freq_units then convert
        # in frequency units
        if df_temp.index.name[1] in time2freq_units.keys():
            new_index_name = (
                "Frequency",
                time2freq_units[df_temp.index.name[1]],
            )
        else:
            new_index_name = ("Frequency", "")
        #
        f_bins = fft.rfftfreq(N, Ts)
        # NOTE: I had to use pd.Index to preserve the name and being able to
        # replace the values
        df_freq.index = pd.Index(f_bins, name=new_index_name)

        return df_freq.round(NUM_DECIMALS)

    def plot_spectrum(
        self,
        *signals: str | tuple[str, str] | None,
        kind: Literal["amplitude", "power", "psd"] = "power",
        overlap: bool = False,
        linecolor_input: str = "blue",
        linestyle_fg: str = "-",
        alpha_fg: float = 1.0,
        linecolor_output: str = "green",
        linestyle_bg: str = "--",
        alpha_bg: float = 1.0,
        _grid: matplotlib.gridspec.GridSpec | None = None,
        layout: Literal["constrained", "compressed", "tight", "none"] = "tight",
        ax_height: float = 1.8,
        ax_width: float = 4.445,
    ) -> matplotlib.figure.Figure:
        """
        Plot the spectrum of the specified signals in the dataset in different format.

        If some signals have *NaN* values, then the FFT cannot be computed and
        an error is raised.


        The method return a `matplotlib.figure.Figure`, so you can perform further
        plot manipulations by resorting to the *matplotlib* API.


        Example
        -------
        >>> fig = ds.plot_spectrum() # ds is a dymoval Dataset
        >>> fig = ds.plot_spectrum(("u1","y3"),"u2", kind ="amplitude")
        # The following are methods of the class `matplotlib.figure.Figure`
        >>> fig.set_size_inches(10,5)
        >>> fig.set_layout_engine("constrained")
        >>> fig.savefig("my_plot.svg")


        Parameters
        ----------
        *signals:
            The spectrum of these signals will be plotted.
        kind:

            - **amplitude** plot both the amplitude and phase spectrum.
              If the signal has unit V, then the amplitude has unit *V*.
              Angle is in degrees.
            - **power** plot the auto-power spectrum.
              If the signal has unit V, then the amplitude has unit *V^2*.
            - **psd** plot the power density spectrum.
              If the signal has unit V and the time is *s*, then the amplitude has unit *V^2/Hz*.
        overlap:
            If *True* it overlaps the input and the output signals plots.
            The units of the outputs are displayed on the secondary y-axis.
        linecolor_input:
            Line color of the input signals.
        linestyle_fg:
            Line style of the foreground signal in case of overlapping plots.
        alpha_fg:
            Transparency value of the foreground signal in case of overlapping plots.
        linecolor_output:
            Line color for the output signals.
        linestyle_bg:
            Line style for the background signal in case of overlapping plots.
        alpha_bg:
            Transparency value of background signal in case of overlapping plots.
        _grid:
            Grid where the spectrum ploat will be placed *(Used only internally.)*
        layout:
            Figure layout.
        ax_height:
            Approximative height (inches) of each subplot.
        ax_width:
            Approximative width (inches) of each subplot.



        Example
        -------
        >>> fig = ds.plot() # ds is a dymoval Dataset
        # The following are methods of the class `matplotlib.figure.Figure`
        >>> fig.set_size_inches(10,5)
        >>> fig.set_layout_engine("constrained")
        >>> fig.savefig("my_plot.svg")
        """
        #  Raises
        #  ------
        #  ValueError
        #      If *kind* doen not match any allowed values.
        # ==========================================================
        #  Actual plotting functions
        # ==========================================================

        def _plot_abs_angle(
            df_freq: pd.DataFrame,
            signals_tpl: list[tuple[str, ...]],
            grid: matplotlib.gridspec.GridSpec,
            linecolors_tpl: list[tuple[str, ...]],
            ylabels_tpl: list[tuple[str, ...]],
            labels_tpl: list[tuple[str, ...]],
            linestyles_tpl: list[tuple[str, ...]],
            alpha_fg: float,
            alpha_bg: float,
        ) -> matplotlib.figure.Figure:
            # In this case each element of the grid has a nested
            # subgrid of dimension (2,1).
            # axes scan every element of the subgrid and therefore we
            # have 2*m*n axes (assuming the grid has dimension (m.n)

            # This is treated a special case because it double the number of axes
            # on the  same figure.
            # We tried to make a cleaner code but it ended up in a plethora of
            # if-then-else and for loops that made the code pretty much unreadable.
            # One of the reason is that matplotlib is difficult to handle, especially
            # if you need to shuffle things around. Also, pandas plot, which is built upon
            # matplotlib won't take iterables in many arguments
            # (for example df.plot(ylabel=...) can only be a single value, even if you are
            # plotting multiple columns at once.
            # Also, to have nested subplots (for plotting abs and angle) you must
            # use gridspecs.

            # =====================================================
            # Initialize iteration
            fig = grid.figure
            if fig.get_axes():
                # If axes already exist, use the existings (think to compare_datasets)
                #
                # OBS: this part is a bit tricky, so some explanation may be useful.
                # You have a plain list of axes like [ax1, ax2, ax3, ...].
                # The even represent the amplitudes whereas the odd the angles axes.
                # However, you don't know who is left and who is right axes, but you have
                # such an information contained into signals_tpl.
                #
                # Hence, you may want to group the plain axes list by following
                # the same structure as signals_tpl, like: if
                #
                #   signals_tpl = [("u1",),("u2","u3"), ("y1","y2")], then
                #
                # you have 10 axes by considering the pairs (abs,angle).
                # Hence, the axes shall be grouped as
                #
                #   axes_tpl = [[ax1 ax2], [ax3, ax4, ax5, ax6],[ax7, ax8, ax9, ax10]]
                #
                # so, for each sublist in axes_tpl, the indices are
                #
                # 0 - absolute left axes
                # 1 - angle left axes
                # (3 - absolute right axes, if any)
                # (4 - angle right axes, if any)
                axes_tpl = []
                axes_iter = iter(fig.get_axes())
                for val in signals_tpl:
                    tmp = []
                    for jj, _ in enumerate(2 * val):
                        tmp.append(next(axes_iter))
                    axes_tpl.append(tmp)
            else:
                # If no axes available, then create new ones
                axes_tpl = []
                inner_grid = grid[0].subgridspec(2, 1)
                axes = [
                    fig.add_subplot(inner_grid[0]),
                    fig.add_subplot(inner_grid[1]),
                ]
            # Iterations
            for ii, s in enumerate(signals_tpl):
                # At each iteration a new axes is created and it is placed
                # either into a 1D or 2D list.
                # Add a subplot(2,1) to each "main" subplots
                if len(axes_tpl) > 0:
                    axes = [axes_tpl[ii][0], axes_tpl[ii][1]]
                else:
                    inner_grid = grid[ii].subgridspec(2, 1)
                    axes = [
                        fig.add_subplot(g, sharex=axes[0]) for g in inner_grid
                    ]
                # Two colums per time are being plot: (abs,angle)
                # If you do ax.legend(handles,my_labels) and then you do
                # handled, labels = ax.get_legend_handles_labels() pandas will
                # return the columns names rather than my_labels that you previously set
                # Hence, we must droplevels until we have only e.g. "u1" as column names
                df_freq.droplevel(level=[0, 2, 3], axis=1).loc[:, s[0]].plot(
                    subplots=True,
                    grid=True,
                    legend=False,
                    color=linecolors_tpl[ii][0],
                    linestyle=linestyles_tpl[ii][0],
                    alpha=alpha_fg,
                    ax=axes,
                )
                # ylabels (units)
                axes[0].set_ylabel(f"{ylabels_tpl[ii][0]}")
                axes[1].set_ylabel("(deg)")

                # legend (s,abs) and (s,angle) (will be placed at the end)
                # abs
                line_abs_l, _ = axes[0].get_legend_handles_labels()
                label_abs_l = [f"{s[0]}, abs"]
                line_abs_l[0].set_label(*label_abs_l)

                # angle
                line_angle_l, _ = axes[1].get_legend_handles_labels()
                label_angle_l = [f"{s[0]}, angle"]
                line_angle_l[0].set_label(*label_angle_l)

                # In case the user wants to overlap plots...
                # If the overlapped plots have the same units, then there is
                # no point in using a secondary_y
                line_abs_r = []
                label_abs_r = []
                line_angle_r = []
                label_angle_r = []
                if len(s) == 2:  # tuple like ("u1","u2")
                    if (
                        len(axes_tpl) > 0
                    ):  # TODO: this may be always true. To check.
                        # indices 3 and 4 represent the abs and phase axes of the
                        # right axes
                        axes = [axes_tpl[ii][2], axes_tpl[ii][3]]
                    else:
                        axes_right = [axes[0].twinx(), axes[1].twinx()]
                    df_freq.droplevel(level=[0, 2, 3], axis=1).loc[
                        :, s[1]
                    ].plot(
                        subplots=True,
                        grid=False,
                        color=linecolors_tpl[ii][1],
                        legend=False,
                        linestyle=linestyles_tpl[ii][1],
                        alpha=alpha_bg,
                        ax=axes_right,
                    )
                    # ylabels (units)
                    if ylabels_tpl[ii][0] != ylabels_tpl[ii][1]:
                        axes_right[0].set_ylabel(f"{ylabels_tpl[ii][1]}")

                    # legend (s,abs) and (s,angle) (will be placed at the end)
                    # abs
                    line_abs_r, _ = axes_right[0].get_legend_handles_labels()
                    label_abs_r = [f"{s[1]}, abs"]
                    line_abs_r[0].set_label(*label_abs_r)

                    # angle
                    line_angle_r, _ = axes_right[1].get_legend_handles_labels()
                    label_angle_r = [f"{s[1]}, angle"]
                    line_angle_r[0].set_label(*label_angle_r)

                # legend handling
                # Set nice legend
                axes[0].legend(
                    line_abs_l + line_abs_r, label_abs_l + label_abs_r
                )
                axes[1].legend(
                    line_angle_l + line_angle_r,
                    label_angle_l + label_angle_r,
                )
                # Set x_label
                axes[1].set_xlabel(
                    f"{df_freq.index.name[0]} ({df_freq.index.name[1]})"
                )

            # Remove all dummy axes, i.e. axes with no line2D objects
            # TODO: as seen, inspite you explictely set the legends,
            # pandas decide what are the legends.
            # print(fig.get_axes()[2].get_legend_handles_labels())
            axes_all = fig.get_axes()
            for ax in axes_all:
                if len(ax.get_lines()) == 0:
                    ax.remove()

            return fig

        # ============================================
        # Main function begin
        # ============================================

        # A small check
        if kind not in SPECTRUM_KIND:
            raise ValueError(f"Argument 'kind' must be one of {SPECTRUM_KIND}")

        # ===================================================
        # Selection of signals
        # ===================================================
        (
            u_dict,
            y_dict,
            signals_tpl,
            signals_lst_plain,
        ) = self._select_signals_to_plot(*signals, overlap=overlap)

        # ============================================
        # Compute Spectrums values of selected signals
        # ============================================

        # For real signals, the spectrum is Hermitian anti-simmetric, i.e.
        # the amplitude is symmetric wrt f=0 and the phase is antisymmetric wrt f=0.
        # See e.g. https://ccrma.stanford.edu/~jos/ReviewFourier/Symmetries_Real_Signals.html
        df_freq = self.fft(*signals_lst_plain)
        # Frequency and number of samples
        # Ts = self.sampling_period
        # N = len(self.dataset.index)  # number of samples
        # Compute frequency bins
        # f_bins = fft.rfftfreq(N, Ts)
        # Update DataFame index.
        # NOTE: I had to use pd.Index to preserve the name and being able to
        # replace the values
        # df_freq.index = pd.Index(f_bins, name=df_freq.index.name)

        # Switch between the kind
        if kind == "amplitude":
            # Add another level to specify abs and phase
            df_freq = df_freq.agg([np.abs, lambda x: np.angle(x, deg=True)])
            df_freq = df_freq.rename(columns={"<lambda>": "angle"}, level=3)
            df_freq = df_freq.rename(columns={"absolute": "abs"}, level=3)
            # df_freq = df_freq.agg([np.abs, np.angle])

        elif kind == "power":
            df_freq = df_freq.abs() ** 2
            # We take half spectrum, so for conserving the energy we must consider
            # also the negative frequencies with the exception of the DC compontent
            # because that is not mirrored. This is why we multiply by 2.
            # The same happens for the psd.
            df_freq[1:-1] = 2 * df_freq[1:-1]
        elif kind == "psd":
            Ts = self.sampling_period
            N = len(self.dataset.index)  # number of samples
            Delta_f = 1 / (Ts * N)  # Size of each frequency bin
            df_freq = df_freq.abs() ** 2 / Delta_f
            df_freq[1:-1] = 2 * df_freq[1:-1]

        # ===================================================
        # Arrange colors, ylabels (=units) and linestyles
        # ===================================================
        (
            linecolors_tpl,
            ylabels_tpl,
            linestyles_tpl,
            labels_tpl,
        ) = self._create_plot_args(
            kind,
            u_dict,
            y_dict,
            signals_lst_plain,
            signals_tpl,
            linecolor_input,
            linecolor_output,
            linestyle_fg,
            linestyle_bg,
        )

        # ===================================================
        # Actual plot
        # ===================================================
        # create a figure and a grid first.
        # Subplots will be dynamically created
        if not _grid:
            fig = plt.figure()
            n = len(signals_tpl)
            nrows, ncols = factorize(n)  # noqa
            grid = fig.add_gridspec(nrows, ncols)
        else:
            grid = _grid

        # Switch type of plot: abs_angle (which has double number of Subplots
        # or the others (power and psd)
        # Subplots will be added automatically
        if kind == "amplitude":
            fig = _plot_abs_angle(
                df_freq,
                signals_tpl,
                grid,
                linecolors_tpl,
                ylabels_tpl,
                labels_tpl,
                linestyles_tpl,
                alpha_fg,
                alpha_bg,
            )
        else:
            fig = self._plot_actual(
                df_freq,
                signals_tpl,
                grid,
                linecolors_tpl,
                ylabels_tpl,
                labels_tpl,
                linestyles_tpl,
                alpha_fg,
                alpha_bg,
            )

        # Title
        fig.suptitle(
            f"Dataset '{self.name}' {kind} spectrum. \n {linecolor_input} lines are inputs and {linecolor_output} lines are outputs."
        )
        #

        # Adjust fig size and layout
        nrows = fig.get_axes()[0].get_gridspec().get_geometry()[0]
        ncols = fig.get_axes()[0].get_gridspec().get_geometry()[1]
        if kind == "amplitude":
            fig.set_size_inches(
                ncols * ax_width * 2, nrows * ax_height * 2 + 1.25
            )
        else:
            fig.set_size_inches(ncols * ax_width, nrows * ax_height + 1.25)
        fig.set_layout_engine(layout)

        return fig

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
        u_dict, y_dict = self._classify_signals(*signals)

        u_names = list(u_dict.keys())
        y_names = list(y_dict.keys())

        u_units = list(u_dict.values())
        y_units = list(y_dict.values())

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


        Example
        -------
        >>> fig = ds.remove_offset(("u1",0.4),("u2",-1.5)) # ds is a Dataset


        Parameters
        ----------
        *signals:
            Tuples of the form *(name, offset)*.
            The *name* parameter must match the name of a signal stored
            in the
            :py:class:`Dataset <dymoval.dataset.Dataset>`.
            The *offset* parameter is the value to remove to the *name* signal.
        """
        # TODO: can be refactored
        # Safe copy
        ds_temp = deepcopy(self)
        df_temp = ds_temp.dataset

        # Validate passed arguments
        (
            u_dict,
            y_dict,
            u_list,
            y_list,
        ) = self._validate_name_value_tuples(*signals_values)

        u_names = list(u_dict.keys())
        y_names = list(y_dict.keys())

        u_units = list(u_dict.values())
        y_units = list(y_dict.values())

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

        return ds_temp

    def low_pass_filter(
        self,
        *signals_values: tuple[str, float],
    ) -> Dataset:
        """
        Low-pass filter a list of specified signals.

        The low-pass filter is first-order IIR filter.


        Parameters
        ----------
        *signals_values:
            Tuples of the form *(signal_name, cutoff_frequency)*.
            The values of *signal_name* are low-pass
            filtered with a first-order low-pass filter with cutoff frequency
            *cutoff_frequency*
        """
        #  Raises
        #  ------
        #  TypeError
        #      If no arguments are passed.
        #  ValueError
        #      If any of the passed cut-off frequencies is negative.

        # TODO: can be refactored
        # Safe copy
        ds_temp = deepcopy(self)
        df_temp = ds_temp.dataset

        # Validate passed arguments
        (
            u_dict,
            y_dict,
            u_list,
            y_list,
        ) = self._validate_name_value_tuples(*signals_values)

        u_names = list(u_dict.keys())
        y_names = list(y_dict.keys())

        u_units = list(u_dict.values())
        y_units = list(y_dict.values())

        # Sampling frequency
        fs = 1 / (ds_temp.dataset.index[1] - ds_temp.dataset.index[0])
        N = len(ds_temp.dataset.index)

        # INPUT LPF
        # TODO: Here you may want to refactor a bit.
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
        to dump the Dataset into Signals through :py:meth:`~dymoval.dataset.dump_to_signals()`,
        manipulate them, and then create a brand new Dataset object.


        Warning
        -------
        This function may be removed in the future.


        Parameters
        ----------
        signal_function:
            Signals where to apply a function.
            This argument shall have the form *(name, func, new_unit)*.
        **kwargs:
            Additional keyword arguments to pass as keywords arguments to
            the underlying pandas DataFrame *apply* method.

        """
        #  Raises
        #  ------
        #  ValueError:
        #      If the passed signal name does not exist in the current Dataset.

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
        ds_temp.coverage = self._find_dataset_coverage()

        return ds_temp

    def remove_NaNs(
        self,
        **kwargs: Any,
    ) -> Dataset:  # pragma: no cover
        """Replace *NaN:s* values in the
        :py:class:`Dataset <dymoval.dataset.Dataset>`.

        It uses pandas *DataFrame.interpolate()* method, so that the **kwargs are directly
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

    def add_input(self, *signals: Signal) -> Dataset:
        """Add input signals to the
        :py:class:`Dataset <dymoval.dataset.Dataset>` object.

        Signals will be trimmed to the length of the
        :py:class:`Dataset <dymoval.dataset.Dataset>`.
        Shorter signals will be padded with *NaN:s*.

        Parameters
        ----------
        *signals:
            Input signals to be added.
        """
        kind: Signal_type = "INPUT"
        ds = deepcopy(self)
        return ds._add_signals(kind, *signals)

    def add_output(self, *signals: Signal) -> Dataset:
        """Add output signals to the
        :py:class:`Dataset <dymoval.dataset.Dataset>` object.

        Signals will be trimmed to the length of the
        :py:class:`Dataset <dymoval.dataset.Dataset>`.
        Shorter signals will be padded with *NaN:s*.


        Parameters
        ----------
        *signals:
            Output signals to be added.
        """
        kind: Signal_type = "OUTPUT"
        ds = deepcopy(self)
        return ds._add_signals(kind, *signals)

    def remove_signals(self, *signals: str) -> Dataset:
        """Remove signals from the
        :py:class:`Dataset <dymoval.dataset.Dataset>`.

        Parameters
        ----------
        signals:
            Signal name to be removed.
        """
        #  Raises
        #  ------
        #  KeyError:
        #      If signal(s) not found in the *Dataset* object.
        #  KeyError:
        #      If the reminder Dataset object has less than one input or one output.

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
    tpl: list[tuple[str, ...]], lst: list[str]
) -> list[tuple[str, ...]]:
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


def change_axes_layout(
    fig: matplotlib.axes.Figure,
    nrows: int,
    ncols: int,
) -> tuple[matplotlib.axes.Figure, matplotlib.axes.Axes]:
    """Change *Axes* layout of an existing Matplotlib *Figure*.


    Parameters
    ----------
    fig:
        Reference figure.
    nrwos:
        New number or rows.
    ncols:
        New number of columns
    """

    axes = fig.get_axes()
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
    list of :py:class:`Signals <dymoval.dataset.Signal>`
    can be used to create a Dataset.

    Every :py:class:`Signal <dymoval.dataset.Signal>` in *signals*
    must have all the attributes adequately set.

    A :py:class:`Signal <dymoval.dataset.Signal>` is a *TypedDict* with the
    following keys

    1. name: str
    2. values: 1D np.array
    3. signal_unit: str
    4. sampling_period: float
    5. time_unit: str


    Example
    -------
    >>> # How to create a Dataset object from a list of Signals
    >>>
    >>> import numpy as np
    >>> import dymoval as dmv
    >>>
    >>> signal_names = [
    >>>     "SpeedRequest",
    >>>     "AccelPedalPos",
    >>>     "OilTemp",
    >>>     "ActualSpeed",
    >>>     "SteeringAngleRequest",
    >>> ]
    >>> signal_values = np.random.rand(5, 100)
    >>> signal_units = ["m/s", "%", "°C", "m/s", "deg"]
    >>> sampling_periods = [0.1, 0.1, 0.1, 0.1, 0.1]
    >>> # Create dymoval signals
    >>> signals = []
    >>> for ii, val in enumerate(signal_names):
    >>>     tmp: dmv.Signal = {
    >>>         "name": val,
    >>>         "values": signal_values[ii],
    >>>         "signal_unit": signal_units[ii],
    >>>         "sampling_period": sampling_periods[ii],
    >>>         "time_unit": "s",
    >>>     }
    >>>     signals.append(tmp)
    >>> # Validate signals
    >>> dmv.validate_signals(*signals)
    >>> # Specify which signals are inputs and which signals are output
    >>> input_labels = ["SpeedRequest", "AccelPedalPos", "SteeringAngleRequest"]
    >>> output_labels = ["ActualSpeed", "OilTemp"]
    >>> # Create dymoval Dataset objects
    >>> ds = dmv.Dataset("my_dataset", signals, input_labels, output_labels)
    >>>
    >>> # At this point you can plot, trim, manipulate, analyze, etc you dataset
    >>> # through the Dataset class methods.



    Parameters
    ----------
    *signals :
        Signal to be validated.

    """
    #  Raises
    #  ------
    #  ValueError
    #      If values is not a *1-D numpy ndarray*,
    #      or if sampling period must positive or the signals *time_unit* keys are
    #      not the same.
    #  KeyError
    #      If signal attributes are not found or not allowed or if signal names are
    #      not unique,
    #  IndexError
    #      If signal have less than two samples.
    #  TypeError
    #      If values is not a *1-D numpy ndarray*, or if sampling period is not a *float*.

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
) -> None:
    """
    Check if a *pandas DataFrame* is suitable for instantiating
    a :py:class:`Dataset <dymoval.dataset.Dataset>` object.

    The index of the *DataFrame* shall represent the common timeline for all the
    signals, whereas the *j-th* column values shall represents the realizations
    of the *j-th* signal.

    The column names are tuples of strings of the form *(signal_name, signal_unit)*.

    It must be specified which signal(s) are the input through the *u_names* and
    which signal(s) is the output through the  *y_names* parameters.

    The candidate *DataFrame* shall meet the following requirements

    - Columns names shall be unique,
    - Columns name shall be a tuple of *str* of the form *(name, unit)*,
    - The index shall represent the common timeline for all  and its
      name shall be *'(Time, time_unit)'*, where *time_unit* is a string.
    - Each signal must have at least two samples (i.e. the DataFrame has at least two rows),
    - Only one index and columns levels are allowed (no *MultiIndex*),
    - There shall be at least two signals representing one input and one output,
    - Both the index values and the column values must be *float* and the index values
      must be a a 1D vector of monotonically, equi-spaced, increasing *floats*.


    Example
    -------
    >>> # How to create a Dataset object from a pandas DataFrame
    >>>
    >>> import dymoval as dmv
    >>> import pandas as pd
    >>>
    >>> # Signals names, units and values
    >>> signal_names = [
    >>>     "SpeedRequest",
    >>>     "AccelPedalPos",
    >>>     "OilTemp",
    >>>     "ActualSpeed",
    >>>     "SteeringAngleRequest",
    >>> ]
    >>> signal_units = ["m/s", "%", "°C", "m/s", "deg"]
    >>> signal_values = np.random.rand(100, 5)
    >>>
    >>> # time axis
    >>> sampling_period = 0.1
    >>> timestamps = np.arange(0, 10, sampling_period)
    >>>
    >>> # Build a candidate DataFrame
    >>> cols = list(zip(signal_names, signal_units))
    >>> index_name = ("Time", "s")
    >>>
    >>> # Create dymoval signals
    >>> df = pd.DataFrame(data=signal_values, columns=cols)
    >>> df.index = pd.Index(data=timestamps, name=index_name)
    >>>
    >>> # Check if the dataframe is suitable for a dymoval Dataset
    >>> dmv.validate_dataframe(df)
    >>>
    >>> # Specify input and output signals
    >>> input_labels = ["SpeedRequest", "AccelPedalPos", "SteeringAngleRequest"]
    >>> output_labels = ["ActualSpeed", "OilTemp"]
    >>>
    >>> # Create dymoval Dataset objects
    >>> ds = dmv.Dataset("my_dataset", df, input_labels, output_labels)



    Parameters
    ----------
    df :
        DataFrame to be validated.
    """
    #  Raises
    #  ------
    #  Error:
    #      Depending on the issue found an appropriate error will be raised.

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

    # The following is automatically tested by the previous.
    #  # Check that the DataFrame must have only one index and columns levels
    #  if df.columns.nlevels > 1 or df.index.nlevels > 1:
    #      raise IndexError(
    #          "The number index levels must be one for both the index and the columns.",
    #          "The index shall represent a time vector of equi-distant time instants",
    #          "and each column shall correspond to one signal values.",
    #      )

    # At least two samples
    if df.index.size < 2:
        raise IndexError("A signal needs at least two samples.")

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


def plot_signals(*signals: Signal) -> matplotlib.figure.Figure:
    """Plot :py:class:`Signals <dymoval.dataset.Signal>`.


    The method return a `matplotlib.figure.Figure`, so you can perform further
    plot manipulations by resorting to the *matplotlib* API.


    Example
    -------
    >>> fig = dmv.plot_signals(s1,s2,s3) # s1, s2 and s3 are dymoval Signal objects.
    # The following are methods of the class `matplotlib.figure.Figure`
    >>> fig.set_size_inches(10,5)
    >>> fig.set_layout_engine("constrained")
    >>> fig.savefig("my_plot.svg")


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

    for ii in range(n, len(ax)):
        ax[ii].remove()

    return fig


def compare_datasets(
    *datasets: Dataset,
    kind: Literal["time", "coverage"] | Spectrum_type = "time",
    layout: Literal["constrained", "compressed", "tight", "none"] = "tight",
    ax_height: float = 1.8,
    ax_width: float = 4.445,
) -> matplotlib.figure.Figure:
    """
    Compare different :py:class:`Datasets <dymoval.dataset.Dataset>` graphically
    by overlapping them.

    The method return a `matplotlib.figure.Figure`, so you can perform further
    plot manipulations by resorting to the *matplotlib* API.


    Example
    -------
    >>> import dymoval as dmv
    >>> fig = dmv.compare_datasets(ds,ds1,ds2, kind="coverage")
    # The following are methods of the class `matplotlib.figure.Figure`
    >>> fig.set_size_inches(10,5)
    >>> fig.set_layout_engine("constrained")
    >>> fig.savefig("my_plot.svg")


    Parameters
    ----------
    *datasets :
        :py:class:`Datasets <dymoval.dataset.Dataset>` to be compared.
    kind:
        Kind of graph to be plotted.
    layout:
        Figure layout.
    ax_height:
        Approximative height (inches) of each subplot.
    ax_width:
        Approximative width (inches) of each subplot.
    """

    # Utility function to avoid too much code repetition
    def _adjust_legend(ds_names: list[str], axes: matplotlib.axes.Axes) -> None:
        # Based on the pair (handles, labels), where handles are e.g. Line2D,
        # or other matplotlib Artist (an Artist is everything you can draw
        # on a figure)

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
    ) -> tuple[matplotlib.axes.Figure, matplotlib.gridspec.GridSpec]:
        # When performing many plots on the same figure,
        # this function find number of axes needed
        # n is the number of total signals, no matter if they are INPUT or OUTPUT
        n = max(
            [
                len(
                    df.droplevel(level="kind", axis=1).columns.get_level_values(
                        "names"
                    )
                )
                for df in dfs
            ]
        )

        # Set nrows and ncols
        fig = plt.figure()
        nrows, ncols = factorize(n)  # noqa
        grid = fig.add_gridspec(nrows, ncols)

        return fig, grid

    # ========================================
    #    Main implementation
    # ========================================
    # Small check
    for ds in datasets:
        if not isinstance(ds, Dataset):
            raise TypeError("Input must be a dymoval Dataset type.")

    # Arrange figure
    dfs = [ds.dataset.droplevel(level="units", axis=1) for ds in datasets]
    fig, grid = _arrange_fig_axes(*dfs)
    cmap = plt.get_cmap(COLORMAP)

    # ========================================
    #   Switch case
    # ========================================
    if kind == "time":
        # All the plots made on the same axis
        for ii, ds in enumerate(datasets):
            ds.plot(
                linecolor_input=cmap(ii),
                linecolor_output=cmap(ii),
                _grid=grid,
            )
    elif kind == "coverage":
        for ii, ds in enumerate(datasets):
            ds.plot_coverage(
                linecolor_input=cmap(ii),
                linecolor_output=cmap(ii),
                _grid=grid,
            )
    else:
        for ii, ds in enumerate(datasets):
            ds.plot_spectrum(
                linecolor_input=cmap(ii),
                linecolor_output=cmap(ii),
                _grid=grid,
                kind=kind,
            )

    # Adjust legend
    ds_names = [ds.name for ds in datasets]
    _adjust_legend(ds_names, fig.get_axes())
    fig.suptitle(f"Dataset comparison in {kind}.")

    # Adjust fig size and layout
    nrows = fig.get_axes()[0].get_gridspec().get_geometry()[0]
    ncols = fig.get_axes()[0].get_gridspec().get_geometry()[1]
    if kind == "amplitude":
        fig.set_size_inches(ncols * ax_width * 2, nrows * ax_height * 2 + 1.25)
    else:
        fig.set_size_inches(ncols * ax_width, nrows * ax_height + 1.25)
    fig.set_layout_engine(layout)

    return fig
