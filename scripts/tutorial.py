#  -*- coding: utf-8 -*-
# ===========================================================================
# In this tutorial we show the main functionalities of dymoval.
# The code is divided in code-cell blocks so you can run it piece-by-piece
# and analyze the result.
# ===========================================================================

from copy import deepcopy
import dymoval as dmv
import numpy as np
import matplotlib.pyplot as plt


# ===========================================================================
# Assume that we have some measurements from some experiment.
# Unfortunately, there is some missing data and the signals are not even sampled
# with the same sampling period.
#
# Simulate something with Simulink and export an FMU.
#
# We  map each logged signal into a dymoval Signal.
# To create a dymoval Signal use the syntax and fill the fields accordingly
# to the specified datatypes
# name: dmv.Signal = {
#                   "name": str
#                   "values": np.ndarray
#                   "signal_unit": str | list[str]
#                   "sampling_period": float
#                   "time_unit": str
#                     }
# ===========================================================================
# %%
plt.ion()
# Let's create some Signal
nan_intervals = np.empty(200)
nan_intervals[:] = np.NaN

# INPUT signals
input_signal_names = ["u1", "u2", "u3"]
input_sampling_periods = [0.01, 0.1, 0.1]
input_signal_values = [
    np.hstack(
        (np.random.rand(50), nan_intervals, np.random.rand(400), nan_intervals)
    ),
    np.hstack(
        (
            np.random.rand(20),
            nan_intervals[0:5],
            np.random.rand(30),
            nan_intervals,
        )
    ),
    np.hstack((np.random.rand(80), nan_intervals, np.random.rand(100))),
]


input_signal_units = ["m/s", "%", "°C"]
#
in_lst = []
for ii, val in enumerate(input_signal_names):
    temp_in: dmv.Signal = {
        "name": val,
        "values": input_signal_values[ii],
        "signal_unit": input_signal_units[ii],
        "sampling_period": input_sampling_periods[ii],
        "time_unit": "s",
    }

    in_lst.append(deepcopy(temp_in))
# OUTPUT signals
output_signal_names = ["y1", "y2", "y3", "y4"]
output_sampling_periods = [0.3, 0.2, 0.1, 0.05]
output_signal_values = [
    np.hstack(
        (np.random.rand(50), nan_intervals, np.random.rand(100), nan_intervals)
    ),
    np.hstack(
        (
            np.random.rand(100),
            nan_intervals[0:50],
            np.random.rand(150),
            nan_intervals,
        )
    ),
    np.hstack(
        (
            np.random.rand(10),
            nan_intervals[0:105],
            np.random.rand(50),
            nan_intervals,
        )
    ),
    np.hstack(
        (
            np.random.rand(20),
            nan_intervals[0:85],
            np.random.rand(60),
            nan_intervals,
        )
    ),
]

output_signal_units = ["m/s", "deg", "°C", "kPa"]
out_lst = []
for ii, val in enumerate(output_signal_names):
    # This is the syntax for defining a dymoval signal
    temp_out: dmv.Signal = {
        "name": val,
        "values": output_signal_values[ii],
        "signal_unit": output_signal_units[ii],
        "sampling_period": output_sampling_periods[ii],
        "time_unit": "s",
    }

    out_lst.append(deepcopy(temp_out))
signal_list = [*in_lst, *out_lst]
first_output_idx = len(input_signal_names)

# %% You can check if your Signal are in correct format through the function
# dmv.signals_validation
dmv.signals_validation(signal_list)

# ... and you can visually inspect them through the function dmv.plot_signals
dmv.plot_signals(signal_list)

# ... and you can optionally pass the list of input and outputs.
dmv.plot_signals(signal_list, input_signal_names[0:1], output_signal_names)

# The signals to be included in a dataset must have the same sampling period,
# so you may need to re-sample your signals.
# Dymoval will try to fix the sampling period for you with
# the function fix_sampling_period.
# Such a function will tell you what signals were resampled and what not.

# resampled_signals, excluded_signals = dmv.fix_sampling_periods(
#     signal_list, target_sampling_period=0.1
# )


# %% At this point we can create a dymoval Dataset.
# To do that we need to pass the signal list, and we need to specify who is the input
# and who is the output.
# Note that the initializer will validate the signals, will try to resample them,
# will identify the intervals where there are NaN:s values and will estimate the
# coverage region.
# The user can select the dataset time interval that he/she wants to keep
# both graphically or bt passing some arguments to the initializer.

ds = dmv.Dataset(
    "my_dataset",
    signal_list,
    input_signal_names,
    output_signal_names,
    target_sampling_period=0.1,
    overlap=True,
)


# Once the dymoval Dataset is created, it is possible to handle with the NaN:s
# in a number of ways
ds.replace_NaNs("fillna", 0.0)

# At this point we can visually inspect the resulting Dataset.
# Note how the areas where the NaN:s have been replaced are shaded.
ds.plot()

# %% We can also inspect the Dataset coverage region

ds.plot_coverage()

# Other methods of the class Dataset are self-explanatory.
#
#
#
# ===========================================================================
# Validation Session
# Now that we have a good Dataset, we can create a dymoval ValidationSession
# where we append the simulation results of our models so that we can
# evaluate them
# ===========================================================================
#
# To create a dymoval ValidationSession we only need to pass a dymoval Dataset.
vs = dmv.ValidationSession("my_validation", ds)

# %% Now Pretend that we built two models and we have want to validate them.

sim1_name = "Model 1"
sim1_labels = ["my_y1", "my_y2"]
sim1_values = vs.Dataset.dataset["OUTPUT"].values + np.random.rand(
    len(vs.Dataset.dataset["OUTPUT"].values), 2
)


sim2_name = "Model 2"
sim2_labels = ["your_y1", "your_y2"]

sim2_values = vs.Dataset.dataset["OUTPUT"].values + np.random.rand(
    len(vs.Dataset.dataset["OUTPUT"].values), 2
)

# We use the ValidationSession's method append_simulation to append the simulation
# results.
vs.append_simulation(sim1_name, sim1_labels, sim1_values)
vs.append_simulation(sim2_name, sim2_labels, sim2_values)

# %% We can visually inspect the results...
vs.plot_simulations()

# %% ... or we can get validation metrics

vs.validation_results

# %% At this point we can clear the simulations list
vs.clear()
