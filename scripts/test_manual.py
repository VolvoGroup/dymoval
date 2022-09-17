#  -*- coding: utf-8 -*-
# ===========================================================================
# This script is used for testing the plot functions
# by visual inspection.
# It is recommended to test all the "MIMO", "SISO", "SIMO", "MISO" cases.
# ===========================================================================

from copy import deepcopy
import dymoval as dmv
import numpy as np

# ===========================================================================
# Arange
# ===========================================================================
# Setup the test type
dataset_type = ["MIMO", "SISO", "SIMO", "MISO"]
fixture_type = "MIMO"

# Set test data
nan_thing = np.empty(200)
nan_thing[:] = np.NaN

input_signal_names = ["u1", "u2", "u3"]
input_sampling_periods = [0.01, 0.1, 0.1]
input_signal_values = [
    np.hstack((np.random.rand(50), nan_thing, np.random.rand(400), nan_thing)),
    np.hstack(
        (np.random.rand(20), nan_thing[0:5], np.random.rand(30), nan_thing)
    ),
    np.hstack((np.random.rand(80), nan_thing, np.random.rand(100))),
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
# Output signal
output_signal_names = ["y1", "y2", "y3", "y4"]
output_sampling_periods = [0.1, 0.1, 0.1, 0.1]
output_signal_values = [
    np.hstack((np.random.rand(50), nan_thing, np.random.rand(100), nan_thing)),
    np.hstack(
        (np.random.rand(100), nan_thing[0:50], np.random.rand(150), nan_thing)
    ),
    np.hstack(
        (np.random.rand(10), nan_thing[0:105], np.random.rand(50), nan_thing)
    ),
    np.hstack(
        (np.random.rand(20), nan_thing[0:85], np.random.rand(60), nan_thing)
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


if fixture_type == "SISO":
    # Slice signal list
    # Pick u1 and y1
    signal_list = [signal_list[0], signal_list[first_output_idx]]
    input_signal_names = dmv.str2list(input_signal_names[0])
    output_signal_names = dmv.str2list(output_signal_names[0])
if fixture_type == "MISO":
    signal_list = [
        *signal_list[:first_output_idx],
        signal_list[first_output_idx],
    ]
    output_signal_names = dmv.str2list(output_signal_names[0])
if fixture_type == "SIMO":
    signal_list = [signal_list[0], *signal_list[first_output_idx:]]
    input_signal_names = dmv.str2list(input_signal_names[0])


# %%
# ===========================================================================
# Act: Dataset plots test
# ===========================================================================
# Get a dataset
ds = dmv.Dataset(
    "pippo",
    signal_list,
    input_signal_names,
    output_signal_names,
    target_sampling_period=0.1,
    overlap=True,
)

ds.replace_NaNs("fillna", 0.0)

ds.plot()
ds.plot(
    line_color_input="r",
    linestyle_input=":",
    line_color_output="c",
    alpha_output=0.5,
    overlap=True,
)

# Conditional plot
# ds.plot("u1")

# %% Coverage

ds.plot_coverage()
ds.plot_coverage(line_color_input="r", line_color_output="c", alpha_output=0.5)

# ===========================================================================
# Act: ValidatonSession plots test
# ===========================================================================
# Get a ValidationSession
vs = dmv.ValidationSession("pippo", ds)

# %% Pretend that you ran two simulations of a model with two different settings.
sim1_name = "Model 1"
sim1_labels = ["my_y1", "my_y2", "my_y3", "my_y4"]
if fixture_type == "SISO" or fixture_type == "MISO":
    sim1_labels = dmv.str2list("my_y1")
sim1_values = vs.Dataset.dataset["OUTPUT"].values + np.random.rand(
    len(vs.Dataset.dataset["OUTPUT"].values), 1
)


sim2_name = "Model 2"
sim2_labels = ["your_y1", "your_y2", "your_y3", "your_y4"]
if fixture_type == "SISO" or fixture_type == "MISO":
    sim2_labels = dmv.str2list("your_y1")
sim2_values = vs.Dataset.dataset["OUTPUT"].values + np.random.rand(
    len(vs.Dataset.dataset["OUTPUT"].values), 1
)

vs.append_simulation(sim1_name, sim1_labels, sim1_values)
vs.append_simulation(sim2_name, sim2_labels, sim2_values)

vs.plot_simulations("Model 1", "Model 2")
vs.plot_simulations("Model 1")
vs.plot_simulations(plot_dataset=True, plot_input=True)
