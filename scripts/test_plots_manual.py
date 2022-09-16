#  -*- coding: utf-8 -*-
"""
    Copyright 2022 Volvo Autonomous Solutions
    Author: Ubaldo Tiberi

Redistribution and use in source and binary forms, with or without modification, \n
    are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, \n
    this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, \n
    this list of conditions and the following disclaimer in the documentation \n
    and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors\n
    may be used to endorse or promote products derived from this software \n
    without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" \n
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, \n
    THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR\n
    PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR \n
    CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,\n
    EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, \n
    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;\n
    OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, \n
    WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)\n
    ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF \n
    THE POSSIBILITY OF SUCH DAMAGE.
"""
# ===========================================================================
# This script is used for testing the plot functions
# by visual inspection.
# It is recommended to test all the "MIMO", "SISO", "SIMO", "MISO" cases.
# ===========================================================================

from copy import deepcopy
import dymoval as dmv
import numpy as np
import pandas as pd

# ===========================================================================
# Arange
# ===========================================================================
# Setup the test type
dataset_type = ["MIMO", "SISO", "SIMO", "MISO"]
fixture_type = "SISO"

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
    input_signal_names = input_signal_names[0]
    output_signal_names = output_signal_names[0]
if fixture_type == "MISO":
    signal_list = [
        *signal_list[:first_output_idx],
        signal_list[first_output_idx],
    ]
    output_signal_names = output_signal_names[0]
if fixture_type == "SIMO":
    signal_list = [signal_list[0], *signal_list[first_output_idx:]]
    input_signal_names = input_signal_names[0]
# %%
# ===========================================================================
# Act: Dataset plots test
# ===========================================================================
# Get a dataset
ds = dmv.Dataset(
    "mydataset",
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
if fixture_type == "MIMO" or fixture_type == "MISO":
    ds.plot(u_labels=["u1", "u2"], y_labels="y1", overlap=True)
else:
    ds.plot(u_labels="u1", y_labels="y1")
# %% Coverage

ds.plot_coverage()
ds.plot_coverage(line_color_input="r", line_color_output="c", alpha_output=0.5)

# ===========================================================================
# Act: ValidatonSession plots test
# ===========================================================================
# Get a ValidationSession
vs = dmv.ValidationSession("my_validation", ds)

# %% Pretend that you ran two simulations of a model with two different settings.
sim1_name = "Model 1"
sim1_labels = ["my_y1", "my_y2", "my_y3", "my_y4"]
if fixture_type == "SISO" or fixture_type == "MISO":
    sim1_labels = "my_y1"
sim1_values = vs.Dataset.dataset["OUTPUT"].values + np.random.rand(
    len(vs.Dataset.dataset["OUTPUT"].values), 1
)


sim2_name = "Model 2"
sim2_labels = ["your_y1", "your_y2", "your_y3", "your_y4"]
if fixture_type == "SISO" or fixture_type == "MISO":
    sim2_labels = "your_y1"
sim2_values = vs.Dataset.dataset["OUTPUT"].values + np.random.rand(
    len(vs.Dataset.dataset["OUTPUT"].values), 1
)

vs.append_simulation(sim1_name, sim1_labels, sim1_values)
vs.append_simulation(sim2_name, sim2_labels, sim2_values)

vs.plot_simulations("Model 1", "Model 2")
vs.plot_simulations("Model 1")
vs.plot_simulations(plotdataset=True, plot_input=True)
# %%
vs.clear()

# =========================================================================
# Test frequency response
# =========================================================================
# %%
Ts = 0.001
N = 10000
t = np.linspace(0, Ts * N, N)
c1 = 2
c2 = 3
c3 = 1

f1 = 2
w1 = 2 * np.pi * f1
f2 = 2.4
w2 = 2 * np.pi * f2
f3 = 4.8
w3 = 2 * np.pi * f3

u_labels = ["u1", "u2", "u3"]
u_values = [
    c1 + np.sin(w1 * t) + np.sin(w2 * t),
    c1 + np.sin(w2 * t),
    c1 + np.sin(w3 * t),
]

y_labels = ["y1", "y2", "y3", "y4"]
y_values = [
    c1 + np.sin(w1 * t) + np.sin(w3 * t),
    c3 + np.sin(w3 * t),
    c1 + np.sin(w1 * t) + np.sin(w2 * t) + np.sin(w3 * t),
    np.sin(w1 * t) - np.sin(w2 * t) - np.sin(w3 * t),
]

data = np.vstack((np.asarray(u_values), np.asarray(y_values))).transpose()
df = pd.DataFrame(index=t, columns=[*u_labels, *y_labels], data=data)
ds = dmv.Dataset("mydataset", df, u_labels, y_labels)
ds.plot()

ds.plot_amplitude_spectrum()
ds.plot_amplitude_spectrum(u_labels="u1", y_labels="y4")


# Remove means and offset
u_list = ("u1", 2.0)

ds.remove_offset(u_list=u_list)
