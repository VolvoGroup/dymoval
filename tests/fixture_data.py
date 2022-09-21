# -*- coding: utf-8 -*-

import pytest
import dymoval as dmv
import numpy as np
from copy import deepcopy
import pandas as pd

# For more info on parametrized fixtures, look here:
# https://www.youtube.com/watch?v=aQH7hyJn-No

dataset_type = ["MIMO", "SISO", "SIMO", "MISO"]

# ============================================
# All good test Signals (good)
# ============================================


@pytest.fixture(params=dataset_type)
def good_signals(request):  # type: ignore

    fixture_type = request.param
    # General case (MIMO)
    nan_thing = np.empty(200)
    nan_thing[:] = np.NaN

    # Signals creation
    input_signal_names = ["u1", "u2", "u3"]
    input_sampling_periods = [0.01, 0.1, 0.1]
    input_signal_values = [
        np.hstack(
            (np.random.rand(50), nan_thing, np.random.rand(400), nan_thing)
        ),
        np.hstack(
            (np.random.rand(20), nan_thing[0:5], np.random.rand(30), nan_thing)
        ),
        np.hstack((np.random.rand(80), nan_thing, np.random.rand(100))),
    ]

    input_signal_units = ["m/s", "%", "°C"]

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
        np.hstack(
            (np.random.rand(50), nan_thing, np.random.rand(100), nan_thing)
        ),
        np.hstack(
            (
                np.random.rand(100),
                nan_thing[0:50],
                np.random.rand(150),
                nan_thing,
            )
        ),
        np.hstack(
            (
                np.random.rand(10),
                nan_thing[0:105],
                np.random.rand(50),
                nan_thing,
            )
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

    # Adjust based on fixtures
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
    return signal_list, input_signal_names, output_signal_names, fixture_type


# ============================================
# Good DataFrame
# ============================================
@pytest.fixture(params=dataset_type)
def good_dataframe(request):  # type: ignore

    fixture_type = request.param
    # Create a dummy dataframe
    num_samples = 100
    sampling_period = 0.1
    idx = np.arange(num_samples) * sampling_period
    u_labels = ["u1", "u2", "u3"]
    y_labels = ["y1", "y2"]
    cols_name = [*u_labels, *y_labels]
    df = pd.DataFrame(
        np.random.randn(num_samples, len(cols_name)),
        index=idx,
        columns=cols_name,
    )
    df.index.name = "Time (s)"

    if fixture_type == "SISO":
        # Slice signal list
        u_labels = u_labels[0]
        y_labels = y_labels[0]
        cols = [u_labels, y_labels]
    if fixture_type == "MISO":
        # Slice signal list
        y_labels = y_labels[0]
        cols = [*u_labels, y_labels]
    if fixture_type == "SIMO":
        # Slice signal list
        u_labels = u_labels[0]
        cols = [u_labels, *y_labels]
    if fixture_type == "MIMO":
        cols = [*u_labels, *y_labels]
    df = df.loc[:, cols]
    df = np.round(df, dmv.NUM_DECIMALS)
    return df, u_labels, y_labels, fixture_type


@pytest.fixture(params=dataset_type)
def sine_dataframe(request):  # type: ignore

    fixture_type = request.param

    Ts = 0.1
    N = 100
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
        c1 + np.sin(w1 * t) + np.sin(w2 * t) + c2 * np.sin(w3 * t),
        np.sin(w1 * t) - np.sin(w2 * t) - np.sin(w3 * t),
    ]

    data = np.vstack((np.asarray(u_values), np.asarray(y_values))).transpose()
    df = pd.DataFrame(index=t, columns=[*u_labels, *y_labels], data=data)
    df.index.name = "Time (s)"

    if fixture_type == "SISO":
        # Slice signal list
        u_labels = u_labels[0]
        y_labels = y_labels[0]
        cols = [u_labels, y_labels]
    if fixture_type == "MISO":
        # Slice signal list
        y_labels = y_labels[0]
        cols = [*u_labels, y_labels]
    if fixture_type == "SIMO":
        # Slice signal list
        u_labels = u_labels[0]
        cols = [u_labels, *y_labels]
    if fixture_type == "MIMO":
        cols = [*u_labels, *y_labels]
    df = df.loc[:, cols]
    np.round(df, dmv.NUM_DECIMALS)
    return df, u_labels, y_labels, fixture_type


@pytest.fixture(params=dataset_type)
def constant_ones_dataframe(request):  # type: ignore

    fixture_type = request.param

    N = 10
    idx = np.linspace(0, 1, N)
    u_labels = ["u1", "u2", "u3"]
    y_labels = ["y1", "y2", "y3"]
    values = np.ones((N, 6))
    df = pd.DataFrame(index=idx, columns=[*u_labels, *y_labels], data=values)
    df.index.name = "Time (s)"

    if fixture_type == "SISO":
        # Slice signal list
        u_labels = u_labels[0]
        y_labels = y_labels[0]
        cols = [u_labels, y_labels]
    if fixture_type == "MISO":
        # Slice signal list
        y_labels = y_labels[0]
        cols = [*u_labels, y_labels]
    if fixture_type == "SIMO":
        # Slice signal list
        u_labels = u_labels[0]
        cols = [u_labels, *y_labels]
    if fixture_type == "MIMO":
        cols = [*u_labels, *y_labels]
    df = df.loc[:, cols]
    np.round(df, dmv.NUM_DECIMALS)
    return df, u_labels, y_labels, fixture_type
