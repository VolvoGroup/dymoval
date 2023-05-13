"""Config file."""

import pathlib
from typing import Literal
import typing

# Constants exposed to the user
config = {
    "NUM_DECIMALS": 4,
    "COLORMAP": "tab10",
}  # Defaults
mapping_dict = {
    "num_decimals": "NUM_DECIMALS",
    "color_map": "COLORMAP",
}

# TODO If you remove python 3.10 remove the dependency (also from pyproject) from tomli as tomlib is
# part of the standard python package starting from 3.11
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

try:
    with open(
        pathlib.Path.home().joinpath(".dymoval/config.toml"), mode="rb"
    ) as fp:
        data = tomllib.load(fp)
    for k, val in data.items():
        config[mapping_dict[k]] = val
except FileNotFoundError:  # pragma: no cover
    pass


locals().update(config)

ATOL = 10**-NUM_DECIMALS  # noqa

# Internal constants
Signal_type = Literal["INPUT", "OUTPUT"]
SIGNAL_KIND: list[Signal_type] = list(typing.get_args(Signal_type))

Spectrum_type = Literal["amplitude", "power", "psd"]
SPECTRUM_KIND: list[Spectrum_type] = list(typing.get_args(Spectrum_type))

Allowed_keys_type = Literal[
    "name", "values", "signal_unit", "sampling_period", "time_unit"
]
SIGNAL_KEYS: list[Allowed_keys_type] = list(typing.get_args(Allowed_keys_type))
