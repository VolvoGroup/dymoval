"""Config file."""

import pathlib
from typing import Literal
import typing

# Constants exposed to the user
config = {
    "NUM_DECIMALS": 4,
    "COLORMAP": "tab10",
    "AX_HEIGHT": 2.5,
    "AX_WIDTH": 4.445,
}  # Defaults
mapping_dict = {
    "num_decimals": "NUM_DECIMALS",
    "color_map": "COLORMAP",
    "ax_height": "AX_HEIGHT",
    "ax_width": "AX_WIDTH",
}

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type:ignore

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

# Internal constants
Signal_type = Literal["INPUT", "OUTPUT"]
SIGNAL_KIND: list[Signal_type] = list(typing.get_args(Signal_type))

Spectrum_type = Literal["amplitude", "power", "psd"]
SPECTRUM_KIND: list[Spectrum_type] = list(typing.get_args(Spectrum_type))

Allowed_keys_type = Literal[
    "name", "values", "signal_unit", "sampling_period", "time_unit"
]
SIGNAL_KEYS: list[Allowed_keys_type] = list(typing.get_args(Allowed_keys_type))
