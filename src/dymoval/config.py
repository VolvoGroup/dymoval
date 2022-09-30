"""Config file."""

import pathlib

config = {"NUM_DECIMALS": 3, "COLORMAP": "tab10"}  # Defaults
mapping_dict = {"num_decimals": "NUM_DECIMALS", "color_map": "COLORMAP"}

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
