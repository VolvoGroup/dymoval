# -*- coding: utf-8 -*-
"""Module containing some useful functions."""

import numpy as np
from .config import *  # noqa
import scipy.signal as signal  # noqa
from typing import Any
import sys
import subprocess
from importlib import resources


def factorize(n: int) -> tuple[int, int]:
    r"""Find the smallest and closest integers *(a,b)* such that :math:`n \le ab`."""
    a = int(np.ceil(np.sqrt(n)))
    b = int(np.ceil(n / a))
    return a, b


def difference_lists_of_str(
    # Does it work only for strings?
    A: str | list[str],
    B: str | list[str],
) -> list[str]:
    r"""Return the strings contained in the list *A* but not in the list *B*.

    In set formalism, this function return a list representing the set difference
    :math:`A \backslash ( A \cap B)`.
    Note that the operation is not commutative.

    Parameters
    ----------
    A:
        list of strings A.
    B:
        list of strings B.
    """
    A = str2list(A)
    B = str2list(B)

    return list(set(A) - set(B))


def str2list(x: str | list[str]) -> list[str]:
    """
    Cast a *str* type to a *list[str]* type.

    If the input is alread a string, then it return it as-is.

    Parameters
    ----------
    x :
        Input string or list of strings.
    """
    if not isinstance(x, list):
        x = [x]
    return x


def _get_tutorial():
    with resources.path("dymoval.tutorial", "tutorial.ipynb") as f:
        tutorial_file_path = f
    return tutorial_file_path


def open_tutorial() -> tuple[Any, Any]:
    """Open the *Dymoval* tutorial.

    More precisely, it copies a IPython notebook named
    dymoval_tutorial.ipynb from your installation
    to your home folder and open it.

    If a dymoval_tutorial.ipynb file already exists in your home
    folder, then it will be overwritten.

    """

    # site_packages = next(p for p in sys.path if "site-packages" in p)
    # src = site_packages
    # home = str(Path.home())
    filename = _get_tutorial()
    if sys.platform == "win32":
        # dst = shutil.copyfile(src + filename, home + "\\dymoval_tutorial.ipynb")
        shell_process = subprocess.Popen(filename, shell=True)
    else:
        # filename = "/dymoval/src/tutorial/tutorial.ipynb"
        # dst = shutil.copyfile(src + filename, home + "/dymoval_tutorial.ipynb")
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        shell_process = subprocess.call([opener, filename])

    return shell_process, filename
