# -*- coding: utf-8 -*-
"""Module containing some useful functions."""

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
import scipy.signal as signal  # noqa
from typing import Union
import os, sys, subprocess


def factorize(n: int) -> tuple[int, int]:
    """Find the smallest and closest integers *(a,b)* such that :math: `n \\le ab`."""
    a = int(np.ceil(np.sqrt(n)))
    b = int(np.ceil(n / a))
    return a, b


def list_belonging_check(
    # Does it work only for strings?
    A: Union[str, list[str]],
    B: Union[str, list[str]],
) -> list[str]:
    r"""Return the strings contained in the list *A* but not in the list *B*.

    In set formalism, this function return a list representing the set difference
    *A \\ ( A & B)*.
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
    if len(A) == 0 or len(B) == 0:
        raise IndexError("One of the argument is empty.")
    A_set = set(A)
    B_set = set(B)

    elements_not_found = set()
    if not A_set.issubset(B_set):
        elements_found = B_set & A_set  # Set intersection
        elements_not_found = (
            A_set - elements_found
        )  # Elements in A but not in B
    return list(elements_not_found)


def str2list(x: Union[str, list[str]]) -> list[str]:
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


def save_plot_as(fig: matplotlib.figure.Figure, name: str) -> None:
    """Save matplotlib figure on disk.

    Parameters
    ----------
    fig:
        Figure to be saved.
    name:
        Figure filename.
    """
    if not name:
        raise Exception(
            "You must specify a filename for the figure you want to save."
        )
    fig.tight_layout()
    fig.savefig(name)
    plt.close()


def open_tutorial() -> None:
    """Test"""
    site_packages = next(p for p in sys.path if "site-packages" in p)
    if sys.platform == "win32":
        filename = "\\dymoval\\scripts\\tutorial.py"
        os.startfile(site_packages + filename)
    else:
        filename = "/dymoval/scripts/tutorial.py"
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.call([opener, site_packages + filename])
