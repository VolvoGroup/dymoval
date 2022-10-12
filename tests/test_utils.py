# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 11:43:21 2022

@author: yt75534
"""

import pytest
import dymoval as dmv
from .fixture_data import *  # noqa
import matplotlib.pyplot as plt
import os

# import sys
# import subprocess
# import psutil


class Test_difference_lists_of_str:
    # This function returns elements not found.
    @pytest.mark.parametrize(
        "A,B, expected",
        [
            (
                ["dog", "cat", "donkey"],
                ["iguana", "dog", "cat", "donkey", "snake", "cow"],
                [],
            ),
            (
                ["banana", "dog", "cat", "orange", "apple"],
                ["iguana", "dog", "cat", "donkey", "snake", "cow"],
                ["banana", "orange", "apple"],
            ),
            ("iguana", ["iguana", "dog", "cat", "donkey", "snake", "cow"], []),
            (
                "orange",
                ["iguana", "dog", "cat", "donkey", "snake", "cow"],
                ["orange"],
            ),
            (
                ["iguana", "dog", "cat", "donkey", "snake", "cow"],
                "orange",
                ["iguana", "dog", "cat", "donkey", "snake", "cow"],
            ),
            (
                ["iguana", "dog", "cat", "donkey", "snake", "cow"],
                "iguana",
                ["dog", "cat", "donkey", "snake", "cow"],
            ),
        ],
    )
    def test_difference_lists_of_str(
        self,
        A: str | list[str],
        B: str | list[str],
        expected: list[str],
    ) -> None:
        # Nominal
        elements_not_found = dmv.difference_lists_of_str(A, B)
        assert sorted(elements_not_found) == sorted(expected)

    def test_difference_lists_of_str_empty_set(self) -> None:
        # Error
        B = [0, 1, 2, 3, 4, 5]
        with pytest.raises(IndexError):
            dmv.difference_lists_of_str([], B)


class Test_str2list:
    @pytest.mark.parametrize(
        "x, expected",
        [
            ("u1", ["u1"]),  # str input
            (
                ["a", "b", "c", "d"],
                ["a", "b", "c", "d"],
            ),  # list input
        ],
    )
    def test_str2list(self, x: str | list[str], expected: list[str]) -> None:
        actual = dmv.str2list(x)
        assert sorted(actual) == sorted(expected)


class Test_save_plot_as:
    def test_nominal(self, tmp_path: str) -> None:

        t = np.arange(1, 10)
        x = np.sin(2 * np.pi * 0.1 * t)
        filename = "potato"

        plt.plot(t, x)
        fig = plt.gcf()

        tmp_path_str = str(tmp_path)
        filename = tmp_path_str + "/potato"
        dmv.save_plot_as(fig, filename)
        assert os.path.exists(filename + ".png")


class Test_open_tutorial:
    @pytest.mark.open_tutorial
    def test_open_tutorial(self) -> None:
        shell_process, filename = dmv.open_tutorial()
        # Check that the file exist, i.e. during the installation the folder
        # script was installed in the right place.
        assert os.path.exists(filename)

        # keep it open for a while
        # time.sleep(5)

        # # Check that it opens and then kill the associated process
        # parent = psutil.Process(shell_process.pid)
        # while parent.children() == []:
        #     continue
        # children = parent.children()
        # child_pid = children[0].pid
        # subprocess.check_output("Taskkill /PID %d /F" % child_pid)
