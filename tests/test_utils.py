# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 11:43:21 2022

@author: yt75534
"""

import pytest
import dymoval as dmv
from .fixture_data import *  # noqa
from typing import Union


class Test_list_belonging_check:
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
    def test_list_belonging_check(
        self,
        A: Union[str, list[str]],
        B: Union[str, list[str]],
        expected: list[str],
    ) -> None:
        # Nominal
        elements_not_found = dmv.list_belonging_check(A, B)
        assert sorted(elements_not_found) == sorted(expected)

    def test_list_belonging_check_empty_set(self) -> None:
        # Error
        B = [0, 1, 2, 3, 4, 5]
        with pytest.raises(IndexError):
            dmv.list_belonging_check([], B)


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
    def test_str2list(
        self, x: Union[str, list[str]], expected: list[str]
    ) -> None:
        actual = dmv.str2list(x)
        assert sorted(actual) == sorted(expected)
