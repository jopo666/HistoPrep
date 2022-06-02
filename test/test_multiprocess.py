import functools
import time

from histoprep.helpers._multiprocess import is_lambda, is_local, multiprocess_loop


def divide_by_2(x: int, round_numbers: bool):
    x = x / 2
    if round_numbers:
        x = round(x, 0)
    return x


def divide_by_zero(i):
    1 / 0


def sleeping_beaty(i):
    time.sleep(1)


def test_multiprocess_loop():
    loop_rounding = multiprocess_loop(
        divide_by_2,
        range(20),
        num_workers=1,
        round_numbers=True,
    )
    for i, output in enumerate(loop_rounding):
        assert round(i / 2, 0) == output
    loop_no_rounding = multiprocess_loop(
        divide_by_2,
        range(20),
        num_workers=1,
        round_numbers=False,
    )
    for i, output in enumerate(loop_no_rounding):
        assert i / 2 == output


def test_multiprocess_loop_lambda_function_fail():
    assert is_lambda(lambda x: x)
    assert not is_lambda(functools.partial(divide_by_2, {"round_numbers": True}))
    assert not is_lambda(sleeping_beaty)


def test_multiprocess_loop_local_function_fail():
    def local_func(x):
        pass

    assert is_local(local_func)
    assert not is_local(functools.partial(divide_by_2, {"round_numbers": True}))
    assert not is_local(sleeping_beaty)
