import craftground_native


def test_add():
    assert craftground_native.add(2, 3) == 5


def test_safe_divide():
    try:
        craftground_native.safe_divide(1, 0)
    except ZeroDivisionError:
        assert True
    else:
        assert False
