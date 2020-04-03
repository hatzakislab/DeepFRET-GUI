class UiError(ValueError):
    pass
    a = 2


if a != 3:
    raise UiError("unexpected")
