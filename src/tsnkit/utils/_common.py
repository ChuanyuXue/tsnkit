"""
Author: Chuanyu (skewcy@gmail.com)
_common.py (c) 2023
Desc: description
Created:  2023-10-08T17:51:27.418Z
"""

def _interface(name: str)->property:
    ## Create a property for class
    def getter(self):
        return getattr(self, f"_{name}")

    return property(getter)


if __name__ == "__main__":
    ## Test _interface
    class Test:

        def __init__(self):
            self._a = 1

        a = _interface("a")

    t = Test()
    assert t.a == 1
    assert t._a == 1