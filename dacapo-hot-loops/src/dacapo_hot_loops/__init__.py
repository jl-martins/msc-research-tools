"""Find hot loops in DaCapo benchmark profiles"""


class HotLoopsError(Exception):
    """Error reported to the user without a traceback."""
