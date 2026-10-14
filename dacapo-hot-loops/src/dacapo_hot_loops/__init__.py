"""Find and analyze hot loops in DaCapo benchmarks"""


class HotLoopsError(Exception):
    """Error reported to the user without a traceback."""
