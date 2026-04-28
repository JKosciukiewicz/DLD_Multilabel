import numpy as np
import sys

# Monkey-patch numpy.typing for older numpy versions (e.g., 1.20 or older)
# which might be missing NDArray, especially when newer Pillow versions expect it.
try:
    import numpy.typing as npt
    if not hasattr(npt, "NDArray"):
        npt.NDArray = np.ndarray
except (ImportError, AttributeError):
    # If numpy.typing doesn't exist at all, create it in sys.modules
    from types import ModuleType
    npt = ModuleType("numpy.typing")
    npt.NDArray = np.ndarray
    sys.modules["numpy.typing"] = npt
