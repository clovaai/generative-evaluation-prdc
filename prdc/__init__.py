from .prdc import compute_prdc

try:
    from .prdc_torch import compute_prdc_torch
except ModuleNotFoundError:
    # Error handling
    pass
