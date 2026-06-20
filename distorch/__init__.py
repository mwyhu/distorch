from torch.torch_version import TorchVersion

from .metrics import boundary_metrics, pixel_center_metrics, overlap_metrics

__version__ = TorchVersion('0.1.4')  # allows version comparison with strings
