import pytest
import torch

from distorch.min_pairwise_distance import minimum_distances


@pytest.mark.parametrize("device_type", ("cpu", "cuda"))
@pytest.mark.parametrize("d", (2, 3))
@pytest.mark.parametrize("dtype", (torch.float, torch.double))
@pytest.mark.parametrize("n,m", ((3, 5), (64, 64), (64, 128), (128, 64)))
def test_minin_sqdist(device_type: str, d: int, dtype: torch.dtype, n: int, m: int) -> None:
    device = torch.device(device_type)

    t1 = torch.randn(n, d, device=device, dtype=dtype)
    t2 = torch.randn(m, d, device=device, dtype=dtype)

    distances = torch.cdist(t1, t2, p=2)
    min_distances_ref = distances.amin(dim=1).square()

    min_distances = minimum_distances(t1, t2, sqrt=False)
    torch.testing.assert_close(min_distances, min_distances_ref)
