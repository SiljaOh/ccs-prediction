import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
torch = pytest.importorskip("torch")

from ccs.models.deep import CNNRegressor, LSTMRegressor  # noqa: E402


@pytest.mark.parametrize("model_cls", [LSTMRegressor, CNNRegressor])
def test_forward_shapes(model_cls):
    seq = torch.zeros(5, 32, 20)
    seq[:, :10, 0] = 1  # length-10 peptides
    out = model_cls()(seq, torch.eye(4)[torch.tensor([0, 1, 2, 3, 1])], torch.zeros(5, 3))
    assert out.shape == (5,)
