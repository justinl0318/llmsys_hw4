import numpy as np

import minitorch


def test_view_backward_with_noncontiguous_gradient() -> None:
    x = minitorch.tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    weights = minitorch.tensor([[[1, 2], [3, 4], [5, 6]]])

    output = x.view(1, 2, 3).permute(0, 2, 1)
    (output * weights).sum().backward()

    assert x.grad is not None
    np.testing.assert_array_equal(
        x.grad.to_numpy(),
        np.array([[1, 3, 5], [2, 4, 6]], dtype=np.float32),
    )
