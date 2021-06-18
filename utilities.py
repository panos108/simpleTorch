import torch
import math
from torch.nn import functional as F
from torch.nn import _reduction as _Reduction
from torch import Tensor
from torch.nn.modules.module import Module
from typing import Callable, Optional

def agregate_outputs(f_aug):

    return torch.cat((f_aug[0],f_aug[1]))


def NLL(input, target):
    n = input[0].numel()
    nll =(((input[0] - target).pow(2) / (input[1] + 1e-7))).sum()
    for i in range(input[0].shape[0]):
        nll += torch.log(torch.det(torch.diag(input[1][i,:] + 1e-7)))
    return nll/n

def gaussian_nll_loss(input, target, var, *, full=False, eps=1e-6, reduction='mean'):
    r"""Gaussian negative log likelihood loss.
    See :class:`~torch.nn.GaussianNLLLoss` for details.
    Args:
        input: expectation of the Gaussian distribution.
        target: sample from the Gaussian distribution.
        var: tensor of positive variance(s), one for each of the expectations
            in the input (heteroscedastic), or a single one (homoscedastic).
        full: ``True``/``False`` (bool), include the constant term in the loss
            calculation. Default: ``False``.
        eps: value added to var, for stability. Default: 1e-6.
        reduction: specifies the reduction to apply to the output:
            `'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the output is the average of all batch member losses,
            ``'sum'``: the output is the sum of all batch member losses.
            Default: ``'mean'``.
    """
    if not torch.jit.is_scripting():
        tens_ops = (input, target, var)

    # Inputs and targets much have same shape
    input = input.view(input.size(0), -1)
    target = target.view(target.size(0), -1)
    if input.size() != target.size():
        raise ValueError("input and target must have same size")

    # Second dim of var must match that of input or be equal to 1
    var = var.view(input.size(0), -1)
    if var.size(1) != input.size(1) and var.size(1) != 1:
        raise ValueError("var is of incorrect size")

    # Check validity of reduction mode
    if reduction != 'none' and reduction != 'mean' and reduction != 'sum':
        raise ValueError(reduction + " is not valid")

    # Entries of var must be non-negative
    if torch.any(var < 0):
        raise ValueError("var has negative entry/entries")

    # Clamp for stability
    var = var.clone()
    with torch.no_grad():
        var.clamp_(min=eps)

    # Calculate loss (without constant)
    loss = 0.5 * (torch.log(var) + (input - target)**2 / var).view(input.size(0), -1).sum(dim=1)

    # Add constant to loss term if required
    if full:
        D = input.size(1)
        loss = loss + 0.5 * D * math.log(2 * math.pi)

    # Apply reduction
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

class _Loss(Module):
    reduction: str

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction


class _WeightedLoss(_Loss):
    def __init__(self, weight: Optional[Tensor] = None, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(_WeightedLoss, self).__init__(size_average, reduce, reduction)
        self.register_buffer('weight', weight)

class GaussianNLLLoss(_Loss):
    r"""Gaussian negative log likelihood loss.

    The targets are treated as samples from Gaussian distributions with
    expectations and variances predicted by the neural network. For a
    D-dimensional ``target`` tensor modelled as having heteroscedastic Gaussian
    distributions with a D-dimensional tensor of expectations ``input`` and a
    D-dimensional tensor of positive variances ``var`` the loss is:

    .. math::
        \text{loss} = \frac{1}{2}\sum_{i=1}^D \left(\log\left(\text{max}\left(\text{var}[i],
        \ \text{eps}\right)\right) + \frac{\left(\text{input}[i] - \text{target}[i]\right)^2}
        {\text{max}\left(\text{var}[i], \ \text{eps}\right)}\right) + \text{const.}

    where :attr:`eps` is used for stability. By default, the constant term of
    the loss function is omitted unless :attr:`full` is ``True``. If ``var`` is
    a scalar (implying ``target`` tensor has homoscedastic Gaussian
    distributions) it is broadcasted to be the same size as the input.


    Args:
        full (bool, optional): include the constant term in the loss
            calculation. Default: ``False``.
        eps (float, optional): value used to clamp ``var`` (see note below), for
            stability. Default: 1e-6.
        reduction (string, optional): specifies the reduction to apply to the
            output:``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
            will be applied, ``'mean'``: the output is the average of all batch
            member losses, ``'sum'``: the output is the sum of all batch member
            losses. Default: ``'mean'``.

    Shape:
        - Input: :math:`(N, *)` where :math:`*` means any number of additional
          dimensions
        - Target: :math:`(N, *)`, same shape as the input
        - Var: :math:`(N, 1)` or :math:`(N, *)`, same shape as the input
        - Output: scalar if :attr:`reduction` is ``'mean'`` (default) or
          ``'sum'``. If :attr:`reduction` is ``'none'``, then :math:`(N)`

    Examples::
        #
        # >>> loss = nn.GaussianNLLLoss()
        # >>> input = torch.randn(5, 2, requires_grad=True)
        # >>> target = torch.randn(5, 2)
        # >>> var = torch.ones(5, 2, requires_grad=True) #heteroscedastic
        # >>> output = loss(input, target, var)
        # >>> output.backward()
        #
        #
        # >>> loss = nn.GaussianNLLLoss()
        # >>> input = torch.randn(5, 2, requires_grad=True)
        # >>> target = torch.randn(5, 2)
        # >>> var = torch.ones(5, 1, requires_grad=True) #homoscedastic
        # >>> output = loss(input, target, var)
        # >>> output.backward()

    Note:
        The clamping of ``var`` is ignored with respect to autograd, and so the
        gradients are unaffected by it.

    Reference:
        Nix, D. A. and Weigend, A. S., "Estimating the mean and variance of the
        target probability distribution", Proceedings of 1994 IEEE International
        Conference on Neural Networks (ICNN'94), Orlando, FL, USA, 1994, pp. 55-60
        vol.1, doi: 10.1109/ICNN.1994.374138.
    """
    __constants__ = ['full', 'eps', 'reduction']
    full: bool
    eps: float

    def __init__(self, *, full: bool = False, eps: float = 1e-6, reduction: str = 'mean') -> None:
        super(GaussianNLLLoss, self).__init__(None, None, reduction)
        self.full = full
        self.eps = eps

    def forward(self, input: Tensor, target: Tensor, var: Tensor) -> Tensor:
        return gaussian_nll_loss(input, target, var, full=self.full, eps=self.eps, reduction=self.reduction)