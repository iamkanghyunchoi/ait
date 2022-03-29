import torch
import torch.optim._functional as F
from torch.optim.optimizer import Optimizer, required

def clamp(input, min, max, inplace=False):
    """
    Clamp tensor input to (min, max).
    input: input tensor to be clamped
    """

    if inplace:
        input.clamp_(min, max)
        return input
    return torch.clamp(input, min, max)


def linear_quantize(input, scale, zero_point, inplace=False):
    """
    Quantize single-precision input tensor to integers with the given scaling factor and zeropoint.
    input: single-precision input tensor to be quantized
    scale: scaling factor for quantization
    zero_pint: shift for quantization
    """

    # reshape scale and zeropoint for convolutional weights and activation
    if len(input.shape) == 4:
        scale = scale.view(-1, 1, 1, 1)
        zero_point = zero_point.view(-1, 1, 1, 1)
    # reshape scale and zeropoint for linear weights
    elif len(input.shape) == 2:
        scale = scale.view(-1, 1)
        zero_point = zero_point.view(-1, 1)
    # mapping single-precision input to integer values with the given scale and zeropoint
    if inplace:
        input.mul_(scale).sub_(zero_point).round_()
        return input
    return torch.round(scale * input - zero_point)


def linear_dequantize(input, scale, zero_point, inplace=False):
    """
    Map integer input tensor to fixed point float point with given scaling factor and zeropoint.
    input: integer input tensor to be mapped
    scale: scaling factor for quantization
    zero_pint: shift for quantization
    """

    # reshape scale and zeropoint for convolutional weights and activation
    if len(input.shape) == 4:
        scale = scale.view(-1, 1, 1, 1)
        zero_point = zero_point.view(-1, 1, 1, 1)
    # reshape scale and zeropoint for linear weights
    elif len(input.shape) == 2:
        scale = scale.view(-1, 1)
        zero_point = zero_point.view(-1, 1)
    # mapping integer input to fixed point float point value with given scaling factor and zeropoint
    if inplace:
        input.add_(zero_point).div_(scale)
        return input
    return (input + zero_point) / scale, scale, zero_point


def asymmetric_linear_quantization_params(num_bits,
                                          saturation_min,
                                          saturation_max,
                                          integral_zero_point=True,
                                          signed=True):
    """
    Compute the scaling factor and zeropoint with the given quantization range.
    saturation_min: lower bound for quantization range
    saturation_max: upper bound for quantization range
    """
    n = 2**num_bits - 1
    scale = n / torch.clamp((saturation_max - saturation_min), min=1e-8)
    zero_point = scale * saturation_min

    if integral_zero_point:
        if isinstance(zero_point, torch.Tensor):
            zero_point = zero_point.round()
        else:
            zero_point = float(round(zero_point))
    if signed:
        zero_point += 2**(num_bits - 1)
    return scale, zero_point

def get_perc(x, quant_x, alpha, lr, grad, num_param, q_bit):
    changed_x = x-alpha*lr*grad
    x_transform = changed_x.clone().contiguous().view(x.shape[0], -1) ## outchannels
    w_min = x_transform.min(dim=1).values
    w_max = x_transform.max(dim=1).values
    scale, zero_point = asymmetric_linear_quantization_params(q_bit, w_min, w_max)
    new_quant_x = linear_quantize(changed_x, scale, zero_point, inplace=False)
    n = 2**(q_bit - 1)
    new_quant = torch.clamp(new_quant_x, -n, n - 1)

    num_changed = torch.count_nonzero(quant_x-new_quant)
    return num_changed / num_param

def gisgd(params, d_p_list, momentum_buffer_list, weight_decay,
        momentum,
        lr,
        dampening,
        nesterov,
        alpha_list):
    r"""Functional API that performs SGD algorithm computation.
    See :class:`~torch.optim.SGD` for details.
    """

    for i, param in enumerate(params):

        d_p = d_p_list[i]
        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf

        alpha = -lr*alpha_list[i]
        param.add_(d_p, alpha=alpha)

class SGD_adalr(Optimizer):

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, setting=None):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        self.settings = setting
        self.disable_adalr = True

        super(SGD_adalr, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD_adalr, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        alpha_total = []

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            lr = group['lr']

            for i,p in enumerate(group['params']):
                if p.grad is not None:
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                        b_t = p.grad
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])
                        b_t = self.settings.momentum*state['momentum_buffer'] + p.grad

                    g_t = p.grad + self.settings.momentum*b_t # Nesterov

                    x_transform = p.clone().contiguous().view(p.shape[0], -1) ## outchannels
                    w_min = x_transform.min(dim=1).values
                    w_max = x_transform.max(dim=1).values
                    scale, zero_point = asymmetric_linear_quantization_params(self.settings.qw, w_min, w_max)
                    new_quant_x = linear_quantize(p, scale, zero_point, inplace=False)
                    n = 2**(self.settings.qw - 1)
                    new_quant_x = torch.clamp(new_quant_x, -n, n - 1)

                    alpha = 1.0
                    num_param = p.numel()
                    
                    perc = get_perc(p, new_quant_x, alpha, lr, g_t, num_param,self.settings.qw)

                    if perc < self.settings.passing_threshold:
                        while True:
                            alpha = alpha*2
                            perc = get_perc(p, new_quant_x, alpha, lr, g_t, num_param,self.settings.qw)

                            if perc>=self.settings.passing_threshold:
                                break
                            if self.disable_adalr and alpha>100:
                                break
                        
                        start = alpha / 2
                        end = alpha

                        for i in range(self.settings.alpha_iter):
                            mid = (start+end)/2
                            perc = get_perc(p, new_quant_x, mid, lr, g_t, num_param,self.settings.qw)
                            if torch.abs(perc-self.settings.passing_threshold) < self.settings.passing_threshold/100:
                                break
                            
                            if perc < self.settings.passing_threshold:
                                start = mid
                            else:
                                end = mid

                        alpha = mid
 
                    alpha_total.append(alpha)

                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

            gisgd(params_with_grad,
                  d_p_list,
                  momentum_buffer_list,
                  weight_decay=weight_decay,
                  momentum=momentum,
                  lr=lr,
                  dampening=dampening,
                  nesterov=nesterov,
                  alpha_list=alpha_total)

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        return loss

