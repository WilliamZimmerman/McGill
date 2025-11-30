import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    #############################################################################
    #############################################################################
    nelements = w.shape[1]
    out = np.dot(x.reshape(x.shape[0], -1), w) + b
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    dw = np.dot(dout.T, x.reshape(x.shape[0], -1)).T
    db = np.sum(dout, axis=0)
    dx = np.dot(dout, w.T).reshape(x.shape)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return dx, dw, db

def conv_forward_naive(x, w, b, conv_param):
    stride, pad = conv_param['stride'], conv_param['pad']
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    
    H_out = 1 + (H + 2 * pad - HH) // stride
    W_out = 1 + (W + 2 * pad - WW) // stride
    
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    out = np.zeros((N, F, H_out, W_out))
    
    for i in range(N):
        for f in range(F):
            for h in range(H_out):
                for w_ in range(W_out):
                    h_start, w_start = h * stride, w_ * stride
                    h_end, w_end = h_start + HH, w_start + WW
                    out[i, f, h, w_] = np.sum(
                        x_padded[i, :, h_start:h_end, w_start:w_end] * w[f]
                    ) + b[f]
    
    cache = (x, w, b, conv_param)
    return out, cache

def conv_backward_naive(dout, cache):
    x, w, b, conv_param = cache
    stride, pad = conv_param['stride'], conv_param['pad']
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    H_out, W_out = dout.shape[2], dout.shape[3]
    
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    dx_padded = np.zeros_like(x_padded)
    dw = np.zeros_like(w)
    db = np.sum(dout, axis=(0, 2, 3))
    
    for i in range(N):
        for f in range(F):
            for h in range(H_out):
                for w_ in range(W_out):
                    h_start, w_start = h * stride, w_ * stride
                    h_end, w_end = h_start + HH, w_start + WW
                    
                    dw[f] += x_padded[i, :, h_start:h_end, w_start:w_end] * dout[i, f, h, w_]
                    dx_padded[i, :, h_start:h_end, w_start:w_end] += w[f] * dout[i, f, h, w_]
    
    dx = dx_padded[:, :, pad:-pad, pad:-pad] if pad > 0 else dx_padded
    return dx, dw, db

def max_pool_forward_naive(x, pool_param):
    pool_height, pool_width, stride = (
        pool_param['pool_height'],
        pool_param['pool_width'],
        pool_param['stride']
    )
    
    N, C, H, W = x.shape
    H_out = (H - pool_height) // stride + 1
    W_out = (W - pool_width) // stride + 1
    
    out = np.zeros((N, C, H_out, W_out))
    
    for i in range(N):
        for c in range(C):
            for h in range(H_out):
                for w in range(W_out):
                    h_start, w_start = h * stride, w * stride
                    h_end, w_end = h_start + pool_height, w_start + pool_width
                    
                    out[i, c, h, w] = np.max(x[i, c, h_start:h_end, w_start:w_end])
    
    cache = (x, pool_param)
    return out, cache

def max_pool_backward_naive(dout, cache):
    x, pool_param = cache
    pool_height, pool_width, stride = (
        pool_param['pool_height'],
        pool_param['pool_width'],
        pool_param['stride']
    )
    
    N, C, H, W = x.shape
    H_out, W_out = dout.shape[2], dout.shape[3]
    dx = np.zeros_like(x)
    
    for i in range(N):
        for c in range(C):
            for h in range(H_out):
                for w in range(W_out):
                    h_start, w_start = h * stride, w * stride
                    h_end, w_end = h_start + pool_height, w_start + pool_width
                    
                    x_pool = x[i, c, h_start:h_end, w_start:w_end]
                    mask = (x_pool == np.max(x_pool))
                    dx[i, c, h_start:h_end, w_start:w_end] += dout[i, c, h, w] * mask
    
    return dx

def batchnorm_forward(x, gamma, beta, bn_param):
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)
    running_mean = bn_param.get('running_mean', np.zeros(x.shape[1], dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(x.shape[1], dtype=x.dtype))
    
    if mode == 'train':
        sample_mean = np.mean(x, axis=0)
        sample_var = np.var(x, axis=0)
        x_hat = (x - sample_mean) / np.sqrt(sample_var + eps)
        out = gamma * x_hat + beta
        
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
        
        cache = (x, sample_mean, sample_var, x_hat, gamma, beta, eps)
    elif mode == 'test':
        x_hat = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * x_hat + beta
        cache = None
    
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var
    
    return out, cache


def batchnorm_backward(dout, cache):
    x, mean, var, x_hat, gamma, beta, eps = cache
    N, D = dout.shape
    
    dx_hat = dout * gamma
    dvar = np.sum(dx_hat * (x - mean) * -0.5 * (var + eps)**-1.5, axis=0)
    dmean = np.sum(dx_hat * -1 / np.sqrt(var + eps), axis=0) + dvar * np.sum(-2 * (x - mean), axis=0) / N
    
    dx = dx_hat / np.sqrt(var + eps) + dvar * 2 * (x - mean) / N + dmean / N
    dgamma = np.sum(dout * x_hat, axis=0)
    dbeta = np.sum(dout, axis=0)
    
    return dx, dgamma, dbeta

