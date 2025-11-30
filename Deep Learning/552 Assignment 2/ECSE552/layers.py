import numpy as np
# for reference
# Σ x̂ μ ∂

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
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and width
    W. We convolve each input with F different filters, where each filter spans
    all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    #############################################################################
    # TODO: Implement the convolutional forward pass.                           #
    # Hint: you can use the function np.pad for padding.                        #
    # Hint: You can also use im2col or im2col_indices see the file im2col.py    #
    #       for further information                                             #
    #############################################################################
    #Extract Needed Shapes
    N, C, H , W = x.shape
    F, _, HH, WW = w.shape

    #Extract Params
    stride = conv_param['stride']
    pad = conv_param['pad']

    #Calculate Size of conv output
    H_prime = 1 + (H + 2 * pad - HH) // stride
    W_prime = 1 + (W + 2 * pad - WW) // stride

    #pad input with zeros
    x_padded = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), mode='constant')

    #Create output buffer
    out = np.zeros((N, F, H_prime, W_prime))

    for n in range(N): #per image
        for f in range(F): #per filter
            for h in range(H_prime): # per row
                for w_index in range(W_prime): #per column

                    h_start = h * stride #Start y index for this convolution
                    h_end = h_start + HH #end y index for this convolution
                    w_start = w_index * stride #start x index for this conv
                    w_end = w_start + WW #end x index convolution

                    window = x_padded[n, :, h_start:h_end, w_start:w_end] #pixels from original image

                    #apply conv weights buy summing across all channels and pixels in the window
                    out[n, f, h, w_index] = np.sum(window * w[f, :, :, :]) + b[f] 
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    x, w, b, conv_param = cache
    dx, dw, db = None, None, None
    #############################################################################
    # TODO: Implement the convolutional backward pass.                          #
    #############################################################################
    # Extract shapes from input and parameters
    N, C, H, W = x.shape                    # Input shape
    F, C, HH, WW = w.shape                  # Filter shape
    stride = conv_param['stride']          # Stride for convolution
    pad = conv_param['pad']                # Padding applied to input
    N, F, H_out, W_out = dout.shape        # Output gradient shape (same as output from forward pass)

    # Pad the input to match forward pass
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')

    # Initialize gradients
    dx_padded = np.zeros_like(x_padded)    # Gradient w.r.t. padded input
    dw = np.zeros_like(w)                  # Gradient w.r.t. filters
    db = np.zeros_like(b)                  # Gradient w.r.t. biases

    # Loop over all images in the batch
    for n in range(N):
        # Loop over all filters
        for f in range(F):
            # Loop over output height
            for h in range(H_out):
                # Loop over output width
                for w_out in range(W_out):
                    
                    # Define the slice corners in the input
                    h_start = h * stride
                    h_end = h_start + HH
                    w_start = w_out * stride
                    w_end = w_start + WW

                    # Extract the corresponding input window
                    window = x_padded[n, :, h_start:h_end, w_start:w_end]

                    # Gradient w.r.t. filter weights: accumulate over batch and spatial positions
                    dw[f] += window * dout[n, f, h, w_out]

                    # Gradient w.r.t. input (padded): distribute upstream gradient through the filter
                    dx_padded[n, :, h_start:h_end, w_start:w_end] += w[f] * dout[n, f, h, w_out]

                    # Gradient w.r.t. bias: just sum dout across spatial locations
                    db[f] += dout[n, f, h, w_out]

    # Remove padding from dx_padded to get gradient w.r.t. original input
    dx = dx_padded[:, :, pad:pad+H, pad:pad+W]

    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None
    #############################################################################
    # TODO: Implement the max pooling forward pass                              #
    #############################################################################
    
    #Extract shapes & params
    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    pool_stride = pool_param['stride']
    
    #calculate output image dimensions
    H_out = 1 + (H - pool_height) // pool_stride
    W_out = 1 + (W - pool_width) // pool_stride
    
    #Create output buffer
    out = np.zeros((N, C, H_out, W_out))

    for n in range(N): #per image
        for c in range(C): #per channel
            for h in range(H_out): #per row
                for w_ind in range(W_out):
                    #y start and end
                    h_start = h * pool_stride 
                    h_end = h_start + pool_height

                    #x start and end
                    w_start = w_ind * pool_stride
                    w_end = w_start + pool_width

                    #get pixels that we are pooling
                    window = x[n, c, h_start:h_end, w_start:w_end]

                    #Store max value from the window
                    out[n, c, h, w_ind] = np.max(window)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """

    
    #############################################################################
    # TODO: Implement the max pooling backward pass                             #
    #############################################################################
    
    x, pool_param = cache # get cache
    #Extract shapes & params
    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    pool_stride = pool_param['stride']

    #calculate output image dimensions
    H_out = 1 + (H - pool_height) // pool_stride
    W_out = 1 + (W - pool_width) // pool_stride
    
    # Initialize gradient w.r.t. input as zeros

    dx = np.zeros_like(x)
    for n in range(N):
        for c in range(C):
            for h in range(H_out):
                for w_ind in range(W_out):
                  # get the index in the region i,j where the value is the maximum
                  i_t, j_t = np.where(np.max(x[n, c, h * pool_stride : h * pool_stride + pool_height, w_ind * pool_stride : w_ind * pool_stride + pool_width]) == x[n, c, h * pool_stride : h * pool_stride + pool_height, w_ind * pool_stride : w_ind * pool_stride + pool_width])
                  i_t, j_t = i_t[0], j_t[0]
                  
                  # Assign upstream gradient only to the max location
                  dx[n, c, h * pool_stride : h * pool_stride + pool_height, w_ind * pool_stride : w_ind * pool_stride + pool_width][i_t, j_t] = dout[n, c, h, w_ind]



    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return dx

def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the mean
    and variance of each feature, and these averages are used to normalize data
    at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7 implementation
    of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    #Extract Shape
    N, D = x.shape

   
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #############################################################################
        # TODO: Implement the training-time forward pass for batch normalization.   #
        # Use minibatch statistics to compute the mean and variance, use these      #
        # statistics to normalize the incoming data, and scale and shift the        #
        # normalized data using gamma and beta.                                     #
        #                                                                           #
        # You should store the output in the variable out. Any intermediates that   #
        # you need for the backward pass should be stored in the cache variable.    #
        #                                                                           #
        # You should also use your computed sample mean and variance together with  #
        # the momentum variable to update the running mean and running variance,    #
        # storing your result in the running_mean and running_var variables.        #
        #############################################################################
        
        mu = np.mean(x, axis=0) # calculate mean

        sigma_sq = np.mean((x - mu)**2, axis=0) #calc variance
        
        x_norm = (x-mu)/np.sqrt(sigma_sq + eps) #normalize with eps to avoid instability

        out = gamma * x_norm + beta #scale and shift normalized values

        running_mean = momentum * running_mean + (1 - momentum) * mu #update running terms
        running_var = momentum * running_var + (1 - momentum) * sigma_sq

        cache = (x, x_norm, mu, sigma_sq, gamma, beta, eps) #save for the cache
        
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
    elif mode == 'test':
        #############################################################################
        # TODO: Implement the test-time forward pass for batch normalization. Use   #
        # the running mean and variance to normalize the incoming data, then scale  #
        # and shift the normalized data using gamma and beta. Store the result in   #
        # the out variable.                                                         #
        #############################################################################
        x_norm = (x-running_mean)/np.sqrt(running_var + eps) #normalize

        out = gamma * x_norm + beta #scale and shift
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """

    dx, dgamma, dbeta = None, None, None
    x, xhat, mu, var, gamma, beta, eps = cache
    N, D = x.shape

    # Intermediate values
    x_mu = x - mu
    sqrtvar = np.sqrt(var + eps)
    

    #############################################################################
    # TODO: Implement the backward pass for batch normalization. Store the      #
    # results in the dx, dgamma, and dbeta variables.                           #
    #############################################################################
    dgammaxhat = dout  # Just to match computation graph notation: this is the upstream gradient flowing into x̂ (x_hat)

    dgamma = np.sum(dout * xhat, axis=0)  # Gradient of loss w.r.t. gamma (scale parameter)
    dbeta = np.sum(dout, axis=0)         # Gradient of loss w.r.t. beta (shift parameter)

    # Gradient of loss w.r.t. x̂ (normalized input)
    dxhat = gamma * dgammaxhat

    # Backprop through x̂ = (x - μ) / sqrt(var + eps)
    # This is the chain rule part for the sqrt(var + eps)
    dsqrtvar = np.sum(dxhat * (-x_mu) / (sqrtvar**2), axis=0)  # (∂L/∂sqrtvar)
    dvar = dsqrtvar * 0.5 / sqrtvar                            # ∂sqrtvar/∂var

    # Gradient of loss w.r.t. x_mu (x - μ)
    dx_mu_1 = dxhat * 1/sqrtvar     # Direct path from dxhat to x

    # Backprop through var = (1/N) * Σ(x - μ)^2
    # This contributes indirectly to x via ∂var/∂x
    dx_mu_2 = (2.0 / N) * x_mu * dvar  # ∂L/∂x_mu from var

    # Total gradient flowing to the mean μ
    dmu = -1 * np.sum(dx_mu_1 + dx_mu_2, axis=0)  # because x - μ: mean affects all x equally

    # Combine partial derivatives w.r.t. x
    dx1 = dx_mu_1 + dx_mu_2         # gradient from two paths
    dx2 = 1/N * np.ones((N, D)) * dmu  # from mean μ back to x

    # Total gradient w.r.t. input x
    dx = dx1 + dx2
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return dx, dgamma, dbeta