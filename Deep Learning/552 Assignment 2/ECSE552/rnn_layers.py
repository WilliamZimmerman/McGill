import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
  """
  Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
  activation function.

  The input data has dimension D, the hidden state has dimension H, and we use
  a minibatch size of N.

  Inputs:
  - x: Input data for this timestep, of shape (N, D).
  - prev_h: Hidden state from previous timestep, of shape (N, H)
  - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
  - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
  - b: Biases of shape (H,)

  Returns a tuple of:
  - next_h: Next hidden state, of shape (N, H)
  - cache: Tuple of values needed for the backward pass.
  """
  next_h, cache = None, None
  ##############################################################################
  # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
  # hidden state and any values you need for the backward pass in the next_h   #
  # and cache variables respectively.                                          #
  ##############################################################################

  next_h = np.tanh(np.dot(x, Wx) + np.dot(prev_h, Wh) + b) # Compute activation tanh(x @ Wx + ht-1 @ Wh+b)
  cache = (next_h, prev_h, x, b, Wx, Wh) #Store
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return next_h, cache


def rnn_step_backward(dnext_h, cache):
  """
  Backward pass for a single timestep of a vanilla RNN.
  
  Inputs:
  - dnext_h: Gradient of loss with respect to next hidden state
  - cache: Cache object from the forward pass
  
  Returns a tuple of:
  - dx: Gradients of input data, of shape (N, D)
  - dprev_h: Gradients of previous hidden state, of shape (N, H)
  - dWx: Gradients of input-to-hidden weights, of shape (N, H)
  - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
  - db: Gradients of bias vector, of shape (H,)
  """
  dx, dprev_h, dWx, dWh, db = None, None, None, None, None
  next_h, prev_h,x, b ,Wx,Wh = cache
  ##############################################################################
  # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
  #                                                                            #
  # HINT: For the tanh function, you can compute the local derivative in terms #
  # of the output value from tanh.                                             #
  ##############################################################################
  # next_h = tanh(x @ Wx + prev_h @ Wh + b)
  # The local gradient of tanh 
  da = dnext_h * (1 - next_h ** 2)  # (N, H)

  # Gradient of loss w.r.t. input x

  dx = np.dot(da, Wx.T)            # (N, D)

  # Gradient of loss w.r.t. previous hidden state
  dprev_h = np.dot(da, Wh.T)       # (N, H)

  # Gradient of loss w.r.t. input-to-hidden weights Wx
  dWx = np.dot(x.T, da)            # (D, H)

  # Gradient of loss w.r.t. hidden-to-hidden weights Wh
  dWh = np.dot(prev_h.T, da)       # (H, H)

  # Gradient of loss w.r.t. bias (sum over batch dimension)
  db = np.sum(da, axis=0)          # (H,)

  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
  """
  Run a vanilla RNN forward on an entire sequence of data. We assume an input
  sequence composed of T vectors, each of dimension D. The RNN uses a hidden
  size of H, and we work over a minibatch containing N sequences. After running
  the RNN forward, we return the hidden states for all timesteps.
  
  Inputs:
  - x: Input data for the entire timeseries, of shape (N, T, D).
  - h0: Initial hidden state, of shape (N, H)
  - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
  - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
  - b: Biases of shape (H,)
  
  Returns a tuple of:
  - h: Hidden states for the entire timeseries, of shape (N, T, H).
  - cache: Values needed in the backward pass
  """
  h, cache = None, None
  ##############################################################################
  # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
  # input data. You should use the rnn_step_forward function that you defined  #
  # above.                                                                     #
  ##############################################################################
  N, T, D = x.shape
  H = h0.shape[1]
    
  h = np.zeros((N, T, H))
  cache = []
    
  prev_h = h0
  for t in range(T): # for timestep 
        xt = x[:, t, :]  # shape (N, D)
        next_h, step_cache = rnn_step_forward(xt, prev_h, Wx, Wh, b) 
        h[:, t, :] = next_h # store hidden state for current time step
        prev_h = next_h # store current for next iteration
        cache.append(step_cache) #append to cache for backprop
        
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return h, cache


def rnn_backward(dh, cache):
  """
  Compute the backward pass for a vanilla RNN over an entire sequence of data.
  
  Inputs:
  - dh: Upstream gradients of all hidden states, of shape (N, T, H)
  
  Returns a tuple of:
  - dx: Gradient of inputs, of shape (N, T, D)
  - dh0: Gradient of initial hidden state, of shape (N, H)
  - dWx: Gradient of input-to-hidden weights, of shape (D, H)
  - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
  - db: Gradient of biases, of shape (H,)
  """
  dx, dh0, dWx, dWh, db = None, None, None, None, None
  ##############################################################################
  # TODO: Implement the backward pass for a vanilla RNN running an entire      #
  # sequence of data. You should use the rnn_step_backward function that you   #
  # defined above.                                                             #
  ##############################################################################
  caches = cache  # Unpack the cache which stores per-timestep values
  N, T, H = dh.shape  # Batch size, sequence length, hidden dimension
  x0 = caches[0][2]   # Extract x from timestep 0 to determine input dimension
  D = x0.shape[1]     # Input feature dimension

  # Initialize gradient arrays
  dx = np.zeros((N, T, D))    # Gradient w.r.t. input sequence
  dWx = np.zeros((D, H))      # Gradient w.r.t. input-to-hidden weights
  dWh = np.zeros((H, H))      # Gradient w.r.t. hidden-to-hidden weights
  db = np.zeros((H,))         # Gradient w.r.t. bias
  dh_prev = np.zeros((N, H))  # Gradient flowing from next timestep

  # Iterate backward through time
  for t in reversed(range(T)):
      dnext_h = dh[:, t, :] + dh_prev  # Add upstream gradient from next timestep
      dxt, dh_prev, dWxt, dWht, dbt = rnn_step_backward(dnext_h, caches[t])  # Step backward

      dx[:, t, :] = dxt    # Store gradient w.r.t. input at time t
      dWx += dWxt          # Accumulate gradient w.r.t. Wx
      dWh += dWht          # Accumulate gradient w.r.t. Wh
      db += dbt            # Accumulate gradient w.r.t. bias

  dh0 = dh_prev  # The final dh_prev becomes gradient w.r.t. initial hidden state
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return dx, dh0, dWx, dWh, db


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
  """
  Forward pass for a single timestep of an LSTM.
  
  The input data has dimension D, the hidden state has dimension H, and we use
  a minibatch size of N.
  
  Inputs:
  - x: Input data, of shape (N, D)
  - prev_h: Previous hidden state, of shape (N, H)
  - prev_c: previous cell state, of shape (N, H)
  - Wx: Input-to-hidden weights, of shape (D, 4H)
  - Wh: Hidden-to-hidden weights, of shape (H, 4H)
  - b: Biases, of shape (4H,)
  
  Returns a tuple of:
  - next_h: Next hidden state, of shape (N, H)
  - next_c: Next cell state, of shape (N, H)
  - cache: Tuple of values needed for backward pass.
  """
  next_h, next_c, cache = None, None, None
  #############################################################################
  # TODO: Implement the forward pass for a single timestep of an LSTM.        #
  # You may want to use the numerically stable sigmoid implementation above.  #
  #############################################################################
  H = prev_h.shape[1]  # hidden size

    # Linear combination
  a = x @ Wx + prev_h @ Wh + b

  # Gates
  i = sigmoid(a[:, :H])           # input gate
  f = sigmoid(a[:, H:2*H])        # forget gate
  o = sigmoid(a[:, 2*H:3*H])      # output gate
  g = np.tanh(a[:, 3*H:])         # candidate cell content (a.k.a. block input)

   # Next cell state
  next_c = f * prev_c + i * g

    # Next hidden state
  next_h = o * np.tanh(next_c)
 
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  cache = (x, prev_h, prev_c, Wx, Wh, b, i, f, o, g, next_c)

  return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
  """
  Backward pass for a single timestep of an LSTM.
  
  Inputs:
  - dnext_h: Gradients of next hidden state, of shape (N, H)
  - dnext_c: Gradients of next cell state, of shape (N, H)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient of input data, of shape (N, D)
  - dprev_h: Gradient of previous hidden state, of shape (N, H)
  - dprev_c: Gradient of previous cell state, of shape (N, H)
  - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
  - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
  - db: Gradient of biases, of shape (4H,)
  """
  dx, dh, dc, dWx, dWh, db = None, None, None, None, None, None
  #############################################################################
  # TODO: Implement the backward pass for a single timestep of an LSTM.       #
  #                                                                           #
  # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
  # the output value from the nonlinearity.                                   #
  #############################################################################
  x, prev_h, prev_c, Wx, Wh, b, i, f, o, g, next_c = cache  # Unpack forward pass cache
  N, H = dnext_h.shape  # Batch size and hidden state dimension

  # Backprop through output gate and tanh
  # dnext_h = o * tanh(next_c), so chain through tanh
  dtanh_next_c = dnext_h * o
  dnext_c_total = dnext_c + dtanh_next_c * (1 - np.tanh(next_c) ** 2)  # Total gradient w.r.t. next_c

  # Backprop through cell update: next_c = f * prev_c + i * g
  dprev_c = dnext_c_total * f     # Gradient w.r.t. prev_c
  di = dnext_c_total * g          # Gradient w.r.t. input gate
  df = dnext_c_total * prev_c     # Gradient w.r.t. forget gate
  do = dnext_h * np.tanh(next_c)  # Gradient w.r.t. output gate
  dg = dnext_c_total * i          # Gradient w.r.t. candidate

  # Backprop through non-linearities
  dai = di * i * (1 - i)      # Sigmoid gradient
  daf = df * f * (1 - f)
  dao = do * o * (1 - o)
  dag = dg * (1 - g ** 2)     # Tanh gradient

  # Concatenate gate gradients into one matrix da
  da = np.hstack((dai, daf, dao, dag))  # Shape: (N, 4H)

  # Backprop through affine transform: a = x @ Wx + prev_h @ Wh + b
  dx = da @ Wx.T        # Gradient w.r.t. x
  dWx = x.T @ da        # Gradient w.r.t. Wx

  dprev_h = da @ Wh.T   # Gradient w.r.t. prev_h
  dWh = prev_h.T @ da   # Gradient w.r.t. Wh

  db = np.sum(da, axis=0)  # Gradient w.r.t. biases


  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
  """
  Forward pass for an LSTM over an entire sequence of data. We assume an input
  sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
  size of H, and we work over a minibatch containing N sequences. After running
  the LSTM forward, we return the hidden states for all timesteps.
  
  Note that the initial cell state is passed as input, but the initial cell
  state is set to zero. Also note that the cell state is not returned; it is
  an internal variable to the LSTM and is not accessed from outside.
  
  Inputs:
  - x: Input data of shape (N, T, D)
  - h0: Initial hidden state of shape (N, H)
  - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
  - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
  - b: Biases of shape (4H,)
  
  Returns a tuple of:
  - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
  - cache: Values needed for the backward pass.
  """
  h, cache = None, None
  #############################################################################
  # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
  # You should use the lstm_step_forward function that you just defined.      #
  #############################################################################
  N, T, D = x.shape
  H = h0.shape[1]
    
  h = np.zeros((N, T, H))
  c = np.zeros((N, H))


  cache = []
    
  prev_h = h0
  prev_c = np.zeros((N, H)) #initial state 0

  for t in range(T): # For each timestep
        xt = x[:, t, :]  #get input at timestep
        next_h, next_c, step_cache = lstm_step_forward(xt, prev_h, prev_c, Wx, Wh, b) #collect next steps
        h[:, t, :] = next_h #store hidden state at current time
        #next step in time
        prev_h = next_h #store variables for next itter
        prev_c = next_c
        cache.append(step_cache) #save for backprop
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return h, cache


def lstm_backward(dh, cache):
  """
  Backward pass for an LSTM over an entire sequence of data.]
  
  Inputs:
  - dh: Upstream gradients of hidden states, of shape (N, T, H)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient of input data of shape (N, T, D)
  - dh0: Gradient of initial hidden state of shape (N, H)
  - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
  - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
  - db: Gradient of biases, of shape (4H,)
  """
  dx, dh0, dWx, dWh, db = None, None, None, None, None
  #############################################################################
  # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
  # You should use the lstm_step_backward function that you just defined.     #
  #############################################################################
  N, T, H = dh.shape                 # Batch size, sequence length, hidden dim
  D = cache[0][0].shape[1]           # Input feature dimension (from x in timestep 0)

  # Initialize gradient arrays
  dx = np.zeros((N, T, D))          # Gradient w.r.t. input sequence
  dprev_h = np.zeros((N, H))        # Upstream gradient from next timestep
  dprev_c = np.zeros((N, H))        # Cell state gradient from next timestep
  dWx = np.zeros((D, 4 * H))        # Gradient w.r.t. Wx
  dWh = np.zeros((H, 4 * H))        # Gradient w.r.t. Wh
  db = np.zeros((4 * H,))           # Gradient w.r.t. biases

  # Loop backward through time
  for t in reversed(range(T)):
      dnext_h = dh[:, t, :] + dprev_h  # Add upstream hidden gradient
      dnext_c = dprev_c                # Cell state gradient from next timestep

      # Backward step for this timestep
      dxt, dh_prev, dprev_c, dWxt, dWht, dbt = lstm_step_backward(dnext_h, dnext_c, cache[t])

      dx[:, t, :] = dxt     # Store gradient w.r.t. input at time t
      dWx += dWxt           # Accumulate gradients
      dWh += dWht
      db += dbt
      dprev_h = dh_prev     # Pass hidden state gradient to next iteration

  dh0 = dh_prev  # dh0 is the gradient of the loss w.r.t. the initial hidden state
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  
  return dx, dh0, dWx, dWh, db
