from medmnist import OrganAMNIST
from matplotlib import pyplot as plt
import numpy as np
import torch
import time
#%% Task 1
def flatten_and_norm(dataset, normalize=True):
    data, labels = zip(*dataset)
    data = np.stack(data).reshape(len(data), -1).astype(np.float32)
    labels = np.array(labels)

    if normalize:
        mean = np.mean(data, axis=0, keepdims=True)
        std = np.std(data, axis=0, keepdims=True) + 1e-8  # Add epsilon to avoid division by zero
        data = (data - mean) / std

    return data, labels

def plot_class_distrib(labels, title):
    classes = ['bladder', 'femur-left', 'femur-right', 'heart', 'kidney-left', 'kidney-right','liver', 'lung-left',
               'lung-right', 'pancreas', 'spleen']
    n, bins, patches = plt.hist(labels, bins=len(classes), edgecolor='black')
    plt.xticks([(bins[i] + bins[i + 1]) / 2 for i in range(len(classes))], classes, rotation=30, fontsize=7)
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.title('Class distribution of'+title+' dataset')
    plt.show()

def one_hot(labels): # one-hot encoding to binarize the 11 classes
    one_hot = np.eye(11)[labels]
    return one_hot.reshape(-1, 11)

# load datasets from medmnist:
train_dataset = OrganAMNIST(split="train", download=True, size=28)
test_dataset = OrganAMNIST(split="test", download=True, size=28)
train_dataset_128 = OrganAMNIST(split="train", download=True, size=128)
test_dataset_128 = OrganAMNIST(split="test", download=True, size=128)

# flatten and normalize:
# 28x28
X_train, y_train = flatten_and_norm(train_dataset)
X_test, y_test = flatten_and_norm(test_dataset)
X_train_128, y_train_128 = flatten_and_norm(train_dataset_128)
X_test_128, y_test_128 = flatten_and_norm(test_dataset_128)

# preliminary data analysis - plot class distributions:
plot_class_distrib(y_train, title='train')
plot_class_distrib(y_test, title='test')

# one-hot encode labels:
y_train = one_hot(y_train)
y_test = one_hot(y_test)
y_train_128 = one_hot(y_train_128)
y_test_128 = one_hot(y_test_128)

#%% Task 2: implementing MLP


def softmax(z):
    z -= np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# activation functions:
def relu(x, derivative=False):
    if derivative:
        return np.where(x > 0, 1, 0)

    return np.maximum(0, x)

def leaky_relu(x, derivative=False, alpha=0.01):
    if derivative:
        return np.where(x > 0, 1, alpha)
    return np.where(x > 0, x, alpha * x)

def tanh(x, derivative=False):
    if derivative:
        return 1 - np.square(np.tanh(x))
    return np.tanh(x)


class MLP:
    def __init__(self, act_func=relu, img_dim=28, num_classes=11, hidden_layers=2, layer_sizes=[128, 128], include_bias=True):
        self.act_func = act_func
        self.input_dim = img_dim * img_dim
        self.num_classes = num_classes
        self.hidden_layers = hidden_layers
        self.layer_sizes = layer_sizes
        self.include_bias = include_bias
        self.params = {'weights': [], 'biases': []}
        

        if hidden_layers > 0:
            self.params['weights'].append(np.random.randn(self.input_dim, layer_sizes[0]) * 0.01)
            if include_bias:
                self.params['biases'].append(np.random.randn(1, layer_sizes[0]) * 0.01)
            for idx in range(1, hidden_layers):
                self.params['weights'].append(np.random.randn(layer_sizes[idx - 1], layer_sizes[idx]) * 0.01)
                if include_bias:
                    self.params['biases'].append(np.random.randn(1, layer_sizes[idx]) * 0.01)
            self.params['weights'].append(np.random.randn(layer_sizes[-1], num_classes) * 0.01)
            if include_bias:
                self.params['biases'].append(np.random.randn(1, num_classes) * 0.01)
        else:
            self.params['weights'].append(np.random.randn(self.input_dim, num_classes) * 0.01)
            if include_bias:
                self.params['biases'].append(np.random.randn(1, num_classes) * 0.01)
    
    def fit(self, optimizer,  X_train ,y_train, X_test, y_test, epochs=10,batch_size=128):
        self.val_acc = []
        self.train_acc = []
        self.time=0
        N = X_train.shape[0]
        def calc_gradients(data, labels, parameters):
            activations, z_values = [data], []
            
            # Forward pass
            activations, z_values = self.forward_pass(data,parameters,self.act_func, self.include_bias)
            

            # Backward pass
            grads_w, grads_b = self.backward_pass(activations, z_values, labels, parameters, self.act_func, self.include_bias)
    
            return grads_w, grads_b
        
        for epoch in range(epochs):
            start = time.time()
            for i in range(0, N, batch_size):
                end = i + batch_size
                X_batch = X_train[i:end]
                y_batch = y_train[i:end]
                
                self.params['weights'], self.params['biases'] = optimizer.run(calc_gradients, X_batch, y_batch, self.params)
            
            y_hat_tr = self.predict(X_train)
           
            train_acc=self.compute_accuracy(y_hat_tr,y_train)
            self.train_acc.append(train_acc)
            y_hat_test = self.predict(X_test)
            val_acc = self.compute_accuracy(y_hat_test, y_test)
            self.val_acc.append(val_acc)
            self.time+=(end-start)
            print(f"Epoch:{epoch+1}/{epochs} Train Acc: {train_acc} Val Acc: {val_acc}")
            
        
        # Plot accuracies
        plt.subplot(1, 2, 2)
        plt.plot(self.train_acc, label='Train')
        plt.plot(self.val_acc, label='Validation')
        plt.title(f'Model Accuracy for 128 Pixels')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.show()
        return self

    def predict(self, X):
        for idx in range(self.hidden_layers):
            X = np.matmul(X, self.params['weights'][idx])
            if self.include_bias:
                X += self.params['biases'][idx]
            X = self.act_func(X)
        scores = softmax(np.matmul(X, self.params['weights'][-1]) + (self.params['biases'][-1]*self.include_bias))
        return scores
    
    def forward_pass(self, data, parameters, activation_fn, include_bias):
        activations, z_values = [data], []
        
        # Iterate through hidden layers
        for idx in range(len(parameters['weights']) - 1):  # Exclude the output layer
            z = np.matmul(data, parameters['weights'][idx])
            if include_bias:
                z += parameters['biases'][idx]
            z_values.append(z)
            data = activation_fn(z)  # Apply activation
            activations.append(data)
        
        # Output layer
        z = np.matmul(data, parameters['weights'][-1])
        if include_bias:
            z += parameters['biases'][-1]
        z_values.append(z)
        preds = softmax(z)  # Use softmax for final predictions
        activations.append(preds)
    
        return activations, z_values
    def backward_pass(self, activations, z_values, labels, parameters, activation_fn, include_bias):
        grads_w = [np.zeros_like(w) for w in parameters['weights']]
        grads_b = [np.zeros_like(b) for b in parameters['biases']]
        
        # Output layer gradients
        preds = activations.pop(-1)
        error = preds - labels
        grads_w[-1] = np.dot(activations[-2].T, error) / len(labels)
        if include_bias:
            grads_b[-1] = np.mean(error, axis=0)

        # Hidden layers gradients
        for idx in range(len(parameters['weights']) - 2, -1, -1):  # Iterate backward excluding the output layer
            dz = np.dot(error, parameters['weights'][idx + 1].T) * activation_fn(z_values[idx], derivative=True)
            grads_w[idx] = np.dot(activations[idx].T, dz) / len(labels)
            if include_bias:
                grads_b[idx] = np.mean(dz, axis=0)
            error = dz
        
        return grads_w, grads_b
    def compute_accuracy(self, y, y_hat):
        pred_classes = np.argmax(y_hat, axis=1) #softmax
        true_classes = np.argmax(y, axis=1)
        return np.mean(pred_classes == true_classes) # Total 1s over total
# %%
class GradientDescent:
    def __init__(self, lr=0.001, epsilon=1, max_steps=1e4, reg_l1=0.0, reg_l2=0.0):
        self.lr = lr
        self.max_steps = max_steps
        self.reg_l1 = reg_l1
        self.epsilon = epsilon
        self.reg_l2 = reg_l2

    def run(self, grad_fn, x, y, parameters):
        weights, biases = parameters['weights'], parameters['biases']
        step_count, max_grad = 0, np.inf

        while max_grad > self.epsilon and step_count < self.max_steps:
            grad_w, grad_b = grad_fn(x, y, parameters)

            for idx, w in enumerate(weights):
                if self.reg_l1:
                    grad_w[idx] += (self.reg_l1 * np.sign(w)) 
                    div = len(x)
                if self.reg_l2:
                    grad_w[idx] += (self.reg_l2 * w) 
                    div=len(x)
                weights[idx] -= self.lr * (grad_w[idx]/div)

            if biases:
                for idx, b in enumerate(biases):
                    biases[idx] -= self.lr * grad_b[idx]

            all_gradients = grad_w + grad_b
            max_grad = -float('inf')
            for g in all_gradients:
                if g is not None:
                    max_grad = max(max_grad, np.linalg.norm(g))
            
            step_count += 1

        return weights, biases
#%%
mlp = MLP(relu, 28,11,2,[256,256], True)
optimizer = GradientDescent(epsilon=1, reg_l2=.17)

mlp.fit(optimizer, X_train, y_train, X_test,y_test,batch_size=16, epochs=10)
# %%
print(mlp.time)
#%%
mlp128 = MLP(relu, 128,11,2,[256,256],True)

optimizer = GradientDescent(epsilon=.8)
start = time.time()
mlp128.fit(optimizer, X_train_128, y_train_128, X_test_128, y_test_128)
end = time.time()

print(f"TIME ELAPSED:{end-start}")
# %%
#%%
mlp = MLP(relu, 28,11,2,[256,256], True)
optimizer = GradientDescent(epsilon=1, reg_l2=.17)

mlp.fit(optimizer, X_train, y_train, X_test,y_test,batch_size=16, epochs=10)
# %%
print(mlp.time)
#%%
mlp128 = MLP(relu, 128,11,2,[256,256],True)
import time
optimizer = GradientDescent(epsilon=.8)
start = time.time()
mlp128.fit(optimizer, X_train_128, y_train_128, X_test_128, y_test_128)
end = time.time()

print(f"TIME ELAPSED:{end-start}")

#%% Task 3.1: different layers
# 0 hidden layers:
mlp_0_hidden = MLP(img_dim=28, num_classes=11, hidden_layers=0, layer_sizes=[], act_func=relu)
optimizer = GradientDescent()
mlp_0_hidden.fit(optimizer, epochs=6, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, batch_size=16)
print('Finished 0 hidden layer model')

# 1 hidden layer:
mlp_1_hidden = MLP(img_dim=28, num_classes=11, hidden_layers=1, layer_sizes=[256], act_func=relu)
optimizer = GradientDescent()
mlp_1_hidden.fit(optimizer, epochs=6, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, batch_size=16)
print('Finished 1 hidden layer model')

# 2 hidden layers
mlp_2_hidden = MLP(img_dim=28, num_classes=11, hidden_layers=2, layer_sizes=[256, 256], act_func=relu)
import time
optimizer = GradientDescent()
start = time.time()
mlp_2_hidden.fit(optimizer, epochs=6, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, batch_size=16)
end = time.time()
print('Finished 2 hidden layers model')
print(f"Time elapsed:{end-start}")

#%% Task 3.2: different activation functions

# 2 hidden layers with tanh activation
mlp_2_tanh = MLP(img_dim=28, num_classes=11, hidden_layers=2, layer_sizes=[256, 256], act_func=tanh)
optimizer = GradientDescent()
mlp_2_tanh.fit(optimizer, epochs=6, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, batch_size=16)
print('Finished tanh activation model')

# 2 hidden layers with leaky relu activation (with leaky relu alph = 0.01)
mlp_2_leaky = MLP(img_dim=28, num_classes=11, hidden_layers=2, layer_sizes=[256, 256], act_func=leaky_relu)
optimizer = GradientDescent()
mlp_2_leaky.fit(optimizer, epochs=5, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, batch_size=16)
print('Finished leaky relu activation model')

# try different leaky relu alpha values:

#%% Task 3.4: normalization

# NOT normalized data
X_train_no_norm, y_train_no_norm = flatten_and_norm(train_dataset, normalize=False)
X_test_no_norm, y_test_no_norm = flatten_and_norm(test_dataset, normalize=False)

# not normalized model
mlp_2_no_norm = MLP(img_dim=28, num_classes=11, hidden_layers=2, layer_sizes=[256, 256], act_func=relu)
optimizer = GradientDescent()
mlp_2_no_norm.fit(optimizer, epochs=6, X_train=X_train_no_norm, y_train=y_train, X_test=X_test_no_norm, y_test=y_test, batch_size=16)
print('Finished Un-normalized model')
