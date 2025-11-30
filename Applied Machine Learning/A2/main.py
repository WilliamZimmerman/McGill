# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from copy import deepcopy
from sklearn.metrics import root_mean_squared_error
import warnings
import itertools
warnings.filterwarnings("ignore", category=RuntimeWarning)
from IPython.core.debugger import set_trace


#%%
class LinearRegression:

    def __init__(self, add_bias=True):
        self.add_bias = add_bias                # add a bias (intercept) term to input features
                   
    def fit(self, X, y, lambda_=0):
        # Convert pandas dataframes to numpy arrays
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.values
        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y = y.values

        if self.add_bias:
            X = np.column_stack((np.ones(X.shape[0]), X))  # Adds bias feature X0

        # Add L2 regularization term lambda_ * I
        I = np.eye(X.shape[1])
        if self.add_bias:
            I[0, 0] = 0  # Do not regularize bias term

        # Regularized least squares solution
        w = np.linalg.pinv(X.T @ X + lambda_ * I) @ X.T @ y
        return w

    def fit_QR(self, X, y, lambda_=0):
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.values
        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y = y.values
        if self.add_bias:
            X = np.column_stack((np.ones(X.shape[0]), X))
        
        # Perform QR decomposition
        Q, R = np.linalg.qr(X)

        # Add L2 regularization (lambda * I) to R^T R part
        I = np.eye(R.shape[1])
        if self.add_bias:
            I[0, 0] = 0  # Do not regularize bias term
        
        if R.shape[0] == R.shape[1]:
            # Regularized solution
            beta = np.linalg.solve(R.T @ R + lambda_ * I, Q.T @ y)  
        else:
            # Use pseudo-inverse for non-square R
            beta = np.linalg.pinv(R) @ (Q.T @ y)

        return beta

    def fit_SVD(self, X, y, lambda_=0):
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.values
        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y = y.values
            
        if self.add_bias:
            X = np.column_stack((np.ones(X.shape[0]), X))
        
        # Perform Singular Value Decomposition (SVD)
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        
        # Apply L2 regularization by modifying singular values
        S_inv = np.diag(S / (S**2 + lambda_))  # Regularized inverse of the singular values

        # Calculate the weights using the pseudo-inverse from SVD
        beta = Vt.T @ S_inv @ U.T @ y
        
        return beta

    
    def predict(self, X, w):
        if self.add_bias:
            X = np.column_stack((np.ones(X.shape[0]), X))

        y_hat = X @ w
        return y_hat
    
    def find_gradient(self, X_batch, y_batch, W, max_grad_norm=50):
        N, D = X_batch.shape
        y_hat = X_batch @ W  # This is the dot product of X and W
                
        residuals = (y_hat - y_batch.flatten()) # Flatten y_batch if it's (N, 1)
        dW = (1/N) * np.dot(X_batch.T, residuals)  # X.T @ residuals

        grad_norm = np.linalg.norm(dW)

        if grad_norm > max_grad_norm:
            dW = dW * (max_grad_norm / grad_norm)
        return dW

    import numpy as np

    def compute_loss(self, X, y, W, lambda_l1=0, lambda_l2=0):
        N, D = X.shape
        y_pred = X @ W
        loss = (1/(2*N)) * np.sum((y_pred - y.flatten())**2)
        
        # Adding L1 and L2 penalty terms
        l1_penalty = lambda_l1 * np.sum(np.abs(W))
        l2_penalty = (lambda_l2 / 2) * np.sum(W**2)
    
        return loss + l1_penalty + l2_penalty

    def fit_stochastic(self, X, y, learning_rate=5e-5, epochs=500, batch_size=4, tolerance=1e-6, max_grad_norm=50, momentum=0, lambda_l1=0, lambda_l2=0):
        X = X.values if isinstance(X, pd.DataFrame) else X
        y = y.values if isinstance(y, pd.DataFrame) else y

        if self.add_bias:
            X = np.column_stack((np.ones(X.shape[0]), X))

        m, n = X.shape
        weights = np.zeros(n)  # Initialize weights
        dW_momentum = weights.copy()
        losses = []  # Track losses for each epoch
        total_iterations = 0
        
        for epoch in range(epochs):
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0

            for i in range(0, m, batch_size):
                total_iterations += 1
                # Mini-batch selection
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]
                b = X_batch.shape[0]  # Batch size

                # Compute gradient
                dW = self.find_gradient(X_batch, y_batch, weights, max_grad_norm)

                # Add L1 penalty gradient (sub-gradient) for each weight
                dW += lambda_l1 * np.sign(weights) + lambda_l2 * weights

                # Update momentum with gradient and L1 penalty

                dW_momentum = momentum * dW_momentum + (1 - momentum) * dW           
                weights -= learning_rate * dW_momentum

                #if lambda_l1 > 0: #soft thresholding as extra
                    #weights = np.sign(weights) * np.maximum(0, np.abs(weights) - learning_rate * lambda_l1)
                # Compute loss with L1 regularization
                batch_loss = self.compute_loss(X_batch, y_batch, weights, lambda_l1, lambda_l2)
                epoch_loss += np.sum(batch_loss)  # accumulate total loss for each epoch


            # Record average loss for this epoch
            if batch_size <= 0 or m < batch_size:
                print("Warning: Batch size larger than dataset size or batch size is zero")
            losses.append(epoch_loss / (m // batch_size))

            # Stopping condition based on loss
            if np.linalg.norm(dW) < tolerance or losses[-1] < tolerance:
                print(f"Converged at epoch: {epoch + 1} and at loss: {losses[-1]:.6f}")
                break

        return weights, losses, total_iterations
    def fit_stochastic_with_path(self, X, y, learning_rate=5e-5, epochs=1000, batch_size=32, tolerance=1e-6, max_grad_norm=50, momentum=0, lambda_l1=0, lambda_l2=0):
        X = X.values if isinstance(X, pd.DataFrame) else X
        y = y.values if isinstance(y, pd.DataFrame) else y

        if self.add_bias:
            X = np.column_stack((np.ones(X.shape[0]), X))

        m, n = X.shape
        weights = np.zeros(n)  # Initialize weights
        dW_momentum = weights.copy()
        losses = []  # Track losses for each epoch
        total_iterations = 0

        weights_path = []  # Store the weights at each iteration

        for epoch in range(epochs):
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0

            for i in range(0, m, batch_size):
                total_iterations += 1
                # Mini-batch selection
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]
                b = X_batch.shape[0]  # Batch size

                # Compute gradient
                dW = self.find_gradient(X_batch, y_batch, weights, max_grad_norm)

                # Add L1 penalty gradient (sub-gradient) for each weight
                dW += lambda_l1 * np.sign(weights) + lambda_l2 * weights

                # Update momentum with gradient and L1 penalty
                dW_momentum = momentum * dW_momentum + (1 - momentum) * dW           
                weights -= learning_rate * dW_momentum

                # Store weights
                weights_path.append(weights.copy())

                # Compute loss with L1/L2 regularization
                batch_loss = self.compute_loss(X_batch, y_batch, weights, lambda_l1, lambda_l2)
                epoch_loss += np.sum(batch_loss)  # accumulate total loss for each epoch

            # Record average loss for this epoch
            losses.append(epoch_loss / (m // batch_size))

            # Stopping condition based on loss
            if np.linalg.norm(dW) < tolerance or losses[-1] < tolerance:
                print(f"Converged at epoch: {epoch + 1} and at loss: {losses[-1]:.6f}")
                break

        return weights, losses, total_iterations, np.array(weights_path)
    
    def fit_gd(self, X, y, learning_rate=5e-3, epochs=1000, tolerance=1e-6, max_grad_norm=50, momentum=0, lambda_l1=0, lambda_l2=0):
        X = X.values if isinstance(X, pd.DataFrame) else X
        y = y.values if isinstance(y, pd.DataFrame) else y

        if self.add_bias:
            X = np.column_stack((np.ones(X.shape[0]), X))

        m, n = X.shape
        weights = np.zeros(n)  # Initialize weights
        dW_momentum = weights.copy()
        losses = []  # Track losses for each epoch
        total_iterations = 0

        for epoch in range(epochs):
            total_iterations += 1

            # Compute gradient using the entire dataset (full batch)
            dW = self.find_gradient(X, y, weights, max_grad_norm)

            # Add L1 penalty gradient (sub-gradient) and L2 penalty
            dW += lambda_l1 * np.sign(weights) + lambda_l2 * weights

            # Update momentum with gradient
            dW_momentum = momentum * dW_momentum + (1 - momentum) * dW
            weights -= learning_rate * dW_momentum

            # Compute loss with L1/L2 regularization
            loss = self.compute_loss(X, y, weights, lambda_l1, lambda_l2)
            losses.append(loss)

            # Stopping condition based on loss
            if np.linalg.norm(dW) < tolerance or loss < tolerance:
                print(f"Converged at epoch: {epoch + 1} with loss: {loss:.6f}")
                break

        return weights, losses, total_iterations
    
    def fit_with_path(self, X, y, learning_rate=5e-5, epochs=1000, tolerance=1e-6, max_grad_norm=50, momentum=0, lambda_l1=0, lambda_l2=0):
        X = X.values if isinstance(X, pd.DataFrame) else X
        y = y.values if isinstance(y, pd.DataFrame) else y

        if self.add_bias:
            X = np.column_stack((np.ones(X.shape[0]), X))

        m, n = X.shape
        weights = np.zeros(n)  # Initialize weights
        dW_momentum = weights.copy()
        losses = []  # Track losses for each epoch
        weights_path = []  # Track the weights at each iteration
        total_iterations = 0

        for epoch in range(epochs):
            total_iterations += 1

            # Compute gradient using the entire dataset (full batch)
            dW = self.find_gradient(X, y, weights, max_grad_norm)

            # Add L1 penalty gradient (sub-gradient) and L2 penalty
            dW += lambda_l1 * np.sign(weights) + lambda_l2 * weights

            # Update momentum with gradient
            dW_momentum = momentum * dW_momentum + (1 - momentum) * dW
            weights -= learning_rate * dW_momentum

            # Store the weights at this iteration
            weights_path.append(weights.copy())

            # Compute loss with L1/L2 regularization
            loss = self.compute_loss(X, y, weights, lambda_l1, lambda_l2)
            losses.append(loss)

            # Stopping condition based on loss
            if np.linalg.norm(dW) < tolerance or loss < tolerance:
                print(f"Converged at epoch: {epoch + 1} with loss: {loss:.6f}")
                break

        return weights, losses, total_iterations, np.array(weights_path)



#%%
def test_train_val_split_wz(X,y,test_size=0.2,val_size=0.2,train_size=0.6):
    if(test_size+val_size+train_size!=1):
        print("Sizes must add to 1")
        return
    # Shuffle the data indices
    X = np.array(X)
    y = np.array(y)
    data_size = len(X)
    indices = np.arange(data_size)
    np.random.shuffle(indices)
    
    # Calculate the split indices
    train_end = int(train_size * data_size)
    val_end = int((train_size + val_size) * data_size)
    
    # Split the indices for training, validation, and test sets
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    # Split the data using these indices
    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    train_sorted_indices = np.argsort(X_train)
    X_train_sorted = X_train[train_sorted_indices]
    y_train_sorted = y_train[train_sorted_indices]

    val_sorted_indices = np.argsort(X_val)
    X_val_sorted = X_val[val_sorted_indices]
    y_val_sorted = y_val[val_sorted_indices]

    test_sorted_indices = np.argsort(X_test)
    X_test_sorted = X_test[test_sorted_indices]
    y_test_sorted = y_test[test_sorted_indices]
    
    return X_train_sorted, y_train_sorted, X_val_sorted, y_val_sorted, X_test_sorted, y_test_sorted

def kfold_wz(X,y,k,test_ratio=0.1):
    X = np.array(X)
    y = np.array(y)
    data_size = len(X)
    indices = np.arange(data_size)
    np.random.shuffle(indices)
    folds=[]

    fold_end = int((1-test_ratio) * data_size)

    fold_indices = indices[:fold_end]
    test_indices = indices[fold_end:]


    X_fold, y_fold = X[fold_indices], y[fold_indices]
    fold_size = len(fold_indices)//k
    for i in range(k):
        
        if i == k - 1: #last fold may be weird
            xf = X_fold[i * fold_size:]
            yf = y_fold[i * fold_size:]
        else:
            xf = X_fold[i * fold_size:(i + 1) * fold_size]
            yf = y_fold[i * fold_size:(i + 1) * fold_size]
        
        folds.append([xf, yf])
    
    #Get test indices out
    X_test, y_test = X[test_indices], y[test_indices]
    #test_sorted_indices = np.argsort(X_test)
    #X_test_sorted = X_test[test_sorted_indices]
    #y_test_sorted = y_test[test_sorted_indices]

    return folds, X_test, y_test

def kfold_true_wz(X,y,yt,k,test_ratio=0.1):
    X = np.array(X)
    y = np.array(y)
    yt = np.array(yt)    
    data_size = len(X)
    indices = np.arange(data_size)
    np.random.shuffle(indices)
    folds=[]

    fold_end = int((1-test_ratio) * data_size)

    fold_indices = indices[:fold_end]
    test_indices = indices[fold_end:]


    X_fold, y_fold, yt_fold = X[fold_indices], y[fold_indices],yt[fold_indices]
    fold_size = len(fold_indices)//k
    for i in range(k):
        
        if i == k - 1: #last fold may be weird
            xf = X_fold[i * fold_size:]
            yf = y_fold[i * fold_size:]
            ytf = yt_fold[i * fold_size:]
        else:
            xf = X_fold[i * fold_size:(i + 1) * fold_size]
            yf = y_fold[i * fold_size:(i + 1) * fold_size]
            ytf = yt_fold[i * fold_size:(i + 1) * fold_size]
        
        folds.append([xf, yf,ytf])
    
    #Get test indices out
    X_test, y_test, y_true_test = X[test_indices], y[test_indices], yt[test_indices]
    #test_sorted_indices = np.argsort(X_test)
    #X_test_sorted = X_test[test_sorted_indices]
    #y_test_sorted = y_test[test_sorted_indices]

    return folds, X_test, y_test, y_true_test

#%%
def gen_samples_wz(num_points, sigma=1):
    X = np.linspace(0,20, num=num_points)

    y=[0]*num_points
    y_true=[0]*num_points
    mu = 0 # mean and standard deviation
    epsilons = np.random.normal(mu, sigma, num_points)
    for i in range(0,num_points):
        x= X[i]
        
        val = np.sin(np.sqrt(x))+np.cos(x)+np.sin(x)
        
        y_true[i]=val
        
        y[i]=val+epsilons[i]
    return X,y,y_true

def gen_samples_task_4_wz(num_points, sigma):
    X = np.linspace(0,10, num=num_points)
    y=[0]*num_points
    y_true=[0]*num_points
    mu=0
    epsilons = np.random.normal(mu, sigma, num_points)

    for i in range(0,num_points):
        x= X[i]
        
        val = (-4)*x+10
        y_true[i]= val
        y[i]=val+2*epsilons[i]
    return X,y,y_true


#%%
#num_features = array with the number of geatures
def gaussian_transform_wz(X, plot,num_features):
   
    phi = np.zeros((len(X),num_features))
    mu_values = np.linspace(0,20, num=num_features) #Create values for 
    sigma =1
    for col in range(num_features):
            phi[:,col] = np.exp(- (X - mu_values[col]) ** 2 / (2 * sigma ** 2))

    # for col in range(num_features):
    #     phi[:, col] = X ** (col + 1)  # Polynomial terms x^1, x^2, ..., x^num_features
    
    # for col in range(num_features):
    #     frequency = (col // 2) + 1  # Increase frequency with each pair of sine/cosine
    #     if col % 2 == 0:
    #         phi[:, col] = np.sin(2 * np.pi * frequency * X)  # sine basis
    #     else:
    #         phi[:, col] = np.cos(2 * np.pi * frequency * X)  # cosine basis

    if(plot):
            plt.figure(figsize=(10, 6))
            for col in range(num_features):
                #plt.plot(X, phi[:, col], label=f'Gaussian {col+1} (mu={mu_values[col]:.2f})')
                 plt.plot(X, phi[:, col], label=f'Polynomial x^{col+1}')
            # Add plot details
            plt.title(f'Gaussian Basis Functions (D={num_features})')
            plt.xlabel('X')
            plt.ylabel('Basis Function Value')
            plt.legend(loc='upper right')
            plt.grid(True)
            plt.show()


    return phi
    
#%%
def fit_gaussian_basis_wz(X,y,y_true,model: LinearRegression, num_features, plot_model, plot_basis):
   
    for num in num_features:
        phi = gaussian_transform_wz(X,plot_basis,num)
 
        weights = model.fit(phi, y)
        weightsQR = model.fit_QR(phi, y)
        #weightsSVD = model.fit_SVD(phi, y)
        
        result = model.predict(phi, weights)
        resultQR = model.predict(phi, weightsQR)
        #resultSVD = model.predict(phi, weightsSVD)
        if(plot_model):
            plt.figure(figsize=(8, 5))
            plt.plot(X, y, label='Noisy Model', color='blue', linewidth=2)
            plt.plot(np.linspace(0,20,100),y_true, label='True Model',color='purple', linewidth=2)
            plt.plot(X, result, label='Prediction', color='red', linestyle='--')
            #plt.plot(X,resultQR, label='Prediction with QR', color='green')
            #plt.plot(X, resultSVD, label = 'Prediction with SVD', color='Orange')
            plt.title(f'Gaussian Basis Functions (D={phi.shape[1]})')
            plt.xlabel('X')
            plt.ylabel('y')
            plt.legend()
            plt.grid(True)
            plt.show()
# %%
def fit_gaussian_basis_test_train_val_wz(X_train,y_train,X_val,y_val,X_test,y_test, model: LinearRegression, num_features, plot_basis, method='regular', error='sse'):
    train_errors = []
    val_errors = []
    test_errors=[]
    
    models=[]
    for num in num_features:
        phi = gaussian_transform_wz(X_train,False,num)
        phi_val = gaussian_transform_wz(X_val,False,num)
        phi_test = gaussian_transform_wz(X_test, False,num)
        if(method.upper()=='REGULAR'):
            weights = model.fit(phi, y_train)
        elif(method.upper()=='QR'):
             weights = model.fit_QR(phi, y_train)
       
        
        result = model.predict(phi, weights)
        
        result_val = model.predict(phi_val, weights)
        results_test = model.predict(phi_test, weights)
        models.append(result)
        
        if(error=='sse'):
            trainerr=sum((y_train-result)**2)
            valerr=sum((y_val-result_val)**2)
            testerr=sum((y_test-results_test)**2)
            
           
        elif(error=='mse'):
            trainerr = root_mean_squared_error(result, y_train)**2
            valerr =  root_mean_squared_error(result_val, y_val)**2
            testerr= root_mean_squared_error(results_test, y_test)**2
        
        # weights_qr = model.fit_QR(phi, y_train)
        # result = model.predict(phi, weights_qr)
        # SSE_train_qr=sum((y_train-result)**2)
        # train_errors_qr.append(SSE_train)
        # SSE_val_qr=sum((y_val-result)**2)
        # val_errors_qr.append(SSE_val)
        train_errors.append(trainerr)
        val_errors.append(valerr)
        test_errors.append(testerr)

    # Add plot details
    if(plot_basis):
        plt.figure(figsize=(8, 6))  # Create a new figure for train error
        plt.plot(num_features, train_errors, label='Training Error', marker='o', linestyle='--', color='b')
        plt.title(f'Training Error vs Number of Basis Functions for {method}')
        plt.xlabel('Number of Basis Functions (D)')
        plt.ylabel(f'Errors ({error})')
        plt.grid(True)
        plt.legend()
        plt.show()  # Show the first plot (training error)

        # plt.figure(figsize=(8, 6))  # Create a new figure for train error
        # plt.plot(num_features, train_errors_qr, label='Training Error', marker='o', linestyle='--', color='b')
        # plt.title('Training Error vs Number of Gaussian Basis Functions with QR Decomp')
        # plt.xlabel('Number of Gaussian Basis Functions (D)')
        # plt.ylabel('Sum of Squared Errors (SSE)')
        # plt.grid(True)
        # plt.legend()
        # plt.show()  # Show the first plot (training error)

        #Plot Validation Error
        plt.figure(figsize=(8, 6))  # Create a new figure for validation error
        plt.plot(num_features, val_errors, label='Validation Error', marker='o', linestyle='-', color='r')
        plt.title(f'Validation Error vs Number of Basis Functions for {method}')
        plt.xlabel('Number of Basis Functions (D)')
        plt.ylabel(f'Errors ({error})')
        plt.grid(True)
        plt.yscale('log')
        plt.legend()
        plt.show()  # Show the second plot (validation error)

        plt.figure(figsize=(8, 6))  # Create a new figure for validation error
        plt.plot(num_features, test_errors, label='Test Error', marker='o', linestyle='-', color='r')
        plt.title(f'Test Error vs Number of Basis Functions for {method}')
        plt.xlabel('Number of Basis Functions (D)')
        plt.ylabel(f'Errors ({error})')
        plt.grid(True)
        plt.yscale('log')
        plt.legend()
        plt.show()  # Show the second plot (validation error)

        # plt.figure(figsize=(8, 6))  # Create a new figure for validation error
        # plt.plot(num_features, val_errors_qr, label='Validation Error', marker='o', linestyle='-', color='r')
        # plt.title('Validation Error vs Number of Gaussian Basis Functions')
        # plt.xlabel('Number of Gaussian Basis Functions (D)')
        # plt.ylabel('Sum of Squared Errors (SSE)')
        # plt.grid(True)
        # plt.legend()
        # plt.show()  # Show the second plot (validation error)

    return models, train_errors, val_errors
#%%
def plot_bias_variance_analysis_wz(X, y_true, guassian_models_loc, num_features_index, avg_fit, method):
    plt.figure(figsize=(10, 6))
    
    # Plot the true underlying function (blue line)
    plt.plot(X, y_true, label='True Function', color='blue', linewidth=2)

    # Plot all model fits (green lines) for the same number of features
    for iteration in range(10):  # Loop over each of the 10 iterations
        #print(guassian_models_loc[num_features_index][iteration])
        #print(num_features_index, iteration)
        plt.plot(np.linspace(0,20,60), guassian_models_loc[num_features_index][iteration], color='green', alpha=0.3, linestyle='--')  # Green lines
    
    # Plot the average of the 10 models (red line)
    
    
    plt.plot(np.linspace(0,20,60), avg_fit, color='red', label=f'Average Fit for D={num_of_features[num_features_index]} for {method}', linewidth=2)

    # Add plot details
    plt.title(f'Bias and Variance Analysis (D={num_of_features[num_features_index]}) for {method}')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_avg_errors_wz(num_features,train_errors,val_errors, method):
        plt.figure(figsize=(8, 6))  # Create a new figure for train error
        plt.plot(num_features, train_errors, label='Training Error', marker='o', linestyle='--', color='b')
        plt.title(f'Average Training Error vs Number of Basis Functions for {method}')
        plt.xlabel('Number of Basis Functions (D)')
        plt.ylabel('Sum of Squared Errors (SSE)')
        plt.grid(True)
        plt.legend()
        plt.show()  # Show the first plot (training error)

        # plt.figure(figsize=(8, 6))  # Create a new figure for train error
        # plt.plot(num_features, train_errors_qr, label='Training Error', marker='o', linestyle='--', color='b')
        # plt.title('Training Error vs Number of Gaussian Basis Functions with QR Decomp')
        # plt.xlabel('Number of Gaussian Basis Functions (D)')
        # plt.ylabel('Sum of Squared Errors (SSE)')
        # plt.grid(True)
        # plt.legend()
        # plt.show()  # Show the first plot (training error)

        #Plot Validation Error
        plt.figure(figsize=(8, 6))  # Create a new figure for validation error
        plt.plot(num_features, val_errors, label='Validation Error', marker='o', linestyle='-', color='r')
        plt.title(f'Average Validation Error vs Number of Basis Functions for {method} least squares')
        plt.xlabel('Number of Basis Functions (D)')
        plt.ylabel('Sum of Squared Errors (SSE)')
        plt.grid(True)
        plt.legend()
        plt.yscale('log')
        plt.show()  # Show the second plot (validation error)

        # plt.figure(figsize=(8, 6))  # Create a new figure for validation error
        # plt.plot(num_features, val_errors_qr, label='Validation Error', marker='o', linestyle='-', color='r')
        # plt.title('Validation Error vs Number of Gaussian Basis Functions')
        # plt.xlabel('Number of Gaussian Basis Functions (D)')
        # plt.ylabel('Sum of Squared Errors (SSE)')
        # plt.grid(True)
        # plt.legend()
        # plt.show()  # Show the second plot (validation error)

# %%
linreg = LinearRegression(True)
# %%
Xg,yg,ytrueg = gen_samples_wz(100)
#num_of_features=[1,2,3,4,5,6,7,8,9,10]
num_of_features=[10,20,30,40,50,60,70,80,90,100]
#%%
X_traing, y_traing, X_valg, y_valg, X_testg, y_testg = test_train_val_split_wz(Xg,yg)

fit_gaussian_basis_wz(X_traing,y_traing,ytrueg,linreg,num_of_features,True,True)
#%%
X_train, y_train, X_val, y_val, X_test, y_test = test_train_val_split_wz(Xg,yg)
fit_gaussian_basis_test_train_val_wz(X_train, y_train, X_val, y_val, X_test, y_test,linreg,num_of_features,True)
# fit_gaussian_basis_test_train_val(X_train, y_train, X_val, y_val, X_test, y_test,linreg,num_of_features,True, method='QR')
# fit_gaussian_basis_test_train_val(X_train, y_train, X_val, y_val, X_test, y_test,linreg,num_of_features,True, method="SVD")



# %% TASK 2
# Repeat the task 10 times
# For each number of bases 
def task2_wz(num_of_features_loc, linreg_loc: LinearRegression, lstsqmethod):
    
    gaussian_models = [[] for _ in range(len(num_of_features))] 
    train_errors=[0]*10
    train_errors=np.array(train_errors)
    val_errors=[0]*10
    val_errors=np.array(val_errors)

    print("TASK 2 STARTS HERE")

    for i in range(10):
        X_new,y_val, y_true = gen_samples_wz(100) #sample data points, X never changes
        X_train_loc, y_train_loc, X_val_loc, y_val_loc, X_test_loc, y_test_loc = test_train_val_split_wz(X_new,y_val)
        returned_models,returned_train_errors,returned_val_errors = fit_gaussian_basis_test_train_val_wz(X_train_loc, y_train_loc, X_val_loc, y_val_loc, X_test_loc, y_test_loc,linreg_loc,num_of_features_loc,False, method=lstsqmethod, error='mse')

        returned_models = np.array(returned_models)
        returned_train_errors = np.array(returned_train_errors)
        returned_val_errors = np.array(returned_val_errors)

        val_errors=np.add(val_errors,returned_val_errors)
        train_errors=np.add(train_errors, returned_train_errors)
        for j in range(len(num_of_features)):
            
            gaussian_models[j].append(deepcopy(returned_models[j]))  # Copy model for each feature size
 
    
    gaussian_models = np.array(gaussian_models)
    train_errors*=1/10
    val_errors*=1/10

    plot_avg_errors_wz(num_of_features,train_errors,val_errors, lstsqmethod)
    
    for k in range(len(num_of_features)):
        average_predictions = np.mean(gaussian_models[k], axis=0)
       
        plot_bias_variance_analysis_wz(X_new, y_true, gaussian_models,k,average_predictions,lstsqmethod)
# %%
def task3_1_wz(X,y, model: LinearRegression, num,k, l1_vals, l2_vals):
    X, y, y_true = gen_samples_wz(20,sigma=3.5)

    folds, X_test, y_test = kfold_wz(X,y,k)
   
    l1_train_errors = []
    l1_val_errors = []
    l2_train_errors = []
    l2_val_errors = []
    for lambda_l1, lambda_l2 in zip(l1_vals, l2_vals):
        validation_errors_l1 = []
        training_errors_l1 = []
        validation_errors_l2 = []
        training_errors_l2 = []
            
        for i in range(k):
            
            val_slice = folds[i]
            train_slices = [folds[j] for j in range(k) if j != i]

            X_train = np.concatenate([data[0] for data in train_slices])
            y_train = np.concatenate([data[1] for data in train_slices])

            # Validation fold
            X_val = val_slice[0]
            y_val = val_slice[1]
            
            phi_train = gaussian_transform_wz(X_train, False, num)
            phi_val = gaussian_transform_wz(X_val, False, num)
           

            weights_l1, _, _ = model.fit_gd(phi_train, y_train, lambda_l1=lambda_l1, lambda_l2=0)
            weights_l2, _, _ = model.fit_gd(phi_train, y_train, lambda_l1=0, lambda_l2=lambda_l2)
            
            #prediction and error calc for lasso
            y_train_pred_l1 = model.predict(phi_train, weights_l1)
            train_error_l1 = np.mean((y_train - y_train_pred_l1) ** 2)
            training_errors_l1.append(train_error_l1)
            
            y_val_pred_l1 = model.predict(phi_val, weights_l1)
            val_error_l1 = np.mean((y_val - y_val_pred_l1) ** 2)
            validation_errors_l1.append(val_error_l1)

            #prediction and error calc for ridge
            y_train_pred_l2 = model.predict(phi_train, weights_l2)
            train_error_l2 = np.mean((y_train - y_train_pred_l2) ** 2)
            training_errors_l2.append(train_error_l2)

            y_val_pred_l2 = model.predict(phi_val, weights_l2)
            val_error_l2 = np.mean((y_val - y_val_pred_l2) ** 2)
            validation_errors_l2.append(val_error_l2)

               
        l1_train_errors.append(np.mean(training_errors_l1))
        l1_val_errors.append(np.mean(validation_errors_l1))
        l2_train_errors.append(np.mean(training_errors_l2))
        l2_val_errors.append(np.mean(validation_errors_l2))
    
    plt.figure(figsize=(10, 6))
    plt.plot(l1_vals, l1_train_errors, label='Training Error (L1)', color='blue', marker='o', linestyle='--')
    plt.plot(l1_vals, np.array(l1_val_errors), label='Scaled Validation Error (L1)', color='red', marker='o', linestyle='-')
    plt.xscale('log')
    plt.xlabel('Regularization Strength (λ)')
    plt.ylabel('Error')
    plt.title('Training and Validation for L1 Regularization')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(l2_vals, l2_train_errors, label='Training Error (L2)', color='blue', marker='o', linestyle='--')
    plt.plot(l2_vals, np.array(l2_val_errors), label='Validation Error (L2)', color='red', marker='o', linestyle='-')
    plt.xscale('log')
    plt.xlabel('Regularization Strength (λ)')
    plt.ylabel('Error')
    plt.title('Training and Validation, and Test Errors for L2 Regularization')
    plt.legend()
    plt.grid(True)
    plt.show()
    index_of_min_l1 = l1_val_errors.index(min(l1_val_errors))
    index_of_min_l2 = l2_val_errors.index(min(l2_val_errors))

    print(f"IDEAL L1 {l1_vals[index_of_min_l1]}")
    print(f"IDEAL L2 {l2_vals[index_of_min_l2]}")
    return l1_train_errors, l1_val_errors, l2_train_errors, l2_val_errors
#%%

def task3_2_wz(model: LinearRegression, num_features, k, l1_vals, l2_vals, num_datasets, noise_variance=1.0):
    # Generate synthetic data
    y_sets = []
    y_truesets = []
    x_true, y_val, y_true = gen_samples_wz(20, sigma=1)  # Validation set for comparison without noise
    l1_train_errors = np.zeros((len(l1_vals), num_datasets))  # Initialize error matrices
    l1_test_errors = np.zeros((len(l1_vals), num_datasets))
    l2_train_errors = np.zeros((len(l2_vals), num_datasets))
    l2_test_errors = np.zeros((len(l2_vals), num_datasets))
   
    l1_biases = np.zeros((len(l1_vals), num_datasets))
    l1_variances = np.zeros((len(l1_vals), num_datasets))
    l2_biases = np.zeros((len(l2_vals), num_datasets))
    l2_variances = np.zeros((len(l2_vals), num_datasets))

    for dataset_idx in range(num_datasets):
        print(f'Starting Dataset {dataset_idx+1}/{num_datasets}')
        x,y,y_true = gen_samples_wz(100) #generate dataset
        
        idx=0
        for lambda1, lambda2 in zip(l1_vals, l2_vals):
            total_bias_l1 = 0
            total_variance_l1 = 0
            total_bias_l2 = 0
            total_variance_l2 = 0
            total_train_error_l1 = 0
            total_test_error_l1 = 0
            total_train_error_l2 = 0
            total_test_error_l2 = 0

            folds, xtest,ytest,y_test_true = kfold_true_wz(x,y,y_true,k) 
            
            phi_test = gaussian_transform_wz(xtest, False, num_features)
            predictions_l1 = []
            predictions_l2 = []
            for f in range(k):
                X_fold, y_fold, yt_fold = folds[f]
                X_train_folds = np.concatenate([folds[i][0] for i in range(k) if i != f], axis=0)
                y_train_folds = np.concatenate([folds[i][1] for i in range(k) if i != f], axis=0)
                yt_train_folds = np.concatenate([folds[i][2] for i in range(k) if i != f], axis=0)

                sorted_indices = np.argsort(X_train_folds)  # Assuming sorting by the first column
                X_train_folds = X_train_folds[sorted_indices]
                y_train_folds = y_train_folds[sorted_indices]
                yt_train_folds = yt_train_folds[sorted_indices]

                phi = gaussian_transform_wz(X_train_folds,False,num_features)

                if(f==0 and idx==0):
                    print(np.shape(phi))
                #get weights
                weightsl1, _, _ = model.fit_gd(phi, y_train_folds,lambda_l1=lambda1, lambda_l2=0)
                weightsl2, _, _ = model.fit_gd(phi, y_train_folds,lambda_l1=0, lambda_l2=lambda2)
                #predict with test values
                pred_testl1 = model.predict(phi_test, weightsl1)
                pred_testl2 = model.predict(phi_test, weightsl2)
                #predict with train values
                pred_trainl1 = model.predict(phi, weightsl1)
                pred_trainl2 = model.predict(phi, weightsl2)
                #compute MSE for train 
                predictions_l1.append(pred_testl1)
                predictions_l2.append(pred_testl2)
                train_error_l1 = root_mean_squared_error(y_train_folds,pred_trainl1)**2
                train_error_l2 = root_mean_squared_error(y_train_folds,pred_trainl2)**2
                #Computer MSE for test
                test_error_l1 = root_mean_squared_error(ytest,pred_testl1)**2
                test_error_l2 = root_mean_squared_error(ytest,pred_testl2)**2

                # Accumulate train and test errors across folds
                total_train_error_l1 += train_error_l1
                total_train_error_l2 += train_error_l2
                total_test_error_l1 += test_error_l1
                total_test_error_l2 += test_error_l2

                #Compute the bias as MSE
                bias_fold_l1 = np.mean((pred_testl1 - y_test_true) ** 2)  # Squared error for bias
                bias_fold_l2 = np.mean((pred_testl2 - y_test_true) ** 2)

                # Accumulate the bias
                total_bias_l1 += bias_fold_l1
                total_bias_l2 += bias_fold_l2

            predictions_l1 = np.array(predictions_l1)
            predictions_l2 = np.array(predictions_l2)

            # Variance is the average deviation of the predictions from the mean prediction
            avg_prediction_l1 = np.mean(predictions_l1, axis=0)
            avg_prediction_l2 = np.mean(predictions_l2, axis=0)

            variance_fold_l1 = np.mean(np.var(predictions_l1, axis=0))  # Variance of predictions
            variance_fold_l2 = np.mean(np.var(predictions_l2, axis=0))

            avg_bias_l1 = total_bias_l1 / k
            avg_bias_l2 = total_bias_l2 / k
            avg_variance_l1 = variance_fold_l1
            avg_variance_l2 = variance_fold_l2

            avg_train_error_l1 = total_train_error_l1 / k
            avg_train_error_l2 = total_train_error_l2 / k
            avg_test_error_l1 = total_test_error_l1 / k
            avg_test_error_l2 = total_test_error_l2 / k

            # After computing averages from K folds
            # Store the average errors, bias, and variance for this dataset and lambda
            l1_train_errors[idx, dataset_idx] = avg_train_error_l1
            l1_test_errors[idx, dataset_idx] = avg_test_error_l1
            l2_train_errors[idx, dataset_idx] = avg_train_error_l2
            l2_test_errors[idx, dataset_idx] = avg_test_error_l2

            l1_biases[idx, dataset_idx] = avg_bias_l1
            l1_variances[idx, dataset_idx] = np.mean(np.var(predictions_l1, axis=0))
            l2_biases[idx, dataset_idx] = avg_bias_l2
            l2_variances[idx, dataset_idx] = np.mean(np.var(predictions_l2, axis=0))
            idx+=1 #iterate idx val
       
    avg_l1_train_error = np.mean(l1_train_errors, axis=1)
    avg_l1_test_error = np.mean(l1_test_errors, axis=1)
    avg_l1_bias = np.mean(l1_biases, axis=1)
    avg_l1_variance = np.mean(l1_variances, axis=1)

    avg_l2_train_error = np.mean(l2_train_errors, axis=1)
    avg_l2_test_error = np.mean(l2_test_errors, axis=1)
    avg_l2_bias = np.mean(l2_biases, axis=1)
    avg_l2_variance = np.mean(l2_variances, axis=1)


    # Compute Bias² for L1 and L2
    bias_squared_l1 = avg_l1_bias ** 2
    bias_squared_l2 = avg_l2_bias ** 2

    # Compute Bias² + Variance for L1 and L2
    bias_variance_l1 = bias_squared_l1 + avg_l1_variance
    bias_variance_l2 = bias_squared_l2 + avg_l2_variance

    # Compute Bias² + Variance + Noise Variance for L1 and L2
    bias_variance_noise_l1 = bias_variance_l1 + noise_variance
    bias_variance_noise_l2 = bias_variance_l2 + noise_variance

    # Create figure and subplots
    fig, axs = plt.subplots(4, 2, figsize=(12, 15))  # Three rows, two columns of subplots

    # Plot Bias² for L1 Regularization (on its own plot)
    axs[0, 0].plot(l1_vals, bias_squared_l1, label='Bias²', marker='o', color='blue')
    axs[0, 0].set_xlabel('L1 Regularization Strength (lambda)')
    axs[0, 0].set_ylabel('Bias²')
    axs[0, 0].set_title('L1 Bias²')
    axs[0, 0].set_xscale('log')
    axs[0, 0].legend()

    # Plot Bias² for L2 Regularization (on its own plot)
    axs[0, 1].plot(l2_vals, bias_squared_l2, label='Bias²', marker='o', color='blue')
    axs[0, 1].set_xlabel('L2 Regularization Strength (lambda)')
    axs[0, 1].set_ylabel('Bias²')
    axs[0, 1].set_title('L2 Bias²')
    axs[0, 1].set_xscale('log')
    axs[0, 1].legend()

    # Plot Bias² + Variance and Bias² + Variance + Noise for L1 Regularization
    axs[1, 0].plot(l1_vals, bias_variance_l1, label='Bias² + Variance', marker='o', color='green')
    axs[1, 0].plot(l1_vals, bias_variance_noise_l1, label='Bias² + Variance + Noise', marker='o', color='red')
    axs[1, 0].set_xlabel('L1 Regularization Strength (lambda)')
    axs[1, 0].set_ylabel('Total Error')
    axs[1, 0].set_title('L1 Bias² + Variance and Total Error')
    axs[1, 0].set_xscale('log')
    axs[1, 0].legend()

    # Plot Bias² + Variance and Bias² + Variance + Noise for L2 Regularization
    axs[1, 1].plot(l2_vals, bias_variance_l2, label='Bias² + Variance', marker='o', color='green')
    axs[1, 1].plot(l2_vals, bias_variance_noise_l2, label='Bias² + Variance + Noise', marker='o', color='red')
    axs[1, 1].set_xlabel('L2 Regularization Strength (lambda)')
    axs[1, 1].set_ylabel('Total Error')
    axs[1, 1].set_title('L2 Bias² + Variance and Total Error')
    axs[1, 1].set_xscale('log')
    axs[1, 1].legend()

    # Plot Variance for L1 Regularization separately
    axs[2, 0].plot(l1_vals, avg_l1_variance, label='Variance', color='orange', marker='o')
    axs[2, 0].set_xlabel('L1 Regularization Strength (lambda)')
    axs[2, 0].set_ylabel('Variance')
    axs[2, 0].set_title('L1 Variance')
    axs[2, 0].set_xscale('log')
    axs[2, 0].legend()

    # Plot Variance for L2 Regularization separately
    axs[2, 1].plot(l2_vals, avg_l2_variance, label='Variance', color='orange', marker='o')
    axs[2, 1].set_xlabel('L2 Regularization Strength (lambda)')
    axs[2, 1].set_ylabel('Variance')
    axs[2, 1].set_title('L2 Variance')
    axs[2, 1].set_xscale('log')
    axs[2, 1].legend()

    # Plot Training and Test Error for L1 Regularization
    axs[3, 0].plot(l1_vals, avg_l1_train_error, label='Train Error', marker='o', color='purple')
    axs[3, 0].plot(l1_vals, avg_l1_test_error, label='Test Error', marker='o', color='brown')
    axs[3, 0].set_xlabel('L1 Regularization Strength (lambda)')
    axs[3, 0].set_ylabel('Error')
    axs[3, 0].set_title('L1 Train vs Test Error')
    axs[3, 0].set_xscale('log')
    axs[3, 0].legend()

    # Plot Training and Test Error for L2 Regularization
    axs[3, 1].plot(l2_vals, avg_l2_train_error, label='Train Error', marker='o', color='purple')
    axs[3, 1].plot(l2_vals, avg_l2_test_error, label='Test Error', marker='o', color='brown')
    axs[3, 1].set_xlabel('L2 Regularization Strength (lambda)')
    axs[3, 1].set_ylabel('Error')
    axs[3, 1].set_title('L2 Train vs Test Error')
    axs[3, 1].set_xscale('log')
    axs[3, 1].legend()

    plt.tight_layout()

    
    plt.show()
               
# %%

def plot_contour_wz(f, x1bound, x2bound, resolution, ax):
    x1range = np.linspace(x1bound[0], x1bound[1], resolution)
    x2range = np.linspace(x2bound[0], x2bound[1], resolution)
    xg, yg = np.meshgrid(x1range, x2range)
    zg = np.zeros_like(xg)
    
    # Evaluate the cost function over the grid
    for i, j in itertools.product(range(resolution), range(resolution)):
        zg[i, j] = f([xg[i, j], yg[i, j]])
    
    ax.contour(xg, yg, zg, 100)  # 25 contour levels
    return ax

# Cost function for contour plot with fixed weights for dimensions > 2
def cost_wz(w, X, y, lambda_l1=0, lambda_l2=0, fixed_weights=None):
    # Construct the full weight vector
    w_full = np.zeros(X.shape[1])
    w_full[0] = w[0]  # Bias term (w0)
    w_full[1] = w[1]  # First feature (w1)
    
    if fixed_weights is not None:
        w_full[2:] = fixed_weights[2:]  # Other weights are fixed or pre-initialized
    
    y_pred = X @ w_full  # Predicted values
    mse_loss = 0.5 * np.mean((y_pred - y)**2)  # Mean squared error
    
    # L2 regularization (Ridge)
    l2_penalty = (lambda_l2 / 2) * np.dot(w_full, w_full)
    
    # L1 regularization (Lasso)
    l1_penalty = lambda_l1 * np.sum(np.abs(w_full))
    
    # Total loss
    total_loss = mse_loss + l2_penalty + l1_penalty
    
    return total_loss


# %%
def task4_wz(model:LinearRegression, reg_list,lambda1=0,lambda2=0):
    
    x,y,y_true = gen_samples_task_4_wz(50,1)
    y=np.array(y)
    fig, axes = plt.subplots(ncols=len(reg_list), nrows=1, constrained_layout=True, figsize=(15, 5))
    phi = gaussian_transform_wz(x,False,50)

    # Zeros are clearler but random is richer
    #fixed_weights = np.random.randn(phi.shape[1])
    fixed_weights = np.zeros(phi.shape[1]) 
    # First plot: Contour of the loss function only
    fig1, axes1 = plt.subplots(ncols=len(reg_list), nrows=1, constrained_layout=True, figsize=(15, 5))

    # Loop over different regularization strengths for the contour plot only
    for i, reg_coef in enumerate(reg_list):
        # Define the current cost function for contour plotting, fixing all weights except w0 and w1
        current_cost = lambda w: cost_wz(w, phi, y, fixed_weights=fixed_weights, lambda_l1=lambda1*reg_coef, lambda_l2=lambda2*reg_coef)
        
        # Plot contour of the loss function
        plot_contour_wz(current_cost, [-20, 20], [-5, 5], 50, axes1[i])
        
        # Set plot labels
        axes1[i].set_xlabel(r'$w_0$')
        axes1[i].set_ylabel(r'$w_1$')
        axes1[i].set_title(f'$\lambda = {reg_coef}$')
        axes1[i].set_xlim([-20, 20])
        axes1[i].set_ylim([-5, 5])

    # Show the first plot with just the contours
    plt.show()

    # Second plot: Contour of the loss function with gradient descent trajectories
    fig2, axes2 = plt.subplots(ncols=len(reg_list), nrows=1, constrained_layout=True, figsize=(15, 5))

    # Loop over different regularization strengths for contour + gradient descent path
    for i, reg_coef in enumerate(reg_list):
        # Fit the model and track the weight path
        weights, losses, total_iterations, w_path = model.fit_with_path(phi, y, learning_rate=0.01, epochs=500, lambda_l1=lambda1*reg_coef, lambda_l2=lambda2*reg_coef)
        
        # Define the current cost function for contour plotting
        current_cost = lambda w: cost_wz(w, phi, y, fixed_weights=fixed_weights, lambda_l1=lambda1*reg_coef, lambda_l2=lambda2*reg_coef)
        
        # Plot contour of the loss function
        plot_contour_wz(current_cost, [-20, 20], [-5, 5], 50, axes2[i])
        
        # Plot the weight history (w_path is already tracked in fit_with_path)
        w_hist = np.array(w_path)
        axes2[i].plot(w_hist[:, 0], w_hist[:, 1], '.r', alpha=0.8, label='GD Path')  # Plot weight updates
        axes2[i].plot(w_hist[:, 0], w_hist[:, 1], '-r', alpha=0.3)  # Plot the trajectory
        
        # Set plot labels
        axes2[i].set_xlabel(r'$w_0$')
        axes2[i].set_ylabel(r'$w_1$')
        axes2[i].set_title(f'$\lambda = {reg_coef}$')
        axes2[i].set_xlim([-20, 20])
        axes2[i].set_ylim([-5, 5])
        axes2[i].legend()

    # Show the second plot with the contours and gradient descent paths
    plt.show()
#%%
def taskSVD_VSREG(X, y, model: LinearRegression, num, k, l2_vals):
    # Generate k folds for cross-validation
    folds, X_test, y_test = kfold_wz(X, y, k)
    
    least_squares_train_errors = []
    least_squares_val_errors = []
    svd_train_errors = []
    svd_val_errors = []
    
    for lambda_l2 in l2_vals:
        validation_errors_ls = []
        training_errors_ls = []
        validation_errors_svd = []
        training_errors_svd = []
        
        for i in range(k):
            val_slice = folds[i]
            train_slices = [folds[j] for j in range(k) if j != i]

            X_train = np.concatenate([data[0] for data in train_slices])
            y_train = np.concatenate([data[1] for data in train_slices])
                
            # Validation fold
            X_val = val_slice[0]
            y_val = val_slice[1]
            
            # Transform with Gaussian basis
            phi_train = gaussian_transform_wz(X_train, False, num)
            phi_val = gaussian_transform_wz(X_val, False, num)
            
            # Fitting with least squares (normal equations)
            weights_ls = model.fit(phi_train, y_train, lambda_=lambda_l2)
            
            # Fitting with SVD
            weights_svd = model.fit_SVD(phi_train, y_train, lambda_=0)
            
            # Prediction and error calc for least squares
            y_train_pred_ls = model.predict(phi_train, weights_ls)
            train_error_ls = np.mean((y_train - y_train_pred_ls) ** 2)
            training_errors_ls.append(train_error_ls)
            
            y_val_pred_ls = model.predict(phi_val, weights_ls)
            val_error_ls = np.mean((y_val - y_val_pred_ls) ** 2)
            validation_errors_ls.append(val_error_ls)

            # Prediction and error calc for SVD
            y_train_pred_svd = model.predict(phi_train, weights_svd)
            train_error_svd = np.mean((y_train - y_train_pred_svd) ** 2)
            training_errors_svd.append(train_error_svd)

            y_val_pred_svd = model.predict(phi_val, weights_svd)
            val_error_svd = np.mean((y_val - y_val_pred_svd) ** 2)
            validation_errors_svd.append(val_error_svd)
               
        least_squares_train_errors.append(np.mean(training_errors_ls))
        least_squares_val_errors.append(np.mean(validation_errors_ls))
        svd_train_errors.append(np.mean(training_errors_svd))
        svd_val_errors.append(np.mean(validation_errors_svd))
    
    # Plot least squares training and validation errors
    plt.figure(figsize=(10, 6))
    plt.plot(l2_vals, least_squares_train_errors, label='Training Error (Least Squares)', color='blue', marker='o', linestyle='--')
    plt.plot(l2_vals, least_squares_val_errors, label='Validation Error (Least Squares)', color='red', marker='o', linestyle='-')
    plt.xscale('log')
    plt.xlabel('Regularization Strength (λ)')
    plt.ylabel('Error')
    
   
    plt.plot(l2_vals, svd_train_errors, label='Training Error (SVD)', color='green', marker='*', linestyle='--')
    plt.plot(l2_vals, svd_val_errors, label='Validation Error (SVD)', color='orange', marker='*', linestyle='-')
    plt.xscale('log')
    plt.xlabel('Regularization Strength (λ)')
    plt.ylabel('Error')
    plt.title('Training and Validation Errors for Least Squares and SVD')
    plt.legend()
    plt.grid(True)
    plt.show()

    return least_squares_train_errors, least_squares_val_errors, svd_train_errors, svd_val_errors

# %%
task2_wz(num_of_features,linreg,'Regular')
#task2(num_of_features,linreg, 'QR')
#task2(num_of_features,linreg,'SVD')
# %%
taskSVD_VSREG(Xg,yg,linreg,20,10,np.logspace(-10, 3, 100))
# %%
task3_1_wz(Xg,yg,linreg,70,10,np.logspace(-3, 2.5, 100),np.logspace(-3, 2.6, 100))
# %%
#model: LinearRegression, num_features, k, l1_vals, l2_vals, num_datasets, noise_variance=1.0
task3_2_wz(linreg,70,10,np.logspace(-3, 2, 30),np.logspace(-3, 2, 30),50,noise_variance=2)
# %%
task4_wz(linreg,[0.1,1,5,10], lambda1=1)
task4_wz(linreg,[0.1,1,5,10], lambda2=1)
# %%

