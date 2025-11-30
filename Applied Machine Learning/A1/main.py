#%%
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

#%%
## Task 1 ##
def load_data(dataset: int):
    if dataset==1:
        infrared_thermography_temperature = fetch_ucirepo(id=925) # fetch dataset 
        
        # data (as pandas dataframes) 
        X = infrared_thermography_temperature.data.features
        y = infrared_thermography_temperature.data.targets

        print(infrared_thermography_temperature.metadata) # metadata
        print(infrared_thermography_temperature.variables) # variable information
        
    if dataset==2:
        cdc_diabetes_health_indicators = fetch_ucirepo(id=891) #fetch dataset

        # data (as pandas dataframes)
        X = cdc_diabetes_health_indicators.data.features
        y = cdc_diabetes_health_indicators.data.targets

        print(cdc_diabetes_health_indicators.metadata) # metadata
        print(cdc_diabetes_health_indicators.variables) # variable information 
    
    return X, y
#%%
def clean_data(X, y):   
    # Remove all instances containing null values
    X_na = X.index[X.isna().any(axis=1)]
    y_na = y.index[y.isna().any(axis=1)]
    na_indices = X_na.union(y_na)

    X = X.drop(na_indices)
    y = y.drop(na_indices)
    # Identify binary, categorical, and continuous columns
    binary_col = X.select_dtypes(include=['int64', 'float64']).nunique() == 2
    binary_col = binary_col[binary_col].index.tolist()      # create list of binary columns

    # Handle categorical data using OneHot Encoding since it removes ordinal ranking of categories and works well with linear models 
    # source: https://www.geeksforgeeks.org/ml-one-hot-encoding/
    cat_data_columns = X.select_dtypes(include=['object', 'category']).columns #identify columns with categorical data

    X_onehot = pd.get_dummies(X, columns=cat_data_columns, drop_first=True)     # drop_first to avoid dummy variable trap
    
    # Scale features: scale only continuous variables (not including binary)
    cont_data_columns = X_onehot.select_dtypes(include=['int64', 'float64']).columns.tolist() # identify columns with continuous data  
    cont_data_columns = [col for col in cont_data_columns if col not in binary_col]

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_onehot[cont_data_columns]), columns=cont_data_columns, index=X_onehot.index)
    X_onehot.drop(columns=cont_data_columns, inplace=True) # Drop 
    # Combine scaled continuous features with one-hot encoded features
    X_final = pd.concat([X_onehot, X_scaled], axis=1)    
    print(X_final.columns)
    
    return X_final, y

#%%
def split_data(X, y, split_ratio: int): # split dataset into training and test 
    # Shuffle instances order
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    X_shuf = X.iloc[indices].reset_index(drop=True)     # shuffle all features 
    y_shuf = y.iloc[indices].reset_index(drop=True)

    train_size = int(split_ratio * len(X_shuf))         # compute number of indices in training data based on split ratio
    
    X_train = X_shuf[:train_size]
    X_test = X_shuf[train_size:]
    y_train = y_shuf[:train_size]
    y_test = y_shuf[train_size:]

    return X_train, y_train, X_test, y_test

#%%
class LinearRegression:

    def __init__(self, add_bias=True):
        self.add_bias = add_bias                # add a bias (intercept) term to input features
                   
    def fit(self, X, y):
        #convert pandas dataframes to numpy arrays
        X = X.values
        y = y.values

        if self.add_bias: 
            X = np.column_stack((np.ones(X.shape[0]), X))       # adds feature Xo to beginning of matrix 
        
        w = (np.linalg.inv(X.T @ X)) @ X.T @ y                  # l2 least squares difference
        return w
    
    def fit_QR(self, X, y):
        X = X.values
        y=y.values
        if self.add_bias:
            X = np.column_stack((np.ones(X.shape[0]), X))
        Q,R = np.linalg.qr(X)
        beta = np.linalg.solve(R, np.dot(Q.T, y))
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

    def compute_loss(self, X, y, W):
        N, D = X.shape
        y_pred = X @ W
        loss = 1/(2*N)*(y_pred-y.flatten())**2
        return loss
        
    def fit_stochastic(self, X, y, learning_rate=5e-5, epochs=1000, batch_size=32, tolerance=1e-6, max_grad_norm=50, momentum=0):
        X = X.values
        y = y.values
        if self.add_bias:
            X = np.column_stack((np.ones(X.shape[0]), X))
        
        m, n = X.shape
        weights = np.zeros(n)  # Initialize weights 
        dW_momentum = weights.copy()
        losses = []             # track losses for each epoch
        total_iterations = 0

        for epoch in range(epochs):
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0
            
            for i in range(0, m, batch_size):
                total_iterations += 1
                # Mini-batch selection
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                b = X_batch.shape[0]        # batch size

                dW = self.find_gradient(X_batch, y_batch, weights, max_grad_norm)
                dW_momentum = momentum * dW_momentum + (1-momentum)*dW           
                weights -= learning_rate * dW_momentum

                batch_loss = self.compute_loss(X_batch, y_batch, weights)
                epoch_loss += np.sum(batch_loss)        # accumulate total loss for each epoch

            # Record average loss and gradient norm for this epoch
            losses.append(epoch_loss / (m // batch_size))
               
            # Stopping condition based on loss
            if np.linalg.norm(dW)< tolerance or losses[-1] < tolerance:
                print(f"Converged at epoch: {epoch+1} and at loss: {losses[-1]:.6f}")
                break

        return weights, losses, total_iterations
    
    def measure_performance(self, y,y_hat):
        mse = mean_squared_error(y, y_hat)      # mean square error
        rmse = np.sqrt(mse)                     # root mean square error
        mae = mean_absolute_error(y, y_hat)     # mean absolute error
        mpe = np.mean((y-y_hat)/y) * 100        # mean percent error
        r_squared = r2_score(y, y_hat)
               
        return [mse, rmse, mae, mpe, r_squared]

#%%
class LogReg: 

    def __init__(self, add_bias=True):
        self.add_bias = add_bias
        pass

    def gradient(self, x, y, w, max_grad_norm=20):
        if isinstance(y, pd.Series):
            y = y.to_numpy()  # Convert pandas Series to numpy array
        N,D = x.shape
        z = np.dot(x, w)
        y_hat = 1./ (1 + np.exp(-z))    #logistic function
       
        grad = (x.T @ (y_hat - y))/N    # Compute Gradient  
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm > max_grad_norm:
            grad = grad * (max_grad_norm / grad_norm)
        
        return grad       
    
    def fit(self,X,y, max_iter=1e6, learning_rate=0.01, error = 1e-4):
        if X.ndim == 1:
            X = X[:, None]
        if self.add_bias:
            N = X.shape[0]
            X = np.column_stack([X,np.ones(N)])
        N,D = X.shape
        
        weights = np.zeros(D)
        weights = weights.reshape(D,1)

        grad = np.inf 
        i = 0
        # the code snippet below is for gradient descent
        while np.linalg.norm(grad) > error and i < max_iter:
            grad = self.gradient(X, y, weights)
            
            weights = weights - learning_rate * grad
            i += 1

            if(i%10000==0):
                print("iteraton: ", i)
        return weights

    def fit_stochastic(self, X, y, learning_rate=1e-6, epochs=1000, batch_size=64, tolerance=1e-4, power = 2, max_grad_norm=20, momentum=0):
        X = X.values
        y = y.values
            
        if self.add_bias:
            X = np.column_stack((np.ones(X.shape[0]), X))  # Add bias term
               
        m, n = X.shape
        weights = np.zeros(n)  # Initialize weights 
        losses = []             # track losses
        total_iterations = 0
        
        weights = weights.reshape(n,1)
        dW_momentum = weights.copy()

        for epoch in range(epochs):
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0
            epoch_grad_norm = 0

            for i in range(0, m, batch_size):
                total_iterations += 1
                # Mini-batch selection
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                dW = self.gradient(X_batch, y_batch, weights)
                dW_momentum = momentum * dW_momentum + (1-momentum)*dW                
                weights -= learning_rate * dW_momentum

                y_pred = self.predict(X_batch, weights, add_bias=False, convert_binary=False)
                batch_loss = -np.mean(y_batch* np.log(y_pred) + (1- y_batch) * np.log(1-y_pred))
                epoch_loss += batch_loss        # accumulate total loss for each epoch

           
            losses.append(epoch_loss / (m // batch_size))
            
            # Stopping condition based on loss
            if np.linalg.norm(dW) < tolerance or losses[-1] < tolerance :
                print(f"Converged at epoch {epoch}")
                break

        return weights, losses, total_iterations

    def predict(self, X, w, add_bias=True, convert_binary=True):
        
        if add_bias:
            X = np.column_stack((np.ones(X.shape[0]), X))
        z = np.dot(X, w)
        y_hat = 1./ (1 + np.exp(-z))    # predict probabilities

        if convert_binary:      # convert probabilities to binary predictions
            y_hat = (y_hat >= 0.5).astype(int)

        return y_hat
    
    def plot_sgd_convergence(self, losses, grad_norms, iterations):
        iters = list(range(1, iterations+1))

        plt.figure(figsize(14,6))

        # plot loss
        plt.subplot(1,2,1)
        plt.plot(losses, labels='Loss', xlabel='Epochs', ylabel='Loss', title='Convergence Speed by Loss')
        plt.legend()

        # plot gradient norm
        plt.subplot(1,2,2)
        plt.plot(iters[:len(grad_norms)], grad_norms, label='Gradient Norm', color='o')
        plt.xlabel('Iterations')
        plt.ylabel('Convergence by Gradient Norm')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def measure_performance(self, y,y_hat):
        tn, fp, fn, tp = confusion_matrix(y, y_hat).ravel()

        accuracy = accuracy_score(y, y_hat)
        precision = precision_score(y, y_hat)
        recall = recall_score(y, y_hat)
        f1 = f1_score(y, y_hat)
        confusion_mat = [[tn, fp], [fn, tp]]

        print(f"Accuracy: {accuracy:.6f}")
        print(f"Precision: {precision:.6f}")
        print(f"Recall: {recall:.6f}")
        print(f"F1 Score: {f1:.6f} ")
        #print(f"Confusion Matrix:\n = [[{tn} {fp}] \n [{fn} {tp}]]")

        return [accuracy, precision, recall, f1]

# %%

def test_minibatch_sizes_logreg(model, X_train, y_train, X_test, y_test, target, epochs=500, learning_rate=1e-6, tolerance=1e-6, momentum=0, trials=3): # currently works for lin regression
    print(f'Testing increasing batch sizes for logistic regression')
    batch_sizes = [8, 16, 32, 64, 128]     # Batch sizes to test
    results = {}
    batchsize_losses = {}
    iterations_to_convergence = {}

    for size in batch_sizes:
        print(f"Testing batch size: {size}")
        all_metrics = np.zeros(4, dtype='float')        # Initialize array to collect all metrics
        losses_data = []
        iterations_data = []

        for i in range(trials):
            # Train the model with current batch size
            weights, losses, iterations = model.fit_stochastic(X_train, y_train, learning_rate=learning_rate, epochs=epochs, batch_size=size, tolerance=tolerance, momentum=momentum)
            
            losses_data.append(losses)
            iterations_data.append(iterations)

            y_pred = model.predict(X_test, weights)
            
            metrics = model.measure_performance(y_test[target], y_pred)      
            all_metrics += np.array(metrics)        # sum collected accuracy metrics to calculate average later
        
        # average metrics over multiple trials
        avg_metrics = all_metrics / trials
        results[size] = avg_metrics

        losses_avg = np.mean(np.array(losses_data), axis=0)
        batchsize_losses[size] = losses_avg

        iterations_to_convergence[size] = np.mean(iterations_data)
    
    # convergence of change in loss function
    plt.figure(figsize=(10,6))
    for size, losses_avg in batchsize_losses.items():
        plt.plot(losses_avg, label=f'Batch Size {size}')

    plt.title('Convergence Speed Based on Loss Function for Different Batch Sizes')
    plt.xlabel('Iterations')
    plt.ylabel('Change in Loss Function')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    # Plot Performance Metrics
    batch_sizes_array = np.array(list(results.keys()))
    accuracy_values = np.array([metrics[0] for metrics in results.values()])
    precision_values = np.array([metrics[1] for metrics in results.values()])
    recall_values = np.array([metrics[2] for metrics in results.values()])
    f1_values = np.array([metrics[3] for metrics in results.values()])

    metrics_df = pd.DataFrame({
        'Batch Size': batch_sizes_array,
        'Accuracy': accuracy_values,
        'Precision': precision_values,
        'Recall': recall_values,
        'F1 Score': f1_values
    })

    print(f'Performance metrics:\n {metrics_df}')

    plt.figure(figsize=(10,6))
    plt.plot(batch_sizes_array, accuracy_values, marker='o', label='Accuracy')
    plt.plot(batch_sizes_array, precision_values, marker='o', label='Precision')
    plt.plot(batch_sizes_array, recall_values, marker='o', label='Recall')
    plt.plot(batch_sizes_array, f1_values, marker='o', label='F1 Score')
    
    plt.xscale('linear')
    plt.yscale('linear')
    plt.title('Metrics vs Mini-Batch Size')
    plt.xlabel('Batch Size')
    plt.ylabel('Value')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
# %%
def test_minibatch_sizes_linreg(model, X_train, y_train, X_test, y_test, target, epochs=1000, learning_rate=1e-4, tolerance=1e-6, momentum=0, trials=10): # currently works for lin regression
    print(f"Testing increasing batch sizes for linear regression")
    batch_sizes = [8, 16, 32, 64, 128]     # Batch sizes to test
    results = {}
    batchsize_losses = {}
    iterations_to_convergence = {}

    for size in batch_sizes:
        print(f"Testing batch size: {size}")
        
        all_metrics = np.zeros(5, dtype='float')        # Initialize array to collect all metrics
        
        losses_data = []
        iterations_data = []
        
        for i in range(trials):
            # Train the model with current batch size
            weights, losses, iterations = model.fit_stochastic(X_train, y_train, learning_rate=learning_rate, epochs=epochs, batch_size=size, tolerance=tolerance, momentum=momentum)
            
            losses_data.append(losses)
            iterations_data.append(iterations)

            y_pred = model.predict(X_test, weights)
            
            metrics = model.measure_performance(y_test[target], y_pred)      
            all_metrics += np.array(metrics)        # sum collected accuracy metrics to calculate average later
        
        # average metrics over multiple trials
        avg_metrics = all_metrics / trials
        results[size] = avg_metrics

        losses_avg = np.mean(np.array(losses_data), axis=0)
        batchsize_losses[size] = losses_avg

        iterations_to_convergence[size] = np.mean(iterations_data)
    

    # Plot Performance Metrics
    batch_sizes_array = np.array(list(results.keys()))
    mse_values = np.array([metrics[0] for metrics in results.values()])
    rmse_values = np.array([metrics[1] for metrics in results.values()])
    mae_values = np.array([metrics[2] for metrics in results.values()])
    mpe_values = np.array([metrics[3] for metrics in results.values()])
    r_squared_values = np.array([metrics[4] for metrics in results.values()])

    metrics_df = pd.DataFrame({
        'Batch Size': batch_sizes_array,
        'Mean Square Error': mse_values,
        'Root Mean Square Error': mae_values,
        'Mean Percent Error': mpe_values,
        'R-Squared': r_squared_values
    })
    
    print(f'Performance metrics:\n {metrics_df}')

    plt.figure(figsize=(10,6))

    # Plot all metrics on the same graph
    plt.plot(batch_sizes_array, mse_values, marker='o', label='MSE')
    plt.plot(batch_sizes_array, rmse_values, marker='o', label='RMSE')
    plt.plot(batch_sizes_array, mae_values, marker='o', label='MAE')
    plt.plot(batch_sizes_array, mpe_values, marker='o', label='MPE')
    plt.plot(batch_sizes_array, r_squared_values, marker='o', label='R-Squared')
    
    plt.title('Performance Metrics vs. Mini-Batch Size')
    plt.yscale('log')
    plt.xlabel('Batch Size')
    plt.ylabel('Error Value')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    # convergence based on number of iterations
    x_indices = np.arange(len(iterations_to_convergence))
    plt.bar(x_indices, iterations_to_convergence.values(), width=0.4)
    plt.title('Linear Regression Convergence Speed of Different Mini-Batch Sizes')
    plt.xticks(x_indices, iterations_to_convergence.keys(), rotation=0)
    plt.xlabel('Batch Size')
    plt.ylabel('Number of Iterations')
    plt.tight_layout()
    plt.show()

    # convergence based on change in loss function
    plt.figure(figsize=(10,6))
    for size, losses_avg in batchsize_losses.items():
        plt.plot(losses_avg, label=f'Batch Size {size}')

    plt.title('Linear Regression Convergence Speed of Different Mini-Batch Sizes')
    plt.xlabel('Iterations')
    plt.ylabel('Change in Loss')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

# %%
def plot_metrics_with_secondary_y(x_vals, mse_values, rmse_values, mae_values, r_squared_values, mean_percent_error_values, label, xscale="linear", yscale="linear", yscale2="linear"):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Primary y-axis (left) for metrics: MSE, RMSE, MAE, Mean Percent Error
    ax1.plot(x_vals, mse_values, marker='o', label='MSE', color='b')
    ax1.plot(x_vals, rmse_values, marker='o', label='RMSE', color='g')
    ax1.plot(x_vals, mae_values, marker='o', label='MAE', color='r')
    ax1.plot(x_vals, mean_percent_error_values, marker='o', label='Mean Percent Error', color='c')

    ax1.set_xscale(xscale) 
    ax1.set_yscale(yscale)
    ax1.set_xlabel(label)
    ax1.set_ylabel('Metric Values (MSE, RMSE, MAE, Percent Error)')
    
    # Limit the y-axis range for better visualization (optional)
   
    
    # Create a secondary y-axis (right) for R-squared
    ax2 = ax1.twinx()  # Create another y-axis sharing the same x-axis
    ax2.plot(x_vals, r_squared_values, marker='o', label='R-squared', color='m')
    ax2.set_ylabel('R-squared')

   
    # Combine legends from both axes
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    # Show the plot
    plt.title(f"Error vs {label} with Secondary Y-Axis for R-squared")
    plt.tight_layout()
    plt.show()

#%%
def plot_all_metrics_on_one_graph(x_vals, mse_values, rmse_values, mae_values, r_squared_values, label, xscale="linear", yscale="linear"):
    plt.figure(figsize=(10, 6))

    # Plot all metrics on the same graph
    plt.plot(x_vals, mse_values, marker='o', label='Accuracy')
    plt.plot(x_vals, rmse_values, marker='o', label='Precision')
    plt.plot(x_vals, mae_values, marker='o', label='Recall')
    plt.plot(x_vals, r_squared_values, marker='o', label='F1')
   
    
    
    plt.xscale(xscale)
    plt.yscale(yscale)

    # Set the y-axis limits to make the scale smaller (adjust as needed)
   

    # Add titles and labels
    plt.title(f"Error vs {label}")
    plt.xlabel(label)
    plt.ylabel('Error Rate')

    # Add a legend to differentiate the lines
    plt.legend(loc='best')
    
    # Display the plot
    plt.tight_layout()
    plt.show()
#%%
def test_learning_rates(model, X_train, y_train, X_test, y_test, epochs=10000, batch_size=32, tolerance=1e-6, momentum=0, trials=10, num_axes=2, data_label = 'aveOralM'):
    powers_of_ten = np.logspace(-10, 0, num=11)  # Generate powers of 10 from 10^-10 to 10^0
    rates = np.array([1 * p if i % 2 == 0 else 5 * p for i, p in enumerate(powers_of_ten)]) #add multiples of 5 in aswell    
    results = {}
    
    for rate in rates:
        print(rate)
        all_metrics = np.zeros(5, dtype='float')  # Initialize array to collect metrics [mse, rmse, mae, r_squared, mean_percent_error]

        for i in range(trials):
            # Train the model with the current learning rate
            weights, losses, grad_norms = model.fit_stochastic(X_train, y_train, learning_rate=rate, epochs=10000, batch_size=batch_size, tolerance=tolerance, momentum=momentum)
            
            # Predict on the test set
            y_pred = model.predict(X_test, weights)
           
            # Collect accuracy metrics
            metrics = model.measure_performance(y_test[data_label], y_pred)
            # Sum the metrics to calculate the average later
            all_metrics += np.array(metrics)

        # Average the metrics over multiple trials
        avg_metrics = all_metrics / trials
        print(avg_metrics)
        results[rate] = avg_metrics  # Store the averaged metrics for this learning rate
    # [mse, rmse, mae, mpe, r_squared]
    learning_rates = np.array(list(results.keys()))
    mse_values = np.array([metrics[0] for metrics in results.values()])
    rmse_values = np.array([metrics[1] for metrics in results.values()])
    mae_values = np.array([metrics[2] for metrics in results.values()])
    r_squared_values = np.array([metrics[4] for metrics in results.values()])
    mean_percent_error_values = np.array([metrics[3] for metrics in results.values()])

    if(num_axes==2):
        plot_metrics_with_secondary_y(x_vals=learning_rates,mse_values=mse_values,rmse_values=rmse_values, mae_values=mae_values,r_squared_values=r_squared_values, mean_percent_error_values=mean_percent_error_values,label="Learning Rate",xscale="log", yscale="log",yscale2="linear")
    elif(num_axes==1):
        plot_all_metrics_on_one_graph(x_vals=learning_rates,mse_values=mse_values,rmse_values=rmse_values, mae_values=mae_values,r_squared_values=r_squared_values, mean_percent_error_values=mean_percent_error_values,label="Learning Rate",xscale="log", yscale="log")
#%%
def test_learning_rates_log_sdg(model, X_train, y_train, X_test, y_test, epochs=1000, batch_size=32, tolerance=1e-6, momentum=0, trials=10, num_axes=2, data_label = 'aveOralM'):
    powers_of_ten = np.logspace(-10, 0, num=11)  # Generate powers of 10 from 10^-10 to 10^0
    rates = np.array([1 * p if i % 2 == 0 else 5 * p for i, p in enumerate(powers_of_ten)])    
    results = {}

    for rate in rates:
        print(rate)
        all_metrics = np.zeros(4, dtype='float')  # Initialize array to collect metrics [mse, rmse, mae, r_squared, mean_percent_error]

        for i in range(trials):
            # Train the model with the current learning rate
            weights, losses, grad_norms = model.fit_stochastic(X_train, y_train, learning_rate=rate, epochs=epochs, batch_size=batch_size, tolerance=tolerance, momentum=momentum)
            
            # Predict on the test set
            y_pred = model.predict(X_test, weights)
           
            # Collect accuracy metrics
            metrics = model.measure_performance(y_test[data_label], y_pred)
            # Sum the metrics to calculate the average later
            all_metrics += np.array(metrics)

        # Average the metrics over multiple trials
        avg_metrics = all_metrics / trials
        print(avg_metrics)
        results[rate] = avg_metrics  # Store the averaged metrics for this learning rate

    learning_rates = np.array(list(results.keys()))
    mse_values = np.array([metrics[0] for metrics in results.values()])
    rmse_values = np.array([metrics[1] for metrics in results.values()])
    mae_values = np.array([metrics[2] for metrics in results.values()])
    r_squared_values = np.array([metrics[3] for metrics in results.values()])

    if(num_axes==2):
        plot_metrics_with_secondary_y(x_vals=learning_rates,mse_values=mse_values,rmse_values=rmse_values, mae_values=mae_values,r_squared_values=r_squared_values,label="Learning Rate",xscale="log", yscale="log",yscale2="linear")
    elif(num_axes==1):
        plot_all_metrics_on_one_graph(x_vals=learning_rates,mse_values=mse_values,rmse_values=rmse_values, mae_values=mae_values,r_squared_values=r_squared_values,label="Learning Rate",xscale="log", yscale="linear")

#%%
def test_learning_rates_gd(model, X_train, y_train, X_test, y_test, tolerance=1e-6, trials=10, num_axes=2, data_label = 'aveOralM'):
    powers_of_ten = np.logspace(-10, 0, num=11)  # Generate powers of 10 from 10^-10 to 10^0
    rates = np.array([1 * p if i % 2 == 0 else 5 * p for i, p in enumerate(powers_of_ten)])    
    results = {}

    for rate in rates:
        print(rate)
        all_metrics = np.zeros(4, dtype='float')  # Initialize array to collect metrics [mse, rmse, mae, r_squared, mean_percent_error]

        for i in range(trials):
            # Train the model with the current learning rate
            weights = model.fit(X_train, y_train, max_iter=1e4, learning_rate=rate, error=tolerance)
            
            # Predict on the test set
            y_pred = model.predict(X_test, weights)
           
            # Collect accuracy metrics
            print(y_test)
            print(y_pred)
            metrics = model.measure_performance(y_test[data_label], y_pred)
            # Sum the metrics to calculate the average later
            all_metrics += np.array(metrics)

        # Average the metrics over multiple trials
        avg_metrics = all_metrics / trials
        print("Learning rate ",avg_metrics)
        results[rate] = avg_metrics  # Store the averaged metrics for this learning rate

    learning_rates = np.array(list(results.keys()))
    mse_values = np.array([metrics[0] for metrics in results.values()])
    rmse_values = np.array([metrics[1] for metrics in results.values()])
    mae_values = np.array([metrics[2] for metrics in results.values()])
    r_squared_values = np.array([metrics[3] for metrics in results.values()])

    if(num_axes==2):
        plot_metrics_with_secondary_y(x_vals=learning_rates,mse_values=mse_values,rmse_values=rmse_values, mae_values=mae_values,r_squared_values=r_squared_values, label="Learning Rate",xscale="log", yscale="log",yscale2="linear")
    elif(num_axes==1):
        plot_all_metrics_on_one_graph(x_vals=learning_rates,mse_values=mse_values,rmse_values=rmse_values, mae_values=mae_values,r_squared_values=r_squared_values, label="Learning Rate",xscale="log", yscale="linear")

# %%
def test_gradient_max(model, X_train, y_train, X_test, y_test, epochs=10000, batch_size=32, tolerance=1e-6, momentum=0, trials=10, label="r-squared", learning_rate=0.00001,num_axes=2):
    norms = np.linspace(0, 10000, num=200)  # Learning rates from 1 to 10^-6
    results = {}

    for norm in norms:
        all_metrics = np.zeros(5, dtype='float')  # Initialize array to collect metrics [mse, rmse, mae, r_squared, mean_percent_error]

        for i in range(trials):
            # Train the model with the current learning rate
            weights = model.fit_stochastic(X_train, y_train, learning_rate=learning_rate, epochs=epochs, batch_size=batch_size, tolerance=tolerance, max_grad_norm = norm, momentum=momentum)
            
            # Predict on the test set
            y_pred = model.predict(X_test, weights)
           
            # Collect accuracy metrics
            metrics = model.measure_performance(y_test['aveOralM'], y_pred)
            # Sum the metrics to calculate the average later
            all_metrics += np.array(metrics)

        # Average the metrics over multiple trials
        avg_metrics = all_metrics / trials
        print("Gradient ",avg_metrics)
        results[norm] = avg_metrics  # Store the averaged metrics for this learning rate
    # [mse, rmse, mae, mpe, r_squared]
    norms = np.array(list(results.keys()))
    mse_values = np.array([metrics[0] for metrics in results.values()])
    rmse_values = np.array([metrics[1] for metrics in results.values()])
    mae_values = np.array([metrics[2] for metrics in results.values()])
    r_squared_values = np.array([metrics[4] for metrics in results.values()])
    mean_percent_error_values = np.array([metrics[3] for metrics in results.values()])

    if(num_axes==2):
        plot_metrics_with_secondary_y(x_vals=norms,mse_values=mse_values,rmse_values=rmse_values, mae_values=mae_values,r_squared_values=r_squared_values, mean_percent_error_values=mean_percent_error_values,label="Gradient")
    elif(num_axes==1):
        plot_all_metrics_on_one_graph(x_vals=norms,mse_values=mse_values,rmse_values=rmse_values, mae_values=mae_values,r_squared_values=r_squared_values, mean_percent_error_values=mean_percent_error_values)

# %%
def test_momentum_values(model, X_train, y_train, X_test, y_test, learning_rate=0.5, epochs=10000, batch_size=32, tolerance=1e-6, trials=3, label="r-squared", num_axes=2):
    momentum_values = np.arange(.85, 1.1, 0.01)  # Momentum values from 0 to 1 in intervals of 0.1
    results = {}

    for momentum in momentum_values:
        # Initialize array to collect metrics as floats [mse, rmse, mae, r_squared, mean_percent_error]
        all_metrics = np.zeros(5, dtype=float)
        print(momentum)
        for i in range(trials):
            # Train the model with the current momentum value
            weights, losses, grad_norms = model.fit_stochastic(X_train, y_train, learning_rate=learning_rate, epochs=epochs, batch_size=batch_size, tolerance=tolerance, momentum=momentum)
            
            # Predict on the test set
            y_pred = model.predict(X_test, weights)
            
            # Collect accuracy metrics
            metrics = model.measure_performance(y_test['aveOralM'], y_pred)

            # Ensure metrics is a numeric NumPy array
            metrics = np.array(metrics, dtype=float)  # Convert metrics to a float array
            
            # Sum the metrics to calculate the average later
            all_metrics += metrics

        # Average the metrics over multiple trials
        avg_metrics = all_metrics / trials
        print("momentum ",momentum, avg_metrics)
        results[momentum] = avg_metrics  # Store the averaged metrics for this momentum value

    # Convert the results to lists for plotting
    momentum_values = np.array(list(results.keys()))
    mse_values = np.array([metrics[0] for metrics in results.values()])
    rmse_values = np.array([metrics[1] for metrics in results.values()])
    mae_values = np.array([metrics[2] for metrics in results.values()])
    r_squared_values = np.array([metrics[4] for metrics in results.values()])
    mean_percent_error_values = np.array([metrics[3] for metrics in results.values()])

    if(num_axes==2):
        plot_metrics_with_secondary_y(x_vals=momentum_values,mse_values=mse_values,rmse_values=rmse_values, mae_values=mae_values,r_squared_values=r_squared_values, mean_percent_error_values=mean_percent_error_values,label="Momentum")
    elif(num_axes==1):
        plot_all_metrics_on_one_graph(x_vals=momentum_values,mse_values=mse_values,rmse_values=rmse_values, mae_values=mae_values,r_squared_values=r_squared_values, mean_percent_error_values=mean_percent_error_values)
# %%
def test_gradient_max_log(model, X_train, y_train, X_test, y_test, epochs=10000, batch_size=32, tolerance=1e-6, momentum=0, trials=10, label="r-squared", learning_rate=0.00001,num_axes=1):
    norms = np.linspace(0, 10000, num=200)  
    results = {}

    for norm in norms:
        all_metrics = np.zeros(4, dtype='float')  # Initialize array to collect metrics [mse, rmse, mae, r_squared, mean_percent_error]

        for i in range(trials):
            # Train the model with the current learning rate
            weights, losses, grad_norms = model.fit_stochastic(X_train, y_train, learning_rate=learning_rate, epochs=epochs, batch_size=batch_size, tolerance=tolerance, max_grad_norm = norm, momentum=momentum)
            
            # Predict on the test set
            y_pred = model.predict(X_test, weights)
           
            # Collect accuracy metrics
            metrics = model.measure_performance(y_test['Dibetes_binary'], y_pred)
            # Sum the metrics to calculate the average later
            all_metrics += np.array(metrics)

        # Average the metrics over multiple trials
        avg_metrics = all_metrics / trials
        print(avg_metrics)
        results[norm] = avg_metrics  # Store the averaged metrics for this learning rate

    norms = np.array(list(results.keys()))
    mse_values = np.array([metrics[0] for metrics in results.values()])
    rmse_values = np.array([metrics[1] for metrics in results.values()])
    mae_values = np.array([metrics[2] for metrics in results.values()])
    r_squared_values = np.array([metrics[3] for metrics in results.values()])

    if(num_axes==2):
        plot_metrics_with_secondary_y(x_vals=norms,mse_values=mse_values,rmse_values=rmse_values, mae_values=mae_values,r_squared_values=r_squared_values, label="Gradient")
    elif(num_axes==1):
        plot_all_metrics_on_one_graph(x_vals=norms,mse_values=mse_values,rmse_values=rmse_values, mae_values=mae_values,r_squared_values=r_squared_values)

# %%
def test_momentum_values_log(model, X_train, y_train, X_test, y_test, learning_rate=0.1, epochs=10000, batch_size=32, tolerance=1e-6, trials=3, label="r-squared", num_axes=1):
    momentum_values = np.arange(0, 1.1, 0.1)  # Momentum values from 0 to 1 in intervals of 0.1
    results = {}

    for momentum in momentum_values:
        # Initialize array to collect metrics as floats [mse, rmse, mae, r_squared, mean_percent_error]
        all_metrics = np.zeros(4, dtype=float)

        for i in range(trials):
            # Train the model with the current momentum value
            weights, losses, grad_norms = model.fit_stochastic(X_train, y_train, learning_rate=learning_rate, epochs=epochs, batch_size=batch_size, tolerance=tolerance, momentum=momentum)
            
            # Predict on the test set
            y_pred = model.predict(X_test, weights)
            
            # Collect accuracy metrics
            metrics = model.measure_performance(y_test['Diabetes_binary'], y_pred)

            # Ensure metrics is a numeric NumPy array
            metrics = np.array(metrics, dtype=float)  # Convert metrics to a float array
            
            # Sum the metrics to calculate the average later
            all_metrics += metrics

        # Average the metrics over multiple trials
        avg_metrics = all_metrics / trials
        results[momentum] = avg_metrics  # Store the averaged metrics for this momentum value

    # Convert the results to lists for plotting
    momentum_values = np.array(list(results.keys()))
    mse_values = np.array([metrics[0] for metrics in results.values()])
    rmse_values = np.array([metrics[1] for metrics in results.values()])
    mae_values = np.array([metrics[2] for metrics in results.values()])
    r_squared_values = np.array([metrics[3] for metrics in results.values()])

    if(num_axes==2):
        plot_metrics_with_secondary_y(x_vals=momentum_values,mse_values=mse_values,rmse_values=rmse_values, mae_values=mae_values,r_squared_values=r_squared_values, label="Momentum")
    elif(num_axes==1):
        plot_all_metrics_on_one_graph(x_vals=momentum_values,mse_values=mse_values,rmse_values=rmse_values, mae_values=mae_values,r_squared_values=r_squared_values)

def test_qr_vs_normal(model, X_train, y_train, X_test, y_test):
    weights_normal = model.fit(X_train, y_train)
    y_hat_normal = model.predict(X_test, weights_normal)
    metrics_normal = model.measure_performance(y_hat_normal, y_test)
    weights_QR = model.fit_QR(X_train, y_train)
    y_hat_QR = model.predict(X_test, weights_QR)
    metrics_QR= model.measure_performance(y_hat_QR, y_test)
    metrics_labels = ['MSE', 'RMSE', 'MAE', 'MPE','R-Squared']
    # Number of metrics
    n_metrics = len(metrics_labels)

    # Set positions for bars
    bar_width = 0.35
    index = np.arange(n_metrics)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot bars for Model 1 and Model 2
    bar1 = ax.bar(index, metrics_normal, bar_width, label='Normal')
    bar2 = ax.bar(index + bar_width, metrics_QR, bar_width, label='QR')

    # Add labels, title, and axes ticks
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Scores')
    ax.set_title('Comparison of QR vs Normal Least Squares')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(metrics_labels)

    # Add legend
    ax.legend()

    # Show plot
    plt.tight_layout()
    plt.show()

# %%
def test_lin_log_and_weights(): # compare analytical lin with fully batched log and plot weights for both models
    
    # linear regression test:
    X1, y1 = load_data(1)
    X1_final, y1 = clean_data(X1, y1)
    y1 = y1['aveOralM']
    lin_reg = LinearRegression()
    X1_train, y1_train, X1_test, y1_test = split_data(X1_final, y1, 0.8)
    w1 = lin_reg.fit(X1_train, y1_train)
    y1_hat_test = lin_reg.predict(X1_test, w1)
    performance_lin = lin_reg.measure_performance(y1_test,y1_hat_test)
    print('Performance metrics for linear regression:')
    print('MSE:'+str(performance_lin[0]))
    print('RMSE:'+str(performance_lin[1]))
    print('MAE:'+str(performance_lin[2]))
    print('MPE:'+str(performance_lin[3]))
    # plot lin reg feature weights
    feature_names = X1_train.columns.values
    plt.bar(feature_names, w1[1:])
    plt.xticks(rotation=90)
    plt.title('Feature weights for Linear Regression')
    plt.ylabel('Weight')
    plt.xlabel('Features')
    plt.show()
                                                    
    # logistic regression test:
    X2, y2 = load_data(2)
    X2_final, y2 = clean_data(X2, y2)
    log_reg = LogReg()
    X2_train, y2_train, X2_test, y2_test = split_data(X2_final, y2, 0.8)
    w2 = log_reg.fit(X2_train, y2_train)
    y2_hat_test = log_reg.predict(X2_test, w2)
    # print('Performance metrics for linear regression:')
    performance_log = log_reg.measure_performance(y2_test,y2_hat_test)
    # plot log reg feature weights
    feature_names_log = X2_train.columns.values
    w2 = w2['Diabetes_binary']
    plt.bar(X2_train.columns.values, w2[1:])
    plt.xticks(rotation=90)
    plt.title('Feature weights for Logistic Regression')
    plt.ylabel('Weight')
    plt.xlabel('Features')
    plt.show()

# %%
def test_data_split_ratios(model_name): # testing model performances over different percentage of training data
    if model_name == 'LinearRegression':
        X, y = load_data(1)
        X_final, y = clean_data(X, y)
        
        training_ratio = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        model = LinearRegression()
        
        train_metrics = []
        test_metrics = []
        
        # evaluate the performance of analytical linear regression
        for ratio in training_ratio:
            # fit and predict model
            X_train, y_train, X_test, y_test = split_data(X_final, y, ratio)
            w = model.fit(X_train, y_train)
            y_hat_train = model.predict(X_train, w)
            y_hat_test = model.predict(X_test, w)
            
            # get performance metrics for train and test set
            performance_train = model.measure_performace(y_train, y_hat_train)
            performance_test = model.measure_performance(y_test,y_hat_test)
            train_metrics.append(performance_train)
            test_metrics.append(performance_test)
        
        train_metrics = np.array(train_metrics)
        test_metrics = np.array(test_metrics)
        
        # plot model performance across the different training set ratios [mse, rmse, mae, mpe, r_squared]
        # training set 
        plt.plot(training_ratio, train_metrics[:,0], label='MSE')
        plt.plot(training_ratio, train_metrics[:,1], label='RMSE')
        plt.plot(training_ratio, train_metrics[:,2], label='MAE')
        plt.plot(training_ratio, train_metrics[:,3], label='MPE')
        plt.plot(training_ratio, train_metrics[:,4], label='R^2')
        plt.legend()
        plt.xlabel('Training data set ratio')
        plt.ylable('Performance metric values')
        plt.title('Performance of Linear Regression at Different Training Data Ratios (Train Set)')
        plt.show()
        # testing set
        plt.plot(training_ratio, test_metrics[:,0], label='MSE')
        plt.plot(training_ratio, test_metrics[:,1], label='RMSE')
        plt.plot(training_ratio, test_metrics[:,2], label='MAE')
        plt.plot(training_ratio, test_metrics[:,3], label='MPE')
        plt.plot(training_ratio, test_metrics[:,4], label='R^2')
        plt.legend()
        plt.xlabel('Training data set ratio')
        plt.ylable('Performance metric values')
        plt.title('Performance of Linear Regression at Different Training Data Ratios (Test Set)')
        plt.show()
        
    if model_name == 'LogisticRegression':
        Xd, yd = load_data(2)
        Xd, yd = clean_data(Xd, yd)
        model = LogReg()
        training_ratio = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        
        train_metrics = []
        test_metrics = []
        
        for ratio in training_ratio:
            Xd_train, yd_train, Xd_test, yd_test = split_data(Xd, yd, ratio)
            weights = model.fit(Xd_train, yd_train, max_iter=1e6, learning_rate=0.01, error=1e-4)
            y_hat_train = model.predict(Xd_train, weights, add_bias=True, convert_binary=True)
            y_hat_test = model.predict(Xd_test, weights, add_bias=True, convert_binary=True)
        
            # get performance metrics for train and test set
            performance_train = model.measure_performance(yd_train, y_hat_train)
            performance_test = model.measure_performance(yd_test, y_hat_test)
            train_metrics.append(performance_train)
            test_metrics.append(performance_test)
        
        train_metrics = np.array(train_metrics)
        test_metrics = np.array(test_metrics)
        
        # plot model performance across the different training set ratios
        # training set - [accuracy, precision, recall, f1]
        plt.plot(training_ratio, train_metrics[:,0], label='Accuracy')
        plt.plot(training_ratio, train_metrics[:,1], label='Precision')
        plt.plot(training_ratio, train_metrics[:,2], label='Recall')
        plt.plot(training_ratio, train_metrics[:,3], label='F1 score')
        plt.legend()
        plt.xlabel('Training data set ratio')
        plt.ylable('Performance metric values')
        plt.title('Performance of Logistic Regression at Different Training Data Ratios (Train Set)')
        plt.show()
        # testing set
        plt.plot(training_ratio, test_metrics[:,0], label='Accuracy')
        plt.plot(training_ratio, test_metrics[:,1], label='Precision')
        plt.plot(training_ratio, test_metrics[:,2], label='Recall')
        plt.plot(training_ratio, test_metrics[:,3], label='F1 score')
        plt.legend()
        plt.xlabel('Training data set ratio')
        plt.ylable('Performance metric values')
        plt.title('Performance of Logistic Regression at Different Training Data Ratios (Test Set)')
        plt.show()

# %%
def test_analytic_vs_sgd(model, X_train, y_train, X_test, y_test, batch_size=64):
    weights_normal = model.fit(X_train, y_train)
    y_hat_normal = model.predict(X_test, weights_normal)
    metrics_normal = model.measure_performance(y_test, y_hat_normal)
    weights_sgd, losses_sgd, iters_sgd = model.fit_stochastic(X_train, y_train)
    y_hat_sgd = model.predict(X_test, weights_sgd)
    metrics_sgd = model.measure_performance(y_test.to_numpy().flatten(), y_hat_sgd)
    metrics_labels = ['MSE', 'RMSE', 'MAE', 'MPE','R-Squared']
    # Number of metrics
    n_metrics = len(metrics_labels)

    # Set positions for bars
    bar_width = 0.35
    index = np.arange(n_metrics)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot bars for Model 1 and Model 2
    bar1 = ax.bar(index, metrics_normal, bar_width, label='Analytical')
    bar2 = ax.bar(index + bar_width, metrics_sgd, bar_width, label='SGD')

    # Add labels, title, and axes ticks
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Scores')
    ax.set_title('Comparison of Analytic vs Stochastic Gradient Descent')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(metrics_labels)

    # Add legend
    ax.legend()

    # Show plot
    plt.tight_layout()
    plt.show()

# %%
#dataset 1 test
regression = LinearRegression(True)
Xt, yt = load_data(1)
yt = yt.drop("aveOralF", axis=1)
Xt, yt = clean_data(Xt, yt)
Xt_train, yt_train, Xt_test, yt_test = split_data(Xt, yt, 0.8)
print(yt_test)
test_minibatch_sizes_linreg(regression, Xt_train, yt_train, Xt_test, yt_test, "aveOralM")
#%%
#I KNOW THIS ISN"T DOING ANYTHING BUT IF YOU REMOVE IT THEN THE SGD STOPS WORKING
data_num =1
X1,y1 = load_data(data_num) #X and Y are both dataframes

y1 = y1.drop('aveOralF', axis=1)

na_indices = X1[X1.isna().any(axis=1)].index.union(y1[y1.isna().any(axis=1)].index)

# Step 2: Drop those rows from both X and y
X1 = X1.drop(na_indices)
y1 = y1.drop(na_indices)


cat_columns = X1.select_dtypes(include=['object', 'category']).columns
X_norm= pd.get_dummies(X1, columns=cat_columns)
X= pd.get_dummies(X1, columns=cat_columns)
#Using One hot encoding here because I don't want to assume some kind of greater magnitude for certain categories
#If I used enumeration where White=1, Asian=2, etc... then this may lead to race being weighted in a way that is not accurate

#Scale CONTINUOUS COLUMNS
num_columns = X_norm.select_dtypes(include=['int64', 'float64']).columns
scaler = StandardScaler()
X_norm[num_columns] = scaler.fit_transform(X[num_columns])

X_train, X_test, y_train, y_test = split_data(X_norm,y1,0.8)

# %%
regression = LinearRegression(True)
test_qr_vs_normal(regression, Xt_train, yt_train, Xt_test, yt_test)
test_learning_rates(regression,Xt_train, yt_train, Xt_test, yt_test, 2)
test_momentum_values(regression,Xt_train, yt_train, Xt_test, yt_test, 2)
test_gradient_max(regression,Xt_train, yt_train, Xt_test, yt_test, 2)
test_analytic_vs_sgd(regression, Xt_train, yt_train, Xt_test, yt_test)
test_minibatch_sizes_linreg(regression, Xt_train, yt_train, Xt_test, yt_test, "aveOralM")

# %%

#dataset 2 test
logregression = LogReg(True)
Xd, yd = load_data(2)
Xd, yd = clean_data(Xd, yd)
Xd_train, yd_train, Xd_test, yd_test = split_data(Xd, yd, 0.8)
#%%
test_minibatch_sizes_logreg(logregression, Xd_train, yd_train, Xd_test, yd_test, "Diabetes_binary")
#%%
test_learning_rates_gd(logregression,Xd_train, yd_train, Xd_test, yd_test, data_label="Diabetes_binary", num_axes=1)
test_learning_rates_log_sdg(logregression,Xd_train, yd_train, Xd_test, yd_test, data_label="Diabetes_binary", num_axes=1)
#%%
test_gradient_max_log(logregression,Xd_train, yd_train, Xd_test, yd_test, num_axes=1)
# %%
test_momentum_values_log(logregression,Xd_train, yd_train, Xd_test, yd_test, num_axes=1)
# %%

# Grace's tests:
# plot feature weights and compare analytical lin and fully batched log:
test_lin_log_and_weights()
# test training dataset ratios and plot performances:
test_data_split_ratios('LinearRegression')
test_data_split_ratios('LogisticRegression')

################## Task 1 ##################

### Compute Basic Statistics of Datasets 
# dataset 1
# distribution of numerical features [histograms]
cont_temps_col = Xt.select_dtypes(include=['int64', 'float64']).columns
Xt[cont_temps_col].hist(bins=30, grid=False)
plt.suptitle('Histograms of Numerical Features for Temperatures Dataset')
plt.tight_layout()
plt.show()

# range of features
temps_ranges = Xt.max() - Xt.min()
temps_ranges.plot(kind='bar', title="Range of Features for Temperatures Dataset")
plt.xticks(fontsize=8)
plt.show()

# plot correlations using a bar chart
t_correlations = Xt.apply(lambda col: col.corr(yt["aveOralM"]))
t_correlations.plot(kind='bar', title='Correlation with Output for Temperatures Dataset', color='skyblue')
plt.ylabel('Correlation')
plt.xticks(fontsize=8)
plt.show()

# plot the output
yt["aveOralM"].plot()
plt.title("aveOralM Target Distribution")
plt.xlabel("Datapoints")
plt.ylabel("Value")
plt.show()

#%%
### compute basic statistics - dataset 2
# distribution of numerical features [histograms]
cont_diab_col = Xd.select_dtypes(include=['int64', 'float64']).columns
binary_diab_col = [col for col in cont_diab_col if Xd[col].nunique() == 2] # identify columns with binary data
non_binary_diab_col = [col for col in cont_diab_col if col not in binary_diab_col] # identify columns without binary data
Xd[non_binary_diab_col].hist(bins=20, grid=False)
plt.suptitle('Histograms of Numerical Features for Diabetes Dataset')
plt.tight_layout()
plt.show()

# range of features
diab_ranges = Xd.max() - Xd.min()
diab_ranges.plot(kind='bar', title="Range of Numerical Features for Diabetes Dataset")
plt.xticks(fontsize=8)
plt.show()

# plot correlations using a bar chart
d_correlations = Xd.apply(lambda col: col.corr(yd["Diabetes_binary"]))
d_correlations.plot(kind='bar', title='Correlation with Output for Diabetes Dataset', color='skyblue')
plt.ylabel('Correlation')
plt.xticks(fontsize=8)
plt.show()

# plot the output
yd["Diabetes_binary"].value_counts().plot(kind='bar', xlabel='Value', ylabel='Count')
plt.title("Diabetes Dataset Class Distribution")
plt.xticks(rotation = 0)
plt.show()



# %%
