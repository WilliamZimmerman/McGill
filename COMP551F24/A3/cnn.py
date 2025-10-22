
# %%
import numpy as np
from medmnist import OrganAMNIST
import matplotlib.pyplot as plt
import ssl
import torch
import torch.backends
import torch.backends.mps
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision.transforms import ToTensor, Compose, Normalize, RandomHorizontalFlip, RandomRotation, RandomAffine, RandomApply, GaussianBlur, Lambda, ColorJitter, Resize
import torch.optim as optim
from torch.quantization import QuantStub, DeQuantStub, prepare, convert, get_default_qconfig
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import sys
import time
import psutil
import subprocess
import pynvml
import torchvision.models as models
from torchvision.models import ResNet18_Weights  # Import weights enum
#import cupy as cp
from torch.amp import autocast, GradScaler
from itertools import product
from sklearn.model_selection import KFold
from medmnist import INFO, Evaluator
from sklearn.metrics import accuracy_score
import torch.utils.data as data
from sklearn.model_selection import GridSearchCV
# load dataset information
import matplotlib.pyplot as plt
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV

#%%
class CNN(nn.Module):

    def __init__(self, num_filters=8, kernel_size=3, pool_size=2, in_channels=1, num_classes=11, batch_size=64, strides=2, padding='same', img_size=28, dropout_rate=0.6, lin_val =2304, *args, **kwargs ):
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.pool_size   = pool_size
        self.num_classes = num_classes
        self.batch_size  = batch_size
        self.strides     = strides
        self.padding     = padding 
        self.in_channels = in_channels
       
        super(CNN, self).__init__()
        
        if self.padding == 'same':
            pad = self.kernel_size // 2
        else:
            pad = self.padding
            
        conv1_size = ((img_size - kernel_size + 2 * pad) // strides) + 1
        pool1_size = conv1_size // 2
        conv2_size = ((pool1_size - kernel_size + 2 * pad) // strides) + 1
        pool2_size = conv2_size // 2
        
        flat_features = pool2_size * pool2_size * 64

        self.conv1 = nn.Sequential( #Each layer below happens one after another
            nn.Conv2d(in_channels=self.in_channels, out_channels=32, kernel_size=self.kernel_size, padding=self.padding), 
            #grayscale image, 32 feature maps, 3x3 filter, output image should be same size as input
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=kernel_size, stride=strides)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=self.kernel_size, padding=self.padding),
            #32 feature maps as input, 64 features maps as output, 3x3 kernel, output size= input size
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.flatten = nn.Flatten()
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc1 = nn.Sequential(
            nn.Linear(lin_val, 256), #2304 when 28 pixels 61504 when 128
            nn.ReLU(),
            nn.Dropout(dropout_rate)  # Add dropout between FC layers
        )
 
        self.fc2 = nn.Linear(256, self.num_classes)

    def forward(self, x):
        # Add print statements to debug
       
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

import torchvision.models as models
import torch.nn as nn

#%%
class PreTrainedModel(nn.Module):
    def __init__(self, num_classes=11):
        super(PreTrainedModel, self).__init__()
        
        # Load pre-trained ResNet18
        self.resnet = models.resnet18(weights=None)
        
        # Freeze all convolutional layers
        for param in self.resnet.parameters():
            param.requires_grad = False
            
        # Get the number of features from the last fully connected layer
        num_ftrs = self.resnet.fc.in_features
        
        # Replace the fully connected layer with Identity
        self.resnet.fc = nn.Identity()  # Removes the final FC layer
        
        # Add new fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(num_ftrs, 512),  # First FC layer
            nn.ReLU(),
            nn.Dropout(0.5),          # Dropout for regularization
            nn.MaxPool1d(kernel_size=3, stride=2),  # Pooling layer
            nn.Linear(255, 256),      # Second FC layer
            nn.ReLU(),
            nn.Dropout(0,5),          # Dropout for regularization
            nn.Linear(256, num_classes)  # Final classification layer
        )

    def forward(self, x):
        # Pass through the ResNet backbone
        with torch.no_grad():  # Ensure no gradient computation for the ResNet backbone
            features = self.resnet(x)
        
        # Pass through the fully connected layers
        out = self.fc_layers(features)
        return out
from torchvision.models import ResNet
from torchvision.models.resnet import BasicBlock

class PreTrainedModelMed(ResNet):
    def __init__(self, num_classes=11):
        super(PreTrainedModelMed, self).__init__(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=num_classes)
        
        # Modify the first convolutional layer to use a 3x3 kernel
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # Explicitly set downsample where needed
        for layer_name in ["layer2", "layer3", "layer4"]:
            layer = getattr(self, layer_name)
            for block in layer:
                if not hasattr(block, "downsample"):
                    block.downsample = None
        # Replace the final fully connected layer
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            torch.nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return super(PreTrainedModelMed, self).forward(x)

    
#%%
def train_transfer_learning(device):
    # Load dataset
    transform = transforms.Compose([
        transforms.Resize(128),  # ResNet expects 224x224
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x),  # Convert grayscale to RGB
        RandomRotation(15),
        RandomHorizontalFlip(),
        Lambda(lambda x: x + torch.randn_like(x) * 0.01),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    
    train_dataset =OrganAMNIST(split="train", size=128, as_rgb=True, download=True, transform=transform)
    test_dataset = OrganAMNIST(split="test", size=128, as_rgb=True, download=True, transform=transform)

    
    # Initialize model
    model = PreTrainedModel()
    model = model.to(device)
    
    # Only optimize the FC layers
    optimizer = torch.optim.Adam(model.fc_layers.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Train
   
    history = train(model, optimizer, train_dataset, test_dataset, num_epochs=10, device=device)
   
    
    return model, history

# Function to experiment with different FC architectures
def experiment_fc_architectures():
    architectures = [
        # Test different FC layer configurations
        [(512,), (256,), (11,)],  # 3 layers
        [(1024, 512), (256,), (11,)],  # 4 layers
        [(2048, 1024), (512, 256), (11,)],  # 5 layers
    ]
    
    results = []
    for arch in architectures:
        print(f"\nTesting architecture: {arch}")
        # Create model with this architecture
        model = PreTrainedModel(fc_architecture=arch)
        history = train_transfer_learning(model)
        results.append({
            'architecture': arch,
            'history': history
        })
    
    return results
    
# %%
def monitor_gpu():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # GPU 0
    try:
        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # convert to watts
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return {
            'power': power,
            'temperature': temp,
            'gpu_util': util.gpu,
            'memory_util': util.memory
        }
    except Exception as e:
        print(f"Error getting GPU stats: {e}")
        return None
# %%

def get_power_usage_mac():
    try:
        # Use powermetrics command to get power usage
        cmd = ["powermetrics", "-n", "1", "-i", "1000", "--samplers", "cpu_power,gpu_power", "--show-process-energy"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.stdout
    except Exception as e:
        print(f"Error getting power metrics: {e}")
        return None

def get_memory_usage_mac():
    try:
        # Get system memory info
        memory = psutil.virtual_memory()
        # Get process memory info
        process = psutil.Process()
        process_memory = process.memory_info()
        
        return {
            'system_total': memory.total / (1024 ** 3),  # GB
            'system_used': memory.used / (1024 ** 3),    # GB
            'system_free': memory.free / (1024 ** 3),    # GB
            'process_used': process_memory.rss / (1024 ** 3)  # GB
        }
    except Exception as e:
        print(f"Error getting memory metrics: {e}")
        return None

def load_pretrained_weights(model, pretrained_weights_path):
    # Load the checkpoint
    state_dict = torch.load(pretrained_weights_path, weights_only=True)['net']

    # Remap keys from 'shortcut' to 'downsample' and 'linear' to 'fc'
    mapped_state_dict = {}
    for key, value in state_dict.items():
        if "shortcut" in key:
            # Replace 'shortcut' with 'downsample'
            new_key = key.replace("shortcut", "downsample")
            mapped_state_dict[new_key] = value
        elif "linear" in key:
            new_key = key.replace("linear", "fc")
            mapped_state_dict[new_key] = value
        else:
            mapped_state_dict[key] = value

    # Load the remapped state_dict into the model
    missing_keys, unexpected_keys = model.load_state_dict(mapped_state_dict, strict=False)

    print(f"Missing keys: {missing_keys}")
    print(f"Unexpected keys: {unexpected_keys}")
    return model

def modify_model(model):
    # Freeze all layers except the fully connected layers
    for param in model.parameters():
        param.requires_grad = False

    # Add custom fully connected layers
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(256, 11)  # 11 classes for OrganAMNIST
    )
    return model
def prepare_datasets(batch_size=64):
    # Define dataset transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Load OrganAMNIST dataset
    train_dataset = OrganAMNIST(split="train", as_rgb=True,transform=transform, download=True)
    test_dataset = OrganAMNIST(split="test", as_rgb=True, transform=transform, download=True)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)

    return train_loader, test_loader

def train_mac(model, optimizer, train_dataset, test_dataset, num_epochs, device):
    train_loader = DataLoader(train_dataset, batch_size=32)
    history = {'loss': [], 'memory': [], 'power': []}
    criterion = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        # Start of epoch monitoring
        mem_start = get_memory_usage_mac()
        power_start = get_power_usage_mac()
        
        model.train()
        running_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx % 100 == 0:  # Monitor every 100 batches
                mem = get_memory_usage_mac()
                power = get_power_usage_mac()
                print(f"\nBatch {batch_idx}")
                print(f"Memory Usage: {mem}")
                print(f"Power Usage: {power}")
            
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # End of epoch monitoring
        mem_end = get_memory_usage_mac()
        power_end = get_power_usage_mac()
        
        avg_loss = running_loss / len(train_loader)
        history['loss'].append(avg_loss)
        history['memory'].append(mem_end)
        history['power'].append(power_end)
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Loss: {avg_loss:.4f}")
        print(f"Memory Usage: {mem_end}")
        print(f"Power Usage: {power_end}")
    
    return history

# %%
def train(model: CNN, optimizer, train_dataset, test_dataset, num_epochs, device, batch_size=128, debug=False):
    # Assuming train_mnist is your MNIST dataset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    # Define the loss function
    criterion = nn.CrossEntropyLoss()
    for images, labels in train_loader:
        print(f"Max label value: {labels.max()}")
        print(f"Min label value: {labels.min()}")
        print(f"Unique labels: {torch.unique(labels)}")
        break
    
   
    # Training loop
    history = {"loss": [], "accuracy": [], "val_accuracy": [], "power": [], "temp": [], "gpu_util": [], "memory_util": [], "time": [], "Inference_Time":[]}
    
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        print(f"epoch: {epoch}")
        start = time.time()
        for images, labels in tqdm(train_loader):
            if images.size(0) == 0 or labels.size(0) == 0:
                print("Skipping empty batch.")
                continue
            if len(labels.shape) == 0:  # Skip batches with scalar labels
                print(f"Skipping batch with scalar labels. Labels shape: {labels.shape}")
                continue
           
            # stats = monitor_gpu()
            # if stats:
            #     history['power'].append(stats['power'])
            #     history['temp'].append(stats['temperature'])
            #     history['gpu_util'].append(stats['gpu_util'])
            #     history['memory_util'].append(stats['memory_util'])
            
            
            if(debug):
                print("\nDEBUG INFO:")
                print(f"Images shape: {images.shape}")
                print(f"Labels shape before processing: {labels.shape}")
                print(f"Unique labels before processing: {torch.unique(labels)}")
            # Move tensors to device
            images = images.to(device)
            # Squeeze the labels from [512, 1] to [512]
            labels = labels.squeeze().long().to(device) 
            if(debug):
                print(f"Labels shape after processing: {labels.shape}")
                print(f"Unique labels after processing: {torch.unique(labels)}")
            # Zero the gradients
            optimizer.zero_grad()
    
            # Forward pass
            outputs = model(images)
            if(debug):
                print(f"Outputs shape: {outputs.shape}")
                print(f"Output min/max: {outputs.min().item():.2f}/{outputs.max().item():.2f}")
                print(f"Label min/max: {labels.min().item()}/{labels.max().item()}")
            # Calculate loss
            
            try:
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            except Exception as e:
                print(f"Error in loss calculation:")
                print(f"Outputs shape: {outputs.shape}, dtype: {outputs.dtype}")
                print(f"Labels shape: {labels.shape}, dtype: {labels.dtype}")
                raise e
        end= time.time()
        inf_start = time.time()

        # Print the average loss for the epoch
        loss_print = running_loss/len(train_loader)
        acc_print = compute_accuracy(model, train_loader, device)
        val_acc_print = compute_accuracy(model, test_loader, device)
        inf_end = time.time()

        history["loss"].append(running_loss/len(train_loader))
        history["accuracy"].append(acc_print)
        history["val_accuracy"].append(val_acc_print)
        history["time"].append(end-start)
        history["Inference_Time"].append(inf_end-inf_start)

        print(f"loss: {running_loss/len(train_loader)}")
        print(f"accuracy: {acc_print}")
        print(f"Val Accuracy: {val_acc_print}")
        #NEED TO INTEGRATE POWER AND THEN NEED TO TAKE AVG UTIL/TEMP

    print('Training complete!')
    return history
# %%
def train_mixed_precision(model, optimizer, train_dataset, test_dataset, num_epochs, device, debug=False):
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, drop_last=True)

    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler("cuda")  # Gradient scaler for mixed precision

    history = {"loss": [], "accuracy": [], "val_accuracy": [], "power": [], "temp": [], "gpu_util": [], "memory_util": [], "time": [], "Inference_Time":[]}

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        print(f"Epoch {epoch + 1}/{num_epochs}")

        start = time.time()

        for images, labels in tqdm(train_loader):
            # Move data to device
            images = images.to(device)
            labels = labels.squeeze().long().to(device)

            # Optional GPU monitoring
            stats = monitor_gpu() if "monitor_gpu" in globals() else None
            if stats:
                history['power'].append(stats['power'])
                history['temp'].append(stats['temperature'])
                history['gpu_util'].append(stats['gpu_util'])
                history['memory_util'].append(stats['memory_util'])

            optimizer.zero_grad()

            # Mixed precision forward pass
            with autocast("cuda"):  # Enables mixed precision
                outputs = model(images)
                loss = criterion(outputs, labels)

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()  # Update the scale factor for the next iteration

            running_loss += loss.item()

        end = time.time()
        inf_start = time.time()
        # Compute metrics
        avg_loss = running_loss / len(train_loader)
        train_acc = compute_accuracy(model, train_loader, device)  # Use FP32 for evaluation
        val_acc = compute_accuracy(model, test_loader, device)  # Use FP32 for evaluation
        inf_end = time.time()
        # Log metrics
        history["loss"].append(avg_loss)
        history["accuracy"].append(train_acc)
        history["val_accuracy"].append(val_acc)
        history["time"].append(end - start)
        history["Inference_Time"].append(inf_end-inf_start)
        print(f"Loss: {avg_loss:.4f}, Train Accuracy: {train_acc:.2f}%, Val Accuracy: {val_acc:.2f}%")

    print("Training complete!")
    return history
# %%
def train_eff_model(model, train_loader, test_loader, num_epochs, device):
    # Define optimizer for only the fully connected layers
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    history = {"loss": [], "accuracy": [], "val_accuracy": [], "power": [], "temp": [], "gpu_util": [], "memory_util": [], "time": [], "Inference_Time":[]}
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        start = time.time()
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.squeeze().long().to(device)
            stats = monitor_gpu() if "monitor_gpu" in globals() else None
            if stats:
                history['power'].append(stats['power'])
                history['temp'].append(stats['temperature'])
                history['gpu_util'].append(stats['gpu_util'])
                history['memory_util'].append(stats['memory_util'])
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        end = time.time()
        train_acc = 100. * correct / total
        val_acc = compute_accuracy(model, test_loader, device)
        
        history["loss"].append(running_loss / len(train_loader))
        history["accuracy"].append(train_acc)
        history["val_accuracy"].append(val_acc)
        history["time"].append(end - start)
       

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss:.4f}, Train Accuracy: {train_acc:.2f}%, Val Accuracy: {val_acc:.2f}%")
    plot_training_history(history)
    plot_gpu_metrics(history)
    return history

def compute_accuracy(model, data_loader, device):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient calculation for evaluation
        for images, labels in data_loader:
            images, labels = images.to(device), labels.squeeze().to(device)

            # Forward pass
            outputs = model(images)  # Get logits
            _, predicted = torch.max(outputs.data, 1)  # Get predicted class indices

            total += labels.size(0)  # Total number of labels
            correct += (predicted == labels).sum().item()  # Count correct predictions
    print(correct)
    print(total)
    accuracy = correct / total * 100  # Convert to percentage
    return accuracy

def plot_training_history(history):

    # Plot training loss
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'], label='Train')
    plt.plot(history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.show()
#%%
def plot_gpu_metrics(history):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    axes[0,0].plot(history['power']*history['time']/len(history['time']))
    axes[0,0].set_title('Energy Usage Per Epoch (J)')
    
    axes[0,1].plot(history['temp'])
    axes[0,1].set_title('Temperature (°C)')
    
    axes[1,0].plot(history['gpu_util'])
    axes[1,0].set_title('GPU Utilization (%)')
    
    axes[1,1].plot(history['memory_util'])
    axes[1,1].set_title('Memory Utilization (%)')
    print(f"TOTAL POWER {sum(history['power'])} ")
    print(f"Average GPU Util {np.mean(history['gpu_util'])}")
    print(f"Average Memory Util {np.mean(history['memory_util'])}")
    plt.tight_layout()
    plt.show()

# %%
def CNN_task1(batch_size):
    #LOAD DATA
    print("CNN TASK 1 STARTING NOW")
    transform_configs = {
    'basic': Compose([
        ToTensor(),
        Normalize(0.5, 1)
    ]),
    
    'rotation_only': Compose([
        ToTensor(),
        RandomRotation(5),
       Normalize(0.5, 1)
    ]),
    
    'noise_only': Compose([
        ToTensor(),
        Lambda(lambda x: x + torch.randn_like(x) * 0.01),
        Normalize(0.5, 1)
    ]),
    
    'affine_only': Compose([
        ToTensor(),
        RandomAffine(degrees=5, translate=(0.05, 0.05)),
        Normalize(0.5, 0.5)
    ]), 
    'affine_and_noise': 
    Compose([
        ToTensor(),
        Lambda(lambda x: x + torch.randn_like(x) * 0.01),
        RandomAffine(degrees=5, translate=(0.05, 0.05)),
        Normalize(0.5, 1)
    ]), 
    'affine_and_rotation': Compose([
        ToTensor(),
        RandomRotation(5),
       RandomAffine(degrees=5, translate=(0.05, 0.05)),
       Normalize(0.5, 1)
    ]),
    'rotation_and_noise': Compose([
        ToTensor(),
        RandomRotation(5),
        Lambda(lambda x: x + torch.randn_like(x) * 0.01),
        Normalize(0.5, 0.5)
    ])}
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    ssl._create_default_https_context = ssl._create_unverified_context
    transforms = transform_configs["rotation_and_noise"]
    train_dataset =OrganAMNIST(split="train", download=True, transform=transforms)
    test_dataset = OrganAMNIST(split="test", download=True, transform=transforms)

    print(f"{len(train_dataset)=}")
    print(f"{len(test_dataset)=}")

    #SET UP DEVICES
    if torch.backends.mps.is_available():
        print("FOUND MPS")
        device = torch.device("mps")  # Apple M-series chip GPU
    elif torch.cuda.is_available():
        device = torch.device("cuda") # Should be able to use 3090
        print(f"Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("FOUND CPU")
        device = torch.device("cpu")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Selected device: {device}")
    print(f"Python Version: {sys.version}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
    print(f"MPS Available: {torch.backends.mps.is_available()}")
    
    
    #SET UP MODEL
    model = CNN()
    model.to(device)
    print(f"\nModel is on device: {next(model.parameters()).device}")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10  # or whatever number you choose
    
    # 4. Train the model
   
    history = train(model, optimizer, train_dataset, test_dataset, num_epochs, device, batch_size)
   
    print(f"TIME ELAPSED: {sum(history['time'])} Seconds")
    print(f"Inference Time Elapsed: {sum(history['Inference_Time'])} Seconds")
    plot_training_history(history)
    plot_gpu_metrics(history)

# %%
def CNN_task2():
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    ssl._create_default_https_context = ssl._create_unverified_context
    transform_configs = {
    'basic': Compose([
        ToTensor(),
        Normalize(0.5, 1)
    ]),
    
    'rotation_only': Compose([
        ToTensor(),
        RandomRotation(5),
       Normalize(0.5, 1)
    ]),
    
    'noise_only': Compose([
        ToTensor(),
        Lambda(lambda x: x + torch.randn_like(x) * 0.01),
        Normalize(0.5, 1)
    ]),
    
    'affine_only': Compose([
        ToTensor(),
        RandomAffine(degrees=5, translate=(0.05, 0.05)),
        Normalize(0.5, 0.5)
    ]), 
    'affine_and_noise': 
    Compose([
        ToTensor(),
        Lambda(lambda x: x + torch.randn_like(x) * 0.01),
        RandomAffine(degrees=5, translate=(0.05, 0.05)),
        Normalize(0.5, 1)
    ]), 
    'affine_and_rotation': Compose([
        ToTensor(),
        RandomRotation(5),
       RandomAffine(degrees=5, translate=(0.05, 0.05)),
       Normalize(0.5, 1)
    ]),
    'rotation_and_noise': Compose([
        ToTensor(),
        RandomRotation(5),
        Lambda(lambda x: x + torch.randn_like(x) * 0.01),
        Normalize(0.5, 0.5)
    ])
    }
    transforms = Compose([ToTensor(), Normalize(0.5, 1)])
    train_dataset =OrganAMNIST(split="train", size=128, download=True, transform=transforms)
    test_dataset = OrganAMNIST(split="test", size=128, download=True, transform=transforms)

    print(f"{len(train_dataset)=}")
    print(f"{len(test_dataset)=}")
    img, label = train_dataset[0]
    
    #SET UP DEVICES
    if torch.backends.mps.is_available():
        print("FOUND MPS")
        device = torch.device("mps")  # Apple M-series chip GPU
    elif torch.cuda.is_available():
        device = torch.device("cuda") # Should be able to use 3090
        print(f"Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("FOUND CPU")
        device = torch.device("cpu")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Selected device: {device}")
    print(f"Python Version: {sys.version}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
    print(f"MPS Available: {torch.backends.mps.is_available()}")
    
    
    #SET UP MODEL
    model = CNN(img_size=128, lin_val=61504)
    model.to(device)
    print(f"\nModel is on device: {next(model.parameters()).device}")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10  # or whatever number you choose
    
    history = train(model, optimizer, train_dataset, test_dataset, num_epochs, device)
   
    print(f"TIME ELAPSED: {sum(history['time'])} Seconds")
    print(f"Inference Time Elapsed: {sum(history['Inference_Time'])} Seconds")
    plot_training_history(history)
    plot_gpu_metrics(history)
#%%
def CNN_task3():
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    ssl._create_default_https_context = ssl._create_unverified_context
    transform_configs = {
    'basic': Compose([
        ToTensor(),
        Normalize(0.5, 1)
    ]),
    
    'rotation_only': Compose([
        ToTensor(),
        RandomRotation(5),
       Normalize(0.5, 1)
    ]),
    
    'noise_only': Compose([
        ToTensor(),
        Lambda(lambda x: x + torch.randn_like(x) * 0.01),
        Normalize(0.5, 1)
    ]),
    
    'affine_only': Compose([
        ToTensor(),
        RandomAffine(degrees=5, translate=(0.05, 0.05)),
        Normalize(0.5, 0.5)
    ]), 
    'affine_and_noise': 
    Compose([
        ToTensor(),
        Lambda(lambda x: x + torch.randn_like(x) * 0.01),
        RandomAffine(degrees=5, translate=(0.05, 0.05)),
        Normalize(0.5, 1)
    ]), 
    'affine_and_rotation': Compose([
        ToTensor(),
        RandomRotation(5),
       RandomAffine(degrees=5, translate=(0.05, 0.05)),
       Normalize(0.5, 1)
    ]),
    'rotation_and_noise': Compose([
        ToTensor(),
        RandomRotation(5),
        Lambda(lambda x: x + torch.randn_like(x) * 0.01),
        Resize((224, 224)),
        Normalize(0.5, 0.5)
    ])
    }
    transforms = transform_configs["rotation_and_noise"]
    train_dataset =OrganAMNIST(split="train", size=128, as_rgb=True, download=True, transform=transforms)
    test_dataset = OrganAMNIST(split="test", size=128, as_rgb=True, download=True, transform=transforms)

    print(f"{len(train_dataset)=}")
    print(f"{len(test_dataset)=}")
    
    #SET UP DEVICES
    if torch.backends.mps.is_available():
        print("FOUND MPS")
        device = torch.device("mps")  # Apple M-series chip GPU
    elif torch.cuda.is_available():
        device = torch.device("cuda") # Should be able to use 3090
        print(f"Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("FOUND CPU")
        device = torch.device("cpu")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Selected device: {device}")
    print(f"Python Version: {sys.version}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
    print(f"MPS Available: {torch.backends.mps.is_available()}")

    model, history = train_transfer_learning(device)
    print(f"TIME ELAPSED: {sum(history['time'])} Seconds")
    print(f"Inference Time Elapsed: {sum(history['Inference_Time'])} Seconds")
    plot_training_history(history)
    plot_gpu_metrics(history)
# %%
def CNN_task3_med():
    if torch.backends.mps.is_available():
        print("FOUND MPS")
        device = torch.device("mps")  # Apple M-series chip GPU
    elif torch.cuda.is_available():
        device = torch.device("cuda") # Should be able to use 3090
        print(f"Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("FOUND CPU")
        device = torch.device("cpu")
    
    # I# Step 1: Create the custom ResNet18 model
    model = PreTrainedModelMed()
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001, weight_decay=1e-4)

    print(f"\nModel is on device: {next(model.parameters()).device}")
    # Step 2: Load pretrained weights
    pretrained_weights_path = "/Users/williamzimmerman/Desktop/McGill/Fall2024/COMP551/comp551-f242/weights_organamnist/resnet18_28_2.pth"
    model = load_pretrained_weights(model, pretrained_weights_path)

    # Step 3: Modify the model
    model = modify_model(model)

    # Step 4: Prepare datasets
    train_loader, test_loader = prepare_datasets(batch_size=64)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001, weight_decay=1e-4)
    # Step 5: Train the model
    history = train_med_model(model, device, train_loader, test_loader, optimizer,num_epochs=10)

    plot_training_history(history)





# Training loop
def train_med_model(model, device,train_loader, test_loader, optimizer, num_epochs=10):
    model.train()
    history = {'train_loss': [], 'val_acc': [], 'time': []}
    
    criterion = nn.CrossEntropyLoss()
   
    
    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.squeeze().long().to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
           
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        elapsed_time = time.time() - start_time
        val_acc = compute_accuracy(model, test_loader, device)
        train_acc = compute_accuracy(model ,train_loader, device )
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['time'].append(elapsed_time)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}% - Val Acc: {val_acc:.2f}% - Time: {elapsed_time:.2f}s")
    
    return history
# %%
def CNN_task3Eff():

    model = models.efficientnet_b0(pretrained=True)

    # Freeze all convolutional layers
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace the classifier with custom fully connected layers
    num_classes = 11  # Replace with the number of classes in your dataset
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier[1].in_features, 512),  # Add a fully connected layer
        nn.ReLU(),
        nn.Dropout(0.5),  # Dropout for regularization
        nn.Linear(512, num_classes)  # Output layer for classification
    )

    # Move model to device
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    model = model.to(device)
    # Transformations for EfficientNet
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # EfficientNet requires 224x224 input
        transforms.ToTensor(),
        RandomRotation(5),
        Lambda(lambda x: x + torch.randn_like(x) * 0.01),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize grayscale data
    ])

    # Load OrganAMNIST dataset
    from medmnist import OrganAMNIST

    train_dataset = OrganAMNIST(split="train", as_rgb=True, transform=transform, download=True)
    test_dataset = OrganAMNIST(split="test", as_rgb=True, transform=transform, download=True)


    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=True)

    history = train_eff_model(model, train_loader, test_loader, 10, device)
    print(f"TIME ELAPSED: {sum(history['time'])} Seconds")
    print(f"Inference Time Elapsed: {sum(history['Inference_Time'])} Seconds")

    plot_training_history(history)
    plot_gpu_metrics(history)

#%%
def mixed_precision_testing():
    print("CNN Mixed Precision Testing")

    # Define transforms
    transform_configs = {
    'basic': Compose([
        ToTensor(),
        Normalize(0.5, 1)
    ]),
    
    'rotation_only': Compose([
        ToTensor(),
        RandomRotation(5),
       Normalize(0.5, 1)
    ]),
    
    'noise_only': Compose([
        ToTensor(),
        Lambda(lambda x: x + torch.randn_like(x) * 0.01),
        Normalize(0.5, 1)
    ]),
    
    'affine_only': Compose([
        ToTensor(),
        RandomAffine(degrees=5, translate=(0.05, 0.05)),
        Normalize(0.5, 0.5)
    ]), 
    'affine_and_noise': 
    Compose([
        ToTensor(),
        Lambda(lambda x: x + torch.randn_like(x) * 0.01),
        RandomAffine(degrees=5, translate=(0.05, 0.05)),
        Normalize(0.5, 1)
    ]), 
    'affine_and_rotation': Compose([
        ToTensor(),
        RandomRotation(5),
       RandomAffine(degrees=5, translate=(0.05, 0.05)),
       Normalize(0.5, 1)
    ]),
    'rotation_and_noise': Compose([
        ToTensor(),
        RandomRotation(5),
        Lambda(lambda x: x + torch.randn_like(x) * 0.01),
        Normalize(0.5, 0.5)
    ])
    }
    transforms = transform_configs["rotation_and_noise"]

    # Dataset setup
    train_dataset = OrganAMNIST(split="train", download=True, transform=transforms)
    test_dataset = OrganAMNIST(split="test", download=True, transform=transforms)
    print(f"{len(train_dataset)=}")
    print(f"{len(test_dataset)=}")

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"CUDA Version: {torch.version.cuda}")

    # Model setup
    model = CNN()
    model.to(device)
    print(f"\nModel is on device: {next(model.parameters()).device}")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10

    # Train the model
    history = train_mixed_precision(model, optimizer, train_dataset, test_dataset, num_epochs, device)
    print(f"TIME ELAPSED: {sum(history['time'])} Seconds")
    print(f"Inference Time Elapsed: {sum(history['Inference_Time'])} Seconds")
    # Plot metrics
    plot_training_history(history)
    plot_gpu_metrics(history)

# # #%%
CNN_task1(256)
# # #%%
# mixed_precision_testing()
# # # %%
CNN_task1(128)
# # #%%
CNN_task2()
# #%%
CNN_task3()
# %%
CNN_task3Eff() # ONLY RUN THIS ON CUDA SYSTEM!!!
if __name__ == "__main__":
     CNN_task3_med() #SLOW!
