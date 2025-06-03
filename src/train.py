#import wandb
from dataloader import load_splitted_data
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import argparse

CHECKPOINT_EVERY = 10
FOLDER = "/absolute/path/to/your/folder/containing/data/in/netCDF/files"  # Update this path to your data folder

def main(network):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device == "cpu":
        print("Warning: Training on CPU. Exiting.")
        return
    else:
        print(device)
        try:
            print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
            a = torch.randn(3,3).cuda()
            print(f"Successfully created a tensor on CUDA: {a}")
        except Exception as e:
            print(f"Error during basic CUDA operation: {e}")
            print("CUDA is not available or not working properly. Exiting.")
            return
            
    if network == "mini_unet":
        from models.mini_unet import UNet
    elif network == "medium_unet":
        from models.medium_unet import UNet
    elif network == "large_unet":
        from models.large_unet import UNet
    else:
        raise ValueError(f"Unknown network type: {network}. Supported types are: unet, mini_unet, medium_unet, large_unet.")
    
    # Training hyperparameters
    lr = 1e-4
    num_epochs = 100
    batch_size = 20
    
    variables = ["Zm_dBZ_tot", "V_dop_IQnoconv", "T_V", "T_H", "continuous_mask"] # All data
    #variables = ["Zm_dBZ_tot", "T_V", "T_H", "continuous_mask"] # No doppler data
    #variables = ["Zm_dBZ_tot", "V_dop_IQnoconv", "continuous_mask"] # No TB data
    #variables = ["V_dop_IQnoconv", "T_V", "T_H", "continuous_mask"] # No reflectivity data
    #variables = ["Zm_dBZ_tot", "continuous_mask"]  # Only reflectivity data, no TB data, no doppler data
    #variables = ["V_dop_IQnoconv", "continuous_mask"]  # Only doppler data, no TB data, no reflectivity data
    train_data, val_data = load_splitted_data(FOLDER, variables, ratio=0.9, shuffle=True)
    print(f"Training data shape: {train_data.shape}")
    print(f"Validation data shape: {val_data.shape}")

    # Assume train_data and val_data have shape (n_files, n_channels, x, y)
    # We use the first n_channels - 1 as input and the last channel as target.
    n_channels_total = train_data.shape[1]
    input_channels = n_channels_total - 1

    # Create PyTorch datasets and loaders
    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data.astype('float32')  # ensure float32 type
        
        def __len__(self):
            return self.data.shape[0]
        
        def __getitem__(self, idx):
            # The first "input_channels" are inputs and the last one is the target.
            x = self.data[idx, :input_channels, :, :]
            y = self.data[idx, -1:, :, :]
            
            # Determine the desired spatial shape from the first channel.
            desired_shape = x[0].shape
            # Loop over channels and if a channel is not 2D with the desired shape (e.g., TB_V, TB_H),
            # then tile its value to match the spatial dimensions.
            for i in range(x.shape[0]):
                if x[i].ndim < 2 or x[i].shape != desired_shape:
                    # If the channel contains a single value, extract it and create a full array.
                    scalar_value = x[i].item() if x[i].size == 1 else x[i]
                    x[i] = np.full(desired_shape, scalar_value)
            
            # Process inputs:
            # For channel 0 (e.g., reflectivity) replace values less than -25 with -25,
            # and replace NaNs with -25.
            x_clean = np.empty_like(x)
            for i in range(x.shape[0]):
                if i == 0:
                    x[i][x[i] < -25] = -25
                    x_clean[i] = np.nan_to_num(x[i], nan=-25.0)
                else:
                    x_clean[i] = np.nan_to_num(x[i], nan=0.0)

            eps = 1e-8

            # Convert to float64 for normalization to avoid precision issues
            x_clean = x_clean.astype(np.float64)

            # Normalize the first channel (e.g., reflectivity) using min-max scaling.
            x0_min, x0_max = x_clean[0].min(), x_clean[0].max()
            if abs(x0_max - x0_min) < eps:
                x_clean[0] = x_clean[0] - x0_min
            else:
                x_clean[0] = (x_clean[0] - x0_min) / (x0_max - x0_min + eps)

            # Normalize the other channels (including TB_V and TB_H) per sample.
            for i in range(1, x_clean.shape[0]):
                xi_min, xi_max = x_clean[i].min(), x_clean[i].max()
                if abs(xi_max - xi_min) < eps:
                    x_clean[i] = x_clean[i] - xi_min
                else:
                    x_clean[i] = (x_clean[i] - xi_min) / (xi_max - xi_min + eps)

            x_clean = x_clean.astype(np.float32)
            y_clean = np.nan_to_num(y, nan=0.0).astype(np.float32)

            return x_clean, y_clean

    train_dataset = CustomDataset(train_data)
    val_dataset = CustomDataset(val_data)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Instantiate the model, loss and optimizer.
    model = UNet(n_channels=input_channels, n_classes=1).to(device)
    pos_weight = torch.tensor(10.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)#, verbose=True)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device, dtype=torch.float32), targets.to(device, dtype=torch.float32)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device, dtype=torch.float32), targets.to(device, dtype=torch.float32)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} - Training Loss: {avg_train_loss:.6f} - Validation Loss: {avg_val_loss:.6f}")

        # Save checkpoint every CHECKPOINT_EVERY epochs
        if (epoch + 1) % CHECKPOINT_EVERY == 0:
            checkpoint_filename = f"../checkpoints/checkpoint_epoch_{epoch+1}_{network}_CM_1-3_1617GB.pth"
            torch.save(model.state_dict(), checkpoint_filename)
            print(f"Checkpoint saved: {checkpoint_filename}")

    # Save the final model checkpoint and log it
    torch.save(model.state_dict(), f"../checkpoints/checkpoint_epoch_{epoch+1}_{network}_CM_1-3_1617GB.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a UNet-based network")
    parser.add_argument(
        "-n", "--network",
        required=True,
        choices=["unet", "mini_unet", "medium_unet", "large_unet"],
        help="Type of network to train"
    )
    args = parser.parse_args()
    network = args.network
    
    main(network)
    
    