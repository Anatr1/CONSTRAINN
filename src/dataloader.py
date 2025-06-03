from netCDF4 import Dataset
import os
import numpy as np

def load_raw_data(folder, channels):
    """
    Load data from multiple NetCDF files in the specified folder.
    For each variable in the channels list, a list of 2D matrices (numpy arrays) is returned.
    
    :param folder: Path to the folder containing the NetCDF files.
    :param channels: List of variable names (channels) to load from each file.
    :return: A dictionary where each key is a variable name and the value is a list of 2D numpy arrays.
    """
    data = {dim: [] for dim in channels}

    # Find all .nc files in the folder
    nc_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".nc")]

    for filename in nc_files:
        if os.path.exists(filename):
            try:
                with Dataset(filename, mode='r') as ds:
                    for var_name in channels:
                        if var_name not in ds.variables:
                            print(f"Variable '{var_name}' not found in file {filename}")
                            continue
                        var = ds.variables[var_name]
                        data[var_name].append(var[:])
            except OSError as e:
                print(f"Error opening file {filename}: {str(e)}")
        else:
            print(f"File not found: {filename}")
    
    return data

def load_same_shape_data(folder, channels):
    """
    Load data from multiple NetCDF files in the specified folder, converting every variable to a uniform 2D shape.
    For each file, the first available 2D channel is used as a reference for the spatial dimensions.
    Monodimensional or singleton channels are tiled to match the reference shape.
    
    :param folder: Path to the folder containing the NetCDF files.
    :param channels: List of variable names (channels) to load from each file.
    :return: A numpy array of shape (n_files, len(channels), H, W) where H and W are the dimensions of the reference 2D arrays.
    """
    data = []

    # Find all .nc files in the folder
    nc_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".nc")]

    for filename in nc_files:
        if os.path.exists(filename):
            #print(f"Loading file n° {len(data) + 1} of {len(nc_files)}")
            try:
                with Dataset(filename, mode='r') as ds:
                    file_data = []
                    for var_name in channels:
                        if var_name not in ds.variables:
                            print(f"Variable '{var_name}' not found in file {filename}")
                            continue
                        var = ds.variables[var_name]
                        file_data.append(var[:])
                    
                    # Only proceed if we got data for all channels
                    if len(file_data) == len(channels):
                        # Find a reference shape from the first available 2D array.
                        ref = None
                        for arr in file_data:
                            if arr.ndim == 2:
                                ref = arr
                                break
                        if ref is None:
                            print(f"No 2D channel found in file {filename}. Skipping file.")
                            continue
                        ref_shape = ref.shape

                        new_file_data = []
                        for arr in file_data:
                            # Already 2D and shape matches
                            if arr.ndim == 2 and arr.shape == ref_shape:
                                new_file_data.append(arr)
                            # Handle 2D arrays with a singleton dimension:
                            elif arr.ndim == 2 and arr.shape[0] == ref_shape[0] and arr.shape[1] == 1:
                                new_arr = np.tile(arr, (1, ref_shape[1]))
                                new_file_data.append(new_arr)
                            elif arr.ndim == 2 and arr.shape[1] == ref_shape[1] and arr.shape[0] == 1:
                                new_arr = np.tile(arr, (ref_shape[0], 1))
                                new_file_data.append(new_arr)
                            # Handle 1D arrays
                            elif arr.ndim == 1:
                                if arr.shape[0] == ref_shape[0]:
                                    new_arr = np.tile(arr.reshape(ref_shape[0], 1), (1, ref_shape[1]))
                                    new_file_data.append(new_arr)
                                elif arr.shape[0] == ref_shape[1]:
                                    new_arr = np.tile(arr.reshape(1, ref_shape[1]), (ref_shape[0], 1))
                                    new_file_data.append(new_arr)
                                else:
                                    print(f"Warning: Unexpected 1D shape {arr.shape} for variable in file {filename}. Using first element as constant.")
                                    new_arr = np.full(ref_shape, arr[0])
                                    new_file_data.append(new_arr)
                            else:
                                # If the array is a scalar, fill the array with its value.
                                if arr.size == 1:
                                    new_file_data.append(np.full(ref_shape, arr.item()))
                                else:
                                    try:
                                        new_arr = np.array(arr).reshape(ref_shape)
                                        new_file_data.append(new_arr)
                                    except Exception as e:
                                        print(f"Error reshaping array with shape {arr.shape} to {ref_shape} for file {filename}: {e}")
                                        new_file_data.append(arr)
                        # Append only if new_file_data contains data for all channels.
                        if len(new_file_data) == len(channels):
                            data.append(new_file_data)
                        else:
                            print(f"File {filename} skipped due to shape mismatch after processing.")
            except OSError as e:
                print(f"Error opening file {filename}: {str(e)}")
        else:
            print(f"File not found: {filename}")

    if data:
        return np.array(data)
    else:
        return np.empty((0, 0, 0, 0))

def split_data(data, ratio=0.9):
    """
    Split the data into training and validation sets.
    
    :param data: A numpy array of shape (n_files, len(channels), H, W).
    :param ratio: Proportion of the data to use for training.
    :return: Two numpy arrays representing the training and validation sets.
    """
    n = int(data.shape[0] * ratio)
    return data[:n], data[n:]

def load_splitted_data(folder, channels, ratio=0.9, shuffle=False):
    """
    Load data from multiple NetCDF files in the specified folder and split it into training and validation sets.
    
    :param folder: Path to the folder containing the NetCDF files.
    :param channels: List of variable names (channels) to load from each file.
    :param ratio: Proportion of the data to use for training.
    :param shuffle: Whether to shuffle the data before splitting.
    :return: Two numpy arrays representing the training and validation sets.
    """
    #Seed the random number generator for reproducibility
    np.random.seed(42)
    data = load_same_shape_data(folder, channels)
    
    if shuffle:
        np.random.shuffle(data)
    
    train, val = split_data(data, ratio)
        
    return train, val


def load_validation_data_only(folder, channels, train_ratio=0.9, seed=42):
    """
    Loads only the validation data segment that would result from load_splitted_data
    when shuffle=True. It identifies the files that would form the validation set
    after shuffling and loads only them.

    :param folder: Path to the folder containing the NetCDF files.
    :param channels: List of variable names (channels) to load from each file.
    :param train_ratio: Proportion of the data that would be used for training (e.g., 0.9).
                        The validation set is the remaining (1 - train_ratio).
    :param seed: The random seed used for shuffling file list, to ensure consistency.
                 This MUST match the seed used in load_splitted_data if shuffle=True.
    :return: A numpy array representing the validation set, or an empty array if no validation files.
    """
    # 1. Get a sorted list of all .nc file paths. Sorting provides a canonical order before shuffling.
    all_nc_filepaths = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".nc")])

    if not all_nc_filepaths:
        print("No .nc files found in the folder for validation loading.")
        return np.empty((0, 0, 0, 0))

    # 2. Shuffle this list of file paths using the given seed.
    # This mimics the np.random.shuffle(data) step in load_splitted_data on the file level.
    np.random.seed(seed)
    shuffled_filepaths = list(all_nc_filepaths) # Make a mutable copy
    np.random.shuffle(shuffled_filepaths) # Shuffles in place

    # 3. Determine the slice of file paths corresponding to the validation set.
    num_total_files = len(shuffled_filepaths)
    split_idx = int(num_total_files * train_ratio)
    
    # These are the paths of files that should constitute the validation set
    validation_files_to_attempt_load = shuffled_filepaths[split_idx:]

    if not validation_files_to_attempt_load:
        print("No files allocated for validation set based on the ratio and total number of files.")
        return np.empty((0, 0, 0, 0))
        
    print(f"Identified {len(validation_files_to_attempt_load)} files to attempt loading for the validation set.")
    
    # 4. Load data *only* for these selected file paths, applying the same processing
    #    logic as in load_same_shape_data for each file.
    processed_validation_data_list = [] 

    for filename in validation_files_to_attempt_load:
        if os.path.exists(filename):
            try:
                with Dataset(filename, mode='r') as ds:
                    current_file_raw_channels = []
                    all_channels_present = True
                    for var_name in channels:
                        if var_name not in ds.variables:
                            all_channels_present = False
                            print(f"Debug: Channel {var_name} not found in {filename}")
                            break
                        var = ds.variables[var_name]
                        current_file_raw_channels.append(var[:])
                    
                    if not all_channels_present or len(current_file_raw_channels) != len(channels):
                        print(f"Skipping file {filename} (validation) due to missing channels or length mismatch.")
                        continue

                    # Find a reference shape from the first available 2D array.
                    # --- Improved Reference Shape Determination ---
                    ref_array = None
                    ref_shape = None
                    # Create a dictionary of raw channels for easier lookup by name
                    raw_channels_dict = dict(zip(channels, current_file_raw_channels))

                    # Prioritize 'continuous_mask' if available and 2D, as it's often the target shape.
                    if 'continuous_mask' in channels and 'continuous_mask' in raw_channels_dict and raw_channels_dict['continuous_mask'].ndim == 2:
                        ref_array = raw_channels_dict['continuous_mask']
                        ref_shape = ref_array.shape
                    else:
                        # Fallback: find the first 2D array from the loaded channels in the specified order
                        for idx, var_name_in_loop in enumerate(channels): # Use a different loop variable name
                            arr_val_in_loop = current_file_raw_channels[idx] # Use the different loop variable name
                            if arr_val_in_loop.ndim == 2:
                                ref_array = arr_val_in_loop
                                ref_shape = ref_array.shape
                                break 
                    
                    if ref_array is None or ref_shape is None:
                        print(f"Skipping file {filename} (validation): No suitable 2D reference channel found to determine shape.")
                        continue
                    # --- End of Improved Reference Shape Determination ---

                    conformed_channels_for_file = []
                    possible_to_conform_all = True
                    for arr_idx, arr_val in enumerate(current_file_raw_channels): # Iterate through raw channels
                        conformed_arr = None
                        current_channel_name = channels[arr_idx] # For debugging
                        if arr_val.ndim == 2 and arr_val.shape == ref_shape:
                            conformed_arr = arr_val
                        elif arr_val.ndim == 2 and arr_val.shape[0] == ref_shape[0] and arr_val.shape[1] == 1:
                            conformed_arr = np.tile(arr_val, (1, ref_shape[1]))
                        elif arr_val.ndim == 2 and arr_val.shape[1] == ref_shape[1] and arr_val.shape[0] == 1:
                            conformed_arr = np.tile(arr_val, (ref_shape[0], 1))
                        elif arr_val.ndim == 1:
                            if arr_val.shape[0] == ref_shape[0]:
                                conformed_arr = np.tile(arr_val.reshape(ref_shape[0], 1), (1, ref_shape[1]))
                            elif arr_val.shape[0] == ref_shape[1]:
                                conformed_arr = np.tile(arr_val.reshape(1, ref_shape[1]), (ref_shape[0], 1))
                            else:
                                print(f"Warning: Unmatched 1D shape {arr_val.shape} for channel '{current_channel_name}' vs ref {ref_shape} in {filename} (validation). Using first element if available.")
                                conformed_arr = np.full(ref_shape, arr_val[0] if arr_val.size > 0 else np.nan)
                        elif arr_val.size == 1:
                            conformed_arr = np.full(ref_shape, arr_val.item())
                        else: 
                            try:
                                # This reshape is risky if total elements don't match.
                                # It should only be attempted if arr_val.size == ref_array.size
                                if arr_val.size == ref_array.size:
                                    conformed_arr = np.array(arr_val).reshape(ref_shape)
                                else:
                                    print(f"Warning: Cannot reshape channel '{current_channel_name}' (shape {arr_val.shape}, size {arr_val.size}) to {ref_shape} (size {ref_array.size}) for file {filename} (validation). Element count mismatch.")
                                    possible_to_conform_all = False
                                    break
                            except Exception as e_reshape:
                                print(f"Error reshaping channel '{current_channel_name}' (shape {arr_val.shape}) to {ref_shape} for file {filename} (validation). Error: {e_reshape}")
                                possible_to_conform_all = False
                                break
                        
                        if conformed_arr is not None:
                            conformed_channels_for_file.append(conformed_arr)
                        else:
                            # This case implies conformed_arr remained None without possible_to_conform_all being set to False and breaking.
                            # This can happen if a 2D array didn't match any tiling/exact shape and also didn't match element count for reshape.
                            print(f"Critical: Channel '{current_channel_name}' (shape {arr_val.shape}) could not be conformed to {ref_shape} in {filename} (validation), and conformed_arr is None.")
                            possible_to_conform_all = False 
                            break 
                    
                    if possible_to_conform_all and len(conformed_channels_for_file) == len(channels):
                        processed_validation_data_list.append(conformed_channels_for_file)
                    # else:
                        # The "skipped due to channel processing/conforming issues" message will be implicitly covered
                        # if files don't make it to the append.
                        # if not possible_to_conform_all:
                        #    print(f"File {filename} (validation) skipped: possible_to_conform_all is False.")
                        # elif len(conformed_channels_for_file) != len(channels):
                        #    print(f"File {filename} (validation) skipped: length mismatch {len(conformed_channels_for_file)} vs {len(channels)}.")

            except OSError as e:
                print(f"Error opening file {filename} (validation): {str(e)}. Skipping file.")
                continue
        else:
            print(f"File not found during validation loading: {filename}")
            continue # Skip this file

    if processed_validation_data_list:
        try:
            final_val_data = np.array(processed_validation_data_list)
            print(f"Successfully loaded and processed {final_val_data.shape[0]} files for validation.")
            return final_val_data
        except ValueError as e:
            # This error occurs if the inner lists (conformed_channels_for_file) are not all the same shape,
            # or if the channels within a file are not the same shape.
            print(f"Critical error: Could not convert validation data to NumPy array: {e}")
            print("This indicates an unrecoverable shape mismatch among processed files/channels for validation.")
            return np.empty((0, 0, 0, 0))
    else:
        print("No validation data loaded after processing selected files.")
        return np.empty((0, 0, 0, 0))
