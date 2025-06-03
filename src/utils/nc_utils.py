from netCDF4 import Dataset
import matplotlib.pyplot as plt
import logging
import numpy as np
import os
import time 
from scipy.interpolate import griddata
from scipy.ndimage import label, binary_dilation

def explore_netcdf(filename):
    """
    Explore the structure of a .nc (NetCDF) file and print its contents.

    :param filename: Path to the NetCDF file
    """
    try:
        with Dataset(filename, mode='r') as ds:
            print(f"=== Exploring file: {filename} ===\n")

            # Print global (file-level) attributes
            print("Global Attributes:")
            for attr_name in ds.ncattrs():
                print(f"  {attr_name}: {ds.getncattr(attr_name)}")
            print()

            # Print dimension information
            print("Dimensions:")
            for dim_name, dim_data in ds.dimensions.items():
                print(f"  {dim_name}: size = {len(dim_data)}")
            print()

            # Print variables and their attributes
            print("Variables:")
            for var_name, var_data in ds.variables.items():
                # Dimensions and shape
                dim_list = ", ".join(var_data.dimensions)
                shape_str = " x ".join(str(s) for s in var_data.shape)
                
                print(f"  {var_name}:")
                print(f"    Dimensions: {dim_list}")
                print(f"    Shape:      {shape_str}")
                print(f"    Max value:  {np.nanmax(var_data)}")
                print(f"    Min value:  {np.nanmin(var_data)}")

                # Variable attributes
                for vattr_name in var_data.ncattrs():
                    vattr_value = var_data.getncattr(vattr_name)
                    print(f"    Attribute '{vattr_name}': {vattr_value}")

                print()  # blank line for readability

    except FileNotFoundError:
        print(f"Error: File not found: {filename}")
    except OSError as e:
        print(f"Error opening file {filename}: {str(e)}")

def display_image(filename, dimension_name):
    """
    Display an image from a NetCDF file 2D matrix with a graded legend.
    """    
    # Suppress font manager messages
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    
    try:
        with Dataset(filename, mode='r') as ds:
            var = ds.variables[dimension_name]
            data = var[:]
            long_name = var.getncattr('long_name') if 'long_name' in var.ncattrs() else dimension_name
            with plt.xkcd():
                plt.imshow(data.T, cmap='bwr', aspect='auto', extent=[0, data.shape[0], -2.7, 15])
                plt.title(f"{long_name}")
                plt.colorbar(label='Value')
                plt.show()
    except FileNotFoundError:
        print(f"Error: File not found: {filename}")
    except OSError as e:
        print(f"Error opening file {filename}: {str(e)}")
        
def display_images(filename, dimension_names, save=False, savefile=None):
    """
    Display a single figure with subplots for each variable (dimension)
    from the provided NetCDF file. If the variable is 2D it is displayed
    as an image, and if it is 1D it is displayed as a line plot.
    
    :param filename: Path to the NetCDF file.
    :param dimension_names: List of variable names (dimensions) to plot.
    :return: The matplotlib figure object.
    """

    try:
        with Dataset(filename, mode='r') as ds:
            num_plots = len(dimension_names)
            # If only one plot, force axs to be a list for uniformity.
            fig, axs = plt.subplots(num_plots, 1, figsize=(8, 4 * num_plots))
            if num_plots == 1:
                axs = [axs]

            for ax, var_name in zip(axs, dimension_names):
                if var_name not in ds.variables:
                    ax.text(0.5, 0.5, f"Variable '{var_name}' not found", 
                            horizontalalignment='center', verticalalignment='center')
                    ax.set_axis_off()
                    continue

                var = ds.variables[var_name]
                data = var[:]
                # If variable is V_AZW0_atm or v_AZW0_atm_vert, divide data by sin(42°)
                if var_name in ["v_AZW0_atm", "v_AZW0_atm_vert"]:
                    data = data / np.sin(np.deg2rad(42))
                    
                long_name = var.getncattr('long_name') if 'long_name' in var.ncattrs() else var_name

                if data.ndim == 2:
                    # Determine colormap and color range for known variables.
                    cmap = 'bwr' if var_name in ["VV", "Vz", "v_AZW0_atm", "v_AZW0_atm_vert", "v_AZW0_atm_vert_masked", "fused_mask"] else 'jet'
                    vmin, vmax = (-10, 10) if var_name in ["VV", "Vz", "v_AZW0_atm", "v_AZW0_atm_vert"] else (None, None)
                    # Plot 2D variable as an image.
                    img = ax.imshow(data.T, cmap=cmap, aspect='auto',
                                    extent=[0, data.shape[0], -3.5, 15], vmin=vmin, vmax=vmax)
                    ax.set_title(long_name)
                    fig.colorbar(img, ax=ax, label='Value')
                elif data.ndim == 1:
                    # Plot 1D variable as a simple line plot.
                    ax.plot(range(len(data)), data, linestyle='-')
                    ax.set_title(long_name)
                    ax.set_xlabel('Index')
                    ax.set_ylabel('Value')
                else:
                    ax.text(0.5, 0.5, f"Variable '{var_name}' has unsupported dimensions: {data.ndim}",
                            horizontalalignment='center', verticalalignment='center')
                    ax.set_axis_off()

            plt.tight_layout()
            if not save:
                plt.show()
            else:
                if savefile is None:
                    plt.savefig("image.png")
                else:
                    plt.savefig(savefile)
            return fig

    except FileNotFoundError:
        print(f"Error: File not found: {filename}")
    except OSError as e:
        print(f"Error opening file {filename}: {str(e)}")

def split_netcdf(filename, variables, target_folder, max_x=2000):
    """
    Filename is a .nc file containing different variables of large size. 
    This function creates several new .nc files selecting only and all of the specified variables 
    while splitting them in chunks of size max_x along the x-axis. It pads the last chunk with NaNs if necessary.

    :param filename: Path to the NetCDF file to split.
    :param variables: List of variable names to split.
    :param target_folder: Path to the folder where the new files will be saved.
    :param max_x: Maximum size of the x-axis in the new files.
    """
    try:
        with Dataset(filename, mode='r') as ds:
            # Create the target folder if it does not exist
            os.makedirs(target_folder, exist_ok=True)
            
            # Get the shape of the first variable to be split to determine the time axis length and others
            t, other_dim = ds.variables[variables[0]].shape
            
            # Calculate the number of chunks
            num_chunks = t // max_x + (1 if t % max_x > 0 else 0)
            
            # Identify the time dimensions
            time_dims = []
            for v in variables:
                time_dims.append(ds.variables[v].dimensions[0])
            
            # Create new files for each chunk
            for i in range(num_chunks):
                start = i * max_x
                end = min((i + 1) * max_x, t)
                available = end - start
                
                # Construct new filename
                new_filename = os.path.join(target_folder, 
                    f"{os.path.basename(filename).replace('.nc', f'_{i}.nc')}")
                with Dataset(new_filename, mode='w') as new_ds:
                    # Copy global attributes
                    for attr_name in ds.ncattrs():
                        new_ds.setncattr(attr_name, ds.getncattr(attr_name))
                    
                    # Copy dimensions: for the time dimension, use max_x, for others copy original size
                    for dim_name, dim_data in ds.dimensions.items():
                        if dim_name in time_dims:
                            new_ds.createDimension(dim_name, max_x)
                        else:
                            new_ds.createDimension(dim_name, len(dim_data))
                    
                    # Process each variable
                    for var_name in variables:
                        var = ds.variables[var_name]
                        # Create the new variable, handling _FillValue properly
                        if '_FillValue' in var.ncattrs():
                            fill_value = var.getncattr('_FillValue')
                            new_var = new_ds.createVariable(var_name, var.dtype, var.dimensions, fill_value=fill_value)
                        else:
                            new_var = new_ds.createVariable(var_name, var.dtype, var.dimensions)
                        
                        # Copy variable attributes (skip _FillValue if already handled)
                        for vattr_name in var.ncattrs():
                            if vattr_name == '_FillValue':
                                continue
                            new_var.setncattr(vattr_name, var.getncattr(vattr_name))
                        
                        # Determine pad value: use fill_value if defined, else NaN.
                        if hasattr(new_var, "fill_value"):
                            pad_val = new_var.fill_value
                        else:
                            pad_val = np.nan
                            
                        # Use the original variable shape to create the new variable’s shape, replacing the time dimension with max_x.
                        orig_shape = var.shape
                        new_shape = list(orig_shape)
                        new_shape[0] = max_x
                        
                        # Initialize new_var with the padded array.
                        new_var[:] = pad_val * np.ones(new_shape, dtype=var.dtype)
                        #new_var[:] = pad_val * np.ones(new_var.shape, dtype=var.dtype)
                        
                        # Define slicers:
                        # Slicer for the original variable: from start to end in the time dimension.
                        orig_slicer = [slice(start, end)]
                        orig_slicer.extend([slice(None)] * (len(orig_shape) - 1))
                        orig_chunk = var[tuple(orig_slicer)]
                        
                        # Slicer for the new variable: fill from index 0 up to available in the time dimension.
                        new_slicer = [slice(0, available)]
                        new_slicer.extend([slice(None)] * (len(new_shape) - 1))

                        # Now assign the chunk to the new variable.
                        new_var[tuple(new_slicer)] = orig_chunk
            
            print(f"Split {filename} into {num_chunks} files in {target_folder}")
    except FileNotFoundError:
        print(f"Error: File not found: {filename}")
    except OSError as e:
        print(f"Error opening file {filename}: {str(e)}")

def extract_netcdf_segment(filename, variables, target_filename, start_index, end_index):
    """
    Extracts a segment of data along the first dimension (typically time) from a NetCDF file
    for specified variables and saves it to a new NetCDF file.

    :param filename: Path to the original NetCDF file.
    :param variables: List of variable names to extract.
    :param target_filename: Path to the new NetCDF file where the segment will be saved.
    :param start_index: The starting index (inclusive) of the segment along the first dimension.
    :param end_index: The ending index (exclusive) of the segment along the first dimension.
    """
    try:
        with Dataset(filename, mode='r') as ds:
            # Validate indices
            first_var_dim_len = ds.variables[variables[0]].shape[0]
            if start_index < 0 or end_index > first_var_dim_len or start_index >= end_index:
                print(f"Error: Invalid start/end indices ({start_index}, {end_index}) for dimension size {first_var_dim_len}.")
                return

            segment_length = end_index - start_index

            # Identify the dimension name associated with the first dimension of the variables
            # Assuming all variables share the same first dimension name (e.g., 'time')
            first_dim_name = ds.variables[variables[0]].dimensions[0]
            first_var_dim_len = ds.variables[variables[0]].shape[0] # Original length of the dimension being segmented

            with Dataset(target_filename, mode='w') as new_ds:
                # Copy global attributes
                for attr_name in ds.ncattrs():
                    new_ds.setncattr(attr_name, ds.getncattr(attr_name))

                # Copy dimensions, adjusting the first dimension's size
                for dim_name, dim_data in ds.dimensions.items():
                    if dim_name == first_dim_name:
                        new_ds.createDimension(dim_name, segment_length)
                    else:
                        # Check if the dimension exists before trying to get its length
                        if dim_name in ds.dimensions:
                           new_ds.createDimension(dim_name, len(dim_data))
                        else:
                           # Handle cases where a dimension might be implicitly defined or missing
                           # Find a variable that uses this dimension to infer its size
                           inferred_size = None
                           for var_name_check in ds.variables:
                               if dim_name in ds.variables[var_name_check].dimensions:
                                   dim_index = ds.variables[var_name_check].dimensions.index(dim_name)
                                   inferred_size = ds.variables[var_name_check].shape[dim_index]
                                   break
                           if inferred_size is not None:
                               new_ds.createDimension(dim_name, inferred_size)
                           else:
                               print(f"Warning: Could not determine size for dimension '{dim_name}'. Skipping.")


                # Process and copy each specified variable
                for var_name in variables:
                    if var_name not in ds.variables:
                        print(f"Warning: Variable '{var_name}' not found in {filename}. Skipping.")
                        continue

                    var = ds.variables[var_name]

                    # Determine the dimension names for the new variable in the target dataset.
                    # Default to original dimension names.
                    target_dims_for_new_var_tuple = var.dimensions
                    
                    # If the variable is not scalar and its first dimension in the source dataset
                    # has the same original length as the dimension we are segmenting (first_var_dim_len),
                    # then this variable's first dimension in the new dataset should use first_dim_name
                    # to ensure it gets the segmented length.
                    if var.dimensions and ds.dimensions[var.dimensions[0]].size == first_var_dim_len:
                        temp_dims_list = list(var.dimensions)
                        temp_dims_list[0] = first_dim_name
                        target_dims_for_new_var_tuple = tuple(temp_dims_list)
                    
                    # Create the new variable in the target file
                    if '_FillValue' in var.ncattrs():
                        fill_value = var.getncattr('_FillValue')
                        new_var = new_ds.createVariable(var_name, var.dtype, target_dims_for_new_var_tuple, fill_value=fill_value)
                    else:
                        new_var = new_ds.createVariable(var_name, var.dtype, target_dims_for_new_var_tuple)

                    # Copy variable attributes (skip _FillValue if already handled)
                    for vattr_name in var.ncattrs():
                        if vattr_name == '_FillValue':
                            continue
                        new_var.setncattr(vattr_name, var.getncattr(vattr_name))

                    # Define slicer for the original variable to extract the segment
                    orig_slicer = [slice(start_index, end_index)]
                    orig_slicer.extend([slice(None)] * (len(var.shape) - 1))

                    # Read the segment and write it to the new variable
                    segment_data = var[tuple(orig_slicer)]
                    new_var[:] = segment_data

            print(f"Extracted segment [{start_index}:{end_index}] from {filename} to {target_filename}")

    except FileNotFoundError:
        print(f"Error: File not found: {filename}")
    except KeyError as e:
        print(f"Error: Variable {e} not found in {filename}.")
    except IndexError as e:
        print(f"Error: Index out of bounds while accessing data. {e}")
    except OSError as e:
        print(f"Error opening or writing file: {str(e)}")
       
def filter_out_nc_files(path, variables, target_folder):
    """
    Checks all .nc files in the given path and filters out those files which have all NaN values
    or all +/-inf values in the specified variables.
    The non-empty files are then saved in the target folder.
    
    :param path: Path to the folder containing the .nc files.
    :param variables: List of variable names to check for NaN and inf values.
    :param target_folder: Path to the folder where the non-empty files will be saved.
    """
    
    # Create the target folder if it does not exist
    os.makedirs(target_folder, exist_ok=True)
    
    # Search for all .nc files in the folder
    nc_files = [f for f in os.listdir(path) if f.endswith(".nc")]
    
    for filename in nc_files:
        try:
            with Dataset(os.path.join(path, filename), mode='r') as ds:
                # Check if any of the specified variables have valid data.
                # Valid data means not all NaN or all inf/-inf.
                has_data = False
                for var_name in variables:
                    var = ds.variables[var_name]
                    data = var[:]
                    if not (np.all(np.isnan(data)) or np.all(np.isinf(data))):
                        has_data = True
                        break
                
                # If the file has non-empty data, save it to the target folder
                if has_data:
                    new_filename = os.path.join(target_folder, filename)
                    os.system(f"cp {os.path.join(path, filename)} {new_filename}")
                    print(f"Saved {filename} to {new_filename}")
                else:
                    print(f"Discarded {filename}")
        except FileNotFoundError:
            print(f"Error: File not found: {filename}")
        except OSError as e:
            print(f"Error opening file {filename}: {str(e)}")
            
def filter_strict_nc_files(path, variables, target_folder):
    """
    Checks all .nc files in the given path and filters out those files which
    contain any NaN or +/-inf values in the specified variables.
    The files that do not contain any invalid data are then saved in the target folder.

    :param path: Path to the folder containing the .nc files.
    :param variables: List of variable names to check for NaN and inf values.
    :param target_folder: Path to the folder where the files without invalid data will be saved.
    """
    
    os.makedirs(target_folder, exist_ok=True)
    nc_files = [f for f in os.listdir(path) if f.endswith(".nc")]

    for filename in nc_files:
        full_path = os.path.join(path, filename)
        try:
            with Dataset(full_path, mode='r') as ds:
                invalid = False
                for var_name in variables:
                    if var_name not in ds.variables:
                        continue
                    data = ds.variables[var_name][:]
                    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                        invalid = True
                        break

                if not invalid:
                    new_filename = os.path.join(target_folder, filename)
                    os.system(f"cp {full_path} {new_filename}")
                    print(f"Saved {filename} to {new_filename}")
                else:
                    print(f"Discarded {filename} due to invalid data")
        except FileNotFoundError:
            print(f"Error: File not found: {filename}")
        except OSError as e:
            print(f"Error opening file {filename}: {str(e)}")
            
def create_vertical_velocity_single_mask(filename, dimension_name, threshold, binary=True):
    """
    Create a mask for vertical velocity (VV) from a NetCDF file.
    The mask is based on the specified threshold value.

    :param filename: Path to the NetCDF file.
    :param dimension_name: Name of the variable to use for the mask.
    :param threshold: Threshold value for the mask.
    :param binary: If True, returns a binary mask (1 where |data| > threshold, 0 otherwise).
                   If False, returns 1 where data > threshold, -1 for data < -threshold, and 0 otherwise.
    :return: A mask array for the specified variable.
    """
    try:
        with Dataset(filename, mode='r') as ds:
            var = ds.variables[dimension_name] / np.sin(np.deg2rad(42))
            data = var[:]
            if binary:
                mask = np.abs(data) > threshold
                return mask.astype(int)
            else:
                mask = np.zeros_like(data, dtype=int)
                mask[data > threshold] = 1
                mask[data < -threshold] = -1
                return mask
    except FileNotFoundError:
        print(f"Error: File not found: {filename}")
    except OSError as e:
        print(f"Error opening file {filename}: {str(e)}")

def create_vertical_velocity_double_mask(filename, dimension_name, threshold, binary=True):
    """
    Create a mask for vertical velocity (VV) from a NetCDF file.
    The mask is based on the specified threshold value.

    :param filename: Path to the NetCDF file.
    :param dimension_name: Name of the variable to use for the mask.
    :param threshold: Threshold value for the mask.
    :param binary: If True, returns a binary mask (1 where |data| > threshold, 0 otherwise).
                   If False, returns 1 where data > threshold, -1 for data < -threshold, and 0 otherwise.
    :return: A mask array for the specified variable.
    """
    try:
        with Dataset(filename, mode='r') as ds:
            var = ds.variables[dimension_name] / np.sin(np.deg2rad(42))
            data = var[:]
            if binary:
                mask = np.abs(data) > threshold
                return mask.astype(int)
            else:
                mask = np.zeros_like(data, dtype=int)
                mask[data > threshold] = 1
                mask[data < -threshold] = -1
                return mask
    except FileNotFoundError:
        print(f"Error: File not found: {filename}")
    except OSError as e:
        print(f"Error opening file {filename}: {str(e)}")
        
def create_vertical_velocity_continous_mask(filename, dimension_name, velocity_lower_bound, velocity_upper_bound):
    """
    Create a continous mask for vertical velocity (VV) from a NetCDF file.
    The mask is based on the absolute values of the data and on the specified lower and upper bounds.
    Values below the lower bound are set to 0, values above the upper bound are set to 1,
    and values in between are mapped linearly between 0 and 1.
    """
    try:
        with Dataset(filename, mode='r') as ds:
            var = ds.variables[dimension_name] / np.sin(np.deg2rad(42))
            data = np.abs(var[:])
            mask = np.zeros_like(data, dtype=float)
            mask[data < velocity_lower_bound] = 0
            mask[data > velocity_upper_bound] = 1
            mask[(data >= velocity_lower_bound) & (data <= velocity_upper_bound)] = \
                (data[(data >= velocity_lower_bound) & (data <= velocity_upper_bound)] - velocity_lower_bound) / \
                (velocity_upper_bound - velocity_lower_bound)           
            return mask
    except FileNotFoundError:
        print(f"Error: File not found: {filename}")
    except OSError as e:
        print(f"Error opening file {filename}: {str(e)}")

def plot_vertical_velocity_mask(mask):
    """
    Plot the vertical velocity mask as a 2D image.
    
    :param mask: Boolean mask array for vertical velocity.
    """
    plt.imshow(mask.T, cmap='gray', aspect='auto')
    plt.title("Vertical Velocity Mask")
    plt.show()
     
def add_mask_to_netcdf(filename, target_filename, variable_name, new_variable_name, threshold, binary=True):
    """
    Add a new variable to a NetCDF file with the specified mask data.
    
    :param filename: Path to the original NetCDF file.
    :param target_filename: Path to the new NetCDF file with the added variable.
    :param variable_name: Name of the variable to use for the mask.
    :param new_variable_name: Name of the new variable to add.
    :param threshold: Threshold value
    """
    
    if binary:
        mask = create_vertical_velocity_single_mask(filename, variable_name, threshold, binary)
    else:
        mask = create_vertical_velocity_double_mask(filename, variable_name, threshold, binary)
    
    try:
        with Dataset(filename, mode='r') as ds:
            # Open the target file for writing
            with Dataset(target_filename, mode='w') as new_ds:
                # Copy global attributes
                for attr_name in ds.ncattrs():
                    new_ds.setncattr(attr_name, ds.getncattr(attr_name))
                
                # Copy dimensions
                for dim_name, dim_data in ds.dimensions.items():
                    new_ds.createDimension(dim_name, len(dim_data))
                
                # Ensure required dimensions exist if not in the source file
                if 'time' not in new_ds.dimensions:
                    new_ds.createDimension('time', mask.shape[0])
                if 'height_bins' not in new_ds.dimensions:
                    new_ds.createDimension('height_bins', mask.shape[1])
                
                # Copy variables
                for var_name, var_data in ds.variables.items():
                    new_var = new_ds.createVariable(var_name, var_data.dtype, var_data.dimensions)
                    new_var[:] = var_data[:]
                    for vattr_name in var_data.ncattrs():
                        new_var.setncattr(vattr_name, var_data.getncattr(vattr_name))
                
                # Add the new variable
                new_var = new_ds.createVariable(new_variable_name, mask.dtype, ('time', 'height_bins'))
                new_var[:] = mask
                new_var.setncattr('long_name', new_variable_name)
                new_var.setncattr('units', 'boolean')
                
                print(f"Added variable '{new_variable_name}' to {target_filename}")
    except FileNotFoundError:
        print(f"Error: File not found: {filename}")
    except OSError as e:
        print(f"Error opening file {filename}: {str(e)}")

def cloud_mask_naive(reflectivity, threshold=-21):
    """
    Compute a simple cloud mask for the given reflectivity.
    The reflectivity input is expected to be a 2D NumPy array (in linear units).
    
    The algorithm creates a mask where each element is 1 if the corresponding
    reflectivity is greater than the threshold, and 0 otherwise.
    
    :param reflectivity: A 2D NumPy array of reflectivity values.
    :param threshold: The threshold value for the cloud mask.
    :return: A 2D mask array for the cloud mask.
    """    
    mask = np.zeros_like(reflectivity)
    mask[reflectivity > threshold] = 1
    return mask
    
def cloud_mask(reflectivity):
    """
    Compute the cloud mask for the given reflectivity.
    The reflectivity input is expected to be a 2D NumPy array (in linear units).
    
    The algorithm creates a mask where each element is 1 if the corresponding
    reflectivity is positive and not NaN, then iteratively updates the mask using
    a random asynchronous update scheme with a digital filter.
    
    :param reflectivity: A 2D NumPy array of reflectivity values.
    :return: A 2D mask array for the cloud mask.
    """
    # Determine the dimensions: number of rays (rows) and number of bins (columns)
    num_rays, num_bins = reflectivity.shape
    
    # Initialize the mask: set to 1 where reflectivity is positive and not NaN; 0 elsewhere.
    mask = np.zeros((num_rays, num_bins))
    mask[(reflectivity > 0) & (~np.isnan(reflectivity))] = 1
    
    # Random seed for reproducibility, and total number of pixels
    random_seed = 42
    total_pixels = num_rays * num_bins
    
    # Create an array of pixel indices (1-indexed to mimic Matlab behavior)
    pixel_indices = np.arange(1, total_pixels + 1)
    
    # Parameters for the asynchronous update scheme
    update_steps = 5
    filter_box_size = 5
    half_box_size = (filter_box_size - 1) // 2  # Equals 2 for filter_box_size of 5
    probability_threshold = 5e-12

    # Iterative asynchronous update
    for step in range(update_steps):
        # Reset the random seed
        np.random.seed(random_seed)
        # Generate a random permutation of numbers from 1 to total_pixels (1-indexed)
        random_positions = np.random.permutation(total_pixels) + 1

        for i in range(total_pixels, 1, -1):
            current_index = i - 1  # Adjust to 0-indexing
            swap_index = int(random_positions[current_index]) - 1  # Adjust to 0-indexing
            pixel_indices[current_index], pixel_indices[swap_index] = (
                pixel_indices[swap_index],
                pixel_indices[current_index],
            )
        
        # Process each pixel index in the shuffled order
        for idx in pixel_indices:
            # Convert the 1-indexed pixel index to 0-indexed row and column positions.
            row_idx = (idx - 1) % num_rays
            col_idx = (idx - 1) // num_rays
            
            # Skip processing if the current pixel is already masked (i.e., mask is 0)
            if mask[row_idx, col_idx] == 0:
                continue
            
            # Define the neighborhood boundaries, respecting the image limits
            row_start = max(row_idx - half_box_size, 0)
            row_end = min(row_idx + half_box_size, num_rays - 1)
            col_start = max(col_idx - half_box_size, 0)
            col_end = min(col_idx + half_box_size, num_bins - 1)
            
            # Extract the neighborhood of the mask
            neighborhood_mask = mask[row_start:row_end + 1, col_start:col_end + 1]
            ones_count = np.sum(neighborhood_mask)
            total_elements = (row_end - row_start + 1) * (col_end - col_start + 1)
            zeros_count = total_elements - ones_count
            
            # Compute the probability based on the counts in the neighborhood
            probability = (0.84 ** zeros_count) * (0.16 ** ones_count)
            if probability >= probability_threshold:
                mask[row_idx, col_idx] = 0

    return mask

def create_cloud_mask(filename, reflectivity_variable):
    """
    Create a cloud mask for the given reflectivity variable from a NetCDF file.
    
    :param filename: Path to the NetCDF file.
    :param reflectivity_variable: Name of the reflectivity variable in the NetCDF file.
    :return: A 2D mask array for the cloud mask.
    """
    try:
        with Dataset(filename, mode='r') as ds:
            reflectivity = ds.variables[reflectivity_variable][:]
            mask = cloud_mask(reflectivity)
            return mask
    except FileNotFoundError:
        print(f"Error: File not found: {filename}")
    except OSError as e:
        print(f"Error opening file {filename}: {str(e)}")

def plot_cloud_mask(filename, reflectivity_variable, save=False, savefile=None):
    """
    Plot the cloud mask for the given reflectivity variable from a NetCDF file.
    Also plot the original reflectivity data for reference.
    
    :param filename: Path to the NetCDF file.
    :param reflectivity_variable: Name of the reflectivity variable in the NetCDF file.
    :param save: If True, save the plot to a file.
    :param savefile: Path to the file to save the plot.
    """
    try:
        with Dataset(filename, mode='r') as ds:
            reflectivity = ds.variables[reflectivity_variable][:]
            mask = create_cloud_mask(reflectivity)
            
            # Create a figure with two subplots: reflectivity and cloud mask
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            axs[0].imshow(reflectivity.T, cmap='jet', aspect='auto')
            axs[0].set_title("Reflectivity")
            axs[1].imshow(mask.T, cmap='bwr', aspect='auto')
            axs[1].set_title("Cloud Mask")
            fig.colorbar(axs[0].imshow(reflectivity.T, cmap='jet', aspect='auto'), ax=axs[0], label='Reflectivity')
            fig.colorbar(axs[1].imshow(mask.T, cmap='bwr', aspect='auto'), ax=axs[1], label='Cloud Mask')
            plt.tight_layout()
            
            if not save:
                plt.show()
            else:
                if savefile is None:
                    plt.savefig("cloud_mask.png")
                else:
                    plt.savefig(savefile)
    except FileNotFoundError:
        print(f"Error: File not found: {filename}")
    except OSError as e:
        print(f"Error opening file {filename}: {str(e)}")

def add_cloud_mask_to_netcdf(filename, target_filename, reflectivity_variable, mask_variable):
    """
    Add a cloud mask variable to a NetCDF file based on the reflectivity variable.
    
    :param filename: Path to the original NetCDF file.
    :param target_filename: Path to the new NetCDF file with the added cloud mask.
    :param reflectivity_variable: Name of the reflectivity variable in the NetCDF file.
    :param mask_variable: Name of the new cloud mask variable to add.
    """
    try:
        with Dataset(filename, mode='r') as ds:
            reflectivity = ds.variables[reflectivity_variable][:]
            mask = create_cloud_mask(reflectivity)
            
            # Open the target file for writing
            with Dataset(target_filename, mode='w') as new_ds:
                # Copy global attributes
                for attr_name in ds.ncattrs():
                    new_ds.setncattr(attr_name, ds.getncattr(attr_name))
                
                # Copy dimensions
                for dim_name, dim_data in ds.dimensions.items():
                    new_ds.createDimension(dim_name, len(dim_data))
                
                # Ensure required dimensions exist if not in the source file
                if 'time' not in new_ds.dimensions:
                    new_ds.createDimension('time', mask.shape[0])
                if 'height_bins' not in new_ds.dimensions:
                    new_ds.createDimension('height_bins', mask.shape[1])
                
                # Copy variables
                for var_name, var_data in ds.variables.items():
                    new_var = new_ds.createVariable(var_name, var_data.dtype, var_data.dimensions)
                    new_var[:] = var_data[:]
                    for vattr_name in var_data.ncattrs():
                        new_var.setncattr(vattr_name, var_data.getncattr(vattr_name))
                
                # Add the new variable
                new_var = new_ds.createVariable(mask_variable, mask.dtype, ('time', 'height_bins'))
                new_var[:] = mask
                new_var.setncattr('long_name', mask_variable)
                new_var.setncattr('units', 'boolean')
                
                print(f"Added variable '{mask_variable}' to {target_filename}")
    except FileNotFoundError:
        print(f"Error: File not found: {filename}")
    except OSError as e:
        print(f"Error opening file {filename}: {str(e)}")
        
def fuse_masks_light(velocity_mask, cloud_mask, threshold=0.5):
    """
    Two masks are passed. Velocity mask is a binary mask with 1 for high velocity and 0 for low velocity.
    Cloud mask is a binary mask with 1 for cloud and 0 for no cloud.
    The fused mask assigns to each cloud area in the cloud mask a value of 1 or 0 based on whether
    more than threshold fraction of the cloud area has high velocity. It assigns nans to areas where
    the cloud mask is 0.
    
    :param velocity_mask: Binary mask for high velocity.
    :param cloud_mask: Binary mask for cloud.
    :param threshold: Fraction of high velocity required to assign a value of 1 to the fused mask.
    :return: Fused mask with values 1, 0, or nan.
    """
    # Initialize the fused mask with nans
    fused_mask = np.full_like(cloud_mask, np.nan)
    
    # Iterate over the cloud mask
    for i in range(cloud_mask.shape[0]):
        for j in range(cloud_mask.shape[1]):
            # If the cloud mask is 0, assign nan
            if cloud_mask[i, j] == 0:
                fused_mask[i, j] = np.nan
            else:
                # Count the number of high velocity points in the neighborhood
                neighborhood = velocity_mask[max(0, i-1):min(i+2, cloud_mask.shape[0]),
                                             max(0, j-1):min(j+2, cloud_mask.shape[1])]
                high_velocity_count = np.sum(neighborhood)
                # Assign 1 if the fraction of high velocity points is above the threshold
                if high_velocity_count / neighborhood.size > threshold:
                    fused_mask[i, j] = 1
                else:
                    fused_mask[i, j] = 0
    
    return fused_mask

def fuse_masks_clusters(velocity_mask, cloud_mask, threshold=0.5):
    """
    Fuse the velocity and cloud masks by first identifying cloud clusters.
    
    Cloud clusters are defined as connected groups of 1's in the cloud_mask.
    Two groups will be merged into a single cluster if they are separated by
    fewer than three consecutive 0's (using a 3x3 dilation to bridge small gaps).
    
    For each cluster, if the fraction of high-velocity points (velocity_mask==1)
    within the original cloud region exceeds the threshold, the entire cluster is
    assigned a value of 1; otherwise 0. Areas with no cloud remain np.nan.
    
    :param velocity_mask (np.ndarray): 2D binary array with 1 for high velocity, 0 for low.
    :param cloud_mask (np.ndarray): 2D binary array with 1 for cloud, 0 for no cloud.
    :param threshold (float): Fraction threshold for high velocity.
    :return np.ndarray: Fused mask with values 1, 0, or np.nan.
    """
    # Dilate the cloud mask with a 3x3 structure. This merges clusters that are separated
    # by gaps of fewer than three consecutive 0's.
    structure = np.ones((3, 3), dtype=int)
    dilated_cloud = binary_dilation(cloud_mask, structure=structure)
    
    # Label the connected regions in the dilated cloud mask.
    labeled, num_features = label(dilated_cloud)
    
    # Initialize fused mask with np.nan everywhere.
    fused_mask = np.full(cloud_mask.shape, np.nan, dtype=float)
    
    # Process each connected region.
    for region_label in range(1, num_features + 1):
        # Restrict the region to the original cloud mask (to avoid including dilated pixels that are not cloud).
        region_mask = (labeled == region_label) & (cloud_mask == 1)
        if np.sum(region_mask) == 0:
            continue
        # Compute the fraction of high velocity pixels in this cluster.
        fraction_high = np.sum(velocity_mask[region_mask]) / np.sum(region_mask)
        # Assign 1 if the fraction exceeds the threshold, else 0.
        cluster_value = 1 if fraction_high > threshold else 0
        fused_mask[region_mask] = cluster_value
        
    return fused_mask

def find_cloud_regions(cloud_profile, tolerance=0):
    """
    Find the cloud regions in a single profile.

    A cloud region is defined as a group of cloud pixels (marked as 1 in the profile).
    Non-consecutive cloud pixels are merged into the same region if the gap between them
    is no more than the specified tolerance.

    :param cloud_profile: 1D binary array with 1 for cloud, 0 for no cloud.
    :param tolerance: Maximum number of consecutive non-cloud pixels allowed within a region.
    :return: List of tuples with start and end indices of cloud regions.
    """
    # Get the indices where cloud pixels are present.
    cloud_indices = [i for i, value in enumerate(cloud_profile) if value == 1]
    if not cloud_indices:
        return []

    # Initialize with the first cloud index.
    regions = []
    region_start = cloud_indices[0]
    region_end = cloud_indices[0]

    for idx in cloud_indices[1:]:
        # If the gap between the current index and the last index in the region is within tolerance,
        # extend the current region.
        if idx - region_end - 1 <= tolerance:
            region_end = idx
        else:
            # If the gap is too large, finish the current region and start a new one.
            regions.append((region_start, region_end))
            region_start = idx
            region_end = idx

    regions.append((region_start, region_end))
    return regions

def find_cloud_regions_with_nans(cloud_profile, tolerance=0):
    """
    Find the regions of non-NaN values in a single profile.

    A region is defined as a group of consecutive non-NaN pixels.
    Non-consecutive non-NaN pixels are merged into the same region if the gap
    (number of consecutive NaN pixels) between them is no more than the specified tolerance.

    :param cloud_profile: 1D NumPy array containing numerical values and potentially NaNs.
    :param tolerance: Maximum number of consecutive NaN pixels allowed within a region.
    :return: List of tuples with start and end indices of the regions.
    """
    # Get the indices where the profile is not NaN.
    non_nan_indices = np.where(~np.isnan(cloud_profile))[0]

    if non_nan_indices.size == 0:
        return []

    regions = []
    # Initialize with the first non-NaN index.
    region_start = non_nan_indices[0]
    region_end = non_nan_indices[0]

    # Iterate through the rest of the non-NaN indices.
    for idx in non_nan_indices[1:]:
        # Calculate the gap (number of NaNs) between the current index and the previous end of the region.
        gap = idx - region_end - 1

        # If the gap is within tolerance, extend the current region.
        if gap <= tolerance:
            region_end = idx
        else:
            # If the gap is too large, finish the current region and start a new one.
            regions.append((region_start, region_end))
            region_start = idx
            region_end = idx

    # Append the last region found.
    regions.append((region_start, region_end))

    return regions

def plot_cloud_regions(cloud_mask, tolerance=0):
    """
    Plot the cloud profile with regions highlighted each one with a different color.
    
    :param cloud_mask: Binary mask for cloud.
    :param tolerance: Maximum number of consecutive non-cloud pixels allowed within a region.
    """
    # Create a color map for the regions
    cmap = plt.cm.get_cmap('tab20', 20)
    
    # Create a figure with a single subplot
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Plot the cloud mask
    ax.imshow(cloud_mask.T, cmap='binary', aspect='auto')
    
    # Global counter so each region gets a different color
    region_counter = 0
    
    # Loop through each profile 
    for j in range(cloud_mask.shape[0]):
        cloud_profile = cloud_mask[j, :]
        cloud_regions = find_cloud_regions(cloud_profile, tolerance)
        for region_start, region_end in cloud_regions:
            region_color = cmap(region_counter % 20)
            region_counter += 1
            # Compute the relative position for the current profile along x
            xmin = j / cloud_mask.shape[1]
            xmax = (j + 1) / cloud_mask.shape[1]
            ax.axhspan(region_start, region_end, xmin=xmin, xmax=xmax,
                       color=region_color, alpha=0.5)
    
    ax.set_title(f"Cloud Regions (Tolerance = {tolerance})")
    plt.show()

def fuse_masks_profiles(velocity_mask, cloud_mask, threshold=0.5, tolerance=0):
    """
    Fuse the velocity and cloud masks on the vertical profiles.
    For each vertical profile (columns of each mask), identify cloud regions
    from the cloud mask. A cloud region is defined as a group of consecutive
    cloud pixels ALONG THE SINGLE PROFILE. If the fraction of high-velocity
    points in the cloud region exceeds the threshold, the entire region is
    assigned a value of 1; otherwise 0. Areas with no cloud remain np.nan.
    
    :param velocity_mask: Binary mask for high velocity.
    :param cloud_mask: Binary mask for cloud.
    :param threshold: Fraction of high velocity required to assign a value of 1 to the fused mask.
    :param tolerance: Maximum number of consecutive non-cloud pixels allowed within a region.
    :return: Fused mask with values 1, 0, or nan.
    """
    #plot_cloud_regions(cloud_mask, tolerance)
    
    # Initialize the fused mask with nans
    fused_mask = np.full_like(cloud_mask, np.nan)
    
    for i in range(cloud_mask.shape[0]):
        cloud_regions = find_cloud_regions(cloud_mask[i, :], tolerance)
        
        # Process each cloud region
        for region_start, region_end in cloud_regions:
            # Extract the region from the cloud and velocity masks
            cloud_region = cloud_mask[i, region_start:region_end + 1]
            velocity_region = velocity_mask[i, region_start:region_end + 1]
            
            # Compute the fraction of high velocity points in the region
            high_velocity_count = np.sum(velocity_region)
            region_size = region_end - region_start + 1
            if high_velocity_count / region_size > threshold:
                fused_mask[i, region_start:region_end + 1] = 1
            else:
                fused_mask[i, region_start:region_end + 1] = 0
    
    return fused_mask

def fuse_masks_profiles_with_sliding_window(velocity_mask, cloud_mask, threshold=0.5, tolerance=0, window_size=75):
    """
    Fuse the velocity and cloud masks on the vertical profiles.
    For each vertical profile (row), identify cloud regions from the cloud mask.
    A cloud region is defined as a group of consecutive cloud pixels ALONG THE SINGLE PROFILE.
    
    If the region length is shorter than 'window_size', compute the fraction of 
    high velocity points for the entire region and assign 1 if it exceeds the threshold, else 0.
    
    If the region is longer than window_size, use a sliding window (stride=1 pixel) 
    over that region. For each window, compute the fraction of high velocity points.
    If the fraction exceeds the threshold, assign the entire window to 1; otherwise 0.
    Overlapping windows may update values from previous windows.
    
    Areas with no cloud remain np.nan.
    
    :param velocity_mask: Binary mask for high velocity.
    :param cloud_mask: Binary mask for cloud.
    :param threshold: Fraction of high velocity required to assign a value of 1.
    :param tolerance: Maximum number of consecutive non-cloud pixels allowed within a region.
    :param window_size: Size of the sliding window (in pixels).
    :return: Fused mask with values 1, 0, or np.nan.
    """
    # Initialize the fused mask with NaNs
    fused_mask = np.full_like(cloud_mask, np.nan, dtype=float)
    
    # Process each vertical profile (each row)
    for i in range(cloud_mask.shape[0]):
        cloud_regions = find_cloud_regions(cloud_mask[i, :], tolerance)
        # Process each cloud region
        for region_start, region_end in cloud_regions:
            region_length = region_end - region_start + 1
            if region_length < window_size:
                # Use the whole region
                velocity_region = velocity_mask[i, region_start:region_end + 1]
                fraction = np.sum(velocity_region) / region_length
                fused_value = 1 if fraction > threshold else 0
                fused_mask[i, region_start:region_end + 1] = fused_value
            else:
                # Use a sliding window (stride 1)
                # Iterate over all possible windows within the region
                for j in range(region_start, region_end - window_size + 2):
                    window = velocity_mask[i, j:j+window_size]
                    fraction = np.sum(window) / window_size
                    fused_value = 1 if fraction > threshold else 0
                    # Update the fused mask for this window. In overlapping areas, later evaluations may override earlier ones.
                    fused_mask[i, j:j+window_size] = fused_value
                    
    return fused_mask

def fuse_continuous_velocity_mask_and_cloud_mask(velocity_mask, cloud_mask):
    """
    Fuse the continuous velocity mask and cloud mask.
    The function creates a fused mask where each pixel contains the correspongind velocity_mask value
    except for the pixels where the cloud mask is 0. In those cases, the fused mask
    is set to NaN.
    
    :param velocity_mask: Continuous velocity mask.
    :param cloud_mask: Cloud mask.
    :return: Fused mask with values from velocity_mask where cloud_mask is 1, NaN otherwise.
    """
    # Create a copy of the velocity mask to initialize the fused mask
    fused_mask = np.copy(velocity_mask)
    
    # Set the values of the fused mask to NaN where the cloud mask is 0
    fused_mask[cloud_mask == 0] = np.nan
    
    return fused_mask

def plot_fused_mask(filename, reflectivity_variable, velocity_variable, threshold_velocity=1.5, threshold_cloud=0.5, save=False, savefile=None):
    """
    Plot the fused mask for the given reflectivity and velocity variables from a NetCDF file,
    along with the two intermediate masks (cloud mask and velocity mask).
    
    This function displays five plots:
      1. Reflectivity
      2. Raw Velocity
      3. Cloud Mask (derived from reflectivity)
      4. Velocity Mask (derived from velocity)
      5. Fused Mask (combining the two masks)
    
    :param filename: Path to the NetCDF file.
    :param reflectivity_variable: Name of the reflectivity variable in the NetCDF file.
    :param velocity_variable: Name of the velocity variable in the NetCDF file.
    :param threshold_velocity: Threshold for vertical velocity mask.
    :param threshold_cloud: Fraction threshold for fusing masks.
    :param save: If True, save the plot to a file.
    :param savefile: Path to the file to save the plot.
    """
    try:
        with Dataset(filename, mode='r') as ds:
            reflectivity = ds.variables[reflectivity_variable][:]
            velocity = ds.variables[velocity_variable][:] / np.sin(np.deg2rad(42))
            cloud = cloud_mask_naive(reflectivity)
            vel_mask = create_vertical_velocity_single_mask(filename, velocity_variable, threshold_velocity, binary=True)
            fused = fuse_masks_profiles_with_sliding_window(vel_mask, cloud, threshold=threshold_cloud)
            print("Velocity mask shape:", vel_mask.shape)
            print("Cloud mask shape:", cloud.shape)
            print("Fused mask shape:", fused.shape)
            print("Fused mask unique values:", np.unique(fused))
            print("Fused mask range:", np.nanmin(fused), np.nanmax(fused))
            
            # Create a figure with a gridspec layout: a top row with 2 small plots and a bottom row with 1 large plot
            fig = plt.figure(figsize=(25, 10))
            gs = fig.add_gridspec(2, 1, height_ratios=[1, 2])
            
            # Top row: 2 subplots
            gs_top = gs[0].subgridspec(1, 2, wspace=0.1)
            axs_top = gs_top.subplots()
            
            # Plot reflectivity with saturation at -25 and 20 dBZ
            im0 = axs_top[0].imshow(np.clip(reflectivity.T, -25, 20), cmap='jet', aspect='auto', 
                                     extent=[0, reflectivity.shape[0], -4, 15], vmin=-25, vmax=20)
            axs_top[0].set_title("Reflectivity")
            axs_top[0].set_xlabel("km")
            axs_top[0].set_ylabel("km")
            cbar0 = fig.colorbar(im0, ax=axs_top[0], orientation='vertical')
            cbar0.set_label("dBZ")
            
            # Add cloud mask contours to the reflectivity plot (using 0.5 as the threshold between 0 and 1)
            x_ref = np.arange(reflectivity.shape[0])
            y_ref = np.linspace(-4, 15, reflectivity.shape[1])
            X_ref, Y_ref = np.meshgrid(x_ref, y_ref)
            axs_top[0].contour(X_ref, np.flipud(Y_ref), cloud.T, levels=[0.5], colors='white', linewidths=1)
            
            # Plot raw velocity
            im1 = axs_top[1].imshow(velocity.T, cmap='jet', aspect='auto', 
                                     extent=[0, velocity.shape[0], -4, 15],
                                     vmin=-4, vmax=4)
            axs_top[1].set_title("Velocity")
            axs_top[1].set_xlabel("km")
            axs_top[1].set_ylabel("km")
            cbar1 = fig.colorbar(im1, ax=axs_top[1], orientation='vertical')
            cbar1.set_label("m/s")
            
            # Create the coordinate grid matching the velocity image
            x = np.arange(velocity.shape[0])
            y = np.linspace(-4, 15, velocity.shape[1])
            X, Y = np.meshgrid(x, y)

            # Add a contour line where velocity equals threshold_velocity without text labels
            axs_top[1].contour(X, np.flipud(Y), velocity.T, levels=[-threshold_velocity, threshold_velocity],
                                colors='black', linewidths=1)
            
            # Bottom row: fused mask plot spanning the whole row
            ax_bottom = fig.add_subplot(gs[1])
            
            # Prepare the fused mask plot: mask invalid (NaN) values and set a distinct color.
            fused_masked = np.ma.masked_invalid(fused.T)
            cmap_custom = plt.cm.get_cmap('bwr').copy()
            cmap_custom.set_bad(color='gray')
            
            #TEST: fused mask = fused but Nans are put to 0
            #fused_masked = np.nan_to_num(fused.T, nan=0)
            
            im_bottom = ax_bottom.imshow(fused_masked, cmap=cmap_custom, aspect='auto', 
                                         extent=[0, fused_masked.shape[1], -4, 15])
            ax_bottom.set_title("Convection/Stratiform Mask")
            ax_bottom.set_xlabel("km")
            ax_bottom.set_ylabel("km")
            
            # Add legend for fused mask: red = convection, blue = stratiform
            import matplotlib.patches as mpatches
            red_patch = mpatches.Patch(color='red', label='Convection')
            blue_patch = mpatches.Patch(color='blue', label='Stratiform')
            ax_bottom.legend(handles=[red_patch, blue_patch], loc='upper right')
            
            plt.tight_layout()
            
            if not save:
                plt.show()
            else:
                if savefile is None:
                    plt.savefig("fused_mask.png")
                else:
                    plt.savefig(savefile)
                    
    except FileNotFoundError:
        print("Error: File not found")
    except OSError as e:
        print(f"Error opening file: {str(e)}")

def add_fused_mask_to_netcdf(filename, target_filename, reflectivity_variable, velocity_variable, threshold_velocity=1.5, threshold_cloud=0.5, window_size=75):
    """
    Add a fused mask variable to a NetCDF file based on the reflectivity and velocity variables.
    
    :param filename: Path to the original NetCDF file.
    :param target_filename: Path to the new NetCDF file with the added fused mask.
    :param reflectivity_variable: Name of the reflectivity variable in the NetCDF file.
    :param velocity_variable: Name of the velocity variable in the NetCDF file.
    :param mask_variable: Name of the new fused mask variable to add.
    :param threshold_velocity: Threshold for vertical velocity mask.
    :param threshold_cloud: Fraction threshold for fusing masks.
    """
    try:
        with Dataset(filename, mode='r') as ds:
            reflectivity = ds.variables[reflectivity_variable][:]
            velocity = ds.variables[velocity_variable][:] / np.sin(np.deg2rad(42))
            cloud = cloud_mask_naive(reflectivity)
            vel_mask = create_vertical_velocity_single_mask(filename, velocity_variable, threshold_velocity, binary=True)
            fused = fuse_masks_profiles_with_sliding_window(vel_mask, cloud, threshold=threshold_cloud, window_size=window_size)
            
            # Open the target file for writing
            with Dataset(target_filename, mode='w') as new_ds:
                # Copy global attributes
                for attr_name in ds.ncattrs():
                    new_ds.setncattr(attr_name, ds.getncattr(attr_name))
                
                # Copy dimensions
                for dim_name, dim_data in ds.dimensions.items():
                    new_ds.createDimension(dim_name, len(dim_data))
                
                # Ensure required dimensions exist if not in the source file
                if 'time' not in new_ds.dimensions:
                    new_ds.createDimension('time', fused.shape[0])
                if 'height_bins' not in new_ds.dimensions:
                    new_ds.createDimension('height_bins', fused.shape[1])
                
                # Copy variables
                for var_name, var_data in ds.variables.items():
                    new_var = new_ds.createVariable(var_name, var_data.dtype, var_data.dimensions)
                    new_var[:] = var_data[:]
                    for vattr_name in var_data.ncattrs():
                        new_var.setncattr(vattr_name, var_data.getncattr(vattr_name))
                
                # Add the new variable
                new_var = new_ds.createVariable('fused_mask', fused.dtype, ('time', 'height_bins'))
                new_var[:] = fused
                new_var.setncattr('long_name', 'Fused Mask')
                new_var.setncattr('units', 'boolean')
                
                print(f"Added variable 'fused_mask' to {target_filename}")
    except FileNotFoundError:
        print(f"Error: File not found: {filename}")
    except OSError as e:
        print(f"Error opening file {filename}: {str(e)}")
        
def add_continuous_mask_to_netcdf(filename, target_filename, reflectivity_variable, velocity_variable, velocity_lower_bound, velocity_upper_bound):
    """
    Add a continous mask to a NetCDF file based on the reflectivity and velocity variables.
    A continous mask is computed by taking the absolute value of the velocity and mapping to 0 the values <= to velocity_lower_bound
    and to 1 the values >= velocity_upper_bound. The values in between are mapped linearly.
    Finally, the final mask is fused with the cloud mask.
    
    :param filename: Path to the original NetCDF file.
    :param target_filename: Path to the new NetCDF file with the added fused mask.
    :param reflectivity_variable: Name of the reflectivity variable in the NetCDF file.
    :param velocity_variable: Name of the velocity variable in the NetCDF file.
    :param velocity_lower_bound: Lower bound for the velocity mask.
    :param velocity_upper_bound: Upper bound for the velocity mask.
    """
    try:
        with Dataset(filename, mode='r') as ds:
            reflectivity = ds.variables[reflectivity_variable][:]
            velocity = ds.variables[velocity_variable][:] / np.sin(np.deg2rad(42))
            cloud = cloud_mask_naive(reflectivity)
            vel_mask = create_vertical_velocity_continous_mask(filename, velocity_variable, velocity_lower_bound, velocity_upper_bound)
            fused = fuse_continuous_velocity_mask_and_cloud_mask(vel_mask, cloud)
            
            # Open the target file for writing
            with Dataset(target_filename, mode='w') as new_ds:
                # Copy global attributes
                for attr_name in ds.ncattrs():
                    new_ds.setncattr(attr_name, ds.getncattr(attr_name))
                
                # Copy dimensions
                for dim_name, dim_data in ds.dimensions.items():
                    new_ds.createDimension(dim_name, len(dim_data))
                
                # Ensure required dimensions exist if not in the source file
                if 'time' not in new_ds.dimensions:
                    new_ds.createDimension('time', fused.shape[0])
                if 'height_bins' not in new_ds.dimensions:
                    new_ds.createDimension('height_bins', fused.shape[1])
                
                # Copy variables
                for var_name, var_data in ds.variables.items():
                    new_var = new_ds.createVariable(var_name, var_data.dtype, var_data.dimensions)
                    new_var[:] = var_data[:]
                    for vattr_name in var_data.ncattrs():
                        new_var.setncattr(vattr_name, var_data.getncattr(vattr_name))
                
                # Add the new variable
                new_var = new_ds.createVariable('continuous_mask', fused.dtype, ('time', 'height_bins'))
                new_var[:] = fused
                new_var.setncattr('long_name', 'Continuous Mask')
                new_var.setncattr('units', 'boolean')
                
                print(f"Added variable 'continuous_mask' to {target_filename}")
    except FileNotFoundError:
        print(f"Error: File not found: {filename}")
    except OSError as e:
        print(f"Error opening file {filename}: {str(e)}")

def plot_continuous_mask(filename, reflectivity_variable, velocity_variable, velocity_lower_bound, velocity_upper_bound, save=False, savefile=None):
    """
    Plot the continuous mask for the given reflectivity and velocity variables from a NetCDF file,
    along with the two intermediate masks (cloud mask and velocity mask).
    
    This function displays five plots:
      1. Reflectivity
      2. Raw Velocity
      3. Cloud Mask (derived from reflectivity)
      4. Velocity Mask (derived from velocity)
      5. Continuous Mask (combining the two masks)
    
    :param filename: Path to the NetCDF file.
    :param reflectivity_variable: Name of the reflectivity variable in the NetCDF file.
    :param velocity_variable: Name of the velocity variable in the NetCDF file.
    :param velocity_lower_bound: Lower bound for the velocity mask.
    :param velocity_upper_bound: Upper bound for the velocity mask.
    :param save: If True, save the plot to a file.
    :param savefile: Path to the file to save the plot.
    """
    try:
        with Dataset(filename, mode='r') as ds:
            reflectivity = ds.variables[reflectivity_variable][:]
            velocity = ds.variables[velocity_variable][:] / np.sin(np.deg2rad(42))
            cloud = cloud_mask_naive(reflectivity)
            vel_mask = create_vertical_velocity_continous_mask(filename, velocity_variable, velocity_lower_bound, velocity_upper_bound)
            fused = fuse_continuous_velocity_mask_and_cloud_mask(vel_mask, cloud)
            print("Velocity mask shape:", vel_mask.shape)
            print("Cloud mask shape:", cloud.shape)
            print("Fused mask shape:", fused.shape)
            #print("Fused mask unique values:", np.unique(fused))
            print("Fused mask range:", np.nanmin(fused), np.nanmax(fused))
            
            # Create a figure with a gridspec layout: a top row with 2 small plots and a bottom row with 1 large plot
            fig = plt.figure(figsize=(25, 10))
            gs = fig.add_gridspec(2, 1, height_ratios=[1, 2])
            
            # Top row: 2 subplots
            gs_top = gs[0].subgridspec(1, 2, wspace=0.1)
            axs_top = gs_top.subplots()
            
            # Plot reflectivity with saturation at -25 and 20 dBZ
            im0 = axs_top[0].imshow(np.clip(reflectivity.T, -25, 20), cmap='jet', aspect='auto', 
                                     extent=[0, reflectivity.shape[0], -4, 15], vmin=-25, vmax=20)
            axs_top[0].set_title("Reflectivity")
            axs_top[0].set_xlabel("km")
            axs_top[0].set_ylabel("km")
            cbar0 = fig.colorbar(im0, ax=axs_top[0], orientation='vertical')
            cbar0.set_label("dBZ")
            
            # Add cloud mask contours to the reflectivity plot (using 0.5 as the threshold between 0 and 1)
            x_ref = np.arange(reflectivity.shape[0])
            y_ref = np.linspace(-4, 15, reflectivity.shape[1])
            X_ref, Y_ref = np.meshgrid(x_ref, y_ref)
            axs_top[0].contour(X_ref, np.flipud(Y_ref), cloud.T, levels=[0.5], colors='white', linewidths=1)
            
            # Plot raw velocity
            im1 = axs_top[1].imshow(velocity.T, cmap='bwr', aspect='auto', 
                                     extent=[0, velocity.shape[0], -4, 15],
                                     vmin=-4, vmax=4)
            axs_top[1].set_title("Velocity")
            axs_top[1].set_xlabel("km")
            axs_top[1].set_ylabel("km")
            cbar1 = fig.colorbar(im1, ax=axs_top[1], orientation='vertical')
            cbar1.set_label("m/s")
            
            # Create the coordinate grid matching the velocity image
            x = np.arange(velocity.shape[0])
            y = np.linspace(-4, 15, velocity.shape[1])
            X, Y = np.meshgrid(x, y)

            # Add a contour line where velocity equals velocity upper bound both in positive and negative direction
            axs_top[1].contour(X, np.flipud(Y), velocity.T, levels=[-velocity_upper_bound, velocity_upper_bound],
                                colors='black', linewidths=1)
            
            # Bottom row: fused mask plot spanning the whole row
            ax_bottom = fig.add_subplot(gs[1])
            
            # Prepare the fused mask plot: mask invalid (NaN) values and set a distinct color.
            fused_masked = np.ma.masked_invalid(fused.T)
            cmap_custom = plt.cm.get_cmap('bwr').copy()
            cmap_custom.set_bad(color='gray')
            
            #TEST: fused mask = fused but Nans are put to 0
            #fused_masked = np.nan_to_num(fused.T, nan=0)
            
            im_bottom = ax_bottom.imshow(fused_masked, cmap=cmap_custom, aspect='auto', 
                                         extent=[0, fused_masked.shape[1], -4, 15])
            ax_bottom.set_title("Convection/Stratiform Mask")
            ax_bottom.set_xlabel("km")
            ax_bottom.set_ylabel("km")
            
            # Add legend for fused mask: red = convection, blue = stratiform
            import matplotlib.patches as mpatches
            red_patch = mpatches.Patch(color='red', label='Convection')
            blue_patch = mpatches.Patch(color='blue', label='Stratiform')
            ax_bottom.legend(handles=[red_patch, blue_patch], loc='upper right')
            
            plt.tight_layout()
            
            if not save:
                plt.show()
            else:
                if savefile is None:
                    plt.savefig("fused_mask.png")
                else:
                    plt.savefig(savefile)
                    
    except FileNotFoundError:
        print("Error: File not found")
    except OSError as e:
        print(f"Error opening file: {str(e)}")
            
def postprocess_profiles(image, threshold, horizontal_tolerance=0):
    """
    Post processing to masks and model outputs.
    image is a 2D numpy array with values between 0 and 1, containing also NaN values.
    The postprocessing consists in:
    1. Create an initial mask based on vertical regions:
        - Find all separate non-NaN regions in each column.
        - For each region, assign 1 if the average value > threshold, else 0. NaN remains NaN.
    2. Refine the mask horizontally:
        - Mark a pixel initially marked as 0 to 1 if it is horizontally situated
          between pixels marked as 1 (at the same row) within a distance specified by horizontal_tolerance,
          ignoring NaN values in the search.
    3. Vertical consistency check:
        - Ensure that within each column, 0s do not appear below 1s unless separated by at least one NaN.
        - If a 0 is found below a 1 without a NaN gap, change all 0s in that column to 1s.

    :param image: 2D numpy array with values between 0 and 1, containing also NaN values. Shape (rows, columns).
    :param threshold: Threshold for the initial vertical region processing.
    :param horizontal_tolerance: Maximum horizontal distance (in columns) to bridge '0' pixels between '1' pixels.
    :return: 2D numpy array with the same shape as the image, containing only 0, 1, or NaN values.
    """

    # Step 1: Initial mask based on vertical regions
    initial_mask = np.full_like(image, np.nan, dtype=float) # Use float for NaN
    rows, cols = image.shape

    for c in range(cols):
        column = image[:, c]
        # Find non-NaN regions in the column (vertical tolerance for gaps is 0)
        cloud_regions = find_cloud_regions_with_nans(column, tolerance=0)

        for region_start, region_end in cloud_regions:
            # Ensure indices are valid
            if region_start <= region_end:
                # Calculate mean, ignoring potential NaNs within the segment
                avg_value = np.nanmean(column[region_start:region_end + 1])

                if not np.isnan(avg_value):
                    # Assign 1 or 0 based on threshold
                    value_to_assign = 1.0 if avg_value > threshold else 0.0
                    initial_mask[region_start:region_end + 1, c] = value_to_assign
                # else: remains NaN

    # Step 2: Refine the mask horizontally
    if horizontal_tolerance > 0:
        intermediate_mask = initial_mask.copy() # Work on a copy for horizontal refinement
        for r in range(rows):
            row_data = initial_mask[r, :]
            for c in range(cols):
                # Check only pixels that are currently 0 in the initial mask
                if initial_mask[r, c] == 0:
                    # Define search ranges, ensuring they are within bounds
                    left_start = max(0, c - horizontal_tolerance)
                    right_end = min(cols - 1, c + horizontal_tolerance)

                    # Check if there is a '1' within the tolerance to the left (ignoring NaNs)
                    found_left_1 = np.any(row_data[left_start:c] == 1)

                    # Check if there is a '1' within the tolerance to the right (ignoring NaNs)
                    found_right_1 = np.any(row_data[c + 1:right_end + 1] == 1)

                    # If a '1' is found on both sides within the tolerance, change the '0' to '1'
                    if found_left_1 and found_right_1:
                        intermediate_mask[r, c] = 1.0
    else:
        intermediate_mask = initial_mask # No horizontal refinement needed

    # Step 3: Vertical consistency check
    final_mask = intermediate_mask.copy()
    for c in range(cols):
        col_data = final_mask[:, c]
        zero_indices = np.where(col_data == 0)[0]
        one_indices = np.where(col_data == 1)[0]

        # Proceed only if both 0s and 1s exist in the column
        if zero_indices.size > 0 and one_indices.size > 0:
            # Iterate through each index where the value is 0
            for z_idx in zero_indices:
                # Find indices of '1's that are located above the current '0'
                ones_above = one_indices[one_indices < z_idx]

                # Check if there is at least one '1' above this '0'
                if ones_above.size > 0:
                    # Find the index of the '1' that is closest (immediately above) the current '0'
                    max_o_idx_above = np.max(ones_above)

                    # Define the vertical segment between this '1' and the current '0'
                    # The segment includes indices from max_o_idx_above + 1 up to z_idx - 1
                    segment_start = max_o_idx_above + 1
                    segment_end = z_idx # Slice end index is exclusive

                    # Check if the segment is valid (at least one element)
                    if segment_start < segment_end:
                        segment_to_check = col_data[segment_start:segment_end]
                        # If there are NO NaN values in this specific segment
                        if not np.any(np.isnan(segment_to_check)):
                            # This '0' violates the vertical consistency rule, change it to '1'
                            final_mask[z_idx, c] = 1.0
                            # Note: We modify final_mask, but continue iterating based on the original zero_indices.
                            # This is generally safe as we are only changing 0s to 1s.
                            # A potential optimization could be to update one_indices, but it adds complexity.
                        else:
                            # If segment_start >= segment_end, the '0' is immediately below the '1'.
                            # This is a violation, change the '0' to '1'.
                            final_mask[z_idx, c] = 1.0


    return final_mask

def postprocess_window(image, threshold):
    """
    Post processing to masks and model outputs.
    image is a 2D numpy array with values between 0 and 1, containing also NaN values.
    The postprocessing consists in:
    1. for each non-nan pixel in the image, run a sliding window centered on the pixel
        centered on the pixel, with an height of 11 pixels and a width of 3 pixels.
    2. if the average value of the pixels in the window is greater than threshold, assign 1 to the pixel,
        otherwise assign 0.
    3. NaN values remain NaN.
    
    :param image: 2D numpy array with values between 0 and 1, containing also NaN values. Shape (rows, columns).
    :param threshold: Threshold for the initial vertical region processing.
    """
    # Step 1: Initialize the mask with NaN values
    rows, cols = image.shape
    fused_mask = np.full_like(image, np.nan, dtype=float) # Use float for NaN

    # Step 2: Sliding window processing
    for r in range(rows):
        for c in range(cols):
            if not np.isnan(image[r, c]):
                # Define the window limits
                row_start = max(0, r - 5)
                row_end = min(rows - 1, r + 5)
                col_start = max(0, c - 1)
                col_end = min(cols - 1, c + 1)

                # Extract the window and compute the mean, ignoring NaNs
                window = image[row_start:row_end + 1, col_start:col_end + 1]
                mean_value = np.nanmean(window)

                # Assign value based on threshold
                if mean_value > threshold:
                    fused_mask[r, c] = 1.0
                else:
                    fused_mask[r, c] = 0.0
                    
    return fused_mask
     
def postprocess(mode, image, threshold, horizontal_tolerance=0):
    if mode == "profiles":
        return postprocess_profiles(image, threshold, horizontal_tolerance)
    elif mode == "window":
        return postprocess_window(image, threshold)
    else:
        raise ValueError("Invalid mode. Use 'profiles' or 'window'.")
