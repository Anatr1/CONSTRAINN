import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from dataloader import load_validation_data_only
from utils.metrics import *
from utils.nc_utils import postprocess
from mpl_toolkits.axes_grid1 import make_axes_locatable

N_INPUT_CHANNELS = 4
THRESHOLD = 0.05
HORIZONTAL_TOLERANCE = 5
POSTPROCESSING_MODE = "window"
FOLDER = "/absolute/path/to/your/folder/containing/data/in/netCDF/files"  # Update this path to your data folder

def evaluate_checkpoint(checkpoint, no_plot=False, savefig=False):
    local_non_thresholded_metrics = {
        'BCE': [],
        'MSE': [],
        'MAE': [],
    }
    all_postprocess_gt_flat_chkpt = []
    all_postprocess_pred_flat_chkpt = []
    
    variables = ["Zm_dBZ_tot", "V_dop_IQnoconv", "T_V", "T_H", "continuous_mask", "v_AZW0_atm_vert"]

     # Load all files from the folder using load_splitted_data with ratio=1.
    data = load_validation_data_only(FOLDER, variables, train_ratio=0.9)
    if data.shape[0] == 0:
        print("No data loaded from folder.")
        return

    # Set device and load model checkpoint.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(n_channels=N_INPUT_CHANNELS, n_classes=1, final_activation='sigmoid').to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    for i, sample in enumerate(data):        
        # Here we assume each sample is a list/array of 6 channels in the defined ordering.
        # Set inputs (channels 0–3), ground truth (channel 4), and control (channel 5).
        input_channels = N_INPUT_CHANNELS
        raw_input = sample[:input_channels]                # channels 0-3 for model input.
        ground_truth = sample[4]                           # fused_mask as ground truth.
        control_data = sample[5] / np.sin(np.deg2rad(42))  # v_AZW0_atm_vert, for control plotting.

        # Use the first channel's shape as a reference.
        desired_shape = raw_input[0].shape
        for j in range(raw_input.shape[0]):
            if raw_input[j].ndim < 2 or raw_input[j].shape != desired_shape:
                scalar_value = raw_input[j].item() if raw_input[j].size == 1 else raw_input[j]
                raw_input[j] = np.full(desired_shape, scalar_value)
                
        eps = 1e-9

        # Process and normalize input channels:
        # Channel 0: Reflectivity - replace NaNs and values below -25 dBZ.
        # Other channels (e.g., Doppler, TB_V, TB_H): replace NaNs with 0.
        x_clean = np.empty_like(raw_input)
        for j in range(raw_input.shape[0]):
            if j == 0:
                channel = raw_input[j]
                mask = np.isnan(channel) | np.isneginf(channel)
                x_clean[j] = np.where(mask, -25.0, channel)
                x_clean[j][x_clean[j] < -25] = -25
            else:
                x_clean[j] = np.nan_to_num(raw_input[j], nan=0.0)

        # Normalize each input channel separately.
        x0_min, x0_max = x_clean[0].min(), x_clean[0].max()
        if abs(x0_max - x0_min) < eps:
            x_clean[0] = x_clean[0] - x0_min
        else:
            x_clean[0] = (x_clean[0] - x0_min) / (x0_max - x0_min + eps)

        for j in range(1, x_clean.shape[0]):
            xi_min, xi_max = x_clean[j].min(), x_clean[j].max()
            if abs(xi_max - xi_min) < eps:
                x_clean[j] = x_clean[j] - xi_min
            else:
                x_clean[j] = (x_clean[j] - xi_min) / (xi_max - xi_min + eps)

        # Prepare input tensor by adding a batch dimension.
        input_tensor = torch.tensor(x_clean).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor.to(device, dtype=torch.float32))
        pred_image = output.cpu().numpy()[0, 0, :, :]
        pred_image = np.clip(pred_image, 0.0, 1.0)
             
        postprocess_ground_truth = postprocess(POSTPROCESSING_MODE, ground_truth, THRESHOLD, horizontal_tolerance=HORIZONTAL_TOLERANCE)
        pred_image_masked = np.where(np.isnan(ground_truth), np.nan, pred_image)
        postprocess_pred_image = postprocess(POSTPROCESSING_MODE, pred_image_masked, THRESHOLD, horizontal_tolerance=HORIZONTAL_TOLERANCE)

        # --------- Plotting ----------
        if no_plot:
            pass
        else:
            #fig, axes = plt.subplots(3, 3, figsize=(18, 12))
            fig, axes = plt.subplots(3, 2, figsize=(11, 12))

            # --- Column 1 ---
            # Left column, top: Line graph for T_V and T_H.
            ax_line = axes[0, 0]
            ref_len = ground_truth.shape[0]
            x_values = np.arange(ref_len)
            y_TV = np.nanmean(raw_input[2], axis=1)
            y_TH = np.nanmean(raw_input[3], axis=1)
            ax_line.plot(x_values, y_TV, label=r"$T^V_B$")
            ax_line.plot(x_values, y_TH, label=r"$T^H_B$")
            ax_line.set_xlim(0, ref_len - 1)
            ax_line.set_title(f"Brightness Temperature")
            ax_line.set_xlabel("Along Track Distance[Km]")
            ax_line.set_ylabel("Brightness Temperature[K]")
            ax_line.legend()
            divider_line = make_axes_locatable(ax_line)
            cax_line = divider_line.append_axes("right", size="5%", pad=0.05)
            cax_line.axis('off')
            
            # Left column, middle: Reflectivity plot.
            ax_refl = axes[1, 0]
            im_refl = ax_refl.imshow(raw_input[0].T, cmap='jet', aspect='auto',
                        extent=[0, ground_truth.shape[0], -5, 19], vmin=-25, vmax=20)
            ax_refl.set_ylim(-0.5, 19)
            ax_refl.set_title(f"Reflectivity")
            ax_refl.set_xlabel("Along Track Distance[Km]")
            ax_refl.set_ylabel("Height[Km]")
            divider_refl = make_axes_locatable(ax_refl)
            cax_refl = divider_refl.append_axes("right", size="5%", pad=0.05)
            cbar_refl = fig.colorbar(im_refl, cax=cax_refl)
            cbar_refl.set_label('Z [dBZ]')


            # Left column, bottom: Doppler Velocity plot.
            ax_doppler = axes[2, 0]
            divider_doppler = make_axes_locatable(ax_doppler)
            if raw_input.shape[0] > 1:
                im_doppler = ax_doppler.imshow(raw_input[1].T, cmap='jet', aspect='auto',
                            extent=[0, raw_input[1].shape[0], -5, 19])
                ax_doppler.set_ylim(-0.5, 19)
                ax_doppler.set_title(f"Doppler Velocity")
                ax_doppler.set_xlabel("Along Track Distance[Km]")
                ax_doppler.set_ylabel("Height[Km]")
                cax_doppler_cb = divider_doppler.append_axes("right", size="5%", pad=0.05)
                cbar_doppler = fig.colorbar(im_doppler, cax=cax_doppler_cb)
                cbar_doppler.set_label(r'$V_{LoS}$ [m/s]')
            else:
                ax_doppler.text(0.5, 0.5, "Channel 1 not available",
                        horizontalalignment='center', transform=ax_doppler.transAxes)
                ax_doppler.set_title(f"Doppler Velocity")
                ax_doppler.set_xlabel("Along Track Distance[Km]")
                ax_doppler.set_ylabel("Height[Km]")
                cax_doppler_cb = divider_doppler.append_axes("right", size="5%", pad=0.05)
                cax_doppler_cb.axis('off')

            # --- Column 2 ---
            # Middle column, top: Vertical Velocity plot.
            ax_mask_top = axes[0, 1]
            im_vv = ax_mask_top.imshow(control_data.T, cmap='bwr', aspect='auto',
                        extent=[0, control_data.shape[0], -5, 19], vmin=-10, vmax=10)
            ax_mask_top.set_ylim(-0.5, 19)
            ax_mask_top.set_title(f"Vertical Velocity")
            ax_mask_top.set_xlabel("Along Track Distance[Km]")
            ax_mask_top.set_ylabel("Height[Km]")
            divider_vv = make_axes_locatable(ax_mask_top)
            cax_vv_cb = divider_vv.append_axes("right", size="5%", pad=0.05)
            cbar_vv = fig.colorbar(im_vv, cax=cax_vv_cb)
            cbar_vv.set_label('W [m/s]')

            # Middle column, middle: C/S Mask plot.
            ax_mask_mid = axes[1, 1]
            im_cs = ax_mask_mid.imshow(ground_truth.T, cmap='bwr', aspect='auto',
                        extent=[0, ground_truth.shape[0], -5, 19], vmin=0, vmax=1)
            ax_mask_mid.set_ylim(-0.5, 19)
            ax_mask_mid.set_title(f"Target Mask")
            ax_mask_mid.set_xlabel("Along Track Distance[Km]")
            ax_mask_mid.set_ylabel("Height[Km]")
            divider_cs = make_axes_locatable(ax_mask_mid)
            cax_cs_cb = divider_cs.append_axes("right", size="5%", pad=0.05)
            cbar_cs = fig.colorbar(im_cs, cax=cax_cs_cb)
            cbar_cs.set_label('Convective Index')

            # Middle column, bottom: Predicted (Reconstructed) image.
            ax_pred = axes[2, 1]
            im_pred = ax_pred.imshow(pred_image_masked.T, cmap='bwr', aspect='auto',
                        extent=[0, pred_image_masked.shape[0], -5, 19], vmin=0, vmax=1)
            ax_pred.set_ylim(-0.5, 19)
            ax_pred.set_title(f"Reconstructed Image")
            ax_pred.set_xlabel("Along Track Distance[Km]")
            ax_pred.set_ylabel("Height[Km]")
            divider_pred = make_axes_locatable(ax_pred)
            cax_pred_cb = divider_pred.append_axes("right", size="5%", pad=0.05)
            cbar_pred = fig.colorbar(im_pred, cax=cax_pred_cb)
            cbar_pred.set_label('Convective Index')

            '''# --- Column 3 ---
            # Right column, top: Vertical Velocity plot (Replicated).
            ax_mask_top_rep = axes[0, 2]
            im_vv_rep = ax_mask_top_rep.imshow(control_data.T, cmap='bwr', aspect='auto',
                        extent=[0, control_data.shape[0], -5, 19], vmin=-5, vmax=5)
            ax_mask_top_rep.set_ylim(-0.5, 19)
            ax_mask_top_rep.set_title(f"Vertical Velocity (Rep)")
            ax_mask_top_rep.set_xlabel("Along Track Distance[Km]")
            ax_mask_top_rep.set_ylabel("Height[Km]")
            divider_vv_rep = make_axes_locatable(ax_mask_top_rep)
            cax_vv_rep_cb = divider_vv_rep.append_axes("right", size="5%", pad=0.05)
            cbar_vv_rep = fig.colorbar(im_vv_rep, cax=cax_vv_rep_cb)
            cbar_vv_rep.set_label('Vertical Velocity [m/s]') # Added label

            # Right column, middle: C/S Mask plot (Replicated).
            ax_mask_mid_rep = axes[1, 2]
            im_cs_rep = ax_mask_mid_rep.imshow(postprocess_ground_truth.T, cmap='bwr', aspect='auto',
                        extent=[0, postprocess_ground_truth.shape[0], -5, 19], vmin=0, vmax=1)
            ax_mask_mid_rep.set_ylim(-0.5, 19)
            ax_mask_mid_rep.set_title(f"Postprocessed Target Mask")
            ax_mask_mid_rep.set_xlabel("Along Track Distance[Km]")
            ax_mask_mid_rep.set_ylabel("Height[Km]")
            divider_cs_rep = make_axes_locatable(ax_mask_mid_rep)
            cax_cs_rep_cb = divider_cs_rep.append_axes("right", size="5%", pad=0.05)
            cbar_cs_rep = fig.colorbar(im_cs_rep, cax=cax_cs_rep_cb)
            cbar_cs_rep.set_label('Mask Value') # Added label

            # Right column, bottom: Predicted (Reconstructed) image (Replicated).
            ax_pred_rep = axes[2, 2]
            # pred_image_masked is already computed
            im_pred_rep = ax_pred_rep.imshow(postprocess_pred_image.T, cmap='bwr', aspect='auto',
                        extent=[0, postprocess_pred_image.shape[0], -5, 19], vmin=0, vmax=1)
            ax_pred_rep.set_ylim(-0.5, 19)
            ax_pred_rep.set_title(f"Postprocessed Reconstructed Image")
            ax_pred_rep.set_xlabel("Along Track Distance[Km]")
            ax_pred_rep.set_ylabel("Height[Km]")
            divider_pred_rep = make_axes_locatable(ax_pred_rep)
            cax_pred_rep_cb = divider_pred_rep.append_axes("right", size="5%", pad=0.05)
            cbar_pred_rep = fig.colorbar(im_pred_rep, cax=cax_pred_rep_cb)
            cbar_pred_rep.set_label('Predicted Value') # Added label'''
            

            rand = np.random.rand()
            plt.tight_layout()
            if not savefig:
                plt.show()
            else:
                # Random chance in 10% of the cases to save the figure.
                if rand < 0.1:
                    plt.savefig(f"./output_sample_{i+1}.png")
                    
            # New figure for the next plot
            plt.close(fig)

            # Create the second figure for comparison
            fig_comp = plt.figure(figsize=(12, 10))
            gs = plt.GridSpec(2, 2, height_ratios=[1, 1.5], width_ratios=[1,1], hspace=0.3, wspace=0.3)

            # --- Top Row ---
            # Top-left: Postprocessed Target Mask
            ax_pp_gt = fig_comp.add_subplot(gs[0, 0])
            im_cs_rep = ax_pp_gt.imshow(postprocess_ground_truth.T, cmap='bwr', aspect='auto',
                        extent=[0, postprocess_ground_truth.shape[0], -5, 19], vmin=0, vmax=1)
            ax_pp_gt.set_ylim(-0.5, 19)
            ax_pp_gt.set_title(f"Postprocessed Target Mask")
            ax_pp_gt.set_xlabel("Along Track Distance[Km]")
            ax_pp_gt.set_ylabel("Height[Km]")
            divider_cs_rep = make_axes_locatable(ax_pp_gt)
            cax_cs_rep_cb = divider_cs_rep.append_axes("right", size="5%", pad=0.05)
            cbar_cs_rep = fig_comp.colorbar(im_cs_rep, cax=cax_cs_rep_cb)
            cbar_cs_rep.set_label('Mask Value')

            # Top-right: Postprocessed Reconstructed Image
            ax_pp_pred = fig_comp.add_subplot(gs[0, 1])
            im_pred_rep = ax_pp_pred.imshow(postprocess_pred_image.T, cmap='bwr', aspect='auto',
                        extent=[0, postprocess_pred_image.shape[0], -5, 19], vmin=0, vmax=1)
            ax_pp_pred.set_ylim(-0.5, 19)
            ax_pp_pred.set_title(f"Postprocessed Reconstructed Image")
            ax_pp_pred.set_xlabel("Along Track Distance[Km]")
            ax_pp_pred.set_ylabel("Height[Km]")
            divider_pred_rep = make_axes_locatable(ax_pp_pred)
            cax_pred_rep_cb = divider_pred_rep.append_axes("right", size="5%", pad=0.05)
            cbar_pred_rep = fig_comp.colorbar(im_pred_rep, cax=cax_pred_rep_cb)
            cbar_pred_rep.set_label('Predicted Value')

            # --- Bottom Row ---
            # Comparison Map
            ax_comp = fig_comp.add_subplot(gs[1, :]) # Spans both columns

            # Create comparison map: 0=TN, 1=TP, 2=FN, 3=FP, NaN=NaN
            comparison_map = np.full(postprocess_ground_truth.shape, np.nan, dtype=float)

            # Define conditions, handling potential NaNs in inputs first
            gt_nan = np.isnan(postprocess_ground_truth)
            pred_nan = np.isnan(postprocess_pred_image)
            any_nan = gt_nan | pred_nan

            # Use np.where for cleaner assignment, ensuring non-NaN inputs for comparisons
            gt_safe = np.where(gt_nan, -1, postprocess_ground_truth) # Use placeholder for NaN
            pred_safe = np.where(pred_nan, -1, postprocess_pred_image) # Use placeholder for NaN

            tn_cond = (~any_nan) & (gt_safe == 0) & (pred_safe == 0)
            tp_cond = (~any_nan) & (gt_safe == 1) & (pred_safe == 1)
            fn_cond = (~any_nan) & (gt_safe == 1) & (pred_safe == 0)
            fp_cond = (~any_nan) & (gt_safe == 0) & (pred_safe == 1)

            comparison_map[tn_cond] = 0 # True Negative
            comparison_map[tp_cond] = 1 # True Positive
            comparison_map[fn_cond] = 2 # False Negative
            comparison_map[fp_cond] = 3 # False Positive

            # Define colormap and norm
            colors = ['blue', 'red', 'lime', 'yellow'] # TN, TP, FN, FP
            cmap = plt.cm.colors.ListedColormap(colors)
            cmap.set_bad('grey', alpha=0.3) # Color for NaN values
            bounds = [-0.5, 0.5, 1.5, 2.5, 3.5] # Boundaries for the 4 categories
            norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

            # Plotting the comparison map
            im_comp = ax_comp.imshow(comparison_map.T, cmap=cmap, norm=norm, aspect='auto',
                                     extent=[0, comparison_map.shape[0], -5, 19], interpolation='nearest')
            ax_comp.set_ylim(-0.5, 19)
            ax_comp.set_title("Comparison Map (Target vs. Prediction)")
            ax_comp.set_xlabel("Along Track Distance[Km]")
            ax_comp.set_ylabel("Height[Km]")

            # Add colorbar for comparison map
            divider_comp = make_axes_locatable(ax_comp)
            cax_comp_cb = divider_comp.append_axes("right", size="3%", pad=0.05)
            cbar_comp = fig_comp.colorbar(im_comp, cax=cax_comp_cb, ticks=[0, 1, 2, 3])
            cbar_comp.set_ticklabels(['TN', 'TP', 'FN', 'FP'])
            cbar_comp.set_label('Classification Result')

            #plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            if not savefig:
                plt.show()
            else:
                # Random chance in 10% of the cases to save the figure.
                if rand < 0.1:
                    plt.savefig(f"./output_comparison_sample_{i+1}.png")
            plt.close(fig_comp) # Close the comparison figure
            
                        
        # ------ Metrics computation ------
        # Only evaluate metrics for valid (non-NaN) ground truth points.
        flat_mask = ~np.isnan(ground_truth.flatten())
        y_true = ground_truth.flatten()[flat_mask]
        y_true = np.nan_to_num(y_true, nan=0.0)
        y_pred = np.clip(pred_image, 0, 1).flatten()[flat_mask]
        
        # --- Non-thresholded metrics ---
        bce = binary_cross_entropy(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        
        local_non_thresholded_metrics['BCE'].append(bce)
        local_non_thresholded_metrics['MSE'].append(mse)
        local_non_thresholded_metrics['MAE'].append(mae)
        
        # --- Thresholded metrics ---
        # Accumulate flattened arrays for aggregated computation later
        # NaNs are handled here before appending, similar to eval_w_sc.py approach
        current_gt_flat = np.nan_to_num(postprocess_ground_truth.flatten(), nan=0.0)
        current_pred_flat = np.nan_to_num(postprocess_pred_image.flatten(), nan=0.0)
        
        all_postprocess_gt_flat_chkpt.append(current_gt_flat)
        all_postprocess_pred_flat_chkpt.append(current_pred_flat)
        
        # Substitute NaN values in the ground truth with 0.0 for metric computation.
        postprocess_ground_truth = np.nan_to_num(postprocess_ground_truth, nan=0.0)
        postprocess_pred_image = np.nan_to_num(postprocess_pred_image, nan=0.0)

        # Average metrics after the final sample.
        if i == len(data) - 1:
            output = "\n"
            print(f"Model: {checkpoint}")
            output += f"Model: {checkpoint}\n"
            
            print("Non-thresholded metrics for all samples:")
            output += "Non-thresholded metrics for all samples:\n"
            for key in local_non_thresholded_metrics.keys():
                if local_non_thresholded_metrics[key]: # Check if list is not empty
                    avg_metric = np.mean(local_non_thresholded_metrics[key])
                    print(f"{key}: {avg_metric:.4f}")
                    output += f"{key}: {avg_metric:.4f}\n"
                else:
                    print(f"{key}: N/A (no samples)")
                    output += f"{key}: N/A (no samples)\n"
            
            print("\nThresholded metrics for all samples:")
            output += "\nThresholded metrics for all samples:\n"
            
            if all_postprocess_gt_flat_chkpt and all_postprocess_pred_flat_chkpt:
                final_gt_all_samples = np.concatenate(all_postprocess_gt_flat_chkpt)
                final_pred_all_samples = np.concatenate(all_postprocess_pred_flat_chkpt)

                # Ensure integer type for compute_metrics, assuming binary classification (0 or 1)
                final_gt_all_samples = final_gt_all_samples.astype(np.int32)
                final_pred_all_samples = final_pred_all_samples.astype(np.int32)

                if final_gt_all_samples.size > 0: # Ensure there's data after concatenation
                    aggregated_thresholded_metrics = compute_metrics(final_gt_all_samples, final_pred_all_samples)
                    
                    if aggregated_thresholded_metrics:
                        print_metrics(aggregated_thresholded_metrics)
                        output += metrics_to_string(aggregated_thresholded_metrics) # Assumes metrics_to_string can handle the dict
                    else:
                        msg = "Could not compute aggregated thresholded metrics (e.g., no positive cases or other issue).\n"
                        print(msg, end='')
                        output += msg
                else:
                    msg = "No valid data points for aggregated thresholded metrics.\n"
                    print(msg, end='')
                    output += msg
            else:
                msg = "No data accumulated for thresholded metrics.\n"
                print(msg, end='')
                output += msg
            
            print()
            output += "\n"
            # Append the output to a file.
            # Ensure the file path is correct and directory exists if uncommented
            # with open("output_metrics_eval.txt", "a") as f:
            #    f.write(output)

if __name__ == "__main__":
    checkpoints = []
    # Fill the checkpoints list with all the .pth files in the specified directory.
    checkpoint_dir = "../checkpoints"

    for filename in os.listdir(checkpoint_dir):
        if filename.endswith(".pth"):
            checkpoints.append(os.path.join(checkpoint_dir, filename))
                      
    for checkpoint in checkpoints: 
        print(f"Evaluating checkpoint: {checkpoint}")      
        # Import the right UNet model based on the checkpoint name.
        if "mini" in checkpoint:
            from models.mini_unet import UNet
        elif "medium" in checkpoint:
            from models.medium_unet import UNet
        elif "large" in checkpoint:
            from models.large_unet import UNet
        else:
            print("Unknown model type in checkpoint name. Please check the checkpoint path.")
        
        evaluate_checkpoint(checkpoint, no_plot=False, savefig=True)
