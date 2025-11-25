#!/usr/bin/env python3
"""
ndvi_kmeans_fullgui_edited_4band.py

Full GUI (edited) — selects Blue (B2), Green (B3), Red (B4) and NIR (B5) TIFFs,
computes NDVI, masks vegetation, downsamples for KMeans (aspect-ratio preserved),
runs KMeans on non-vegetation using 4-band features (B,G,R,NIR), upsamples clusters,
saves outputs including a 4-band masked TIFF, creates a preview PNG and pops it up.

Requirements:
 pip install rasterio numpy matplotlib scipy
"""

import os
import threading
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from tkinter import Tk, Label, Entry, Button, StringVar, filedialog, messagebox
from scipy.ndimage import zoom

# -----------------------
# K-means implemented from scratch 
# -----------------------
def kmeans_from_scratch(X, K, max_iters=200, tol=1e-4, seed=0):
    """
    Implements K-Means clustering algorithm using NumPy primitives.
    """
    rng = np.random.RandomState(seed)
    N, D = X.shape
    init_idx = rng.choice(N, size=K, replace=False)
    centers = X[init_idx].astype(float)
   
    labels = np.zeros(N, dtype=int)
    for it in range(max_iters):
        # Calculate squared Euclidean distances using broadcasting
        dists = np.sum((X[:, None, :] - centers[None, :, :])**2, axis=2)  # (N, K)
        new_labels = np.argmin(dists, axis=1)

        new_centers = np.zeros_like(centers)
        
        # Recalculate centroids
        for k in range(K):
            members = X[new_labels == k]
            if len(members) == 0:
                # Handle empty cluster: reinitialize randomly
                new_centers[k] = X[rng.randint(N)]
            else:
                new_centers[k] = members.mean(axis=0)

        # Check for convergence
        if np.max(np.abs(new_centers - centers)) < tol:
            labels = new_labels
            centers = new_centers
            break

        centers = new_centers
        labels = new_labels

    return labels, centers

# -----------------------
# Aesthetic green thresholding / display helper
# -----------------------
def plot_aesthetic_preview_and_save(ndvi, veg_mask, clustered_full, K, ndvi_threshold, preview_png):
    """
    Creates a 2-panel figure: Aesthetic green NDVI mask and KMeans clusters.
    Saves to preview_png and then pops up the figure.
    """
    # Create custom green colormap
    colors_high = ['lightgreen', 'green', 'darkgreen']
    high_ndvi_cmap = LinearSegmentedColormap.from_list("custom_high", colors_high, N=256)

    # Prepare display array for clusters: map -1 (veg) to last color index
    disp_vis = clustered_full.copy().astype(np.int32)
    disp_vis[disp_vis == -1] = K  # veg -> last color index (K)

    fig = plt.figure(figsize=(14, 5))

    # Panel 1 - Aesthetic green mask (NDVI values below threshold -> NaN)
    ax2 = fig.add_subplot(1, 2, 1)
    high_ndvi_mask = ndvi > ndvi_threshold
    masked_ndvi = ndvi.copy()
    masked_ndvi[~high_ndvi_mask] = np.nan

    im2 = ax2.imshow(masked_ndvi, cmap=high_ndvi_cmap, vmin=ndvi_threshold, vmax=1)
    ax2.set_title(f"Vegetation (NDVI > {ndvi_threshold})")
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.02)
    cbar2.set_label("NDVI Value")
    cbar2.set_ticks(np.linspace(ndvi_threshold, 1, 5))

    # Panel 2 - Clusters
    ax3 = fig.add_subplot(1, 2, 2)
    cmap_clusters = plt.get_cmap('tab20', K + 1)  # K clusters + veg color
    im3 = ax3.imshow(disp_vis, cmap=cmap_clusters, vmin=0, vmax=K)
    ax3.set_title(f"KMeans clusters (K={K})")
    cbar3 = plt.colorbar(im3, ax=ax3, ticks=list(range(K + 1)), fraction=0.046, pad=0.02)
    labels_cbar = [str(i) for i in range(K)] + ["veg"]
    cbar3.ax.set_yticklabels(labels_cbar)

    plt.tight_layout()
    try:
        fig.savefig(preview_png, dpi=200)
    except Exception as e:
        print("Warning: could not save preview PNG:", e)

    plt.show()

# -----------------------
# Processing pipeline (Updated for 4-band)
# -----------------------
def process_and_save(blue_path, green_path, red_path, nir_path, out_folder, ndvi_threshold, K, downsample_target=1000):
    try:
        # Read inputs (B,G,R,NIR)
        with rasterio.open(blue_path) as b:
            blue = b.read(1).astype('float32')
            meta = b.meta.copy()
            blue_profile = b.profile

        with rasterio.open(green_path) as g:
            green = g.read(1).astype('float32')
            green_profile = g.profile

        with rasterio.open(red_path) as r:
            red = r.read(1).astype('float32')
            red_profile = r.profile

        with rasterio.open(nir_path) as n:
            nir = n.read(1).astype('float32')
            nir_profile = n.profile

        # Alignment checks (shapes and spatial metadata must match)
        shapes = {blue.shape, green.shape, red.shape, nir.shape}
        if len(shapes) != 1:
            raise RuntimeError(f"Shape mismatch among bands: blue {blue.shape}, green {green.shape}, red {red.shape}, nir {nir.shape}")

        if (blue_profile.get('transform') != green_profile.get('transform') or
            blue_profile.get('transform') != red_profile.get('transform') or
            blue_profile.get('transform') != nir_profile.get('transform') or
            blue_profile.get('crs') != green_profile.get('crs') or
            blue_profile.get('crs') != red_profile.get('crs') or
            blue_profile.get('crs') != nir_profile.get('crs')):
            raise RuntimeError("Bands appear to have different transform/CRS. Ensure all four bands are aligned and from same scene.")

        rows, cols = blue.shape

        # Normalize to 0-1 if data are in high DN ranges (e.g., 12-bit or 16-bit)
        max_val = max(blue.max(), green.max(), red.max(), nir.max())
        if max_val > 100:
            divisor = 10000.0
            if max_val > 65535:
                divisor = 65535.0
            blue = blue / divisor
            green = green / divisor
            red = red / divisor
            nir = nir / divisor

        # Compute NDVI (full resolution)
        eps = 1e-8
        ndvi = (nir - red) / (nir + red + eps)

        # Diagnostic
        max_ndvi = ndvi.max()
        count_above_threshold = np.sum(ndvi > ndvi_threshold)
        print("-" * 50)
        print(f"DIAGNOSTIC for NDVI Threshold: {ndvi_threshold}")
        print(f"Max NDVI value found in image: {max_ndvi:.4f}")
        print(f"Total pixels above threshold: {count_above_threshold} (out of {rows * cols})")
        print("-" * 50)

        # Vegetation mask (full resolution)
        veg_mask = ndvi > ndvi_threshold  # boolean mask (True = vegetation)

        # Downsample for clustering, preserving aspect ratio
        target = downsample_target
        max_dim = max(rows, cols)
        scale_factor = target / max_dim
        scale_tuple = (scale_factor, scale_factor)

        # Downsample bands with bilinear (order=1)
        blue_small  = zoom(blue,  scale_tuple, order=1)
        green_small = zoom(green, scale_tuple, order=1)
        red_small   = zoom(red,   scale_tuple, order=1)
        nir_small   = zoom(nir,   scale_tuple, order=1)

        # Downsample mask with nearest neighbor (order=0)
        veg_small = zoom(veg_mask.astype(np.uint8), scale_tuple, order=0).astype(bool)

        # Prepare feature matrix for non-vegetated pixels only (4 bands)
        mask_nonveg_small = ~veg_small
        if np.sum(mask_nonveg_small) == 0:
            raise RuntimeError("No non-vegetation pixels found after thresholding/downsampling. Try lowering the NDVI threshold.")

        features = np.vstack([
            blue_small.ravel(),
            green_small.ravel(),
            red_small.ravel(),
            nir_small.ravel()
        ]).T
        X = features[mask_nonveg_small.ravel()]

        # Normalize features (Crucial for K-Means)
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0) + 1e-8
        Xn = (X - X_mean) / X_std

        # Run K-means on 4-band features
        labels_small, centers = kmeans_from_scratch(Xn, K, max_iters=500, tol=1e-4, seed=42)

        # Build small clustered image (with -1 for veg)
        clustered_small_flat = np.full(blue_small.size, -1, dtype=np.int16)
        clustered_small_flat[mask_nonveg_small.ravel()] = labels_small
        clustered_small = clustered_small_flat.reshape(blue_small.shape)

        # Upscale clustered_small back to full resolution (nearest neighbor, order=0)
        scale_back_r = rows / clustered_small.shape[0]
        scale_back_c = cols / clustered_small.shape[1]
        clustered_full = zoom(clustered_small, (scale_back_r, scale_back_c), order=0)

        # Crop/trim to original dimensions and re-apply vegetation mask
        clustered_full = clustered_full[:rows, :cols]
        clustered_full[veg_mask] = -1  # ensure veg are -1

        # Save outputs
        os.makedirs(out_folder, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(red_path))[0]
        ndvi_path = os.path.join(out_folder, base_name + "_NDVI.tif")
        vegmask_path = os.path.join(out_folder, base_name + "_VEGMASK.tif")
        masked_path = os.path.join(out_folder, base_name + "_MASKED_BGRNIR.tif")
        clustered_path = os.path.join(out_folder, base_name + "_CLUSTERS.tif")
        preview_png = os.path.join(out_folder, base_name + "_PREVIEW.png")

        # Save NDVI (float32)
        ndvi_meta = meta.copy()
        ndvi_meta.update(driver='GTiff', dtype='float32', count=1)
        with rasterio.open(ndvi_path, 'w', **ndvi_meta) as dst:
            dst.write(ndvi.astype('float32'), 1)

        # Save veg mask (uint8)
        mask_meta = meta.copy()
        mask_meta.update(driver='GTiff', dtype='uint8', count=1)
        with rasterio.open(vegmask_path, 'w', **mask_meta) as dst:
            dst.write(veg_mask.astype('uint8'), 1)

        # Save clustered image (int16)
        cl_meta = meta.copy()
        cl_meta.update(driver='GTiff', dtype='int16', count=1)
        with rasterio.open(clustered_path, 'w', **cl_meta) as dst:
            dst.write(clustered_full.astype('int16'), 1)

        # Save masked four-band image (Blue, Green, Red, NIR) - vegetation set to 0
        blue_masked = blue.copy()
        green_masked = green.copy()
        red_masked = red.copy()
        nir_masked = nir.copy()
        blue_masked[veg_mask] = 0.0
        green_masked[veg_mask] = 0.0
        red_masked[veg_mask] = 0.0
        nir_masked[veg_mask] = 0.0

        mb_meta = meta.copy()
        mb_meta.update(driver='GTiff', dtype='float32', count=4)
        with rasterio.open(masked_path, 'w', **mb_meta) as dst:
            dst.write(blue_masked.astype('float32'), 1)
            dst.write(green_masked.astype('float32'), 2)
            dst.write(red_masked.astype('float32'), 3)
            dst.write(nir_masked.astype('float32'), 4)

        # Create preview PNG and popup
        plot_aesthetic_preview_and_save(ndvi, veg_mask, clustered_full, K, ndvi_threshold, preview_png)

        return {
            "ndvi": ndvi_path,
            "vegmask": vegmask_path,
            "masked": masked_path,
            "clusters": clustered_path,
            "preview": preview_png
        }

    except Exception as exc:
        raise

# -----------------------
# GUI (full interface - Updated for 4-band)
# -----------------------
class FullGUI:
    def __init__(self, master):
        self.master = master
        master.title("NDVI mask + KMeans (Blue, Green, Red, NIR)")

        # Variables
        self.blue_var = StringVar()
        self.green_var = StringVar()
        self.red_var = StringVar()
        self.nir_var = StringVar()
        self.outdir_var = StringVar()
        self.ndvi_var = StringVar(value="0.2")
        self.k_var = StringVar(value="6")

        # Widgets - Blue
        Label(master, text="Select Blue band (B2) file:").grid(row=0, column=0, sticky='w', padx=6, pady=4)
        Entry(master, textvariable=self.blue_var, width=60).grid(row=0, column=1, padx=6)
        Button(master, text="Browse", command=self.browse_blue).grid(row=0, column=2, padx=6)

        # Widgets - Green
        Label(master, text="Select Green band (B3) file:").grid(row=1, column=0, sticky='w', padx=6, pady=4)
        Entry(master, textvariable=self.green_var, width=60).grid(row=1, column=1, padx=6)
        Button(master, text="Browse", command=self.browse_green).grid(row=1, column=2, padx=6)

        # Widgets - Red
        Label(master, text="Select Red band (B4) file:").grid(row=2, column=0, sticky='w', padx=6, pady=4)
        Entry(master, textvariable=self.red_var, width=60).grid(row=2, column=1, padx=6)
        Button(master, text="Browse", command=self.browse_red).grid(row=2, column=2, padx=6)

        # Widgets - NIR
        Label(master, text="Select NIR band (B5) file:").grid(row=3, column=0, sticky='w', padx=6, pady=4)
        Entry(master, textvariable=self.nir_var, width=60).grid(row=3, column=1, padx=6)
        Button(master, text="Browse", command=self.browse_nir).grid(row=3, column=2, padx=6)

        # Output folder
        Label(master, text="Output folder:").grid(row=4, column=0, sticky='w', padx=6, pady=4)
        Entry(master, textvariable=self.outdir_var, width=60).grid(row=4, column=1, padx=6)
        Button(master, text="Browse", command=self.browse_out).grid(row=4, column=2, padx=6)

        # NDVI threshold
        Label(master, text="NDVI threshold (mask if > threshold):").grid(row=5, column=0, sticky='w', padx=6, pady=4)
        Entry(master, textvariable=self.ndvi_var, width=20).grid(row=5, column=1, sticky='w', padx=6)

        # K value
        Label(master, text="K (number of clusters):").grid(row=6, column=0, sticky='w', padx=6, pady=4)
        Entry(master, textvariable=self.k_var, width=20).grid(row=6, column=1, sticky='w', padx=6)

        Button(master, text="Run (non-blocking)", bg='green', fg='white', command=self.run_thread).grid(row=7, column=1, pady=10)
        Button(master, text="Quit", command=master.quit).grid(row=7, column=2, pady=10)

    def browse_blue(self):
        p = filedialog.askopenfilename(title="Select Blue band (B2) TIFF", filetypes=[("TIFF", "*.tif *.tiff"), ("All files", "*.*")])
        if p:
            self.blue_var.set(p)

    def browse_green(self):
        p = filedialog.askopenfilename(title="Select Green band (B3) TIFF", filetypes=[("TIFF", "*.tif *.tiff"), ("All files", "*.*")])
        if p:
            self.green_var.set(p)

    def browse_red(self):
        p = filedialog.askopenfilename(title="Select Red band (B4) TIFF", filetypes=[("TIFF", "*.tif *.tiff"), ("All files", "*.*")])
        if p:
            self.red_var.set(p)

    def browse_nir(self):
        p = filedialog.askopenfilename(title="Select NIR band (B5) TIFF", filetypes=[("TIFF", "*.tif *.tiff"), ("All files", "*.*")])
        if p:
            self.nir_var.set(p)

    def browse_out(self):
        p = filedialog.askdirectory(title="Select output folder (will be created if not exist)")
        if p:
            self.outdir_var.set(p)

    def run_thread(self):
        t = threading.Thread(target=self.run, daemon=True)
        t.start()

    def run(self):
        try:
            blue_path = self.blue_var.get().strip()
            green_path = self.green_var.get().strip()
            red_path = self.red_var.get().strip()
            nir_path = self.nir_var.get().strip()
            outdir = self.outdir_var.get().strip()
            ndvi_thr = float(self.ndvi_var.get())
            K = int(self.k_var.get())

            if not blue_path or not green_path or not red_path or not nir_path:
                messagebox.showerror("Input error", "Please select Blue, Green, Red and NIR TIFF files.")
                return
            if not outdir:
                messagebox.showerror("Output error", "Please select an output folder.")
                return
            if K <= 0:
                messagebox.showerror("Input error", "K must be > 0.")
                return

            # Run processing
            messagebox.showinfo("Processing", "Processing started. This runs in background; you will be notified when done.")
            try:
                results = process_and_save(blue_path, green_path, red_path, nir_path, outdir, ndvi_thr, K, downsample_target=1000)
            except Exception as e:
                messagebox.showerror("Processing error", str(e))
                return

            # Success
            msg = "Processing complete. Files saved:\n\n"
            msg += "\n".join(f"{k}: {v}" for k, v in results.items())
            messagebox.showinfo("Done", msg)

        except Exception as e:
            messagebox.showerror("Error", str(e))

# -----------------------
# Run GUI
# -----------------------
if __name__ == "__main__":
    root = Tk()
    gui = FullGUI(root)
    root.mainloop()
