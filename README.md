# GNR607 – NDVI & K-Means Clustering

## Project Objective
Compute NDVI to isolate vegetation using a user-defined threshold, then apply K-Means clustering on non-vegetated areas using 4 spectral bands (B2–B5).

---

## Input Data
- **B2 (Blue), B3 (Green), B4 (Red), B5 (NIR)**: Georeferenced TIFFs, same scene & aligned (CRS, transform, resolution, dimensions).  
- **Output Folder**: Directory to save results.

---

## User Parameters
- **NDVI Threshold**: Float; separates vegetation from non-vegetation.
    ```
    NDVI = (NIR - Red) / (NIR + Red)
    vegetation = NDVI > threshold
    ```
- **K (Clusters)**: Integer; number of K-Means clusters (excluding vegetation).  


---

## Outputs
| File | Description |
|------|-------------|
| `_NDVI.tif` | NDVI raster (float32). |
| `_VEGMASK.tif` | Binary mask (1=vegetation, 0=non-vegetation). |
| `_MASKED_4BANDS.tif` | Four-band raster with vegetation pixels set to 0. |
| `_CLUSTERS.tif` | K-Means clusters; vegetation = -1, others = 0 to K-1. |
| `_PREVIEW.png` | Visualization: vegetation in green + cluster map. |

---

## Workflow
1. Load B2–B5 bands.  
2. Compute NDVI and vegetation mask.  
3. Downsample bands & mask for efficiency.  
4. Extract non-vegetation pixels, normalize features `[B2,B3,B4,B5]`.  
5. Apply K-Means clustering.  
6. Upsample clusters, restore vegetation as -1.  
7. Save outputs + preview.

---

## Summary
- Accurate vegetation extraction using NDVI.  
- Efficient 4-band K-Means clustering on non-vegetated areas.  
- Outputs include raster maps and a preview for quick verification.  

