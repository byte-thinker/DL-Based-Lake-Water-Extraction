# Deep Learning Water Body Extraction

This repository provides example code for **pixel-wise water body extraction** using a DenseNet-style fully convolutional neural network.  
It includes training and inference scripts, along with example images and labels to demonstrate the required input format.

---

## Requirements

- Python >= 3.9
- PyTorch
- numpy
- pandas
- matplotlib
- rasterio
- rioxarray

## Input Data Format

The model expects **six Sentinel-2 bands** as input in the following order:

```
'B2', 'B3', 'B4', 'B8', 'B11', 'B12'
```

Input images should be multi-band rasters containing these six bands in this order. Pixel values are normalized by 10000 in the scripts.

## CSV File Format

The CSV file lists input images and corresponding labels. Example:

```
scene_file	truth_file
/path/to/your/test/img/Img001.tif	/path/to/your/test/label/Img001.tif
/path/to/your/test/img/Img002.tif	/path/to/your/test/label/Img002.tif
```

- `scene_file` → path to input image
- `truth_file` → path to corresponding label
- Users can prepare their own datasets following the **format shown in the `example/` folder** (example images are 512×512, but users can adjust the size as needed).

## Training

Run the training script:

```
python train_model.py
```

- Uses `train_files.csv` and `val_files.csv` to load data
- Outputs:
  - Model
  - Training metrics
  - Loss curve visualization

## Inference / Prediction

Run the inference script:

```
python model_inference.py
```

- Uses a trained model and an input raster image
- Outputs:
  - Predicted water mask (results/pred.tif)
  - Visualization of original image and predicted mask

## Visualization

- `visualize_sentinel2_example.ipynb` → Visualize example Sentinel-2 images and labels
- `visualize_prediction_result.ipynb` → Visualize prediction results and compare with original images

## License

This repository is licensed under the **MIT License**. 