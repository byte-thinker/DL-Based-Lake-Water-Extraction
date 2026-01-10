import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.nn as nn
import numpy as np
import rioxarray as rio
import matplotlib.pyplot as plt


# -------------------------------------
# Model definition
# -------------------------------------
class DenseNetConvNet(nn.Module):
    """
    A DenseNet-style fully convolutional network
    for pixel-wise classification.
    """
    def __init__(self, n_channels, n_classes):
        super(DenseNetConvNet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=n_channels,
            out_channels=16,
            kernel_size=3,
            padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=16 + n_channels,
            out_channels=32,
            kernel_size=3,
            padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=32 + 16 + n_channels,
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        self.conv4 = nn.Conv2d(
            in_channels=64 + 32 + 16 + n_channels,
            out_channels=n_classes,
            kernel_size=1
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x1_cat = torch.cat((x, x1), dim=1)

        x2 = self.relu(self.conv2(x1_cat))
        x2_cat = torch.cat((x1_cat, x2), dim=1)

        x3 = self.relu(self.conv3(x2_cat))
        x3_cat = torch.cat((x2_cat, x3), dim=1)

        x4 = self.conv4(x3_cat)
        return x4


# -------------------------------------
# Model loading
# -------------------------------------
def load_model(model_path, model, device=None):
    """
    Load a pretrained model checkpoint.
    """
    if device is None:
        device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

    checkpoint = torch.load(
        model_path,
        map_location=device
    )
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)

    return model, device


# -------------------------------------
# Inference function
# -------------------------------------
def model_predict(net, img, device, scale=10000):
    """
    Perform model inference on a multi-band raster image.
    """
    net.eval()

    image = np.expand_dims(img.values, axis=0) / scale
    img_tensor = torch.from_numpy(image).float().to(device)

    with torch.no_grad():
        outputs = net(img_tensor)
        preds = torch.sigmoid(outputs).squeeze(0).cpu().numpy()

    prediction = (preds >= 0.5).astype(np.uint8)

    # ---- Visualization ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    if img.sizes['band'] >= 3:
        rgb = img.isel(band=[0, 1, 2]).values.astype(np.float32)
        rgb = rgb / np.percentile(rgb, 98)
        rgb = np.clip(rgb, 0, 1)
        ax1.imshow(np.transpose(rgb, (1, 2, 0)))
    else:
        ax1.imshow(img.isel(band=0), cmap='gray')

    ax1.set_title('Original Image')
    ax2.imshow(prediction[0], cmap='gray')
    ax2.set_title('Predicted Water Mask')
    plt.show()

    # ---- Output raster ----
    pred_da = img.isel(band=0).copy()
    pred_da.values = prediction[0].astype(np.uint8)
    pred_da.rio.write_nodata(0, inplace=True)

    return pred_da


# -------------------------------------
# Wrapper for prediction
# -------------------------------------
def predict_image(net, image_path, device, scale=10000):
    img = rio.open_rasterio(image_path)
    pred = model_predict(net, img, device, scale)
    pred.attrs['long_name'] = 'Water body prediction'
    return pred

# -------------------------------------
# Example usage (paths are placeholders)
# -------------------------------------
if __name__ == "__main__":

    # Replace with paths to your own data and model
    image_path = "example.tif"
    model_path = "model.pth"

    net = DenseNetConvNet(
        n_channels=6,
        n_classes=1
    )

    net, device = load_model(
        model_path,
        net
    )

    prediction = predict_image(
        net,
        image_path,
        device,
        scale=10000
    )

    prediction.rio.to_raster(
        "pred.tif",
        compress="lzw"
    )
    print("Prediction saved to pred.tif")