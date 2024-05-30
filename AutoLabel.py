import torch
import os
from pathlib import Path
from PIL import Image
import numpy as np
from utils.general import non_max_suppression, scale_coords, xyxy2xywh
from utils.datasets import letterbox

# Paths
image_folder = Path(r'Jnx03\DSC07001-9000')
label_folder = Path(r'Jnx03\label') #Save Label folder

label_folder.mkdir(parents=True, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load('best.pt', map_location=device)['model'].float()
model.to(device).eval()

for image_path in image_folder.glob('*.jpg'):
    img = Image.open(image_path)
    img = np.array(img)
    original_shape = img.shape[:2]
    img, ratio, (dw, dh) = letterbox(img, new_shape=640, auto=True)
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img)[0]
    pred = non_max_suppression(pred, 0.25, 0.45, agnostic=False)

    for i, det in enumerate(pred):
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], original_shape).round()

            labels = []
            for *xyxy, conf, cls in det:
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()
                xywh[0] /= original_shape[1]
                xywh[1] /= original_shape[0]
                xywh[2] /= original_shape[1]
                xywh[3] /= original_shape[0]
                labels.append(f"{int(cls)} " + " ".join(map(lambda x: f"{x:.6f}", xywh)))

            label_path = label_folder / f"{image_path.stem}.txt"
            with open(label_path, 'w') as f:
                f.write("\n".join(labels))

print("Labeling complete.")
