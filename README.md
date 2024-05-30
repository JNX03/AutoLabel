# AutoLabelX
##### AutoLabel Image as Txt on Yolov7/Yolov5 Using Pre-Train Model

---
## You NEED a pre-train model or train model to use
### Setting

```bash
# Paths
image_folder = Path(r'Jnx03\DSC07001-9000') #Image you want to label
label_folder = Path(r'Jnx03\label') #Save Label folder

# Model
model = torch.load('best.pt', map_location=device)['model'].float() #Change Best.pt to you pre-train model
```

### Instructions

1. **Install PyTorch and other dependencies**:
   ```bash
   pip install torch torchvision pillow numpy
   ```

2. **Download and set up YOLOv7**:
   ```bash
   git clone https://github.com/WongKinYiu/yolov7
   cd yolov7
   ```

3. **Place `best.pt` in the YOLOv7 directory**.

4. **Run the script**:
   Save the script above in a Python file, for example `auto_label.py`, and run it:
   ```bash
   python AutoLabel.py
   ```

