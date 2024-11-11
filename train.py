from ultralytics import YOLO
# import torch
# torch.cuda.empty_cache()

# Load a model
model = YOLO('yolo11.yaml')  # build a new model from YAML
model = YOLO('yolo11n.pt')  # load a pretrained model (recommended for training)
model = YOLO('yolo11.yaml').load('yolo11n.pt')  # build from YAML and transfer weights

# Train the model
model.train(data='data.yaml', epochs=100, imgsz=[1024,1920],batch=16)
