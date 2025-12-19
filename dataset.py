from ultralytics import YOLO

# Load mô hình gốc
model = YOLO('yolov8n.pt')  # nhẹ, nhanh

# Huấn luyện với dataset trái cây
model.train(
    data='Nhandientraicay/data.yaml', 
    epochs=50, 
    imgsz=640
)
