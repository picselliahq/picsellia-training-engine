from ultralytics import YOLO

# Load a model
model = YOLO("/Users/alexis/Downloads/yolov8n-seg.pt")  # pretrained YOLOv8n model

# Run batched inference on a list of images
results = model(
    ["/Users/alexis/Downloads/AP7S7SPQ2NJTJLBWW7ROT6W3VQ.jpg"]
)  # return a list of Results objects

model.track(source=1, imgsz=640, save=False, show=True)
# # Process results list
# for result in results:
#     boxes = result.boxes  # Boxes object for bounding box outputs
#     masks = result.masks  # Masks object for segmentation masks outputs
#     keypoints = result.keypoints  # Keypoints object for pose outputs
#     probs = result.probs  # Probs object for classification outputs
#     obb = result.obb  # Oriented boxes object for OBB outputs
#     result.show()  # display to screen
#     result.save(filename="result.jpg")  # save to disk
