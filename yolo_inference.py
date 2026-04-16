from ultralytics import YOLO

model = YOLO('yolov8l')

results = model.predict('input_videos/08fd33_4.mp4', save=True)
print(results[0])
print("=========================================================")
for box in results[0].boxes:
    print(box)


# import torch

# print(torch.cuda.is_available())   # True = có GPU
# print(torch.cuda.current_device())
# print(torch.cuda.get_device_name(0))