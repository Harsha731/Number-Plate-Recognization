from ultralytics import YOLO

model = YOLO(r"C:\Users\Dell\Desktop\CV_Project\runs\detect\train11\weights\last.pt")

results = model.train(data="data.yaml", epochs=100)