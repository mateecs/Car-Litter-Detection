from ultralytics import YOLO

model = YOLO('models/best.pt')

results  = model.predict('input video/08fd33_4.mp4', save = True)
results(results[0])