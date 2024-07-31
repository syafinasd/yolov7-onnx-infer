from ultralytics import YOLOv10
import onnxruntime as ort

model = YOLOv10("best_10N.onnx")

model.predict(source="0", show=True, conf=0.25, save=False)
