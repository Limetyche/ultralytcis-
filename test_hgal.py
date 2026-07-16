from ultralytics import YOLO

model = YOLO("/content/drive/MyDrive/ultralytics/ultralytics/cfg/models/v8-hgal/yolov8n-hgal.yaml")
head = model.model.model[-1]

print("head:", type(head).__name__)
print("use_hypergraph:", head.use_hypergraph)
print("aux_hyper:", type(head.aux_hyper).__name__)
print("aux_weight:", head.aux_weight)
print("aux_stride:", head.aux_stride)
print("criterion:", type(model.model.init_criterion()).__name__)