import numpy as np
from keras.utils import load_img
from keras_cv import visualization
from keras_cv.models import YOLOV8Detector
from keras_cv.layers import Resizing, NonMaxSuppression

class_ids = [
    "Aeroplane", "Bicycle", "Bird", "Boat", "Bottle",
    "Bus", "Car", "Cat", "Chair", "Cow", "Dining Table",
    "Dog", "Horse", "Motorbike", "Person", "Potted Plant",
    "Sheep", "Sofa", "Train", "Tvmonitor", "Total",
]
class_mapping = dict(zip(range(len(class_ids)), class_ids))
# 載入測試圖片
img = np.array(load_img("people.jpg"))
inference_resizing = Resizing(
    640, 640, pad_to_aspect_ratio=True, 
    bounding_box_format="xywh"
)
resize_img = inference_resizing([img])
visualization.plot_image_gallery(
    np.array([img]),
    value_range=(0, 255),
    rows=1,
    cols=1,
    scale=5,
)
# 建立NonMaxSuppression物件
prediction_decoder = NonMaxSuppression(
    bounding_box_format="xywh",
    from_logits=True,
    iou_threshold=1.0,
    confidence_threshold=0.7,
)
# 建立YOLO模型
model = YOLOV8Detector.from_preset(
    "yolo_v8_m_pascalvoc",
    bounding_box_format="xywh",
    prediction_decoder=prediction_decoder
)
# 使用模型進行預測
y_pred = model.predict(resize_img)
# 繪出模型的偵測結果
visualization.plot_bounding_box_gallery(
    resize_img,
    value_range=(0, 255),
    rows=1,
    cols=1,
    y_pred=y_pred,
    scale=5,
    font_scale=0.7,
    bounding_box_format="xywh",
    class_mapping=class_mapping,
)

