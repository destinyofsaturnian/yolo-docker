import onnxruntime as ort
import cv2
import numpy as np
import time
import os

# =========================
# 設定
# =========================
MODEL_PATH = "yolov8n.onnx"
IMAGE_DIR = "images"
OUTPUT_DIR = "output"
CONF_TH = 0.5

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# 前処理
# =========================
def preprocess(img):
    h0, w0 = img.shape[:2]

    img_resized = cv2.resize(img, (640, 640))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    img_norm = img_rgb.astype(np.float32) / 255.0
    img_chw = np.transpose(img_norm, (2, 0, 1))
    input_tensor = np.expand_dims(img_chw, axis=0)

    return input_tensor, h0, w0

# =========================
# 後処理
# =========================
def postprocess(outputs, h0, w0):
    output = outputs[0]
    output = np.transpose(output, (0, 2, 1))[0]

    boxes, scores, class_ids = [], [], []

    for row in output:
        x, y, w, h = row[:4]
        class_scores = row[4:]

        class_id = np.argmax(class_scores)
        score = class_scores[class_id]

        if score > CONF_TH:
            boxes.append([x, y, w, h])
            scores.append(float(score))
            class_ids.append(class_id)

    # xywh → xyxy
    boxes_xyxy = []
    for x, y, w, h in boxes:
        boxes_xyxy.append([x - w/2, y - h/2, x + w/2, y + h/2])

    # NMS
    indices = cv2.dnn.NMSBoxes(boxes_xyxy, scores, CONF_TH, 0.5)
    indices = indices.flatten() if len(indices) > 0 else []

    final = []
    for i in indices:
        final.append((boxes_xyxy[i], scores[i], class_ids[i]))

    # 元サイズに戻す
    scale_x = w0 / 640
    scale_y = h0 / 640

    scaled = []
    for box, score, cls in final:
        x1, y1, x2, y2 = box
        x1 *= scale_x
        x2 *= scale_x
        y1 *= scale_y
        y2 *= scale_y
        scaled.append(([int(x1), int(y1), int(x2), int(y2)], score, cls))

    return scaled

# =========================
# メイン処理
# =========================
session = ort.InferenceSession(MODEL_PATH)

image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg")]

total_time = 0

for img_name in image_files:
    img_path = os.path.join(IMAGE_DIR, img_name)
    img = cv2.imread(img_path)

    input_tensor, h0, w0 = preprocess(img)

    # FPS測定
    start = time.time()
    outputs = session.run(None, {"images": input_tensor})
    end = time.time()

    infer_time = end - start
    total_time += infer_time

    results = postprocess(outputs, h0, w0)

    # 描画
    for (box, score, cls) in results:
        x1, y1, x2, y2 = box

        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(img, f"{cls}:{score:.2f}",
                    (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0,255,0), 2)

    # 保存
    out_path = os.path.join(OUTPUT_DIR, img_name)
    cv2.imwrite(out_path, img)

    print(f"[{img_name}] Det: {len(results)}  FPS: {1/infer_time:.2f}")

# 全体FPS
avg_fps = len(image_files) / total_time
print(f"\n=== AVG FPS: {avg_fps:.2f} ===")