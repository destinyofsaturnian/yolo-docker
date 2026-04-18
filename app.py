import os
import time
import logging
import numpy as np
import cv2
import onnxruntime as ort

# =========================
# 設定
# =========================
MODEL_PATH = "yolov8s.onnx"
IMAGE_DIR = "images"
OUTPUT_DIR = "output"

CONF_TH = 0.5
NMS_TH = 0.5

# =========================
# ログ設定
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# =========================
# 初期化
# =========================
def init_session(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    try:
        session = ort.InferenceSession(model_path)
        return session
    except Exception as e:
        raise RuntimeError(f"ONNX load failed: {e}")

# =========================
# 入力
# =========================
def load_images(image_dir):
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Image dir not found: {image_dir}")

    files = [f for f in os.listdir(image_dir)
             if f.lower().endswith((".jpg", ".png"))]

    if len(files) == 0:
        raise RuntimeError("No images found")

    return files

def read_image(path):
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return img

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

    return input_tensor, (h0, w0)

# =========================
# 推論
# =========================
def inference(session, input_tensor):
    try:
        outputs = session.run(None, {"images": input_tensor})
        return outputs[0]
    except Exception as e:
        raise RuntimeError(f"Inference failed: {e}")

# =========================
# 後処理
# =========================
def postprocess(output, orig_shape):
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

    if len(boxes) == 0:
        return [], [], []

    boxes_xyxy = []
    for box in boxes:
        x, y, w, h = box
        x1 = x - w/2
        y1 = y - h/2
        x2 = x + w/2
        y2 = y + h/2
        boxes_xyxy.append([x1, y1, x2, y2])

    indices = cv2.dnn.NMSBoxes(
        boxes_xyxy,
        scores,
        score_threshold=CONF_TH,
        nms_threshold=NMS_TH
    )

    if len(indices) == 0:
        return [], [], []

    indices = indices.flatten()

    h0, w0 = orig_shape
    scale_x = w0 / 640
    scale_y = h0 / 640

    final_boxes, final_scores, final_classes = [], [], []

    for i in indices:
        x1, y1, x2, y2 = boxes_xyxy[i]

        x1 *= scale_x
        x2 *= scale_x
        y1 *= scale_y
        y2 *= scale_y

        final_boxes.append([int(x1), int(y1), int(x2), int(y2)])
        final_scores.append(scores[i])
        final_classes.append(class_ids[i])

    return final_boxes, final_scores, final_classes

# =========================
# 出力
# =========================
def draw(img, boxes, scores, classes):
    for box, score, cls in zip(boxes, scores, classes):
        x1, y1, x2, y2 = box

        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(img, f"{cls}:{score:.2f}",
                    (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0,255,0), 2)

    return img

def save_image(img, filename):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, filename)

    if not cv2.imwrite(out_path, img):
        raise RuntimeError(f"Failed to save: {out_path}")

# =========================
# ログ
# =========================
def log_result(name, det_count, fps):
    logging.info(f"[{name}] Det: {det_count} FPS: {fps:.2f}")

# =========================
# メイン
# =========================
def main():
    session = init_session(MODEL_PATH)
    files = load_images(IMAGE_DIR)

    fps_list = []

    for file in files:
        path = os.path.join(IMAGE_DIR, file)

        try:
            img = read_image(path)

            start = time.time()

            input_tensor, shape = preprocess(img)
            output = inference(session, input_tensor)
            boxes, scores, classes = postprocess(output, shape)

            result_img = draw(img, boxes, scores, classes)
            save_image(result_img, file)

            end = time.time()

            fps = 1 / (end - start)
            fps_list.append(fps)

            log_result(file, len(boxes), fps)

        except Exception as e:
            logging.error(f"[{file}] ERROR: {e}")
            continue

    if len(fps_list) > 0:
        avg_fps = sum(fps_list) / len(fps_list)
        logging.info(f"=== AVG FPS: {avg_fps:.2f} ===")

if __name__ == "__main__":
    main()

