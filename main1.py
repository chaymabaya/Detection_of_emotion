import cv2
import numpy as np
import time
from collections import deque
from tensorflow.keras.models import load_model

# ========= Config =========
EMO_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']  # 7 classes
AGE_BINS = [(0,9),(10,19),(20,29),(30,39),(40,49),(50,59),(60,69),(70,79),(80,89),(90,120)]
AGE_BIN_MEAN = np.array([4, 15, 25, 35, 45, 55, 65, 75, 85, 95], dtype=np.float32)

# Confiance (émotions)
CONF_MIN_BASE = 0.40          # seuil général
CONF_MIN_SURPRISE = 0.30      # seuil plus bas pour surprise
SURPRISE_GAIN = 1.15          # boost léger pour "surprise"
SURPRISE_MARGIN = 0.05        # marge min. vs 2e meilleure classe

# Mode SENIOR (gating)
SENIOR_ONLY = True            # n'afficher l'émotion que pour les seniors
SENIOR_MIN_AGE = 50           # seuil d'âge "senior"
SENIOR_STRICT = False         # si True: exige aussi proba(>=50) suffisante
SENIOR_PROB_MIN = 0.45        # utilisé si SENIOR_STRICT=True

# Stabilité âge
AGE_WINDOW = 15               # médiane sur 15 frames
AGE_HOLD_FRAMES = 20          # hold (verrou) durant N frames
AGE_CHANGE_TOL = 6            # tolérance (années) avant changement

# Lissage proba
SMOOTH_ALPHA = 0.7            # EMA fort

# Vote émotion
EMO_VOTE_WINDOW = 12          # vote majoritaire sur 12 frames

# Debug overlay
SHOW_DEBUG_OVERLAY = True
CONSOLE_DEBUG_EVERY_S = 0.5

# I/O
MODEL_PATH = "age_emotion_best.h5"
CAM_INDEX = 0
FRAME_W, FRAME_H = 1280, 720
INPUT_SIZE = 224
USE_GRAYSCALE_CLAHE = True
FACE_MARGIN = 0.25

# ========= Modèle =========
model = load_model(MODEL_PATH)
print("Output shapes:", [m.shape for m in model.outputs])

# ========= OpenCV =========
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

# ========= Utilitaires =========
def crop_with_margin(frame, x, y, w, h, margin=FACE_MARGIN):
    H, W = frame.shape[:2]
    mx = int(w * margin); my = int(h * margin)
    x1 = max(0, x - mx); y1 = max(0, y - my)
    x2 = min(W, x + w + mx); y2 = min(H, y + h + my)
    return frame[y1:y2, x1:x2]

def preprocess_face(bgr, size=INPUT_SIZE):
    if USE_GRAYSCALE_CLAHE:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        eq = clahe.apply(gray)
        rgb = cv2.cvtColor(eq, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size))
    x = rgb.astype(np.float32) / 255.0
    return np.expand_dims(x, axis=0)

def route_outputs(raw_preds):
    if isinstance(raw_preds, (list, tuple)) and len(raw_preds) == 2:
        p0, p1 = raw_preds
        d0 = int(np.asarray(p0).shape[-1]); d1 = int(np.asarray(p1).shape[-1])
        if d0 == len(EMO_LABELS) and (d1 == len(AGE_BINS) or d1 == 1): return p0, p1
        if d1 == len(EMO_LABELS) and (d0 == len(AGE_BINS) or d0 == 1): return p1, p0
        return (p0, p1) if abs(d0 - len(EMO_LABELS)) <= abs(d1 - len(EMO_LABELS)) else (p1, p0)
    return raw_preds, np.array([[65.0]])  # mono-sortie: émotions + âge simulé

def softmax(x):
    x = np.asarray(x, dtype=np.float32)
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)

def decode_age(age_pred):
    age_pred = np.asarray(age_pred)
    last = age_pred.shape[-1] if age_pred.ndim >= 2 else 1
    if last == len(AGE_BINS):  # classes
        probs = age_pred.astype(np.float32)
        if probs.min() < 0 or not np.allclose(probs.sum(axis=-1), 1.0, atol=1e-3):
            probs = softmax(probs)
        exp_age = float(np.sum(probs * AGE_BIN_MEAN))
        cls = int(np.argmax(probs))
        return int(round(exp_age)), cls, probs.squeeze()
    elif last == 1:            # régression
        val = float(age_pred.squeeze())
        return int(round(val)), None, None
    else:                       # fallback
        cls = int(np.argmax(age_pred))
        cls = max(0, min(cls, len(AGE_BIN_MEAN)-1))
        return int(AGE_BIN_MEAN[cls]), cls, None

def decode_emotion(emo_pred):
    probs = np.asarray(emo_pred, dtype=np.float32)
    if probs.min() < 0 or not np.allclose(probs.sum(axis=-1), 1.0, atol=1e-3):
        probs = softmax(probs)
    probs = probs.squeeze().astype(np.float32)
    if probs.ndim == 0:
        probs = np.array([1.0], dtype=np.float32)
    # boost "surprise"
    if len(probs) == len(EMO_LABELS):
        s_idx = EMO_LABELS.index('surprise')
        probs_adj = probs.copy()
        probs_adj[s_idx] *= SURPRISE_GAIN
        probs_adj = probs_adj / probs_adj.sum()
    else:
        probs_adj = probs
    idx = int(np.argmax(probs_adj))
    conf = float(np.max(probs_adj))
    return idx, conf, probs_adj

def majority_vote(labels, num_classes):
    if not labels:
        return None
    counts = np.bincount(labels, minlength=num_classes)
    return int(np.argmax(counts))

def draw_debug_overlay(frame, displayed_age, prob_50plus, is_senior):
    x0, y0 = 10, 20; line_h = 22
    lines = [
        "MODE: Seniors 50+",
        f"Âge affiché: {displayed_age}",
        f"P(age≥50): {prob_50plus:.2f}" if prob_50plus is not None else "P(age≥50): N/A",
        "Gating: SENIOR" if is_senior else "Gating: <50 (bloqué)"
    ]
    for i, txt in enumerate(lines):
        cv2.putText(frame, txt, (x0, y0 + i*line_h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

# ========= Boucle principale =========
def main():
    if not cap.isOpened():
        raise SystemExit("[ERREUR] Caméra introuvable. Modifie CAM_INDEX.")
    prev_age_probs = None; prev_emo_probs = None
    age_buffer = deque(maxlen=AGE_WINDOW); emo_label_buffer = deque(maxlen=EMO_VOTE_WINDOW)
    age_hold_counter = 0; displayed_age = None
    num_classes = len(EMO_LABELS); last_dbg_ts = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(64, 64))

        if len(faces) == 0:
            prev_age_probs = prev_emo_probs = None
            age_buffer.clear(); emo_label_buffer.clear()
            age_hold_counter = 0; displayed_age = None

        for (x, y, w, h) in faces:
            roi0 = crop_with_margin(frame, x, y, w, h, FACE_MARGIN)
            x_in = preprocess_face(roi0, INPUT_SIZE)

            raw_preds = model.predict(x_in, verbose=0)
            emo_pred, age_pred = route_outputs(raw_preds)

            # ---- AGE ----
            age_val_raw, age_cls, age_probs = decode_age(age_pred)
            if age_probs is not None:
                prev_age_probs = age_probs if prev_age_probs is None else SMOOTH_ALPHA*prev_age_probs + (1-SMOOTH_ALPHA)*age_probs
                age_val = int(round(float(np.sum(prev_age_probs * AGE_BIN_MEAN))))
            else:
                age_val = age_val_raw

            age_buffer.append(age_val)
            median_age = int(np.median(age_buffer)) if len(age_buffer) > 0 else age_val

            if displayed_age is None:
                displayed_age = median_age; age_hold_counter = AGE_HOLD_FRAMES
            else:
                if age_hold_counter > 0:
                    if abs(median_age - displayed_age) > AGE_CHANGE_TOL:
                        displayed_age = median_age; age_hold_counter = AGE_HOLD_FRAMES
                    else:
                        age_hold_counter -= 1
                else:
                    if abs(median_age - displayed_age) > AGE_CHANGE_TOL:
                        displayed_age = median_age; age_hold_counter = AGE_HOLD_FRAMES

            # ---- Gating SENIOR ----
            prob_50plus = float(np.sum(prev_age_probs[5:])) if prev_age_probs is not None else None
            is_senior = True
            if SENIOR_ONLY:
                is_senior = (displayed_age >= SENIOR_MIN_AGE)
                if SENIOR_STRICT and prob_50plus is not None:
                    is_senior = is_senior and (prob_50plus >= SENIOR_PROB_MIN)

            now = time.time()
            if SHOW_DEBUG_OVERLAY and (now - last_dbg_ts) >= CONSOLE_DEBUG_EVERY_S:
                print(f"[DBG] age_aff={displayed_age} | P(age>=50)={prob_50plus if prob_50plus is not None else 'N/A'} | senior={is_senior}")
                last_dbg_ts = now

            if not is_senior:
                prev_emo_probs = None; emo_label_buffer.clear()
                label = f"Age: {displayed_age} | analyse reserve 50+"
                color = (180, 180, 180)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                y_text = y - 10 if y - 10 > 20 else y + h + 20
                cv2.putText(frame, label, (x, y_text),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
                if SHOW_DEBUG_OVERLAY:
                    draw_debug_overlay(frame, displayed_age, prob_50plus, is_senior)
                continue

            # ---- EMOTION (pour seniors) ----
            emo_idx_raw, emo_conf_raw, emo_probs = decode_emotion(emo_pred)
            prev_emo_probs = emo_probs if prev_emo_probs is None else SMOOTH_ALPHA*prev_emo_probs + (1-SMOOTH_ALPHA)*emo_probs

            # Re-boost surprise après EMA
            if len(prev_emo_probs) == num_classes:
                s_idx = EMO_LABELS.index('surprise')
                probs_adj = prev_emo_probs.copy()
                probs_adj[s_idx] *= SURPRISE_GAIN
                probs_adj = probs_adj / probs_adj.sum()
            else:
                probs_adj = prev_emo_probs

            emo_idx = int(np.argmax(probs_adj))
            emo_conf = float(np.max(probs_adj))

            emo_label_buffer.append(emo_idx)
            voted_idx = majority_vote(list(emo_label_buffer), num_classes)
            if voted_idx is not None:
                emo_idx = voted_idx
                emo_conf = float(probs_adj[emo_idx])

            emo_lbl = EMO_LABELS[emo_idx] if emo_idx < num_classes else f"class_{emo_idx}"
            conf_threshold = CONF_MIN_SURPRISE if emo_lbl == 'surprise' else CONF_MIN_BASE

            if emo_lbl == 'surprise' and len(probs_adj) == num_classes:
                sorted_probs = np.sort(probs_adj)
                second_best = float(sorted_probs[-2]) if len(sorted_probs) >= 2 else 0.0
                if emo_conf < (second_best + SURPRISE_MARGIN):
                    label = f"Age: {displayed_age} | surprise (?) ({emo_conf:.2f})"
                    color = (0, 165, 255)
                else:
                    label = f"Age: {displayed_age} | surprise ({emo_conf:.2f})"
                    color = (0, 200, 0)
            else:
                if emo_conf >= conf_threshold:
                    label = f"Age: {displayed_age} | {emo_lbl} ({emo_conf:.2f})"
                    color = (0, 200, 0)
                else:
                    label = f"Age: {displayed_age} | émotion incertaine ({emo_conf:.2f})"
                    color = (0, 165, 255)

            # ---- Dessin ----
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            y_text = y - 10 if y - 10 > 20 else y + h + 20
            cv2.putText(frame, label, (x, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
            if SHOW_DEBUG_OVERLAY:
                draw_debug_overlay(frame, displayed_age, prob_50plus, True)

        cv2.imshow("Émotions Seniors (50+)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
