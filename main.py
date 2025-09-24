import cv2
import numpy as np
import time, os
from collections import deque
from tensorflow.keras.models import load_model


EMO_LABELS = ['angry','disgust','fear','happy','neutral','sad','surprise']
AGE_BINS = [(0,9),(10,19),(20,29),(30,39),(40,49),(50,59),(60,69),(70,79),(80,89),(90,120)]
AGE_BIN_MEAN = np.array([4,15,25,35,45,55,65,75,85,95], dtype=np.float32)

CONF_MIN_BASE = 0.45
CONF_MIN_SURPRISE = 0.50          
SURPRISE_MARGIN = 0.12            
EMO_LOGIT_BIAS = {                
    'surprise': -0.35,
    'happy':    +0.12,
    'sad':      +0.08,
    'neutral':  +0.05,
    'angry':    +0.00,
    'disgust':  +0.00,
    'fear':     +0.00
}


SENIOR_ONLY = True
SENIOR_MIN_AGE = 50
SENIOR_STRICT = True               
SENIOR_PROB_MIN = 0.50


AGE_WINDOW = 15
AGE_HOLD_FRAMES = 20
AGE_CHANGE_TOL = 6
SMOOTH_ALPHA = 0.7                 
EMO_VOTE_WINDOW = 12


INPUT_SOURCE = "video"            
VIDEO_PATH = r"C:\Users\pc\Desktop\detection_of_emotions\WhatsApp Video 2025-09-16 at 13.55.17_23c3cc96.mp4"
SAVE_OUTPUT = True
OUTPUT_PATH = "annotated_output.mp4"

FRAME_W, FRAME_H = 1280, 720
INPUT_SIZE = 224
USE_GRAYSCALE_CLAHE = True
FACE_MARGIN = 0.25
MODEL_PATH = "age_emotion_best.h5"


SHOW_DEBUG_OVERLAY = True
CONSOLE_DEBUG_EVERY_S = 0.5


FACE_DETECTOR = "auto"  

DNN_FACE_PROTO = "deploy.prototxt"
DNN_FACE_MODEL = "res10_300x300_ssd_iter_140000.caffemodel"
DNN_CONF_THRESH = 0.60
MIN_FACE_SIZE_PX = 120 


WRINKLE_AID_ENABLED = True
WRINKLE_KERNEL = 3
WRINKLE_THRESH = 55.0   
AGE_SENIOR_BUMP = 1.10 


model = load_model(MODEL_PATH)
print("Output shapes:", [m.shape for m in model.outputs])


use_dnn = False
if FACE_DETECTOR in ("dnn","auto"):
    if os.path.exists(DNN_FACE_PROTO) and os.path.exists(DNN_FACE_MODEL):
        try:
            dnn_net = cv2.dnn.readNetFromCaffe(DNN_FACE_PROTO, DNN_FACE_MODEL)
            use_dnn = True
        except Exception as e:
            print(f"[WARN] DNN face init failed: {e} -> fallback Haar")
    else:
        if FACE_DETECTOR=="dnn":
            print("[WARN] DNN files not found -> fallback Haar")

haar = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def open_capture():
    if INPUT_SOURCE.lower()=="camera":
        cap = cv2.VideoCapture(CAM_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    else:
        if not os.path.exists(VIDEO_PATH):
            raise SystemExit(f"[ERREUR] Fichier vidéo introuvable: {VIDEO_PATH}")
        cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise SystemExit("[ERREUR] Impossible d'ouvrir la source vidéo/caméra.")
    return cap

cap = open_capture()
writer = None
if SAVE_OUTPUT:
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or FRAME_W)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or FRAME_H)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (w, h))
    if not writer.isOpened():
        print("[WARN] Writer KO -> SAVE_OUTPUT=False")
        writer = None


def crop_with_margin(frame, x, y, w, h, margin=FACE_MARGIN):
    H,W = frame.shape[:2]
    mx = int(w*margin); my=int(h*margin)
    x1=max(0,x-mx); y1=max(0,y-my)
    x2=min(W,x+w+mx); y2=min(H,y+h+my)
    return frame[y1:y2, x1:x2]

def preprocess_face(bgr, size=INPUT_SIZE):
    if USE_GRAYSCALE_CLAHE:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        eq = clahe.apply(gray)
        rgb = cv2.cvtColor(eq, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size,size))
    x = (rgb.astype(np.float32))/255.0
    return np.expand_dims(x,0)

def route_outputs(raw_preds):
    if isinstance(raw_preds,(list,tuple)) and len(raw_preds)==2:
        p0,p1 = raw_preds
        d0 = int(np.asarray(p0).shape[-1]); d1=int(np.asarray(p1).shape[-1])
        if d0==len(EMO_LABELS) and (d1==len(AGE_BINS) or d1==1): return p0,p1
        if d1==len(EMO_LABELS) and (d0==len(AGE_BINS) or d0==1): return p1,p0
        return (p0,p1) if abs(d0-len(EMO_LABELS))<=abs(d1-len(EMO_LABELS)) else (p1,p0)
    return raw_preds, np.array([[65.0]])

def softmax(x):
    x = np.asarray(x, np.float32)
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x); return e/np.sum(e, axis=-1, keepdims=True)

def texture_score(gray_roi):
    
    v = cv2.Laplacian(gray_roi, cv2.CV_64F, ksize=WRINKLE_KERNEL).var()
    return float(v)

def recalibrate_age_probs(age_probs, tex_sc):
    probs = age_probs.astype(np.float32).copy()
    if WRINKLE_AID_ENABLED and tex_sc is not None and tex_sc >= WRINKLE_THRESH:
       
        probs[5:] *= AGE_SENIOR_BUMP
        probs /= probs.sum()
    return probs

def decode_age(age_pred, roi_gray=None):
    age_pred = np.asarray(age_pred)
    last = age_pred.shape[-1] if age_pred.ndim>=2 else 1
    if last==len(AGE_BINS):
        probs = age_pred.astype(np.float32)
        if probs.min()<0 or not np.allclose(probs.sum(axis=-1),1.0,atol=1e-3):
            probs = softmax(probs)
        
        tex = texture_score(roi_gray) if (roi_gray is not None) else None
        probs = recalibrate_age_probs(probs.squeeze(), tex)
        exp_age = float(np.sum(probs * AGE_BIN_MEAN))
        cls = int(np.argmax(probs))
        return int(round(exp_age)), cls, probs
    elif last==1:
        val = float(age_pred.squeeze())
        return int(round(val)), None, None
    else:
        cls = int(np.argmax(age_pred))
        cls = max(0, min(cls, len(AGE_BIN_MEAN)-1))
        return int(AGE_BIN_MEAN[cls]), cls, None

def apply_logit_bias(probs, biases):
   
    probs = probs.astype(np.float32).clip(1e-8, 1.0)
    logp = np.log(probs)
    for i,lab in enumerate(EMO_LABELS):
        if lab in biases:
            logp[i] += biases[lab]
    return softmax(logp)

def decode_emotion(emo_pred):
    probs = np.asarray(emo_pred, np.float32)
    if probs.min()<0 or not np.allclose(probs.sum(axis=-1),1.0,atol=1e-3):
        probs = softmax(probs)
    probs = probs.squeeze().astype(np.float32)
    
    probs = apply_logit_bias(probs, EMO_LOGIT_BIAS)
    idx = int(np.argmax(probs)); conf=float(np.max(probs))
    return idx, conf, probs

def majority_vote(labels, k):
    if not labels: return None
    counts = np.bincount(labels, minlength=k)
    return int(np.argmax(counts))

def detect_faces(frame):
    H,W = frame.shape[:2]
    faces=[]
    if use_dnn:
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300,300), (104,177,123), False, False)
        dnn_net.setInput(blob)
        det = dnn_net.forward()
        for i in range(det.shape[2]):
            conf = det[0,0,i,2]
            if conf < DNN_CONF_THRESH: 
                continue
            box = det[0,0,i,3:7]*np.array([W,H,W,H])
            x1,y1,x2,y2 = box.astype(int)
            x1=max(0,x1); y1=max(0,y1); x2=min(W-1,x2); y2=min(H-1,y2)
            w=x2-x1; h=y2-y1
            if w<MIN_FACE_SIZE_PX or h<MIN_FACE_SIZE_PX: 
                continue
            faces.append((x1,y1,w,h))
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        det = haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(MIN_FACE_SIZE_PX,MIN_FACE_SIZE_PX))
        faces = [(x,y,w,h) for (x,y,w,h) in det]
    return faces

def draw_debug_overlay(frame, displayed_age, prob_50plus, is_senior):
    if not SHOW_DEBUG_OVERLAY: return
    x0,y0=10,20; lh=22
    lines=["MODE: Seniors 50+",
           f"Âge affiché: {displayed_age}",
           f"P(age≥50): {prob_50plus:.2f}" if prob_50plus is not None else "P(age≥50): N/A",
           "Gating: SENIOR" if is_senior else "Gating: <50 (bloqué)"]
    for i,t in enumerate(lines):
        cv2.putText(frame, t, (x0, y0+i*lh), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

def main():
    prev_age_probs=None; prev_emo_probs=None
    age_buffer=deque(maxlen=AGE_WINDOW); emo_label_buffer=deque(maxlen=EMO_VOTE_WINDOW)
    age_hold_counter=0; displayed_age=None; last_dbg=0.0
    num_classes=len(EMO_LABELS)

    while True:
        ok, frame = cap.read()
        if not ok: break

        faces = detect_faces(frame)

        if len(faces)==0:
            prev_age_probs=prev_emo_probs=None
            age_buffer.clear(); emo_label_buffer.clear()
            age_hold_counter=0; displayed_age=None

        for (x,y,w,h) in faces:
            roi = crop_with_margin(frame, x,y,w,h, FACE_MARGIN)
            x_in = preprocess_face(roi, INPUT_SIZE)

           
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            raw = model.predict(x_in, verbose=0)
            emo_pred, age_pred = route_outputs(raw)

        
            age_val_raw, age_cls, age_probs = decode_age(age_pred, roi_gray=roi_gray)
            if age_probs is not None:
                prev_age_probs = age_probs if prev_age_probs is None else SMOOTH_ALPHA*prev_age_probs + (1-SMOOTH_ALPHA)*age_probs
                age_val = int(round(float(np.sum(prev_age_probs * AGE_BIN_MEAN))))
            else:
                age_val = age_val_raw

            age_buffer.append(age_val)
            median_age = int(np.median(age_buffer)) if len(age_buffer)>0 else age_val

            if displayed_age is None:
                displayed_age = median_age; age_hold_counter = AGE_HOLD_FRAMES
            else:
                if age_hold_counter>0:
                    if abs(median_age - displayed_age) > AGE_CHANGE_TOL:
                        displayed_age = median_age; age_hold_counter = AGE_HOLD_FRAMES
                    else:
                        age_hold_counter -= 1
                else:
                    if abs(median_age - displayed_age) > AGE_CHANGE_TOL:
                        displayed_age = median_age; age_hold_counter = AGE_HOLD_FRAMES

          
            prob_50plus = float(np.sum(prev_age_probs[5:])) if prev_age_probs is not None else None
            is_senior = True
            if SENIOR_ONLY:
                is_senior = (displayed_age >= SENIOR_MIN_AGE)
                if SENIOR_STRICT and prob_50plus is not None:
                    is_senior = is_senior and (prob_50plus >= SENIOR_PROB_MIN)

            if SHOW_DEBUG_OVERLAY and time.time()-last_dbg >= CONSOLE_DEBUG_EVERY_S:
                print(f"[DBG] age_aff={displayed_age} | P(age>=50)={prob_50plus if prob_50plus is not None else 'N/A'} | senior={is_senior}")
                last_dbg = time.time()

            if not is_senior:
                label = f"Age: {displayed_age} | analyse réservée 50+"
                color = (180,180,180)
                cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
                y_text = y-10 if y-10>20 else y+h+20
                cv2.putText(frame, label, (x, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
                draw_debug_overlay(frame, displayed_age, prob_50plus, False)
                continue

      
            emo_idx_raw, emo_conf_raw, emo_probs = decode_emotion(emo_pred)
            prev_emo_probs = emo_probs if prev_emo_probs is None else SMOOTH_ALPHA*prev_emo_probs + (1-SMOOTH_ALPHA)*emo_probs

         
            order = np.argsort(prev_emo_probs)[::-1]
            top1 = order[0]; top2 = order[1] if len(order)>1 else top1
            margin = float(prev_emo_probs[top1] - prev_emo_probs[top2])

            emo_idx = int(top1); emo_conf = float(prev_emo_probs[top1])

         
            emo_label_buffer.append(emo_idx)
            voted = majority_vote(list(emo_label_buffer), num_classes)
            if voted is not None:
                emo_idx = voted
                emo_conf = float(prev_emo_probs[emo_idx])

            emo_lbl = EMO_LABELS[emo_idx]
            conf_thr = CONF_MIN_SURPRISE if emo_lbl=='surprise' else CONF_MIN_BASE

            if emo_lbl=='surprise' and (margin < SURPRISE_MARGIN or emo_conf < conf_thr):
                label = f"Age: {displayed_age} | surprise (?) ({emo_conf:.2f})"
                color = (0,165,255)
            else:
                if emo_conf >= conf_thr:
                    label = f"Age: {displayed_age} | {emo_lbl} ({emo_conf:.2f})"
                    color = (0,200,0)
                else:
                    label = f"Age: {displayed_age} | émotion incertaine ({emo_conf:.2f})"
                    color = (0,165,255)

            cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
            y_text = y-10 if y-10>20 else y+h+20
            cv2.putText(frame, label, (x, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
            draw_debug_overlay(frame, displayed_age, prob_50plus, True)

        cv2.imshow("Émotions Seniors (50+)", frame)
        if writer is not None:
            writer.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    if writer is not None:
        writer.release(); print(f"[INFO] Vidéo annotée sauvegardée: {OUTPUT_PATH}")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()











