import json
import os
import threading

import cv2
import numpy as np

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _locate(name):
    """Find a model file inside model/ or directly in the project folder."""
    for folder in (os.path.join(BASE_DIR, "model"), BASE_DIR):
        path = os.path.join(folder, name)
        if os.path.exists(path):
            return path
    return None

IMG_SIZE = 224

BINS = {
    "blue": {"en": "Blue bin (dry waste)", "te": "నీలం డబ్బా (పొడి చెత్త)", "color": "#1d6fd8"},
    "green": {"en": "Green bin (wet waste)", "te": "ఆకుపచ్చ డబ్బా (తడి చెత్త)", "color": "#2e9e3a"},
}

INFO = {
    "plastic": {
        "icon": "🧴", "name_en": "Plastic", "name_te": "ప్లాస్టిక్", "bin": "blue",
        "tips_en": ["Empty and rinse the plastic item.", "Remove the cap.", "Put it in the plastic recycling bin."],
        "tips_te": ["ప్లాస్టిక్‌ను ఖాళీ చేసి శుభ్రం చేయండి.", "మూత తీసివేయండి.", "ప్లాస్టిక్ రీసైక్లింగ్ డబ్బాలో వేయండి."],
    },
    "paper": {
        "icon": "📄", "name_en": "Paper", "name_te": "కాగితం", "bin": "blue",
        "tips_en": ["Keep the paper clean and dry.", "Wet or oily paper cannot be recycled.", "Give it for paper recycling."],
        "tips_te": ["కాగితాన్ని పొడిగా, శుభ్రంగా ఉంచండి.", "తడిగా లేదా జిడ్డుగా ఉన్న కాగితం రీసైకిల్ కాదు.", "కాగితం రీసైక్లింగ్‌కు ఇవ్వండి."],
    },
    "glass": {
        "icon": "🍾", "name_en": "Glass", "name_te": "గాజు", "bin": "blue",
        "tips_en": ["Rinse the glass and remove the lid.", "Reuse bottles and jars if you can.", "Put it in the glass bin. Careful with broken glass."],
        "tips_te": ["గాజును కడిగి, మూత తీసివేయండి.", "సీసాలు, జాడీలను మళ్లీ వాడండి.", "గాజు డబ్బాలో వేయండి. పగిలిన గాజుతో జాగ్రత్త."],
    },
    "metal": {
        "icon": "🥫", "name_en": "Metal", "name_te": "లోహం", "bin": "blue",
        "tips_en": ["Empty and rinse the can.", "Do not mix it with other waste.", "Give it for metal or scrap recycling. Careful with sharp edges."],
        "tips_te": ["డబ్బాను ఖాళీ చేసి కడగండి.", "ఇతర చెత్తతో కలపకండి.", "లోహం/తుక్కు రీసైక్లింగ్‌కు ఇవ్వండి. పదునైన అంచులతో జాగ్రత్త."],
    },
    "organic": {
        "icon": "🍌", "name_en": "Organic (food, plants)", "name_te": "తడి చెత్త (ఆహారం, మొక్కలు)", "bin": "green",
        "tips_en": ["Keep food and plant waste separate.", "Do not mix it with plastic.", "Make compost or use the wet-waste bin."],
        "tips_te": ["ఆహార, మొక్కల వ్యర్థాలను విడిగా ఉంచండి.", "ప్లాస్టిక్‌తో కలపకండి.", "కంపోస్ట్ చేయండి లేదా తడి చెత్త డబ్బాలో వేయండి."],
    },
}

_interp = None
_class_names = None
_in = None
_out = None
_lock = threading.Lock()


def get_model():
    """Load the TFLite model and class names once, then reuse them."""
    global _interp, _class_names, _in, _out
    if _interp is None:
        model_path = _locate("model.tflite")
        classes_path = _locate("class_names.json")
        if model_path is None or classes_path is None:
            raise FileNotFoundError("model.tflite or class_names.json not found")
        it = Interpreter(model_path=model_path)
        it.allocate_tensors()
        _in = it.get_input_details()[0]
        _out = it.get_output_details()[0]
        with open(classes_path) as f:
            _class_names = json.load(f)
        _interp = it
    return _interp, _class_names


def predict_bytes(data):
    """Predict from raw image bytes (nothing is saved to disk)."""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not read the image.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    x = np.expand_dims(img.astype("float32"), axis=0)  # rescaling happens inside the model

    interp, class_names = get_model()
    with _lock:
        interp.set_tensor(_in["index"], x)
        interp.invoke()
        probs = interp.get_tensor(_out["index"])[0]
    idx = int(np.argmax(probs))
    key = class_names[idx]
    conf = float(probs[idx]) * 100
    info = INFO.get(key, {
        "icon": "♻️", "name_en": key.capitalize(), "name_te": key, "bin": "blue",
        "tips_en": ["Check local recycling rules."], "tips_te": ["స్థానిక రీసైక్లింగ్ నియమాలను చూడండి."],
    })
    b = BINS[info["bin"]]
    return {
        "key": key,
        "category": info["name_en"],
        "confidence": round(conf, 1),
        "guidance": info["tips_en"],
        "icon": info["icon"],
        "name_en": info["name_en"], "name_te": info["name_te"],
        "tips_en": info["tips_en"], "tips_te": info["tips_te"],
        "bin_en": b["en"], "bin_te": b["te"], "bin_color": b["color"],
        "low_confidence": conf < 60,
    }
