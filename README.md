Sure thing. Here’s your **EchoSight README styled with GitHub-friendly icons, emojis, clean formatting, and better visual hierarchy** — perfect for a polished repository.

---

# ✨ **EchoSight – Real-Time Vision-to-Voice Assistant**

> 🔊 *See the world through sound.*

EchoSight is a modular, offline, real-time **vision-to-audio assistive toolkit** designed for visually impaired users.
It integrates **YOLO object detection**, **traffic-light color classification**, **OCR text reading**, and **offline speech output** into one seamless system.

Runs on laptops, desktops, and even **Raspberry Pi** with lightweight models.

---

## 📦 **Repository Contents**

This project includes multiple coordinated subsystems:

📌 **Main unified system**

* `unified_echosight_system.py`
  *Full pipeline: YOLO + OCR + traffic-light color + voice feedback*

📌 **Object detection** (`object_detection/`)

* `obj_detect.py` – YOLO helper utilities
* `single_cam_live_system.py` – live camera detection
* `unified_vision_system.py` – multi-function pipeline

📌 **Traffic-light processing** (`traffic_signal/`)

* `cnn_model.py` – CNN classifier (Red/Yellow/Green)
* `traffic_color_system.py` – real-time color prediction + voice

📌 **OCR system** (`OCR/`)

* `ocr_main.py` – OCR pipeline
* `CRAFT/` – text-detection models & weights

📌 **Voice/TTS system** (`voice/`)

* `voice_engine.py` – Offline speech (pyttsx3 / Coqui)

📌 **Tests**

* `test_voice_system.py`
* `test_ocr_integration.py`
* `test_full_pipeline.py`

📌 **Models**

* `yolov8n.pt` – small YOLO model
* `traffic_cnn.pth` – traffic-light classifier
* `craft_mlt_25k.pth` – OCR detector weights

---

## 🚀 **Quickstart (Windows PowerShell)**

### ✅ 1. Create & activate virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### ✅ 2. Install requirements

```powershell
pip install -r requirements.txt
```

### ✅ 3. Run the full EchoSight system

```powershell
python unified_echosight_system.py
```

---

## 🔍 **Run Individual Modules**

### ▶ Object Detection Only

```powershell
python object_detection\unified_vision_live_system.py
```

### 🚦 Traffic-Light Color Recognition

```powershell
python trafficli\working_voice_system.py
```

### 📖 OCR Reader

```powershell
python OCR\working_ocr_pipeline.py
```

### 🔊 Voice/TTS Test

```powershell
python test_voice_system.py
```

### 🧪 Tests

```powershell
python test_voice_system.py
python test_ocr_integration.py
python test_full_pipeline.py
```

---

## 📁 **Important Entrypoints**

| Module                           | Description                      |
| -------------------------------- | -------------------------------- |
| `unified_echosight_system.py`    | 🔥 Full Vision-to-Voice Pipeline |
| `object_detection/obj_detect.py` | 🧠 YOLO detection utilities      |
| `traffic_signal/cnn_model.py`    | 🚦 Traffic-light classifier      |
| `OCR/ocr_main.py`                | 📚 OCR text reader               |
| `voice/voice_engine.py`          | 🔊 Offline TTS engine            |

---

## 🧠 **Models Used**

| Model               | Purpose                            |
| ------------------- | ---------------------------------- |
| `yolov8n.pt`        | Real-time object detection         |
| `traffic_cnn.pth`   | Traffic-light color classification |
| `craft_mlt_25k.pth` | OCR text detector                  |

Paths can be customized inside the script or passed as CLI parameters.

---

## 🛠️ **Development Notes**

* The system is fully modular — you can use only detection, only OCR, or combine everything.
* For a lean deployment, keep only:

  * `unified_echosight_system.py`
  * `object_detection/obj_detect.py`
  * `traffic_signal/traffic_color_system.py`
  * `OCR/ocr_main.py`
  * `voice/voice_engine.py`
  * `models/`

---