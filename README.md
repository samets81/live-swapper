<div align="center">

# 🎭 Live-Swapper

### Professional Real-Time Face Swapping

[![Website](https://img.shields.io/badge/🌐_Website-liveswapper.ru-blue?style=for-the-badge)](https://liveswapper.ru/)
[![Python](https://img.shields.io/badge/Python-3.11-yellow?style=for-the-badge&logo=python)](https://python.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-green?style=for-the-badge&logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![TensorRT](https://img.shields.io/badge/Backend-TensorRT-76b900?style=for-the-badge&logo=nvidia)](https://developer.nvidia.com/tensorrt)
[![License](https://img.shields.io/badge/License-GPL--3.0-red?style=for-the-badge)](LICENSE)

</div>

---

## 📖 About

**Live-Swapper** is a powerful tool for real-time face swapping via webcam. The application leverages TensorRT for maximum GPU performance on NVIDIA hardware and provides a clean graphical interface with flexible controls.

Ideal for content creators, streamers, creative projects, and developers exploring computer vision technologies.
<img width="1485" height="784" alt="p1" src="https://github.com/user-attachments/assets/ade0c2ef-daf0-4e33-9452-659ba092c113" />
<img width="1485" height="784" alt="p2" src="https://github.com/user-attachments/assets/ecfedf7f-3a2a-4b0a-b69a-ca0ed251e0ea" />
<img width="1767" height="812" alt="p3" src="https://github.com/user-attachments/assets/5befa93f-6664-4975-b4a9-d477fdc4ca3c" />


---

## ⚠️ Disclaimer

> We would like to emphasize that this software is intended for **responsible and ethical use only**. We must stress that **users are solely responsible** for their actions when using this software.
>
> **Intended Usage:** This software is designed to assist users in creating realistic and entertaining content, such as movies, visual effects, virtual reality experiences, and other creative applications. We encourage users to explore these possibilities within the boundaries of legality, ethical considerations, and respect for others' privacy.
>
> **Ethical Guidelines:** Users are expected to adhere to a set of ethical guidelines when using our software. It is **strictly prohibited** to use it to create content without the consent of the depicted individuals, to discredit, deceive, or harm anyone.
>
> By using this software, users acknowledge that they have read, understood, and agreed to abide by the above guidelines and disclaimers. We strongly encourage users to approach this technology with caution, integrity, and respect for the well-being and rights of others. Remember, technology should be used to empower and inspire — not to harm or deceive.

---

## ✨ Features

- 🎥 **Real-time face swap** — live from your webcam with minimal latency
- ⚡ **TensorRT / CUDA backend** — maximum performance on NVIDIA GPUs
- 🖼️ **GFPGAN / Face Restorer** — AI-powered face quality enhancement
- 🎭 **XSeg Mask** — precise face segmentation for natural blending
- 📷 **Multi-camera support** — select from all connected devices
- 📐 **Flexible resolution** — from 640×480 up to 1920×1080
- 🎛️ **Advanced filters** — face sharpness, mask blur, Erosion/Dilation
- 📡 **OBS / Virtual Camera** — direct output to OBS Studio and other apps
- 📊 **Live FPS counter** — real-time performance monitoring
- 🚀 **One-click launch** — simple `.bat` file startup

---

## 🖥️ Interface Overview

| **Main** Tab | **Filters** Tab |
|---|---|
| Select source photo | Face Sharpness (0.0 — 5.0) |
| Choose webcam | Mask Blur |
| Set camera resolution | GFPGAN / Face Restorer + Blend % |
| Swap quality (128 — fast / 256 — quality) | XSeg Mask + Erosion / Dilation |
| Select backend (TensorRT / CUDA) | |

---

## 🖥️ System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| OS | Windows 10 64-bit | Windows 11 64-bit |
| GPU | NVIDIA with CUDA support | RTX 4000/5000 series |
| CUDA | 12.x | 12.x |
| Python | 3.11 | 3.11 |
| RAM | 8 GB | 16 GB |
| VRAM | 4 GB | 8 GB+ |

---

## 🚀 Installation

### 1. Prerequisites

Install the required components:

- **Python 3.11** → [python.org](https://www.python.org/downloads/)
- **Git** → [git-scm.com](https://git-scm.com/downloads)
- **Visual Studio C++ Build Tools** → [visualstudio.microsoft.com](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
- **CUDA 12.x** → [developer.nvidia.com](https://developer.nvidia.com/cuda-toolkit)


### 2. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/live-swapper.git
cd live-swapper
```

### 3. Install Dependencies

```bash
install.bat
```

### 4. Launch

```bash
Start.bat
```

---

## 💡 How to Use

1. **Select a photo** — click "Choose Photo" and pick an image with a face. Wait for `✓ Face recognized`
2. **Configure camera** — choose your webcam and resolution in the **Main** tab
3. **Set swap quality** — choose `128 — fast` or `256 — quality`
4. **Adjust filters** — switch to the **Filters** tab for fine-tuning
5. **Click Start** — face swapping begins in real time
6. **OBS integration** — enable the OBS button to stream to a virtual camera

---

## ⚙️ Filter Parameters

| Parameter | Description | Range |
|---|---|---|
| **Face Sharpness** | Sharpness of the swapped face | 0.0 — 5.0 |
| **Mask Blur** | Smoothness of mask edges | 1 — 99 |
| **GFPGAN** | AI face restoration (reduces FPS) | On/Off |
| **Blend %** | Blending level with original | 0 — 100 |
| **XSeg Mask** | Precise segmentation for better blending | On/Off |
| **Erosion / Dilation** | Shrink or expand the mask boundary | −/+ |

---

## 📊 Performance Benchmarks

| GPU | Resolution | Approx. FPS |
|---|---|---|
| RTX 5070 Ti | 640×480 | ~30 |
| RTX 4060 | 640×480 | ~25 |


> Performance varies depending on the number of faces in frame and enabled filters.

---

## 🔧 Project Structure

```
live-swapper/
├── app/                    # Core application code
├── models/                 # AI model files
│   ├── inswapper_128.onnx
│   └── GFPGANv1.4.pth
│   
├── venv/                   # Python virtual environment
├── run.py                  # Application entry point
├── down_models.py          # Model downloader
├── requirements.txt        # Python dependencies
├── install.bat             # Installer script
├── Start.bat               # Quick launch
└── trt_cache/
```

---

## 🐛 Troubleshooting

**TensorRT fails to initialize**
- Ensure cuDNN is added to your system `PATH`
- Delete the cache folder `trt_cache/` and relaunch

**Low FPS**
- Lower the camera resolution
- Disable GFPGAN
- Close other GPU-intensive applications

**Face not recognized**
- Use a clear, front-facing photo
- Ensure good lighting in the source image

**OBS doesn't see the virtual camera**
- Verify that `pyvirtualcam` installed correctly
- Restart OBS after launching Live-Swapper

---

## 🗺️ Roadmap

- [ ] **Model pre-compilation** — faster startup on subsequent launches
- [ ] **Background replacement** — virtual background in real time
- [ ] **Processing speed boost** — further pipeline optimization
- [ ] **Memory optimization** — reduced VRAM consumption

<div align="center">
<b>You can also always get the latest updates from our website: https://liveswapper.ru/ by supporting the project.</b>

</div>
---

## 🤝 Credits & Technologies

- [InsightFace](https://github.com/deepinsight/insightface) — face analysis and detection
- [GFPGAN](https://github.com/TencentARC/GFPGAN) — face quality restoration
- [ONNX Runtime](https://onnxruntime.ai/) — model inference engine
- [pyvirtualcam](https://github.com/letmaik/pyvirtualcam) — virtual camera output

---

## 📄 License

Distributed under the **GPL-3.0** license. Please note that the InsightFace model is intended **for non-commercial research purposes only**.

---

<div align="center">

🌐 **[liveswapper.ru](https://liveswapper.ru/)** — official project website

Made with ❤️ for the AI community

</div>
