import sys
import os
import time
import traceback
from math import floor, ceil
sys.stdout.reconfigure(line_buffering=True)
#── Корень проекта в sys.path ─────────────────────────────────────────────────
_THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
import cv2
import numpy as np
import torch
from skimage import transform as trans
from torchvision.transforms import v2
from PySide6 import QtWidgets, QtCore
from app.main_ui import MainWindow
from app.processors.models_processor import ModelsProcessor
from app.processors.utils import faceutil

# ── Проверка доступности pyvirtualcam ────────────────────────────────────────
try:
    import pyvirtualcam
    PYVIRTUALCAM_AVAILABLE = True
    print("[CORE] pyvirtualcam доступен")
except ImportError:
    PYVIRTUALCAM_AVAILABLE = False
    print("[WARNING] pyvirtualcam не установлен. Виртуальная камера недоступна.")
    print("          Установите: pip install pyvirtualcam")

#══════════════════════════════════════════════════════════════════════════════
# Поток захвата камеры
#══════════════════════════════════════════════════════════════════════════════
class CameraThread(QtCore.QThread):
    """Захват кадров с камеры.

    Захват и обработка разделены: отдельный поток читает кадры
    и кладёт их в очередь размером 1 (frame-skip), основной поток
    обработки берёт последний доступный кадр.  Если обработка медленнее
    камеры — промежуточные кадры пропускаются, FPS отображения растёт.
    """
    frame_ready = QtCore.Signal(object)  # np.ndarray BGR

    def __init__(self, backend: "FaceSwapApp"):
        super().__init__()
        self._be     = backend
        self._active = False
        import queue
        self._queue: queue.Queue = queue.Queue(maxsize=1)

    def run(self):
        import queue, threading
        self._active = True
        cap = self._be.capture

        def _capture_loop():
            while self._active and cap and cap.isOpened():
                ret, bgr = cap.read()
                if not ret:
                    time.sleep(0.005)
                    continue
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
                self._queue.put(bgr)

        capture_thread = threading.Thread(target=_capture_loop, daemon=True)
        capture_thread.start()

        while self._active:
            try:
                bgr = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            rgb        = bgr[..., ::-1].copy()
            result_rgb = self._be.process_frame(rgb)
            self.frame_ready.emit(result_rgb[..., ::-1].copy())

        capture_thread.join(timeout=2.0)
        self._active = False

    def stop(self):
        self._active = False
        self.wait(5000)

#══════════════════════════════════════════════════════════════════════════════
# Backend
#══════════════════════════════════════════════════════════════════════════════
class FaceSwapApp(QtCore.QObject):
    # Кандидаты для проверки поддерживаемых разрешений камеры
    _RES_CANDIDATES = [
        (640, 360),  (640, 480),
        (800, 448),  (800, 600),
        (960, 540),
        (1024, 576), (1024, 768),
        (1280, 720), (1280, 960),
        (1600, 896), (1600, 1200),
        (1920, 1080),
    ]

    def __init__(self, window: MainWindow):
        super().__init__()
        self.win = window

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.mp = ModelsProcessor(window, device=self.device)

        # ── Все параметры ─────────────────────────────────────────
        self.detect_mode       = 'RetinaFace'
        self.detect_score      = 0.5
        self.detect_max_num    = 1
        self.recognition_model = 'Inswapper128ArcFace'
        self.similarity_type   = 'Opal'
        self.swapper_model     = 'Inswapper128'
        self.gfpgan_enabled    = False
        self.gfpgan_blend      = 60
        self.gfpgan_det_mode   = 'Original'
        self.gfpgan_det_score  = 0.5
        self.gfpgan_type       = 'GFPGAN-v1.4'
        self.xseg_enabled      = False
        self.xseg_erosion      = 0
        self.sharpness         = 0.6
        self.mask_blur         = 3
        self.show_fps          = False
        self.swap_res          = 128
        self.provider          = 'TensorRT'
        self.n_threads         = 2

        self.virtcam_enabled   = False
        self.virtcam: pyvirtualcam.Camera | None = None if not PYVIRTUALCAM_AVAILABLE else None
        self.source_embedding: np.ndarray | None = None
        self.source_latent:    np.ndarray | None = None
        self.capture:     cv2.VideoCapture | None = None
        self.camera_index = 0
        self.cap_w        = 640
        self.cap_h        = 480
        self.running      = False
        self.cam_thread:  CameraThread | None = None
        self._fps_cnt = 0
        self._fps_t   = time.time()
        self._fps_val = 0.0

        self._connect()
        self._find_cameras()

        self.mp.switch_providers_priority(self.provider)
        if self.provider == "CUDA":
            self.mp.set_number_of_threads(self.n_threads)

        # Синхронизируем UI с дефолтами
        self.win.set_gfpgan_state(self.gfpgan_enabled)
        self.win.set_xseg_state(self.xseg_enabled)
        self.win.set_fps_state(self.show_fps)
        self.win.set_virtcam_state(self.virtcam_enabled)
        self.win.set_provider(self.provider)
        self.win.set_slider_value("sharpness", self.sharpness, 0.1)
        self.win.set_slider_value("blur", self.mask_blur, 1)

        print("[CORE] Инициализация завершена")

    # ────────────────────────────────────────────────────────────────────────
    def _connect(self):
        w = self.win
        w.select_photo_requested.connect(self._select_photo)
        w.start_stop_requested.connect(self._start_stop)
        w.toggle_gfpgan_requested.connect(self._toggle_gfpgan)
        w.toggle_xseg_requested.connect(self._toggle_xseg)
        w.toggle_fps_requested.connect(self._toggle_fps)
        w.toggle_virtcam_requested.connect(self._toggle_virtcam)
        w.sharpness_changed.connect(self._set_sharpness)
        w.mask_blur_changed.connect(self._set_blur)
        w.xseg_erosion_changed.connect(self._set_xseg_erosion)
        w.gfpgan_blend_changed.connect(self._set_gfpgan_blend)
        w.swapper_model_changed.connect(self._set_swapper_model)
        w.swap_res_changed.connect(self._set_swap_res)
        w.provider_changed.connect(self._set_provider)
        w.camera_changed.connect(self._on_camera_changed)
        w.resolution_changed.connect(self._on_resolution_changed)
        w.video_win.closed.connect(self._on_video_closed)

    # ────────────────────────────────────────────────────────────────────────
    def _find_cameras(self):
        cameras = []
        cv2.setLogLevel(0)
        for i in range(3):
            cap_test = cv2.VideoCapture(i, cv2.CAP_DSHOW if os.name == 'nt' else 0)
            if cap_test.isOpened():
                cameras.append((i, f"Камера {i}"))
                cap_test.release()
        cv2.setLogLevel(3)
        if not cameras:
            cameras = [(0, "Камера 0")]
        self.win.populate_cameras(cameras)
        self._load_camera_resolutions(cameras[0][0])

    def _load_camera_resolutions(self, cam_idx: int):
        """Проверяем реальные разрешения, которые поддерживает камера."""
        supported = []
        cv2.setLogLevel(0)
        backend = cv2.CAP_DSHOW if os.name == 'nt' else 0
        cap = cv2.VideoCapture(cam_idx, backend)
        if cap.isOpened():
            for tw, th in self._RES_CANDIDATES:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, tw)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, th)
                aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                entry = (aw, ah)
                if entry not in [(r[0], r[1]) for r in supported]:
                    supported.append((aw, ah, f"{aw}×{ah}"))
            cap.release()
        cv2.setLogLevel(3)

        if not supported:
            supported = [(640, 480, "640×480")]

        supported.sort(key=lambda x: x[0] * x[1])
        self.win.populate_resolutions(supported)
        self.cap_w, self.cap_h = supported[0][0], supported[0][1]
        print(f"[CORE] Разрешения камеры {cam_idx}: {[f'{w}×{h}' for w,h,_ in supported]}")

    # ────────────────────────────────────────────────────────────────────────
    @QtCore.Slot()
    def _select_photo(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.win, "Выбрать фото", "",
            "Images (*.png *.jpg *.jpeg *.bmp);;All (*.*)"
        )
        if not path:
            return
        bgr = cv2.imread(path)
        if bgr is None:
            self.win.show_error("Ошибка", f"Не удалось открыть файл:\n{path}")
            return
        rgb = bgr[..., ::-1].copy()
        self._extract_source_face(rgb)

    def _extract_source_face(self, rgb: np.ndarray):
        self.win.set_status("Распознавание фото...", "#f0c040")
        QtWidgets.QApplication.processEvents()
        img_t = torch.from_numpy(rgb).permute(2, 0, 1).to(self.device)
        try:
            bboxes, kpss_5, _ = self.mp.run_detect(
                img_t,
                detect_mode=self.detect_mode,
                max_num=1,
                score=self.detect_score,
                input_size=(512, 512),
            )
            if len(bboxes) == 0:
                self.win.set_status("❌ Лицо не найдено на фото", "#ff4444")
                return

            kps  = kpss_5[0]
            bbox = bboxes[0]

            embedding, _ = self.mp.run_recognize_direct(
                img_t, kps,
                similarity_type=self.similarity_type,
                arcface_model=self.recognition_model,
            )
            self.source_embedding = embedding.astype(np.float32)
            self._recalc_latent()

            x1, y1, x2, y2 = [int(v) for v in bbox]
            pad = 20
            x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
            x2, y2 = min(rgb.shape[1], x2 + pad), min(rgb.shape[0], y2 + pad)
            preview = rgb[y1:y2, x1:x2].copy()
            self.win.update_preview(preview[..., ::-1])
            self.win.set_status("✓ Лицо распознано", "#66ff66")
        except Exception as e:
            traceback.print_exc()
            self.win.set_status(f"Ошибка распознавания: {e}", "#ff4444")

    def _recalc_latent(self):
        if self.source_embedding is None:
            self.source_latent = None
            return
        m = self.swapper_model
        try:
            if m == 'Inswapper128':
                self.mp.load_inswapper_iss_emap('Inswapper128')
                self.source_latent = self.mp.calc_inswapper_latent(self.source_embedding).astype(np.float32)
            elif 'InStyleSwapper' in m:
                ver = m.split()[-1]
                self.mp.load_inswapper_iss_emap(m)
                self.source_latent = self.mp.calc_swapper_latent_iss(self.source_embedding, ver).astype(np.float32)
            elif m == 'SimSwap512':
                self.source_latent = self.mp.calc_swapper_latent_simswap512(self.source_embedding).astype(np.float32)
            elif m.startswith('GhostFace'):
                self.source_latent = self.mp.calc_swapper_latent_ghost(self.source_embedding).astype(np.float32)
            elif m == 'CSCS':
                self.source_latent = self.mp.calc_swapper_latent_cscs(self.source_embedding).astype(np.float32)
        except Exception as e:
            traceback.print_exc()
            self.win.set_status(f"Ошибка вычисления латента: {e}", "#ff4444")

    # ────────────────────────────────────────────────────────────────────────
    @QtCore.Slot()
    def _start_stop(self):
        if not self.running:
            self._start()
        else:
            self._stop()

    def _start(self):
        if self.source_latent is None:
            self.win.show_error("Ошибка", "Сначала загрузите фото!")
            return
        if self.capture and self.capture.isOpened():
            self.capture.release()
        backend = cv2.CAP_DSHOW if os.name == 'nt' else 0
        self.capture = cv2.VideoCapture(self.camera_index, backend)
        if not self.capture.isOpened():
            self.win.show_error("Ошибка", f"Не удалось открыть камеру {self.camera_index}")
            return
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH,  self.cap_w)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cap_h)

        if self.virtcam_enabled and PYVIRTUALCAM_AVAILABLE:
            try:
                self.virtcam = pyvirtualcam.Camera(
                    width=self.cap_w,
                    height=self.cap_h,
                    fps=30,
                    fmt=pyvirtualcam.PixelFormat.BGR
                )
                print(f"[CORE] Виртуальная камера запущена: {self.virtcam.device}")
            except Exception as e:
                print(f"[WARNING] Не удалось запустить виртуальную камеру: {e}")
                self.virtcam = None

        self.running = True
        self.win.set_start_button(True)
        self.win.set_camera_combo_enabled(False)
        self.win.set_resolution_combo_enabled(False)
        self.win.show_video_window()
        self.cam_thread = CameraThread(self)
        self.cam_thread.frame_ready.connect(self._on_frame)
        self.cam_thread.start()
        self.win.set_status("Камера запущена", "#66ff66")

    def _stop(self):
        self.running = False
        if self.cam_thread:
            self.cam_thread.stop()
            self.cam_thread = None
        if self.capture:
            self.capture.release()
            self.capture = None

        if self.virtcam:
            try:
                self.virtcam.close()
                print("[CORE] Виртуальная камера остановлена")
            except Exception as e:
                print(f"[WARNING] Ошибка при закрытии виртуальной камеры: {e}")
            self.virtcam = None

        self.win.set_start_button(False)
        self.win.set_camera_combo_enabled(True)
        self.win.set_resolution_combo_enabled(True)
        self.win.hide_video_window()
        self.win.set_status("Камера остановлена", "#aaa")

    @QtCore.Slot()
    def _on_video_closed(self):
        if self.running:
            self._stop()

    # ── Отправка кадра на виртуальную камеру ────────────────────────────────
    def _send_frame_to_virtualcam(self, frame_bgr: np.ndarray):
        """Отправляет кадр на виртуальную камеру (OBS Virtual Camera)."""
        if not self.virtcam_enabled:
            return

        if not self.virtcam:
            try:
                self.virtcam = pyvirtualcam.Camera(
                    width=self.cap_w,
                    height=self.cap_h,
                    fps=30,
                    fmt=pyvirtualcam.PixelFormat.BGR
                )
                print(f"[CORE] Виртуальная камера автоматически запущена: {self.virtcam.device}")
            except Exception as e:
                print(f"[WARNING] Не удалось автоматически запустить виртуальную камеру: {e}")
                self.virtcam_enabled = False
                self.win.set_virtcam_state(False)
                return

        try:
            h, w = frame_bgr.shape[:2]
            if (w, h) != (self.virtcam.width, self.virtcam.height):
                frame_bgr = cv2.resize(frame_bgr, (self.virtcam.width, self.virtcam.height))
            self.virtcam.send(frame_bgr)
        except Exception as e:
            print(f"[WARNING] Ошибка отправки кадра на виртуальную камеру: {e}")

    # ────────────────────────────────────────────────────────────────────────
    @QtCore.Slot()
    def _toggle_gfpgan(self):
        self.gfpgan_enabled = not self.gfpgan_enabled
        self.win.set_gfpgan_state(self.gfpgan_enabled)

    @QtCore.Slot()
    def _toggle_xseg(self):
        self.xseg_enabled = not self.xseg_enabled
        self.win.set_xseg_state(self.xseg_enabled)

    @QtCore.Slot()
    def _toggle_fps(self):
        self.show_fps = not self.show_fps
        self.win.set_fps_state(self.show_fps)

    @QtCore.Slot()
    def _toggle_virtcam(self):
        """Переключатель виртуальной камеры с поддержкой горячего переключения."""
        if not PYVIRTUALCAM_AVAILABLE:
            self.win.show_error(
                "Виртуальная камера недоступна",
                "pyvirtualcam не установлен.\n"
                "Установите: pip install pyvirtualcam\n\n"
                "Windows: требуется OBS Virtual Camera\n"
                "Linux: требуется v4l2loopback"
            )
            return

        self.virtcam_enabled = not self.virtcam_enabled
        self.win.set_virtcam_state(self.virtcam_enabled)

        if self.running:
            if self.virtcam_enabled:
                try:
                    self.virtcam = pyvirtualcam.Camera(
                        width=self.cap_w,
                        height=self.cap_h,
                        fps=30,
                        fmt=pyvirtualcam.PixelFormat.BGR
                    )
                    print(f"[CORE] Виртуальная камера запущена: {self.virtcam.device}")
                    self.win.set_status("OBS камера включена", "#66ff66")
                except Exception as e:
                    print(f"[WARNING] Не удалось запустить виртуальную камеру: {e}")
                    self.virtcam = None
                    self.virtcam_enabled = False
                    self.win.set_virtcam_state(False)
                    self.win.show_error("Ошибка OBS", f"Не удалось запустить виртуальную камеру:\n{e}")
            else:
                if self.virtcam:
                    try:
                        self.virtcam.close()
                        print("[CORE] Виртуальная камера остановлена")
                        self.win.set_status("OBS камера выключена", "#aaa")
                    except Exception as e:
                        print(f"[WARNING] Ошибка при закрытии виртуальной камеры: {e}")
                    self.virtcam = None

    @QtCore.Slot(float)
    def _set_sharpness(self, val: float):
        self.sharpness = val

    @QtCore.Slot(int)
    def _set_blur(self, val: int):
        self.mask_blur = val

    @QtCore.Slot(int)
    def _set_xseg_erosion(self, val: int):
        self.xseg_erosion = val

    @QtCore.Slot(int)
    def _set_gfpgan_blend(self, val: int):
        self.gfpgan_blend = val

    @QtCore.Slot(str)
    def _set_swapper_model(self, model: str):
        self.swapper_model = model
        self._recalc_latent()

    @QtCore.Slot(int)
    def _set_swap_res(self, res: int):
        self.swap_res = res

    @QtCore.Slot(str)
    def _set_provider(self, prov: str):
        self.provider = prov
        was_running = self.running
        if was_running:
            self._stop()
        self.mp.switch_providers_priority(prov)
        if prov == "CUDA":
            self.mp.set_number_of_threads(self.n_threads)
        self._recalc_latent()
        if was_running:
            self._start()

    @QtCore.Slot(int)
    def _on_camera_changed(self, index: int):
        was_running = self.running
        if was_running:
            self._stop()
        self.camera_index = index
        self._load_camera_resolutions(index)
        if was_running:
            self._start()

    @QtCore.Slot(int, int)
    def _on_resolution_changed(self, w: int, h: int):
        self.cap_w, self.cap_h = w, h
        if self.running and self.capture:
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    # ────────────────────────────────────────────────────────────────────────
    # ОПТИМИЗИРОВАННАЯ ОБРАБОТКА КАДРА
    # ────────────────────────────────────────────────────────────────────────
    def process_frame(self, rgb: np.ndarray) -> np.ndarray:
        """
        Оптимизированная версия обработки кадра:
        - БЕЗ letterbox (детекция на оригинальном изображении)
        - Детекция на полном разрешении 512×512
        - Paste-back через torchvision (без cv2.warpAffine)
        """
        if self.source_latent is None:
            return rgb

        img_t = torch.from_numpy(rgb).permute(2, 0, 1).to(self.device)

        try:
            bboxes, kpss_5, _ = self.mp.run_detect(
                img_t,
                detect_mode=self.detect_mode,
                max_num=self.detect_max_num,
                score=self.detect_score,
                input_size=(512, 512),
            )

            if len(bboxes) > 0:
                for i in range(len(bboxes)):
                    img_t = self._process_one_face(img_t, bboxes[i], kpss_5[i])

        except Exception:
            traceback.print_exc()

        result = img_t.permute(1, 2, 0).cpu().numpy()

        if self.show_fps:
            self._fps_cnt += 1
            now = time.time()
            if now - self._fps_t >= 1.0:
                self._fps_val = self._fps_cnt / (now - self._fps_t)
                self._fps_cnt, self._fps_t = 0, now
                self.win.update_fps(self._fps_val)

        return result

    # ────────────────────────────────────────────────────────────────────────
    def _get_tform(self, kps_5: np.ndarray) -> trans.SimilarityTransform:
        dst   = faceutil.get_arcface_template(image_size=512, mode='arcface128')
        dst   = np.squeeze(dst)
        tform = trans.SimilarityTransform()
        tform.estimate(kps_5, dst)
        return tform

    # ────────────────────────────────────────────────────────────────────────
    def _process_one_face(self, img_t: torch.Tensor, bbox, kps) -> torch.Tensor:
        """
        Обработка одного лица:
        - Paste-back через torchvision.functional.affine (БЕЗ cv2.warpAffine)
        - Все операции на GPU без переносов CPU↔GPU
        """
        mp: ModelsProcessor = self.mp
        dev: str = self.device

        tform = self._get_tform(kps)

        # 1. Вырезаем face_512
        face_512 = v2.functional.affine(
            img_t,
            tform.rotation * 57.2958,
            (tform.translation[0], tform.translation[1]),
            tform.scale, 0,
            center=(0, 0),
            interpolation=v2.InterpolationMode.BILINEAR,
        )
        face_512 = v2.functional.crop(face_512, 0, 0, 512, 512)

        # 2. Тайлинг по swap_res
        res = self.swap_res
        dim = res // 128
        face_res   = v2.Resize((res, res), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)(face_512)
        input_face = face_res.permute(1, 2, 0).float() / 255.0

        # 3. Прогон тайлами через Inswapper
        output = torch.zeros((res, res, 3), dtype=torch.float32, device=dev)
        lat    = torch.from_numpy(self.source_latent).to(dev).contiguous()
        try:
            with torch.no_grad():
                for j in range(dim):
                    for i in range(dim):
                        tile     = input_face[j::dim, i::dim]
                        tile_in  = tile.permute(2, 0, 1).unsqueeze(0).contiguous()
                        tile_out = torch.empty((1, 3, 128, 128), dtype=torch.float32, device=dev).contiguous()
                        mp.run_inswapper(tile_in, lat, tile_out, sharpness=self.sharpness)
                        output[j::dim, i::dim] = tile_out.squeeze(0).permute(1, 2, 0)
        except Exception:
            traceback.print_exc()
            return img_t

        swap_chw = output.mul(255.0).clamp(0, 255).permute(2, 0, 1).to(torch.uint8)

        # 4. Апскейл до 512 + GFPGAN
        swap_512 = v2.Resize((512, 512), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)(swap_chw)
        if self.gfpgan_enabled:
            try:
                swap_512 = mp.apply_facerestorer(
                    swap_512,
                    restorer_det_type=self.gfpgan_det_mode,
                    restorer_type=self.gfpgan_type,
                    restorer_blend=self.gfpgan_blend,
                    fidelity_weight=self.gfpgan_det_score,
                    detect_score=self.gfpgan_det_score,
                )
            except Exception:
                traceback.print_exc()

        # 5. XSeg маска
        swap_mask = torch.ones((1, res, res), dtype=torch.float32, device=dev)
        if self.xseg_enabled:
            try:
                face_256  = v2.Resize((256, 256), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)(face_512)
                xseg      = mp.apply_dfl_xseg(face_256, -self.xseg_erosion)
                xseg_res  = v2.Resize((res, res), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)(xseg)
                swap_mask = swap_mask * (1.0 - xseg_res)
            except Exception:
                traceback.print_exc()

        # 5b. Плавное затухание краёв маски
        fade_size = max(int(res * 0.35), 4)
        ys = torch.arange(res, dtype=torch.float32, device=dev)
        xs = torch.arange(res, dtype=torch.float32, device=dev)
        dist_y    = torch.minimum(ys, res - 1 - ys).clamp(0, fade_size) / fade_size
        dist_x    = torch.minimum(xs, res - 1 - xs).clamp(0, fade_size) / fade_size
        edge_fade = (dist_y.unsqueeze(1) * dist_x.unsqueeze(0)).unsqueeze(0)
        swap_mask = swap_mask * edge_fade

        # 6. Размытие маски
        blur_k        = self.mask_blur | 1
        swap_mask     = v2.GaussianBlur(blur_k, blur_k * 0.2)(swap_mask)
        swap_mask_512 = v2.Resize((512, 512), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)(swap_mask)

        # 7. Применяем маску
        swap_f = swap_512.float() * swap_mask_512

        # 8. Paste-back через torchvision.functional.affine (БЕЗ cv2.warpAffine!)
        img_h, img_w = img_t.shape[1], img_t.shape[2]
        IM = tform.inverse.params[0:2, :]

        corners = np.array([[0, 0], [0, 511], [511, 0], [511, 511]], dtype=np.float32)
        x = IM[0, 0]*corners[:, 0] + IM[0, 1]*corners[:, 1] + IM[0, 2]
        y = IM[1, 0]*corners[:, 0] + IM[1, 1]*corners[:, 1] + IM[1, 2]
        left   = max(0,     floor(np.min(x)))
        top    = max(0,     floor(np.min(y)))
        right  = min(img_w, ceil(np.max(x)))
        bottom = min(img_h, ceil(np.max(y)))

        if right <= left or bottom <= top:
            return img_t

        swap_padded = v2.functional.pad(swap_f, (0, 0, img_w - 512, img_h - 512))
        swap_back   = v2.functional.affine(
            swap_padded,
            tform.inverse.rotation * 57.2958,
            (tform.inverse.translation[0], tform.inverse.translation[1]),
            tform.inverse.scale, 0,
            interpolation=v2.InterpolationMode.BILINEAR,
            center=(0, 0),
        )
        swap_crop = swap_back[0:3, top:bottom, left:right].permute(1, 2, 0)

        mask_padded = v2.functional.pad(swap_mask_512, (0, 0, img_w - 512, img_h - 512))
        mask_back   = v2.functional.affine(
            mask_padded,
            tform.inverse.rotation * 57.2958,
            (tform.inverse.translation[0], tform.inverse.translation[1]),
            tform.inverse.scale, 0,
            interpolation=v2.InterpolationMode.BILINEAR,
            center=(0, 0),
        )
        mask_crop = mask_back[0:1, top:bottom, left:right].permute(1, 2, 0)

        img_crop = img_t[0:3, top:bottom, left:right].permute(1, 2, 0).float()
        blended  = swap_crop + (1.0 - mask_crop) * img_crop
        blended  = blended.clamp(0, 255).to(torch.uint8).permute(2, 0, 1)
        img_t[0:3, top:bottom, left:right] = blended
        return img_t

    @QtCore.Slot(object)
    def _on_frame(self, frame_bgr: np.ndarray):
        self._send_frame_to_virtualcam(frame_bgr)
        self.win.update_video(frame_bgr)

#─────────────────────────────────────────────────────────────────────────────
def main():
    app     = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Live Face Swap")
    window  = MainWindow()
    backend = FaceSwapApp(window)  # noqa: F841
    window.show()
    print("[CORE] Приложение запущено")
    sys.exit(app.exec())

if __name__ == '__main__':
    main()