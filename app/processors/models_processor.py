"""
models_processor.py — менеджер моделей (оптимизированный)
Только: det_10g, GFPGANv1.4, inswapper_128, w600k_r50, XSeg, MODNet, RVM
"""
import threading
import os
import gc
import traceback
from typing import Dict, TYPE_CHECKING
from packaging import version
import numpy as np
import onnxruntime
import torch
import onnx
from PySide6 import QtCore

try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ModuleNotFoundError:
    TENSORRT_AVAILABLE = False

from app.processors.utils.tensorrt_predictor import TensorRTPredictor
from app.processors.face_detectors import FaceDetectors
from app.processors.face_masks import FaceMasks
from app.processors.face_restorers import FaceRestorers
from app.processors.face_swappers import FaceSwappers
from app.processors.models_data import models_list, arcface_mapping_model_dict, models_trt_list

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow

onnxruntime.set_default_logger_severity(4)
onnxruntime.log_verbosity_level = -1


class ModelsProcessor(QtCore.QObject):
    model_loading = QtCore.Signal()
    model_loaded = QtCore.Signal()

    def __init__(self, main_window: 'MainWindow', device='cuda'):
        super().__init__()
        self.main_window = main_window
        self.provider_name = 'TensorRT'
        self.device = device
        self.model_lock = threading.RLock()

        self.trt_ep_options = {
            'trt_engine_cache_enable': True,
            'trt_engine_cache_path': 'trt_cache',
            'trt_timing_cache_enable': True,
            'trt_timing_cache_path': 'trt_cache',
            'trt_layer_norm_fp32_fallback': True,
            'trt_builder_optimization_level': 5,
        }
        self.providers = [('CUDAExecutionProvider'), ('CPUExecutionProvider')]
        self.nThreads = 2
        self.syncvec = torch.empty((1, 1), dtype=torch.float32, device=self.device)

        # Реестр моделей
        self.models: Dict[str, onnxruntime.InferenceSession] = {}
        self.models_path: Dict[str, str] = {}
        self.models_data: Dict[str, dict] = {} 
        for md in models_list:
            name = md['model_name']
            self.models[name] = None
            self.models_path[name] = md['local_path']
            self.models_data[name] = {
                'local_path': md['local_path'],
                'hash': md['hash'],
                'url': md.get('url'),
            }

        if TENSORRT_AVAILABLE:
            self.models_trt: Dict[str, TensorRTPredictor] = {}
            self.models_trt_path: Dict[str, str] = {}
            for md in models_trt_list:
                name = md['model_name']
                self.models_trt[name] = None
                self.models_trt_path[name] = md['local_path']

        self.face_detectors = FaceDetectors(self)
        self.face_masks = FaceMasks(self)
        self.face_restorers = FaceRestorers(self)
        self.face_swappers = FaceSwappers(self)

        self.arcface_dst = np.array(
            [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
             [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)
        self.emap = []

    # ──────────────────────────────────────────────────────────────────────
    # Базовая загрузка моделей
    # ──────────────────────────────────────────────────────────────────────
    def _emit_loading(self):
        """Эмитирует model_loading_signal только вне warmup."""
        if not getattr(self, '_warmup_running', False):
            self.model_loading.emit()

    def _emit_loaded(self):
        """Эмитирует model_loaded_signal только вне warmup."""
        if not getattr(self, '_warmup_running', False):
            self.model_loaded.emit()

    def load_model(self, model_name, session_options=None):
        with self.model_lock:
            self._emit_loading()
            if session_options is None:
                model_instance = onnxruntime.InferenceSession(
                    self.models_path[model_name], providers=self.providers)
            else:
                model_instance = onnxruntime.InferenceSession(
                    self.models_path[model_name],
                    sess_options=session_options, providers=self.providers)
            if self.models[model_name]:
                del model_instance
                gc.collect()
                return self.models[model_name]
            self._emit_loaded()
            return model_instance

    def delete_models(self):
        for name, inst in self.models.items():
            del inst
            self.models[name] = None
        gc.collect()

    def delete_models_trt(self):
        if TENSORRT_AVAILABLE:
            for md in models_trt_list:
                name = md['model_name']
                if isinstance(self.models_trt.get(name), TensorRTPredictor):
                    self.models_trt[name].cleanup()
                    del self.models_trt[name]
                    self.models_trt[name] = None
            gc.collect()

    def switch_providers_priority(self, provider_name):
        match provider_name:
            case "TensorRT" | "TensorRT-Engine":
                providers = [
                    ('TensorrtExecutionProvider', self.trt_ep_options),
                    ('CUDAExecutionProvider'),
                    ('CPUExecutionProvider'),
                ]
                self.device = 'cuda'
                if (version.parse(trt.__version__) < version.parse("10.2.0")
                        and provider_name == "TensorRT-Engine"):
                    provider_name = "TensorRT"
            case "CPU":
                providers = [('CPUExecutionProvider')]
                self.device = 'cpu'
            case "CUDA":
                providers = [('CUDAExecutionProvider'), ('CPUExecutionProvider')]
                self.device = 'cuda'
        self.providers = providers
        self.provider_name = provider_name
        return self.provider_name

    def set_number_of_threads(self, value):
        self.nThreads = value
        self.delete_models_trt()

    def clear_gpu_memory(self):
        self.delete_models()
        self.delete_models_trt()
        torch.cuda.empty_cache()

    # ──────────────────────────────────────────────────────────────────────
    # Face Swap / Detect / Recognize
    # ──────────────────────────────────────────────────────────────────────
    def load_inswapper_iss_emap(self, model_name):
        with self.model_lock:
            if not self.models[model_name]:
                self._emit_loading()
                graph = onnx.load(self.models_path[model_name]).graph
                self.emap = onnx.numpy_helper.to_array(graph.initializer[-1])
                self._emit_loaded()

    def run_detect(self, img, detect_mode='RetinaFace', max_num=1, score=0.5,
                   input_size=(512, 512)):
        return self.face_detectors.run_detect(
            img, detect_mode, max_num, score, input_size)

    def run_recognize_direct(self, img, kps, similarity_type='Opal',
                             arcface_model='Inswapper128ArcFace'):
        return self.face_swappers.run_recognize_direct(
            img, kps, similarity_type, arcface_model)

    def calc_inswapper_latent(self, source_embedding):
        return self.face_swappers.calc_inswapper_latent(source_embedding)

    def run_inswapper(self, image, embedding, output, sharpness: float = 0.0):
        self.face_swappers.run_inswapper(image, embedding, output, sharpness)

    # ──────────────────────────────────────────────────────────────────────
    # Face Masks (XSeg)
    # ──────────────────────────────────────────────────────────────────────
    def run_dfl_xseg(self, image, output):
        self.face_masks.run_dfl_xseg(image, output)

    def apply_dfl_xseg(self, img, amount):
        return self.face_masks.apply_dfl_xseg(img, amount)

    # ──────────────────────────────────────────────────────────────────────
    # Face Restorers (GFPGAN)
    # ──────────────────────────────────────────────────────────────────────
    def apply_facerestorer(self, swapped_face_upscaled, restorer_det_type,
                           restorer_type, restorer_blend, fidelity_weight, detect_score):
        return self.face_restorers.apply_facerestorer(
            swapped_face_upscaled, restorer_det_type, restorer_type,
            restorer_blend, fidelity_weight, detect_score)

    # ──────────────────────────────────────────────────────────────────────
    # Background Removal — RVM (CUDA)
    # ──────────────────────────────────────────────────────────────────────
    def _make_bg_sess_opts(self) -> onnxruntime.SessionOptions:
        opts = onnxruntime.SessionOptions()
        opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = self.nThreads
        opts.inter_op_num_threads = 1
        return opts

    def _cuda_providers(self) -> list:
        return ['CUDAExecutionProvider', 'CPUExecutionProvider']

    def _trt_providers(self, extra: dict = None) -> list:
        trt_opts = dict(self.trt_ep_options)
        if extra:
            trt_opts.update(extra)
        return [
            ('TensorrtExecutionProvider', trt_opts),
            ('CUDAExecutionProvider'),
            ('CPUExecutionProvider'),
        ]

    def _load_session_with_trt_fallback(self, model_path: str,
                                         sess_opts: onnxruntime.SessionOptions,
                                         trt_providers: list,
                                         label: str) -> onnxruntime.InferenceSession:
        try:
            session = onnxruntime.InferenceSession(
                model_path, sess_options=sess_opts, providers=trt_providers)
            print(f"[{label}] provider={session.get_providers()[0]}")
            return session
        except Exception as e:
            print(f"[{label}] TRT ошибка: {e}\n[{label}] Откат на CUDA...")
            session = onnxruntime.InferenceSession(
                model_path, sess_options=sess_opts, providers=self._cuda_providers())
            print(f"[{label}] provider={session.get_providers()[0]}")
            return session

 