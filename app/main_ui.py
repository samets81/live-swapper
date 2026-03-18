from PySide6 import QtWidgets, QtCore, QtGui
import cv2
import numpy as np

#─────────────────────────────────────────────────────────────────────────────
# Превью лица-донора
#─────────────────────────────────────────────────────────────────────────────
class SourceFaceLabel(QtWidgets.QLabel):
    def __init__(self):
        super().__init__()
        self.setFixedSize(140, 140)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setText("Нет\nфото")
        self.setStyleSheet(
            "border: 2px dashed #555; border-radius: 6px; "
            "background: #1e1e1e; color: #777; font-size: 11px;"
        )

    def set_image(self, img_bgr: np.ndarray):
        if img_bgr is None or img_bgr.size == 0:
            return
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_img = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qt_img)

        scaled_pixmap = pixmap.scaled(
            self.width(), self.height(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )

        x = (scaled_pixmap.width() - self.width()) // 2
        y = (scaled_pixmap.height() - self.height()) // 2
        cropped_pixmap = scaled_pixmap.copy(x, y, self.width(), self.height())

        self.setPixmap(cropped_pixmap)
        self.setText("")

#─────────────────────────────────────────────────────────────────────────────
# Отдельное видео-окно
#─────────────────────────────────────────────────────────────────────────────
class VideoWindow(QtWidgets.QWidget):
    """Всплывает при старте, закрывается при стопе."""
    closed = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent, QtCore.Qt.WindowType.Window)
        self.setWindowTitle("Live-Swapper — Видео")
        self.resize(800, 600)
        self.setMinimumSize(320, 240)
        self.setStyleSheet("background:#111;")

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self.video_label = QtWidgets.QLabel()
        self.video_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.video_label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.video_label.setText("⏳  Инициализация камеры…")
        self.video_label.setStyleSheet("color:#666; font-size:14px;")
        lay.addWidget(self.video_label, stretch=1)

        # Нижняя строка: FPS
        bar = QtWidgets.QHBoxLayout()
        bar.setContentsMargins(8, 2, 8, 4)
        self.fps_label = QtWidgets.QLabel("  ")
        self.fps_label.setStyleSheet(
            "color:#4facc9; font-size:14px; font-weight:bold; background:transparent;"
        )
        bar.addWidget(self.fps_label)
        bar.addStretch()
        lay.addLayout(bar)

    def set_frame(self, frame_bgr: np.ndarray):
        if frame_bgr is None or frame_bgr.size == 0:
            return
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_img = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qt_img).scaled(
            self.video_label.width(), self.video_label.height(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(pixmap)
        self.video_label.setText("")

    def update_fps(self, fps: float):
        self.fps_label.setText(f"{fps:.1f} fps")

    def closeEvent(self, event):
        self.closed.emit()
        super().closeEvent(event)

#─────────────────────────────────────────────────────────────────────────────
# Главное окно
#─────────────────────────────────────────────────────────────────────────────
class MainWindow(QtWidgets.QMainWindow):
    #── Сигналы ───────────────────────────────────────────────────────────────
    select_photo_requested   = QtCore.Signal()
    start_stop_requested     = QtCore.Signal()
    toggle_gfpgan_requested  = QtCore.Signal()
    toggle_xseg_requested    = QtCore.Signal()
    toggle_fps_requested     = QtCore.Signal()
    toggle_virtcam_requested = QtCore.Signal()
    sharpness_changed        = QtCore.Signal(float)
    mask_blur_changed        = QtCore.Signal(int)
    xseg_erosion_changed     = QtCore.Signal(int)
    gfpgan_blend_changed     = QtCore.Signal(int)
    swapper_model_changed    = QtCore.Signal(str)
    swap_res_changed         = QtCore.Signal(int)
    provider_changed         = QtCore.Signal(str)
    camera_changed           = QtCore.Signal(int)
    resolution_changed       = QtCore.Signal(int, int)
    model_loading_signal     = QtCore.Signal()
    model_loaded_signal      = QtCore.Signal()

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Live-Swapper — Настройки")
        self.setMinimumWidth(520)
        self.setMaximumWidth(520)

        self._loading_count = 0
        self.model_loading_signal.connect(self._on_model_loading)
        self.model_loaded_signal.connect(self._on_model_loaded)

        self.video_win = VideoWindow()

        self._build_ui()
        self._apply_styles()
        self.adjustSize()

    #──────────────────────────────────────────────────────────────────────────
    # Построение UI
    #──────────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        root = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(root)
        vbox.setContentsMargins(10, 10, 10, 10)
        vbox.setSpacing(8)

        #── Блок донора ───────────────────────────────────────────────────────
        donor_grp = QtWidgets.QGroupBox("Фото для замены")
        donor_hbox = QtWidgets.QHBoxLayout(donor_grp)
        donor_hbox.setSpacing(10)

        self.source_face_label = SourceFaceLabel()
        donor_hbox.addWidget(self.source_face_label)

        right = QtWidgets.QVBoxLayout()
        right.setSpacing(6)

        self.btn_select = QtWidgets.QPushButton("📁  Выбрать фото")
        self.btn_select.setFixedHeight(34)
        self.btn_select.clicked.connect(self.select_photo_requested)
        right.addWidget(self.btn_select)

        self.btn_start = QtWidgets.QPushButton("▶   Старт")
        self.btn_start.setFixedHeight(42)
        self.btn_start.setObjectName("btn_start")
        self.btn_start.clicked.connect(self.start_stop_requested)
        right.addWidget(self.btn_start)

        self.status_label = QtWidgets.QLabel("Ожидание…")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color:#999; font-size:11px;")
        right.addWidget(self.status_label)

        donor_hbox.addLayout(right, stretch=1)
        vbox.addWidget(donor_grp)

        #── Вкладки настроек ──────────────────────────────────────────────────
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self._tab_main(),    "⚙  Основные")
        self.tabs.addTab(self._tab_filters(), "🎨  Фильтры")
        vbox.addWidget(self.tabs)

        #── Нижняя строка ────────────────────────────────────────────────────
        bottom = QtWidgets.QHBoxLayout()
        self.btn_fps = QtWidgets.QPushButton("FPS: ВЫКЛ")
        self.btn_fps.setFixedHeight(26)
        self.btn_fps.setFixedWidth(120)
        self.btn_fps.clicked.connect(self.toggle_fps_requested)
        bottom.addWidget(self.btn_fps)

        self.btn_obs = QtWidgets.QPushButton("OBS: ВЫКЛ")
        self.btn_obs.setFixedHeight(26)
        self.btn_obs.setFixedWidth(120)
        self.btn_obs.clicked.connect(self.toggle_virtcam_requested)
        bottom.addWidget(self.btn_obs)

        bottom.addStretch()
        vbox.addLayout(bottom)

        self.setCentralWidget(root)

    #── Вкладка «Основные» ────────────────────────────────────────────────────
    def _tab_main(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        # Камера
        cam_grp = QtWidgets.QGroupBox("Веб-камера")
        cam_hbox = QtWidgets.QHBoxLayout(cam_grp)
        cam_hbox.addWidget(QtWidgets.QLabel("Камера:"))
        self.camera_combo = QtWidgets.QComboBox()
        self.camera_combo.currentIndexChanged.connect(self._on_camera_changed)
        cam_hbox.addWidget(self.camera_combo, stretch=1)
        lay.addWidget(cam_grp)

        # Разрешение камеры
        res_grp = QtWidgets.QGroupBox("Разрешение камеры")
        res_hbox = QtWidgets.QHBoxLayout(res_grp)
        res_hbox.addWidget(QtWidgets.QLabel("Режим:"))
        self.resolution_combo = QtWidgets.QComboBox()
        self.resolution_combo.currentIndexChanged.connect(self._on_resolution_changed)
        res_hbox.addWidget(self.resolution_combo, stretch=1)
        lay.addWidget(res_grp)

        # Разрешение свапа
        res_sw_grp = QtWidgets.QGroupBox("Разрешение свапа")
        res_sw_hbox = QtWidgets.QHBoxLayout(res_sw_grp)
        res_sw_hbox.addWidget(QtWidgets.QLabel("Качество:"))
        self.swap_res_combo = QtWidgets.QComboBox()
        for label, val in [("128 — быстро", 128), ("256 — среднее", 256), ("384 — качество", 384)]:
            self.swap_res_combo.addItem(label, val)
        self.swap_res_combo.setCurrentIndex(0)
        self.swap_res_combo.currentIndexChanged.connect(self._on_swap_res_changed)
        res_sw_hbox.addWidget(self.swap_res_combo, stretch=1)
        lay.addWidget(res_sw_grp)

        # Провайдер вычислений
        prov_grp = QtWidgets.QGroupBox("Провайдер")
        prov_hbox = QtWidgets.QHBoxLayout(prov_grp)
        prov_hbox.addWidget(QtWidgets.QLabel("Backend:"))
        self.provider_combo = QtWidgets.QComboBox()
        self.provider_combo.addItem("CUDA", "CUDA")
        self.provider_combo.addItem("TensorRT", "TensorRT")
        self.provider_combo.setCurrentIndex(0)
        self.provider_combo.currentIndexChanged.connect(self._on_provider_changed)
        prov_hbox.addWidget(self.provider_combo, stretch=1)
        lay.addWidget(prov_grp)

        lay.addStretch()
        return w

    #── Вкладка «Фильтры» ─────────────────────────────────────────────────────
    def _tab_filters(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        self._add_slider_float(lay, "Резкость лица", 0.0, 5.0, 0.1, 1.0,
                               self.sharpness_changed, "sharpness")

        self._add_slider_int(lay, "Размытие маски (Blur)", 1, 51, 2, 5,
                             self.mask_blur_changed, "blur")

        #── GFPGAN ────────────────────────────────────────────────────────────
        gfpgan_grp = QtWidgets.QGroupBox("GFPGAN / Face Restorer")
        gfpgan_lay = QtWidgets.QVBoxLayout(gfpgan_grp)
        self.btn_gfpgan = QtWidgets.QPushButton("GFPGAN: ВЫКЛ")
        self.btn_gfpgan.setFixedHeight(30)
        self.btn_gfpgan.clicked.connect(self.toggle_gfpgan_requested)
        gfpgan_lay.addWidget(self.btn_gfpgan)
        self._add_slider_int(gfpgan_lay, "Blend %", 0, 100, 1, 60,
                             self.gfpgan_blend_changed, "gfpgan_blend")
        self.slider_gfpgan_blend.setEnabled(False)
        lay.addWidget(gfpgan_grp)

        #── XSeg ──────────────────────────────────────────────────────────────
        xseg_grp = QtWidgets.QGroupBox("XSeg Mask")
        xseg_lay = QtWidgets.QVBoxLayout(xseg_grp)
        self.btn_xseg = QtWidgets.QPushButton("XSeg: ВЫКЛ")
        self.btn_xseg.setFixedHeight(30)
        self.btn_xseg.clicked.connect(self.toggle_xseg_requested)
        xseg_lay.addWidget(self.btn_xseg)

        erosion_grp = QtWidgets.QGroupBox("Erosion / Dilation")
        e_lay = QtWidgets.QVBoxLayout(erosion_grp)
        self.xseg_erosion_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.xseg_erosion_slider.setRange(-10, 10)
        self.xseg_erosion_slider.setValue(0)
        self.xseg_erosion_slider.setEnabled(False)
        self.xseg_erosion_slider.valueChanged.connect(self._on_xseg_erosion_changed)
        self.xseg_erosion_label = QtWidgets.QLabel("Значение: 0")
        self.xseg_erosion_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        hint = QtWidgets.QLabel("← Эрозия (уменьшить)      Расширение (увеличить) →")
        hint.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        hint.setStyleSheet("color:#777; font-size:10px;")
        e_lay.addWidget(self.xseg_erosion_slider)
        e_lay.addWidget(self.xseg_erosion_label)
        e_lay.addWidget(hint)
        xseg_lay.addWidget(erosion_grp)
        lay.addWidget(xseg_grp)

        lay.addStretch()
        return w

    #── Вспомогательные функции для слайдеров ───────────────────────────────
    def _add_slider_float(self, parent_lay, label_text: str, vmin: float, vmax: float,
                          step: float, default: float, signal, name: str):
        grp = QtWidgets.QGroupBox(label_text)
        hbox = QtWidgets.QHBoxLayout(grp)
        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        int_min, int_max = int(vmin / step), int(vmax / step)
        slider.setRange(int_min, int_max)
        slider.setValue(int(default / step))
        val_lbl = QtWidgets.QLabel(f"{default:.2f}")
        val_lbl.setFixedWidth(50)
        val_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        def on_change(v):
            f = v * step
            val_lbl.setText(f"{f:.2f}")
            signal.emit(f)
        slider.valueChanged.connect(on_change)
        hbox.addWidget(slider)
        hbox.addWidget(val_lbl)
        parent_lay.addWidget(grp)
        setattr(self, f"slider_{name}", slider)
        setattr(self, f"label_{name}_val", val_lbl)

    def _add_slider_int(self, parent_lay, label_text: str, vmin: int, vmax: int,
                        step: int, default: int, signal, name: str):
        grp = QtWidgets.QGroupBox(label_text)
        hbox = QtWidgets.QHBoxLayout(grp)
        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider.setRange(vmin, vmax)
        slider.setSingleStep(step)
        slider.setValue(default)
        val_lbl = QtWidgets.QLabel(str(default))
        val_lbl.setFixedWidth(50)
        val_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        def on_change(v):
            val_lbl.setText(str(v))
            signal.emit(v)
        slider.valueChanged.connect(on_change)
        hbox.addWidget(slider)
        hbox.addWidget(val_lbl)
        parent_lay.addWidget(grp)
        setattr(self, f"slider_{name}", slider)
        setattr(self, f"label_{name}_val", val_lbl)

    #──────────────────────────────────────────────────────────────────────────
    # Внутренние обработчики
    #──────────────────────────────────────────────────────────────────────────
    def _on_camera_changed(self, index: int):
        cam_idx = self.camera_combo.itemData(index)
        if cam_idx is not None:
            self.camera_changed.emit(cam_idx)

    def _on_resolution_changed(self, index: int):
        data = self.resolution_combo.itemData(index)
        if data:
            self.resolution_changed.emit(data[0], data[1])

    def _on_swap_res_changed(self, index: int):
        val = self.swap_res_combo.itemData(index)
        if val is not None:
            self.swap_res_changed.emit(val)

    def _on_provider_changed(self, index: int):
        val = self.provider_combo.itemData(index)
        if val is not None:
            self.provider_changed.emit(val)

    def _on_xseg_erosion_changed(self, value: int):
        self.xseg_erosion_label.setText(f"Значение: {value}")
        self.xseg_erosion_changed.emit(value)

    #── Слоты ModelsProcessor ────────────────────────────────────────────────
    @QtCore.Slot()
    def _on_model_loading(self):
        self._loading_count += 1
        self.set_status(f"Загрузка моделей… ({self._loading_count})", "#f0c040")

    @QtCore.Slot()
    def _on_model_loaded(self):
        self._loading_count = max(0, self._loading_count - 1)
        if self._loading_count == 0:
            self.set_status("Модели загружены ✓", "#66ff66")

    #── Публичные методы ─────────────────────────────────────────────────────
    def show_video_window(self):
        self.video_win.show()
        self.video_win.raise_()

    def hide_video_window(self):
        self.video_win.hide()

    def update_video(self, frame_bgr: np.ndarray):
        self.video_win.set_frame(frame_bgr)

    def update_fps(self, fps: float):
        self.video_win.update_fps(fps)

    def update_preview(self, img_bgr: np.ndarray):
        self.source_face_label.set_image(img_bgr)

    def set_status(self, text: str, color: str = "#aaa"):
        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"color:{color}; font-size:11px;")

    def set_start_button(self, running: bool):
        if running:
            self.btn_start.setText("⏹   Стоп")
            self.btn_start.setStyleSheet(
                "background-color:#7a3a3a; color:#eee; font-weight:bold; font-size:13px;"
            )
        else:
            self.btn_start.setText("▶   Старт")
            self.btn_start.setStyleSheet(
                "background-color:#2d5a2d; color:#eee; font-weight:bold; font-size:13px;"
            )

    def set_fps_state(self, enabled: bool):
        self.btn_fps.setText("FPS: ВКЛ" if enabled else "FPS: ВЫКЛ")
        self.btn_fps.setStyleSheet(
            "background-color:#1e3d4e; color:#4facc9; border:1px solid #4facc9;" if enabled
            else ""
        )
        if not enabled:
            self.video_win.fps_label.setText("  ")

    def set_virtcam_state(self, enabled: bool):
        self.btn_obs.setText("OBS: ВКЛ" if enabled else "OBS: ВЫКЛ")
        self.btn_obs.setStyleSheet(
            "background-color:#1e3d4e; color:#4facc9; border:1px solid #4facc9;" if enabled
            else ""
        )

    def set_gfpgan_state(self, enabled: bool):
        self.btn_gfpgan.setText("GFPGAN: ВКЛ" if enabled else "GFPGAN: ВЫКЛ")
        self.btn_gfpgan.setStyleSheet(
            "background-color:#3a7a3a; color:#eee;" if enabled
            else "background-color:#7a3a3a; color:#eee;"
        )
        self.slider_gfpgan_blend.setEnabled(enabled)

    def set_xseg_state(self, enabled: bool):
        self.btn_xseg.setText("XSeg: ВКЛ" if enabled else "XSeg: ВЫКЛ")
        self.btn_xseg.setStyleSheet(
            "background-color:#3a7a3a; color:#eee;" if enabled
            else "background-color:#7a3a3a; color:#eee;"
        )
        self.xseg_erosion_slider.setEnabled(enabled)

    def set_camera_combo_enabled(self, enabled: bool):
        self.camera_combo.setEnabled(enabled)

    def set_resolution_combo_enabled(self, enabled: bool):
        self.resolution_combo.setEnabled(enabled)

    def populate_cameras(self, cameras: list):
        self.camera_combo.blockSignals(True)
        self.camera_combo.clear()
        for idx, name in cameras:
            self.camera_combo.addItem(name, idx)
        self.camera_combo.blockSignals(False)

    def populate_resolutions(self, resolutions: list):
        self.resolution_combo.blockSignals(True)
        self.resolution_combo.clear()
        for w, h, label in resolutions:
            self.resolution_combo.addItem(label, (w, h))
        if self.resolution_combo.count() > 0:
            self.resolution_combo.setCurrentIndex(0)
        self.resolution_combo.blockSignals(False)

    def set_provider(self, provider: str):
        idx = self.provider_combo.findData(provider)
        if idx >= 0:
            self.provider_combo.blockSignals(True)
            self.provider_combo.setCurrentIndex(idx)
            self.provider_combo.blockSignals(False)

    def set_slider_value(self, name: str, value: float, step: float):
        slider = getattr(self, f"slider_{name}", None)
        value_label = getattr(self, f"label_{name}_val", None)
        if slider is None or value_label is None:
            return
        int_value = int(value) if step == 1 else int(round(value / step))
        slider.blockSignals(True)
        slider.setValue(int_value)
        slider.blockSignals(False)
        value_label.setText(f"{value:.2f}" if step < 1 else str(int(value)))

    def show_error(self, title: str, msg: str):
        QtWidgets.QMessageBox.critical(self, title, msg)

    #── Стили ────────────────────────────────────────────────────────────────
    def _apply_styles(self):
        common = """
            QMainWindow, QWidget {
                background-color: #1a1a1a;
                color: #ddd;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 12px;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #3a3a3a;
                border-radius: 5px;
                margin-top: 8px;
                padding-top: 8px;
                font-size: 11px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 4px;
                color: #5ab3d4;
            }
            QSlider::groove:horizontal {
                height: 6px; background: #333; border-radius: 3px;
            }
            QSlider::sub-page:horizontal {
                background: #4facc9; border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #fff; border: 2px solid #4facc9;
                width: 14px; margin: -4px 0; border-radius: 7px;
            }
            QSlider::handle:horizontal:hover { background: #4facc9; }
            QSlider::handle:horizontal:disabled { background: #444; border-color: #555; }
            QTabWidget::pane {
                border: 1px solid #333; border-radius: 4px; background: #1a1a1a;
            }
            QTabBar::tab {
                background: #252525; border: 1px solid #333;
                border-bottom: none; padding: 6px 14px;
                margin-right: 2px; color: #999; font-size: 11px;
            }
            QTabBar::tab:selected {
                background: #1a1a1a; color: #5ab3d4;
                border-top: 2px solid #4facc9;
            }
            QTabBar::tab:hover:!selected { background: #2a2a2a; }
            QPushButton {
                background-color: #2a2a2a; border: 1px solid #444;
                border-radius: 4px; padding: 4px 10px; color: #ddd;
            }
            QPushButton:hover { background-color: #333; border-color: #4facc9; }
            QPushButton:pressed { background-color: #1e3d4e; }
            QPushButton#btn_start {
                background-color: #2d5a2d; color: #eee;
                font-weight: bold; font-size: 13px;
            }
            QPushButton:disabled {
                background-color: #333; color: #666; border-color: #444;
            }
            QComboBox {
                background-color: #252525; border: 1px solid #444;
                border-radius: 3px; padding: 4px 8px; color: #ddd;
            }
            QComboBox QAbstractItemView {
                background: #252525; selection-background-color: #1e3d4e; color: #ddd;
            }
            QComboBox:disabled {
                background-color: #222; color: #666; border-color: #333;
            }
            QLabel { color: #ddd; }
            QLineEdit {
                background-color: #252525; border: 1px solid #444;
                border-radius: 3px; padding: 4px 8px; color: #ddd;
            }
            QLineEdit:disabled {
                background-color: #222; color: #666; border-color: #333;
            }
        """
        self.setStyleSheet(common)
        self.video_win.setStyleSheet("background:#111; color:#ddd;")