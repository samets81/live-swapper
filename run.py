import sys
import os

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from PySide6 import QtWidgets, QtCore
from PySide6.QtCore import Qt
import qdarktheme
from app.main_ui import MainWindow
from app.app import FaceSwapApp


class ProxyStyle(QtWidgets.QProxyStyle):
    def styleHint(self, hint, opt=None, widget=None, returnData=None) -> int:
        res = super().styleHint(hint, opt, widget, returnData)
        if hint == self.StyleHint.SH_Slider_AbsoluteSetButtons:
            res = Qt.LeftButton.value
        return res


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle(ProxyStyle())
    app.setApplicationName("Live-Swapper")

    try:
        with open("app/theme.qss", "r", encoding="utf-8") as f:
            _style = f.read()
        _style = qdarktheme.load_stylesheet(custom_colors={"primary": "#4facc9"}) + '\n' + _style
        app.setStyleSheet(_style)
    except Exception as e:
        print(f"[WARNING] Theme error: {e}")
        app.setStyleSheet(qdarktheme.load_stylesheet(custom_colors={"primary": "#4facc9"}))

    window = MainWindow()
    backend = FaceSwapApp(window)

    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()