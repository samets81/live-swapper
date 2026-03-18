@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ========================================
echo Live-Swapper - Запуск
echo ========================================
echo.
echo [INFO] Папка программы %cd%




echo [INFO] Активация окружения...
call venv\Scripts\activate

echo [INFO] Загрузка программы...
python run.py
pause