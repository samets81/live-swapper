@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ========================================
echo Live-Swapper - Установка
echo ========================================
echo.

echo [INFO] Создание виртуального окружения...
python -m venv venv

echo [INFO] Активация окружения...
call venv\Scripts\activate

python.exe -m pip install --upgrade pip

echo [INFO] Установка Python-зависимостей...
python -m pip install -r requirements.txt

echo [INFO] Скачивание моделей...
python down_models.py

echo.
echo ========================================
echo [INSTALL COMPLETE] 
echo Запуск приложения: run.bat
echo ========================================
pause
