@echo off
chcp 65001 > nul
title Lung CT Scan Search (K-Top 5 Vector DB)

echo =====================================================================
echo   ระบบค้นหาและเปรียบเทียบภาพ CT Scan ปอด (K-Top 5 Vector Database)
echo =====================================================================
echo.

:: ตรวจสอบคำสั่ง uv
where uv >nul 2>nul
if %errorlevel% neq 0 (
    echo [คำเตือน] ไม่พบคำสั่ง uv ในระบบ กำลังรันผ่าน python ปกติ...
    python -m src.ui
    goto end
)

:: ตรวจสอบ virtual environment
if not exist ".venv" (
    echo [*] กำลังสร้าง Virtual Environment ด้วย uv...
    uv venv --python 3.11 --system-site-packages .venv
)

echo [*] รันระบบด้วย uv mode: uv run python -m src.ui
echo [*] เปิดเว็บเบราว์เซอร์ที่: http://127.0.0.1:7860
echo.

uv run python -m src.ui

:end
if %errorlevel% neq 0 (
    echo.
    echo [เกิดข้อผิดพลาดในการรันโปรแกรม]
)
pause
