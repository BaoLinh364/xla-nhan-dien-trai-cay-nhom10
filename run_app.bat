@echo off
rem Chay app nhan dien trai cay. Lan dau tu tao .venv va cai thu vien (can Internet).
cd /d "%~dp0"

if not exist ".venv\installed.ok" (
    echo [1/2] Dang tao moi truong .venv ...
    if not exist ".venv\Scripts\python.exe" (
        py -3.12 -m venv .venv 2>nul || python -m venv .venv || goto :loi
    )
    echo [2/2] Dang cai thu vien, co the mat vai phut ...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip || goto :loi
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :loi
    echo ok> ".venv\installed.ok"
)

".venv\Scripts\python.exe" app.py
if errorlevel 1 pause
goto :eof

:loi
echo.
echo LOI: khong cai dat duoc. Hay cai Python 3.12 tu https://www.python.org/downloads/
echo (nho tick "Add python.exe to PATH"), xoa thu muc .venv roi chay lai file nay.
pause
