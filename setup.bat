@echo off
chcp 65001 >nul

set KAGGLE_USERNAME=azamov21
set KAGGLE_KEY=KGAT_042b05590eda12d3b4e5c1e7316fb62a

echo.
echo #####################################################
echo #       SNORE DETECTOR - Windows Setup              #
echo #####################################################
echo.
echo  Kaggle foydalanuvchi : %KAGGLE_USERNAME%
echo  Ishchi papka         : %CD%
echo  Vaqt                 : %DATE% %TIME%
echo.
echo #####################################################
echo.

:: -------------------------------------------------------
echo [QADAM 1/5]  Python versiyasi tekshirilmoqda...
echo -------------------------------------------------------
python --version
if %ERRORLEVEL% neq 0 (
    echo.
    echo  [XATO] Python topilmadi!
    echo  https://www.python.org/downloads/ dan yuklab o'rnating
    pause & exit /b 1
)
echo  [OK] Python tayyor
echo.

:: -------------------------------------------------------
echo [QADAM 2/5]  Virtual environment yaratilmoqda...
echo -------------------------------------------------------
if exist venv (
    echo  [MA'LUMOT] venv allaqachon mavjud, o'tkazildi
) else (
    python -m venv venv
    echo  [OK] venv yaratildi
)
call venv\Scripts\activate
echo  [OK] venv faollashtirildi
echo.

:: -------------------------------------------------------
echo [QADAM 3/5]  Kutubxonalar o'rnatilmoqda...
echo -------------------------------------------------------
echo  (bu bir necha daqiqa olishi mumkin)
echo.
pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo.
    echo  [XATO] pip install muvaffaqiyatsiz!
    pause & exit /b 1
)
echo.
echo  [OK] Barcha kutubxonalar o'rnatildi
echo.

:: -------------------------------------------------------
echo [QADAM 4/5]  Dataset yuklanmoqda (Kaggle)...
echo -------------------------------------------------------
echo  Mavjud data:
if exist dataset\0 (
    for /f %%A in ('dir /b dataset\0\*.wav 2^>nul ^| find /c ".wav"') do echo   dataset\0\  - %%A ta fayl
) else (
    echo   dataset\0\  - bo'sh
)
if exist dataset\1 (
    for /f %%A in ('dir /b dataset\1\*.wav 2^>nul ^| find /c ".wav"') do echo   dataset\1\  - %%A ta fayl
) else (
    echo   dataset\1\  - bo'sh
)
echo.
python download_data.py
if %ERRORLEVEL% neq 0 (
    echo.
    echo  [XATO] Data yuklash muvaffaqiyatsiz!
    pause & exit /b 1
)
echo.
echo  Yangilangan data:
for /f %%A in ('dir /b dataset\0\*.wav 2^>nul ^| find /c ".wav"') do echo   dataset\0\  - %%A ta fayl
for /f %%A in ('dir /b dataset\1\*.wav 2^>nul ^| find /c ".wav"') do echo   dataset\1\  - %%A ta fayl
echo.

:: -------------------------------------------------------
echo [QADAM 5/5]  Model o'qitilmoqda...
echo -------------------------------------------------------
echo  GPU tekshirilmoqda:
python -c "import torch; print('  GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'TOPILMADI - CPU ishlatiladi')"
echo.
echo  O'qitish boshlanmoqda (uzoq vaqt oladi)...
echo.
python train_and_export.py
if %ERRORLEVEL% neq 0 (
    echo.
    echo  [XATO] O'qitish muvaffaqiyatsiz!
    pause & exit /b 1
)
echo.

:: -------------------------------------------------------
echo [EXPORT]  Android uchun model export qilinmoqda...
echo -------------------------------------------------------
python export_mobile.py
if %ERRORLEVEL% neq 0 (
    echo.
    echo  [XATO] Export muvaffaqiyatsiz!
    pause & exit /b 1
)
echo.

:: -------------------------------------------------------
echo #####################################################
echo #                  HAMMASI TAYYOR!                  #
echo #####################################################
echo.
echo  Yaratilgan fayllar:
if exist best_model.pt (
    for %%A in (best_model.pt) do echo   best_model.pt      - %%~zA bayt  [PyTorch modeli]
)
if exist snore_model.onnx (
    for %%A in (snore_model.onnx) do echo   snore_model.onnx   - %%~zA bayt  [Android ONNX]
)
if exist snore_mobile.ptl (
    for %%A in (snore_mobile.ptl) do echo   snore_mobile.ptl   - %%~zA bayt  [Android Mobile]
)
echo.
echo  Modelni sinash uchun:
echo    python test_model.py audio.wav
echo.
echo #####################################################
echo.
pause
