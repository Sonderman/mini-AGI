@echo off
rem mini-AGI - web arayuzu baslatici (start.bat)
rem
rem Kullanim:
rem   start.bat                servis, ogrenme acik (serve.py varsayilani)
rem   start.bat --no-learn     salt-okunur servis
rem   Ek argumanlar serve.py'ye gecilir. Port degistirmek icin asagidaki
rem   PORT satirini duzenle - tarayici ve cakisma kontrolu da onu kullanir.
rem
rem Sunucu bu pencerede calisir; durdurmak icin Ctrl+C ya da pencereyi kapat.
setlocal EnableExtensions
cd /d "%~dp0"

set "PY=.venv\Scripts\python.exe"
set "PORT=8080"
set "SYS=%SystemRoot%\System32"
title mini-AGI serve - port %PORT%

if not exist "%PY%" (
    echo  [HATA] Sanal ortam bulunamadi: %~dp0.venv
    echo  Once ortami kur; ayrinti icin AGENTS.md - Environment bolumu:
    echo.
    echo    uv venv --python 3.13 .venv
    echo    uv pip install -p .venv\Scripts\python.exe "torch==2.9.1" --index-url https://download.pytorch.org/whl/cu128
    echo    uv pip install -p .venv\Scripts\python.exe numpy pyyaml matplotlib flask chess zstandard datasets scipy
    echo.
    pause
    exit /b 1
)

rem Zaten calisan bir sunucu varsa tarayiciyi ac ve cik.
rem Not: curl ve ping tam yol ile cagrilir; PATH'te baska bir surum
rem olsa bile sistem araclari kullanilir.
if not exist "%SYS%\curl.exe" goto :poll_slow
"%SYS%\curl.exe" -s -o nul --max-time 2 http://127.0.0.1:%PORT%/
if not errorlevel 1 goto :already

rem Sunucu hazir oldugunda tarayiciyi ac - aktif bekleme.
start "" /b cmd /c "@echo off & for /L %%i in (1,1,180) do (%SYS%\curl.exe -s -o nul --max-time 2 http://127.0.0.1:%PORT%/ && (start http://127.0.0.1:%PORT% & exit /b) || %SYS%\ping.exe -n 2 127.0.0.1 >nul)"
goto :serve

:already
echo  Sunucu zaten calisiyor: http://127.0.0.1:%PORT%
echo  Tarayici aciliyor.
start http://127.0.0.1:%PORT%
exit /b 0

:poll_slow
rem curl yok - sabit gecikmeyle tarayiciyi ac.
start "" /b cmd /c "%SYS%\ping.exe -n 13 127.0.0.1 >nul & start http://127.0.0.1:%PORT%"
goto :serve

:serve
echo.
echo  mini-AGI web arayuzu
echo  ====================
echo  Adres: http://127.0.0.1:%PORT%
echo  Model yukleniyor, hazir olunca tarayici otomatik acilir.
echo  Durdurmak icin: bu pencerede Ctrl+C ya da pencereyi kapat.
echo  Ipucu: ogrenme olmadan calistirmak icin: start.bat --no-learn
echo.

"%PY%" serve.py --port %PORT% %*
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
    echo.
    echo  serve.py hata ile cikti - kod %RC%
    pause
)
exit /b %RC%
