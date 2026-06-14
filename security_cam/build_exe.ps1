# Сборка .exe для Security Camera System
# Запускать из папки security_cam:
#   powershell -ExecutionPolicy Bypass -File build_exe.ps1

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $ProjectDir

# 1. Проверка зависимостей
Write-Host "=== Установка зависимостей ===" -ForegroundColor Cyan
pip install -r requirements.txt
pip install pyinstaller

if ($LASTEXITCODE -ne 0) {
    Write-Host "Ошибка установки зависимостей!" -ForegroundColor Red
    exit 1
}

# 2. Сборка
Write-Host "=== Сборка .exe ===" -ForegroundColor Cyan

$PyInstallerArgs = @(
    "--onefile",
    "--windowed",
    "--name", "SecurityGuard",
    "--icon", "NONE",
    "--add-data", "config.py;.",
    "--hidden-import", "tensorflow",
    "--hidden-import", "tensorflow_hub",
    "--hidden-import", "easyocr",
    "--hidden-import", "ultralytics",
    "--hidden-import", "sounddevice",
    "--hidden-import", "PIL._tkinter_finder",
    "--collect-all", "ultralytics",
    "--collect-all", "easyocr",
    "run.py"
)

pyinstaller @PyInstallerArgs

if ($LASTEXITCODE -eq 0) {
    Write-Host "=== Готово! ===" -ForegroundColor Green
    Write-Host "Файл: $ProjectDir\dist\SecurityGuard.exe" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "ВАЖНО: При первом запуске .exe скачает модели:" -ForegroundColor Cyan
    Write-Host "  - YOLOv8n (~6 MB)" -ForegroundColor Gray
    Write-Host "  - YAMNet (~15 MB)" -ForegroundColor Gray
    Write-Host "  - EasyOCR (~30 MB)" -ForegroundColor Gray
    Write-Host "Это может занять несколько минут." -ForegroundColor Yellow
} else {
    Write-Host "Ошибка сборки!" -ForegroundColor Red
    exit 1
}
