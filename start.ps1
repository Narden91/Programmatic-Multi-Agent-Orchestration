param (
    [switch]$Setup,
    [switch]$Build
)

$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host " Programmatic Multi-Agent Orchestration   " -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# 1. Check requirements
if (-not (Get-Command "uv" -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] 'uv' is not installed or not in PATH." -ForegroundColor Red
    exit 1
}
if (-not (Get-Command "npm" -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] 'npm' is not installed or not in PATH." -ForegroundColor Red
    exit 1
}

# 2. Clean Linux artifacts if moving from WSL
if (Test-Path "frontend\node_modules\@rollup\rollup-linux-x64-gnu") {
    Write-Host "[moe] Detected leftover Linux node_modules. Cleaning for Windows native..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force "frontend\node_modules" -ErrorAction SilentlyContinue
}

# 3. Setup Dependencies
if (-not (Test-Path ".venv") -or $Setup) {
    Write-Host "[moe] Syncing Python dependencies (uv sync)..." -ForegroundColor Yellow
    & uv sync
}

if (-not (Test-Path "frontend\node_modules") -or $Setup) {
    Write-Host "[moe] Installing Frontend dependencies (npm install)..." -ForegroundColor Yellow
    Push-Location frontend
    & npm install --no-audit --no-fund
    Pop-Location
}

if ($Build) {
    Write-Host "[moe] Building Frontend for Production..." -ForegroundColor Yellow
    Push-Location frontend
    & npm run build
    Pop-Location
    Write-Host "[moe] Build complete." -ForegroundColor Green
    exit 0
}

# 4. Run Services Concurrently
Write-Host ""
Write-Host "[moe] Starting Services! Press Ctrl+C to Stop." -ForegroundColor Green
Write-Host "Frontend: http://localhost:5173" -ForegroundColor White
Write-Host "Backend : http://127.0.0.1:8000" -ForegroundColor White
Write-Host ""

& npx concurrently -k -c "cyan,magenta" -n "BACKEND,FRONTEND" "uv run uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload" "npm run dev --prefix frontend"
