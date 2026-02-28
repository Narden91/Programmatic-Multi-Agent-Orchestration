param(
  [switch]$SkipFrontendInstall,
  [switch]$SkipBackendSync
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "[MoE] Project root: $root"
Set-Location $root

if (-not $SkipBackendSync) {
  Write-Host "[MoE] Syncing Python env with uv..."
  uv sync
}

if (-not $SkipFrontendInstall) {
  Write-Host "[MoE] Installing frontend dependencies..."
  npm install --prefix frontend
}

Write-Host "[MoE] Starting backend on http://127.0.0.1:8000 ..."
$backend = Start-Process -FilePath "uv" -ArgumentList "run", "uvicorn", "api.main:app", "--reload", "--host", "127.0.0.1", "--port", "8000" -PassThru -WindowStyle Normal

$ready = $false
for ($i = 0; $i -lt 40; $i++) {
  Start-Sleep -Milliseconds 300
  try {
    $resp = Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/health" -UseBasicParsing -TimeoutSec 2
    if ($resp.StatusCode -eq 200) {
      $ready = $true
      break
    }
  } catch {
  }
}

if (-not $ready) {
  Write-Host "[MoE] Backend did not become ready on :8000. Stop process id $($backend.Id) and inspect logs."
  exit 1
}

Write-Host "[MoE] Backend ready. Starting frontend on http://127.0.0.1:5173 ..."
$frontend = Start-Process -FilePath "npm" -ArgumentList "run", "dev", "--prefix", "frontend" -PassThru -WindowStyle Normal

Write-Host "[MoE] ✅ Both services started"
Write-Host "       Frontend: http://127.0.0.1:5173"
Write-Host "       Backend : http://127.0.0.1:8000/api/health"
Write-Host "       Backend PID: $($backend.Id)"
Write-Host "       Frontend PID: $($frontend.Id)"
Write-Host "[MoE] Use Stop-Process -Id <PID> to stop services."
