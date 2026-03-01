<#
.SYNOPSIS
A Windows wrapper to securely start the Programmatic Multi-Agent Orchestration application inside the WSL subsystem.

.DESCRIPTION
This script prevents Windows developers from accidentally downloading `node_modules` or `uv` environments containing Windows native binaries (which crash Linux). By acting as a proxy, it forwards executing commands strictly into the `wsl` linux subsystem.

.EXAMPLE
.\start.ps1
.\start.ps1 --setup
#>

$ScriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
# Replace backslashes so wsl does not treat them as bash escape sequences
$PosixStylePath = $ScriptPath.Replace('\', '/')

# Convert Windows path to WSL path 
# (e.g., C:/Users/emanu/ -> /mnt/c/Users/emanu/)
$WslPath = wsl wslpath "$PosixStylePath"
if ($LASTEXITCODE -ne 0) {
    Write-Host "[moe] wsl command failed. Is Windows Subsystem for Linux (WSL) installed?" -ForegroundColor Red
    exit 1
}

# Collect args array and wrap them into a single bash string
$ArgsString = $args -join " "

Write-Host "[moe] Launching application from inside Windows Subsystem for Linux (WSL)..." -ForegroundColor Cyan
Write-Host "[moe] Forwarding command: bash start.sh $ArgsString" -ForegroundColor DarkGray

# Enter the directory and execute the bash script directly inside WSL
wsl bash -c "cd '$WslPath' && bash start.sh $ArgsString"
