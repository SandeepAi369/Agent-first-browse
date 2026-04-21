<#
.SYNOPSIS
    Agent First IDE — Windows-Native Launcher

.DESCRIPTION
    Launches the Agent First IDE campaign on Windows natively.
    1. Finds or creates a Python venv
    2. Installs dependencies
    3. Optionally starts a browser with CDP debugging
    4. Runs start_campaign.py

.NOTES
    Run from PowerShell:
        .\launch.ps1
    
    Or with browser auto-start:
        .\launch.ps1 -StartBrowser
#>

param(
    [switch]$StartBrowser,
    [switch]$Install,
    [string]$BrowserPreference = "edge"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host ""
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "  AGENT FIRST IDE — Windows-Native Launcher" -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host ""

# ── Step 1: Find Python ──────────────────────────────────────────────────────

$PythonCmd = $null
foreach ($candidate in @("python", "python3", "py")) {
    $found = Get-Command $candidate -ErrorAction SilentlyContinue
    if ($found) {
        $PythonCmd = $candidate
        break
    }
}

if (-not $PythonCmd) {
    Write-Host "[ERROR] Python not found in PATH." -ForegroundColor Red
    Write-Host "Install Python 3.11+ from https://python.org and ensure it is in your PATH."
    Read-Host "Press Enter to exit"
    exit 1
}

$PyVersion = & $PythonCmd --version 2>&1
Write-Host "[OK] Python found: $PyVersion" -ForegroundColor Green

# ── Step 2: Virtual Environment ──────────────────────────────────────────────

$VenvDir = Join-Path $ProjectRoot ".venv-windows"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$VenvPip = Join-Path $VenvDir "Scripts\pip.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "[INFO] Creating virtual environment at $VenvDir..." -ForegroundColor Yellow
    & $PythonCmd -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Failed to create virtual environment." -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
    $Install = $true  # Force install on fresh venv
}

Write-Host "[OK] Virtual environment: $VenvDir" -ForegroundColor Green

# ── Step 3: Install Dependencies ─────────────────────────────────────────────

if ($Install) {
    Write-Host "[INFO] Installing dependencies..." -ForegroundColor Yellow
    & $VenvPip install --upgrade pip | Out-Null
    & $VenvPip install -e "$ProjectRoot[dev]"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Dependency installation failed." -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }

    Write-Host "[INFO] Installing Playwright browsers..." -ForegroundColor Yellow
    & $VenvPython -m playwright install chromium
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[WARNING] Playwright browser install may have failed. Continuing anyway." -ForegroundColor Yellow
    }

    Write-Host "[OK] Dependencies installed successfully." -ForegroundColor Green
}

# ── Step 4: Start Browser (optional) ─────────────────────────────────────────

if ($StartBrowser) {
    $Port = 9222
    
    $EdgePaths = @(
        "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        "C:\Program Files\Microsoft\Edge\Application\msedge.exe"
    )
    $ChromePaths = @(
        "C:\Program Files\Google\Chrome\Application\chrome.exe",
        "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
    )

    if ($BrowserPreference -eq "chrome") {
        $BrowserPaths = $ChromePaths + $EdgePaths
    } else {
        $BrowserPaths = $EdgePaths + $ChromePaths
    }

    $BrowserExe = $BrowserPaths | Where-Object { Test-Path $_ } | Select-Object -First 1

    if ($BrowserExe) {
        $ProfileDir = Join-Path $env:USERPROFILE ".agent-first-ide-cdp"
        New-Item -ItemType Directory -Force -Path $ProfileDir | Out-Null

        $BrowserArgs = @(
            "--remote-debugging-port=$Port",
            "--remote-debugging-address=0.0.0.0",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-blink-features=AutomationControlled",
            "--disable-features=IsolateOrigins,SitePerProcess,OptimizeImageLoading",
            "--password-store=basic",
            "--start-maximized",
            "--user-data-dir=$ProfileDir"
        )

        Write-Host "[INFO] Launching browser: $BrowserExe" -ForegroundColor Yellow
        Start-Process -FilePath $BrowserExe -ArgumentList $BrowserArgs
        Start-Sleep -Seconds 3
        Write-Host "[OK] Browser launched with CDP on port $Port." -ForegroundColor Green
    } else {
        Write-Host "[WARNING] No browser found. Start Edge/Chrome manually with --remote-debugging-port=9222" -ForegroundColor Yellow
    }
}

# ── Step 5: Run Campaign ─────────────────────────────────────────────────────

Write-Host ""
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "  Starting Campaign..." -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host ""

$env:PYTHONPATH = Join-Path $ProjectRoot "python-orchestrator"

& $VenvPython (Join-Path $ProjectRoot "start_campaign.py")

Write-Host ""
Write-Host "[DONE] Campaign launcher finished." -ForegroundColor Green
Read-Host "Press Enter to exit"
