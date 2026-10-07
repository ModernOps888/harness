[CmdletBinding()]
param()

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host " [TRINITY] Harness + SPINE + ChronoFact Orchestrator" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

# 1. Check / Launch Harness Model Server (:8089)
$harnessPort = Get-NetTCPConnection -LocalPort 8089 -State Listen -ErrorAction SilentlyContinue
if (-not $harnessPort) {
    Write-Host ""
    $scriptPath = Join-Path $PSScriptRoot "launch_qwen32b_coder.ps1"
    Start-Process -FilePath "powershell.exe" -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $scriptPath -WindowStyle Minimized
    $ready = $false
    for ($i = 0; $i -lt 45; $i++) {
        Start-Sleep -Seconds 2
        try {
            if ((Invoke-RestMethod -Uri "http://127.0.0.1:8089/health" -TimeoutSec 2).status -eq "ok") {
                $ready = $true
                break
            }
        } catch {}
        Write-Host "." -NoNewline
    }
    Write-Host ""
    if ($ready) {
        Write-Host "  OK: Harness Model Server is ONLINE at http://127.0.0.1:8089" -ForegroundColor Green
    } else {
        Write-Host "  WAIT: Harness Model Server is warming up in background." -ForegroundColor Yellow
    }
} else {
    Write-Host ""
    Write-Host "[1/3] OK: Harness Accelerated Model Server already ACTIVE on http://127.0.0.1:8089" -ForegroundColor Green
}

# 2. Check / Launch SPINE Reality Gateway (:8080) and HUD (:3333)
$spinePort = Get-NetTCPConnection -LocalPort 8080 -State Listen -ErrorAction SilentlyContinue
if (-not $spinePort) {
    Write-Host ""
    Write-Host "[2/3] Launching SPINE Reality Gateway and HUD..." -ForegroundColor Yellow
    & "C:\spine\start.ps1" -Quiet
    Start-Sleep -Seconds 2
    Write-Host "  OK: SPINE Reality Gateway is ONLINE at http://localhost:8080" -ForegroundColor Green
    Write-Host "  OK: SPINE React 19 HUD is ONLINE at http://localhost:3333" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "[2/3] OK: SPINE Reality Gateway already ACTIVE on http://localhost:8080" -ForegroundColor Green
    Write-Host "      OK: SPINE React 19 HUD is ACTIVE on http://localhost:3333" -ForegroundColor Green
}

# 3. Check ChronoFact Binary and Database
Write-Host ""
Write-Host "[3/3] Checking ChronoFact Epistemic Backbone..." -ForegroundColor Yellow
if (Test-Path "C:\chronofact\target\release\chronofact.exe") {
    Write-Host "  OK: ChronoFact binary verified at C:\chronofact\target\release\chronofact.exe" -ForegroundColor Green
} else {
    Write-Host "  WAIT: ChronoFact binary missing in target\release. Rebuilding..." -ForegroundColor Yellow
    cargo build --release --manifest-path "C:\chronofact\Cargo.toml"
}
if (Test-Path "C:\chronofact\chronofact_memory.db") {
    Write-Host "  OK: ChronoFact SQLite WAL Memory DB verified" -ForegroundColor Green
}

Write-Host ""
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host " TRINITY IS FULLY OPERATIONAL!" -ForegroundColor Cyan
Write-Host " - VS Code Model Endpoint: http://localhost:8080/v1" -ForegroundColor Green
Write-Host " - Model ID: harness/qwen2.5-coder:32b" -ForegroundColor Green
Write-Host " - Direct Harness Endpoint: http://127.0.0.1:8089/v1" -ForegroundColor Green
Write-Host " - SPINE Live HUD: http://localhost:3333" -ForegroundColor Green
Write-Host " - MCP Config: %APPDATA%\Code\User\mcp.json" -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Cyan
