param(
    [string]$Config = 'configs/experiments/m2_skill_api_20261011/runner.json'
)
$ErrorActionPreference = 'Stop'
$taskRepo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $env:DEEPSEEK_API_KEY) {
    $env:DEEPSEEK_API_KEY = [Environment]::GetEnvironmentVariable('DEEPSEEK_API_KEY', 'User')
}
if (-not $env:DEEPSEEK_API_KEY) { throw 'DEEPSEEK_API_KEY is required' }
if (-not $env:MOLSTEER_SCNET_PASSWORD) { throw 'MOLSTEER_SCNET_PASSWORD is required in the launching process' }
$taskConfigPath = (Resolve-Path (Join-Path $taskRepo $Config)).Path
$taskSettings = Get-Content -LiteralPath $taskConfigPath -Raw | ConvertFrom-Json
$taskLogDir = Join-Path $taskRepo ('test/m2_skill_api_transport/' + (Split-Path $taskSettings.study -Leaf))
New-Item -ItemType Directory -Force $taskLogDir | Out-Null
$taskPidPath = Join-Path $taskLogDir 'background.pid'
if (Test-Path -LiteralPath $taskPidPath) {
    $taskPreviousPid = [int](Get-Content -LiteralPath $taskPidPath)
    if (Get-Process -Id $taskPreviousPid -ErrorAction SilentlyContinue) { throw "Study already running (PID $taskPreviousPid)" }
}
$taskPython = Join-Path $taskRepo '.venv/Scripts/python.exe'
$taskScript = Join-Path $taskRepo 'scripts/run_m2_skill_api_study.py'
$taskProcess = Start-Process -FilePath $taskPython -ArgumentList @('-u', ('"' + $taskScript + '"'), '--config', ('"' + $taskConfigPath + '"')) `
    -WorkingDirectory $taskRepo -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $taskLogDir 'background.stdout.log') `
    -RedirectStandardError (Join-Path $taskLogDir 'background.stderr.log')
$taskProcess.Id | Set-Content -LiteralPath $taskPidPath
[PSCustomObject]@{pid=$taskProcess.Id; config=$Config; logs=$taskLogDir; model='deepseek-flash'; backend='real_api'} | ConvertTo-Json
