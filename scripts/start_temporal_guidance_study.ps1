param([string]$Config = 'configs/experiments/temporal_guidance_run_20261013/runner.json')
$ErrorActionPreference = 'Stop'
$taskRepo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $env:DEEPSEEK_API_KEY) {
    $env:DEEPSEEK_API_KEY = [Environment]::GetEnvironmentVariable('DEEPSEEK_API_KEY', 'User')
}
if (-not $env:DEEPSEEK_API_KEY) { throw 'DEEPSEEK_API_KEY is required' }
if (-not $env:MOLSTEER_SCNET_PASSWORD) { throw 'MOLSTEER_SCNET_PASSWORD is required' }
$env:PYTHONIOENCODING = 'utf-8'
$taskRunner = Get-Content -LiteralPath (Join-Path $taskRepo $Config) -Raw | ConvertFrom-Json
$taskLogRelative = if ($taskRunner.background_log_dir) { $taskRunner.background_log_dir } else { 'test/temporal_guidance_transport/background' }
$taskLogDir = [IO.Path]::GetFullPath((Join-Path $taskRepo $taskLogRelative))
if (-not $taskLogDir.StartsWith($taskRepo + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Background logs must stay inside the repository'
}
New-Item -ItemType Directory -Force $taskLogDir | Out-Null
$taskPidPath = Join-Path $taskLogDir 'background.pid'
if (Test-Path -LiteralPath $taskPidPath) {
    $taskPreviousPid = [int](Get-Content -LiteralPath $taskPidPath)
    if (Get-Process -Id $taskPreviousPid -ErrorAction SilentlyContinue) { throw "Study already running (PID $taskPreviousPid)" }
}
$taskPython = Join-Path $taskRepo '.venv/Scripts/python.exe'
$taskScript = Join-Path $taskRepo 'scripts/run_temporal_guidance_study.py'
$taskConfigPath = (Resolve-Path (Join-Path $taskRepo $Config)).Path
$taskProcess = Start-Process -FilePath $taskPython -ArgumentList @('-u', ('"' + $taskScript + '"'), '--config', ('"' + $taskConfigPath + '"')) `
    -WorkingDirectory $taskRepo -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $taskLogDir 'background.stdout.log') `
    -RedirectStandardError (Join-Path $taskLogDir 'background.stderr.log')
$taskProcess.Id | Set-Content -LiteralPath $taskPidPath
[PSCustomObject]@{pid=$taskProcess.Id; config=$Config; logs=$taskLogDir; model='deepseek-flash'; error_policy='skip_each_failed_condition_and_continue'} | ConvertTo-Json
