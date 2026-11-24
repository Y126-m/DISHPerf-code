param(
 [ValidateSet('check','prepare','plan','smoke','run','reuse','summarize','plot')][string]$Stage='plan',
 [ValidateSet('all','x264','lrzip','h2')][string]$System='all',
 [int]$Budget=0,[ValidateRange(1,5)][int]$RunCount=5,[ValidateRange(0,4)][int]$StartRep=0,
 [ValidateSet('all','dishperf_native','hyperf')][string]$Model='all',
 [ValidateSet('formal','smoke')][string]$Profile='formal',
 [switch]$RetryFailed,[switch]$SavePredictiveDraws,
 [string]$ReuseRoot='', [string]$OutputRoot='', [string]$MainPython='', [string]$NativePython=''
)
$ErrorActionPreference='Stop'
$config=Get-Content -LiteralPath (Join-Path $PSScriptRoot 'settings.json') -Raw -Encoding UTF8 | ConvertFrom-Json
if (-not $MainPython) {$MainPython=$config.main_python}
if (-not $NativePython) {$NativePython=$config.native_python}
if (-not (Test-Path -LiteralPath $MainPython)) {throw "Experiment Python missing: $MainPython"}
$env:PYTHONIOENCODING='utf-8';$env:PYTHONDONTWRITEBYTECODE='1';$env:MPLBACKEND='Agg'
$cmdArgs=@('-B','-u',(Join-Path $PSScriptRoot 'experiment.py'),'--stage',$Stage,'--system',$System,'--budget',"$Budget",'--run-count',"$RunCount",'--start-rep',"$StartRep",'--model',$Model,'--profile',$Profile,'--main-python',$MainPython,'--native-python',$NativePython)
if ($RetryFailed) {$cmdArgs+='--retry-failed'}
if ($SavePredictiveDraws) {$cmdArgs+='--save-predictive-draws'}
if ($ReuseRoot) {$cmdArgs+=@('--reuse-root',$ReuseRoot)}
if ($OutputRoot) {$cmdArgs+=@('--output-root',$OutputRoot)}
Write-Host "Experiment Python: $MainPython"
Write-Host "Original TensorFlow Python: $NativePython"
& $MainPython @cmdArgs
if ($LASTEXITCODE -eq 2) {Write-Warning 'Queue finished with recorded execution failures; independent tasks were continued. Inspect task_status.csv and logs. -RetryFailed retries the same seeds and preserves checkpoints.'; exit 2}
if ($LASTEXITCODE -ne 0) {throw "Command failed: $LASTEXITCODE. Inspect the message above; no seed replacement or sampler retuning is automatic."}
