param([string]$Destination='D:\aresearch\DISHPerf_HyPerf_original_v4')
$ErrorActionPreference='Stop'
$source=[IO.Path]::GetFullPath($PSScriptRoot)
$dest=[IO.Path]::GetFullPath($Destination)
if ($dest.TrimEnd('\') -eq $source.TrimEnd('\')) {Write-Host "Already in target directory: $dest"; exit 0}
if (Test-Path -LiteralPath $dest) {
  $manifest=Join-Path $dest 'package_manifest.json'
  if (Test-Path -LiteralPath $manifest) {
    $a=Get-Content -LiteralPath (Join-Path $source 'package_manifest.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    $b=Get-Content -LiteralPath $manifest -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($a.package_id -eq $b.package_id) {Write-Host "Same package already installed; results preserved: $dest"; exit 0}
  }
  throw "Destination exists and is not this exact package. Choose a new folder; nothing overwritten: $dest"
}
New-Item -ItemType Directory -Path $dest | Out-Null
$exclude=@('results','summaries','logs','splits','plans','validation','__pycache__','queue.lock','queue_status.json')
Get-ChildItem -LiteralPath $source -Force | Where-Object {$_.Name -notin $exclude} | ForEach-Object {Copy-Item -LiteralPath $_.FullName -Destination $dest -Recurse}
Write-Host "Installed: $dest"
Write-Host "Next: cd '$dest'; powershell -NoProfile -ExecutionPolicy Bypass -File .\run.ps1 -Stage check"
