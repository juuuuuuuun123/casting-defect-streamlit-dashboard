$ErrorActionPreference = "Stop"

$envPath = Join-Path (Get-Location) ".env"
if (-not (Test-Path $envPath)) {
    throw ".env file not found. KAGGLE_USERNAME and KAGGLE_API_TOKEN are required."
}

$vars = @{}
Get-Content $envPath | ForEach-Object {
    if ($_ -match "^([^=]+)=(.*)$") {
        $vars[$matches[1]] = $matches[2]
    }
}

New-Item -ItemType Directory -Force -Path "data", "data\raw", "data\downloads" | Out-Null

$pair = "$($vars['KAGGLE_USERNAME']):$($vars['KAGGLE_API_TOKEN'])"
$b64 = [Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes($pair))
$headers = @{ Authorization = "Basic $b64" }
$zipPath = "data\downloads\casting-product.zip"
$url = "https://www.kaggle.com/api/v1/datasets/download/ravirajsinh45/real-life-industrial-dataset-of-casting-product"

Invoke-WebRequest -Headers $headers -Uri $url -OutFile $zipPath -TimeoutSec 600
Expand-Archive -Path $zipPath -DestinationPath "data\raw" -Force

Get-ChildItem -Recurse -File "data\raw" |
    Select-Object -First 20 FullName, Length |
    Format-Table -AutoSize
