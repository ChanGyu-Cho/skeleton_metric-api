#!/usr/bin/env pwsh
# Ultra-fast no-cache rebuild optimized for development

Write-Host "="*80 -ForegroundColor Cyan
Write-Host "FAST No-Cache Rebuild (Optimized)" -ForegroundColor Cyan
Write-Host "="*80 -ForegroundColor Cyan

# Enable parallel processing and BuildKit
$env:DOCKER_BUILDKIT=1
$env:BUILDKIT_PROGRESS="plain"

Write-Host "`n[1/4] Cleaning Python cache..." -ForegroundColor Yellow
Get-ChildItem -Path . -Recurse -Include __pycache__,*.pyc,*.pyo | Remove-Item -Force -Recurse -ErrorAction SilentlyContinue
Write-Host "✓ Cache files removed" -ForegroundColor Green

Write-Host "`n[2/4] Checking disk space..." -ForegroundColor Yellow
$drive = (Get-Location).Drive.Name
$disk = Get-PSDrive $drive
$freeGB = [math]::Round($disk.Free / 1GB, 2)
Write-Host "  Free space: $freeGB GB" -ForegroundColor Gray
if ($freeGB -lt 10) {
    Write-Host "  ⚠️  Warning: Low disk space. Consider cleaning old images." -ForegroundColor Yellow
    docker system df
}

Write-Host "`n[3/4] Building with optimizations..." -ForegroundColor Yellow
Write-Host "  Optimizations enabled:" -ForegroundColor Gray
Write-Host "    - BuildKit parallel processing" -ForegroundColor Gray
Write-Host "    - pip cache mount (persists between builds)" -ForegroundColor Gray
Write-Host "    - --no-deps for PyTorch (faster install)" -ForegroundColor Gray
Write-Host "    - Minimal layer count" -ForegroundColor Gray

$buildStart = Get-Date

# Build with no-cache but with mount cache for pip
docker build `
    --no-cache `
    --progress=plain `
    -t skeleton-metric-api:latest `
    .

$buildEnd = Get-Date
$duration = ($buildEnd - $buildStart).TotalSeconds

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✓ Build completed successfully" -ForegroundColor Green
    Write-Host "  Build time: $([math]::Round($duration, 1)) seconds" -ForegroundColor Green
    
    Write-Host "`n[4/4] Image information:" -ForegroundColor Yellow
    docker images skeleton-metric-api:latest --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
    
    Write-Host "`n" + "="*80 -ForegroundColor Cyan
    Write-Host "Ready to run!" -ForegroundColor Green
    Write-Host "="*80 -ForegroundColor Cyan
    Write-Host "`nQuick start command:" -ForegroundColor Yellow
    Write-Host "docker run --gpus all -p 19030:19030 \" -ForegroundColor White
    Write-Host "  -e S3_VIDEO_BUCKET_NAME=your-bucket \" -ForegroundColor White
    Write-Host "  -e S3_RESULT_BUCKET_NAME=your-result-bucket \" -ForegroundColor White
    Write-Host "  -e AWS_ACCESS_KEY_ID=key \" -ForegroundColor White
    Write-Host "  -e AWS_SECRET_ACCESS_KEY=secret \" -ForegroundColor White
    Write-Host "  -e AWS_REGION=ap-northeast-2 \" -ForegroundColor White
    Write-Host "  skeleton-metric-api:latest" -ForegroundColor White
} else {
    Write-Host "`n✗ Build failed" -ForegroundColor Red
    Write-Host "  Check error messages above" -ForegroundColor Red
    exit 1
}
