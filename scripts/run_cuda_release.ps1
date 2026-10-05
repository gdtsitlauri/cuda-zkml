$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root
if (Test-Path build-release) { Remove-Item build-release -Recurse -Force }
cmake -S . -B build-release -G "Visual Studio 17 2022" -DZKML_FAST_DEBUG_LOOP=OFF -DZKML_FAST_ITERATION=OFF -DZKML_ULTRA_FAST_COMPILE=OFF
cmake --build build-release --config Release -- /m:1
ctest --test-dir build-release -C Release --output-on-failure
pytest -q tests
New-Item -ItemType Directory -Force -Path results | Out-Null
nvcc --version | Out-File results/cuda_environment.txt
nvidia-smi | Out-File results/cuda_environment.txt -Append
$prove = "build-release/Release/zkml-prove.exe"
if (!(Test-Path $prove)) { $prove = "build-release/zkml-prove.exe" }
& $prove --demo | Tee-Object -FilePath results/cuda_demo.log
