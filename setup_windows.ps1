# Windows equivalent of ARENA's install_uv.sh (which targets Linux cloud boxes: RunPod / Vast.ai).
# Expects this repo to be cloned as ARENA_materials\my_answers. Run from anywhere:
#
#   powershell -ExecutionPolicy Bypass -File ARENA_materials\my_answers\setup_windows.ps1
#
# Safe to re-run, e.g. after `git pull` in ARENA_materials changes requirements.txt.
#
# Differences from requirements.txt, because some packages don't run on Windows or on older GPUs:
#   - torch/torchvision/torchaudio come from PyTorch's CUDA 12.6 index instead of PyPI. PyPI's builds
#     use CUDA 13, which needs NVIDIA driver 580+ and dropped Pascal/Maxwell GPUs (e.g. GTX 10xx).
#   - jax[cuda13] becomes CPU-only jax (JAX has no CUDA wheels for Windows). Only Brax in [2.3] uses it.
#   - envpool (Linux only) and liger-kernel (needs Triton, Linux only) are skipped. They're optional
#     extras for the Atari bonus in [2.3] and for [2.4.2] RLVR, which needs a cloud GPU anyway.

$ErrorActionPreference = "Stop"
$AnswersDir = $PSScriptRoot
$RepoDir = Split-Path $AnswersDir -Parent  # the ARENA_materials clone
if (-not (Test-Path (Join-Path $RepoDir "requirements.txt"))) { throw "Expected ARENA_materials at $RepoDir" }
$VenvDir = Join-Path $RepoDir ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$PythonVersion = "3.13"
$TorchIndex = "https://download.pytorch.org/whl/cu126"
$KernelName = "arena-env"

# Keep uv's Python installs and cache out of AppData. Shells run from packaged apps (e.g. the Claude
# desktop app) get AppData writes redirected into a private folder that VS Code can't see.
$env:UV_PYTHON_INSTALL_DIR = Join-Path $HOME ".uv\python"
$env:UV_CACHE_DIR = Join-Path $HOME ".uv\cache"

# Pick up uv/gh installed by winget in this same session
$env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "=== Installing uv ==="
    winget install --id astral-sh.uv -e --accept-source-agreements --accept-package-agreements
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")
}

# --- Python venv ---
if (-not (Test-Path $VenvPython)) {
    Write-Host "=== Creating venv at $VenvDir (python $PythonVersion) ==="
    uv venv --managed-python --python $PythonVersion --seed $VenvDir
    if ($LASTEXITCODE -ne 0) { throw "uv venv failed" }
} else {
    Write-Host "=== Reusing existing venv at $VenvDir ==="
}

# --- Build a Windows-compatible copy of requirements.txt ---
$TorchSpec = $null
$WinReqs = foreach ($line in Get-Content (Join-Path $RepoDir "requirements.txt")) {
    $name = ($line -replace "#.*", "").Trim()
    if ($name -match "^torch([<>=!~].*)?$") { $TorchSpec = $name; continue }
    if ($name -match "^(torchvision|torchaudio|envpool|liger-kernel)\b") { continue }
    if ($name -match "^jax\[cuda\d+\](.*)$") { "jax$($Matches[1])"; continue }
    $line
}
$ReqFile = Join-Path $env:TEMP "arena_requirements_windows.txt"
[IO.File]::WriteAllLines($ReqFile, [string[]]$WinReqs)  # UTF-8 without BOM

# --- PyTorch (CUDA 12.6 build) first, so the rest of the install keeps it ---
if (-not $TorchSpec) { $TorchSpec = "torch" }
Write-Host "=== Installing $TorchSpec from $TorchIndex ==="
uv pip install --python $VenvPython $TorchSpec torchvision torchaudio --index-url $TorchIndex
if ($LASTEXITCODE -ne 0) { throw "PyTorch install failed" }
$TorchVersion = & $VenvPython -c "import torch; print(torch.__version__)"

# With NVIDIA drivers older than the CUDA 12.6 runtime (e.g. 538.xx = CUDA 12.2), loading every CUDA
# kernel up front fails with "CUDA-capable device(s) is/are busy or unavailable"; lazy loading works.
# A .pth file runs at interpreter startup, so this applies to every script, terminal and Jupyter kernel.
$SitePackages = & $VenvPython -c "import sysconfig; print(sysconfig.get_paths()['purelib'])"
[IO.File]::WriteAllLines((Join-Path $SitePackages "arena_cuda_lazy_loading.pth"),
    [string[]]@('import os; os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")'))

# --- Everything else ---
Write-Host "=== Installing the remaining requirements ==="
uv pip install --python $VenvPython -r $ReqFile "torch==$TorchVersion"
if ($LASTEXITCODE -ne 0) { throw "requirements install failed" }

# --- Jupyter kernel named "arena-env", stored inside the venv ---
& $VenvPython -m ipykernel install --sys-prefix --name $KernelName --display-name $KernelName

# --- VS Code workspace settings (.vscode is gitignored by the repo) ---
Write-Host "=== Configuring VS Code workspace settings ==="
$ExtraPaths = Get-ChildItem -Directory (Join-Path $RepoDir "chapter*") |
    ForEach-Object { Join-Path $_.FullName "exercises" } | Where-Object { Test-Path $_ }
$Settings = [ordered]@{
    "python.defaultInterpreterPath"                       = $VenvPython
    "python.terminal.activateEnvironment"                 = $true
    "python.analysis.extraPaths"                          = @($ExtraPaths)
    "jupyter.interactiveWindow.textEditor.executeSelection" = $true
    "jupyter.notebookFileRoot"                            = '${fileDirname}'
    # Only show this answers repo in Source Control, not the ARENA_materials clone around it
    "git.ignoredRepositories"                             = @($RepoDir)
}
$VsCodeDir = Join-Path $RepoDir ".vscode"
New-Item -ItemType Directory -Force $VsCodeDir | Out-Null
$Settings | ConvertTo-Json | Set-Content -Encoding utf8 (Join-Path $VsCodeDir "settings.json")

# Keep `git status` in the ARENA_materials clone clean
$Exclude = Join-Path $RepoDir ".git\info\exclude"
$AnswersName = Split-Path $AnswersDir -Leaf
if ((Test-Path $Exclude) -and -not (Select-String -Path $Exclude -SimpleMatch "$AnswersName/" -Quiet)) {
    Add-Content -Path $Exclude -Value "$AnswersName/"
}

Write-Host "=== Done. torch $TorchVersion ==="
& $VenvPython -c "import torch; print('CUDA available:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
