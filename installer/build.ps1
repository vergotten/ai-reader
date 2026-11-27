<#
Builds the Windows installer into dist\. Run on Windows (not WSL), from PowerShell or via build-installer*.bat:

    powershell -ExecutionPolicy Bypass -File installer\build.ps1 [-Lite] [-Max] [-Models all|none] [-Extras all]

-Lite    small installer for slow connections (~0.5 GB): Silero only, CPU build of PyTorch, one .exe.
         Without it: every engine and CUDA PyTorch (4-10 GB, .exe plus .bin parts).
-PackOnly  skip the preparation and only run Inno Setup on what the previous run left in build\ (pass -Lite again
           if that run was a lite one).
-Max     full build only: the smallest installer. F5-TTS weights go in fp16 and everything is compressed with the
         strongest LZMA2 as one solid stream. Takes a few hours and several GB of free memory.
-Models  full build only: what is downloaded into models\ first; whatever is in models\ is packed.
-Voices  full build only: ids of fine-tuned voices from %LOCALAPPDATA%\ai-reader\voices to ship (default: vovan).
Needs uv (https://docs.astral.sh/uv/). Inno Setup 6 is downloaded and installed if it is missing.
#>
param(
    [switch]$Lite,
    [switch]$PackOnly,
    [switch]$Max,
    [ValidateSet("all", "none")] [string]$Models = "all",
    [string]$Extras = "all",
    [string[]]$Voices = @("vovan")
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$LitePython = "3.12"
$LiteTorch = @("torch==2.8.0", "torchaudio==2.8.0")
$LiteTorchIndex = "https://download.pytorch.org/whl/cpu"
# PySide6-Essentials has the three Qt modules the app uses; the full PySide6 adds ~400 MB of extras
$LitePackages = @(
    "PySide6-Essentials", "numpy", "lxml", "pymupdf", "charset-normalizer", "num2words", "soundfile",
    "lameenc", "mutagen", "sounddevice", "soxr", "omegaconf", "huggingface_hub"
)
$InnoDownloads = @(
    "https://github.com/jrsoftware/issrc/releases/download/is-6_7_4/innosetup-6.7.4.exe",
    "https://files.jrsoftware.org/is/6/innosetup-6.7.4.exe"
)

function Invoke-Checked([string]$what, [scriptblock]$command) {
    Write-Host "== $what"
    & $command
    if ($LASTEXITCODE -ne 0) { throw "$what failed (exit code $LASTEXITCODE)" }
}

function Copy-Tree([string]$from, [string]$to) {
    robocopy $from $to /E /NFL /NDL /NJH /NJS /NP /XD __pycache__ | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "Copying $from failed (robocopy code $LASTEXITCODE)" }
}

function Find-Inno {
    @(
        "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe",
        "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
        "$env:ProgramFiles\Inno Setup 6\ISCC.exe",
        "$env:LOCALAPPDATA\Programs\Inno Setup 7\ISCC.exe",
        "${env:ProgramFiles(x86)}\Inno Setup 7\ISCC.exe",
        "$env:ProgramFiles\Inno Setup 7\ISCC.exe"
    ) | Where-Object { Test-Path $_ } | Select-Object -First 1
}

function Test-Installer([string]$path) {
    # a failed download is often a small HTML page saved under the .exe name
    if (-not (Test-Path $path) -or (Get-Item $path).Length -lt 1MB) { return $false }
    $bytes = [System.IO.File]::ReadAllBytes($path)
    return ($bytes[0] -eq 0x4D -and $bytes[1] -eq 0x5A)  # "MZ": a Windows program
}

function Install-Inno {
    Write-Host "== Inno Setup not found: downloading and installing it for the current user"
    $setup = Join-Path $env:TEMP "innosetup-6.exe"
    foreach ($url in $InnoDownloads) {
        Remove-Item $setup -ErrorAction SilentlyContinue
        & curl.exe --location --fail --silent --show-error --output $setup $url
        if (Test-Installer $setup) {
            Start-Process $setup -ArgumentList "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/CURRENTUSER" -Wait
            return
        }
        Write-Warning "Could not download Inno Setup from $url"
    }
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw 'uv not found. Install it:  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"  and reopen this window'
}

$modelsDir = Join-Path $root "models"
$stage = Join-Path $root "build\stage\python"
if ($PackOnly) {
    if (-not (Test-Path (Join-Path $stage "pythonw.exe"))) { throw "Nothing to pack: run once without -PackOnly first" }
    if ($Lite) { $modelsDir = Join-Path $root "build\lite-models" }
} elseif ($Lite) {
    $venv = Join-Path $root "build\lite-venv"
    if (Test-Path $venv) { Remove-Item $venv -Recurse -Force }
    $python = Join-Path $venv "Scripts\python.exe"
    Invoke-Checked "Creating the lite environment" { uv venv $venv --python $LitePython }
    Invoke-Checked "Installing PyTorch (CPU build)" { uv pip install --python $python --index-url $LiteTorchIndex @LiteTorch }
    Invoke-Checked "Installing the other packages" { uv pip install --python $python @LitePackages }

    $modelsDir = Join-Path $root "build\lite-models"
    $silero = "torch\hub\snakers4_silero-models_master"
    if (Test-Path (Join-Path $root "models\$silero")) {  # already downloaded: no need to fetch it again
        Copy-Tree (Join-Path $root "models\$silero") (Join-Path $modelsDir $silero)
    }
    $env:AI_READER_MODELS = $modelsDir
    $env:PYTHONPATH = $root
    Invoke-Checked "Getting the Silero model" { & $python -m ai_reader.fetch_models lite }
    Remove-Item Env:AI_READER_MODELS, Env:PYTHONPATH
} else {
    $venv = Join-Path $root ".venv"
    Invoke-Checked "Installing packages (uv sync --extra $Extras)" { uv sync --extra $Extras }
    if ($Models -ne "none") {
        Write-Host "== Downloading models ($Models)"
        & (Join-Path $venv "Scripts\python.exe") -m ai_reader.fetch_models $Models
        if ($LASTEXITCODE -ne 0) { Write-Warning "Some models were not downloaded; the app will fetch them on first use." }
    }
}

if (-not $Lite) {
    # every model once (the download cache keeps each file twice), F5-TTS in fp16 with -Max
    $packModels = Join-Path $root "build\models-pack"
    $slimArgs = @("installer\slim_models.py", $modelsDir, $packModels)
    if ($Max) { $slimArgs += "--fp16" }
    Invoke-Checked "Preparing the models for the installer" { & (Join-Path $root ".venv\Scripts\python.exe") @slimArgs }
    $modelsDir = $packModels
}

if (-not $Lite) {
    # fine-tuned voices: the model, the reference clip and the clips, without the multi-GB training checkpoints
    $voicesPack = Join-Path $root "build\voices-pack"
    if (Test-Path $voicesPack) { Remove-Item $voicesPack -Recurse -Force }
    foreach ($id in $Voices) {
        $src = Join-Path $env:LOCALAPPDATA "ai-reader\voices\$id"
        if (-not (Test-Path (Join-Path $src "model.pt"))) {
            Write-Warning "Voice '$id' is not fine-tuned yet (no $src\model.pt): not shipped"
            continue
        }
        robocopy $src (Join-Path $voicesPack $id) /E /XD train /NFL /NDL /NJH /NJS /NP | Out-Null
        if ($LASTEXITCODE -ge 8) { throw "Copying voice '$id' failed (robocopy exit $LASTEXITCODE)" }
        $global:LASTEXITCODE = 0  # robocopy returns 1 for "files copied"
        Write-Host "== Voice '$id' goes into the installer"
    }
}

if (-not $PackOnly) {
# The venv's base interpreter is a self-contained Python from uv: it can be copied to another PC as is.
$homeLine = Get-Content (Join-Path $venv "pyvenv.cfg") | Where-Object { $_ -match '^home\s*=' } | Select-Object -First 1
$pythonHome = ($homeLine -replace '^home\s*=\s*', '').Trim()
if (-not (Test-Path (Join-Path $pythonHome "pythonw.exe"))) { throw "No pythonw.exe in $pythonHome" }

if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
Write-Host "== Copying Python from $pythonHome"
Copy-Tree $pythonHome $stage
Write-Host "== Copying installed packages"
$sitePackages = Join-Path $stage "Lib\site-packages"
Copy-Tree (Join-Path $venv "Lib\site-packages") $sitePackages
# the project itself is installed in the venv as a link to this folder; the installer ships ai_reader\ separately
Get-ChildItem $sitePackages -Filter "*ai_reader*" | Remove-Item -Recurse -Force

Invoke-Checked "Checking the packed Python" {
    & (Join-Path $stage "python.exe") -c "import PySide6.QtWidgets, numpy, soundfile, torch; print('packed Python OK, torch', torch.__version__)"
}
}

$version = (Select-String -Path ai_reader\__init__.py -Pattern '__version__\s*=\s*"([^"]+)"').Matches[0].Groups[1].Value
$iscc = Find-Inno
if (-not $iscc) { Install-Inno; $iscc = Find-Inno }
if (-not $iscc) {
    throw "Inno Setup 6 not found. Install it from https://jrsoftware.org/isdl.php and run this again with -PackOnly added (the prepared files are kept)."
}

$defines = @("/DAppVersion=$version", "/DModelsDir=$modelsDir")
if ($Lite) { $defines += "/DLite" }
elseif ($Max) { $defines += "/DMax"; Write-Host "== Maximum compression: this takes a few hours" }
# a smaller build makes fewer .bin parts; stale ones from the previous build would get sent along by mistake
$old = if ($Lite) { "AI-Reader-Setup-*-Lite.exe" } else { "AI-Reader-Setup-$version.exe", "AI-Reader-Setup-$version-*.bin" }
Get-ChildItem dist\* -Include $old -File -ErrorAction SilentlyContinue | Remove-Item -Force
Invoke-Checked "Building the installer (takes a while)" { & $iscc @defines installer\ai-reader.iss }
Write-Host ""
Write-Host "Done. Send ALL files from dist\ that belong to this build (the .exe and, for the full build, every .bin):"
Get-ChildItem dist | Format-Table Name, @{ n = "MB"; e = { [math]::Round($_.Length / 1MB) } }
