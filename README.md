# CAESAR_C

C++ implementation of CAESAR using LibTorch. The goal is to provide a C++ version of the CAESAR foundation model for efficient compression of scientific data.

## Overview

CAESAR (Conditional AutoEncoder with Super-resolution for Augmented Reduction) is a unified framework for spatio-temporal scientific data reduction.

The baseline model, CAESAR-V, is built on a variational autoencoder (VAE) with scale hyperpriors and super-resolution modules to achieve high compression. It encodes data into a latent space and uses learned priors for compact, information-rich representation. This repository ports CAESAR into C++ with LibTorch for use in high-performance scientific applications.

**Reference:** Shaw et al., CAESAR: A Unified Framework of Foundation and Generative Models for Efficient Compression of Scientific Data

## Notes

- GPU support currently tested only with NVIDIA GPUs
- **MPI is not used** in this implementation
- **Zstandard (zstd) 1.5+ is required** for compression support
- Model compression requires Python environment to be set up correctly
- **Windows support** is available but note that CompressAI installation may require additional setup (see Windows notes below)

## Build Instructions

### 1. Clone the repository

```bash
git clone https://github.com/E53klasky/CAESAR_C.git
cd CAESAR_C
```

### 2. Create and activate Python virtual environment

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip wheel setuptools
```

**Windows:**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip wheel setuptools
```

### 3. Install dependencies based on your platform

#### For Linux (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get install -y cmake g++ zstd libzstd-dev

source venv/bin/activate

grep -v "^torch" requirements.txt | \
  grep -v "^torchvision" | \
  grep -v "^--extra-index-url" | \
  grep -v "^cupy" | \
  grep -v "^nvidia" | \
  grep -v "^$" > temp_requirements.txt

pip install --no-cache-dir -r temp_requirements.txt
pip install torch==2.9.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install compressai==1.2.6 imageio==2.37.0
rm temp_requirements.txt
```

#### For macOS

```bash
brew install cmake zstd gcc

source venv/bin/activate

grep -v "^torch" requirements.txt | \
  grep -v "^torchvision" | \
  grep -v "^--extra-index-url" | \
  grep -v "^cupy" | \
  grep -v "^nvidia" | \
  grep -v "^$" > temp_requirements.txt

pip install -r temp_requirements.txt
pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install compressai==1.2.6 imageio==2.37.0
rm temp_requirements.txt
```

#### For Windows

**Note:** CompressAI installation on Windows requires Microsoft Visual C++ 14.0 or greater. Install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) if you encounter errors.

```powershell
# Install dependencies via Chocolatey (if not already installed)
choco install cmake zstandard -y

.\venv\Scripts\Activate.ps1

Get-Content requirements.txt | Where-Object { 
    $_ -notmatch "^torch" -and 
    $_ -notmatch "^torchvision" -and 
    $_ -notmatch "^--extra-index-url" -and 
    $_ -notmatch "^cupy" -and 
    $_ -notmatch "^nvidia" -and 
    $_ -notmatch "^$" 
} | Set-Content temp_requirements.txt

pip install -r temp_requirements.txt
pip install torch==2.9.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# CompressAI may fail on Windows - if so, see troubleshooting section
pip install compressai==1.2.6 imageio==2.37.0
Remove-Item temp_requirements.txt
```

**Windows CompressAI Troubleshooting:**

If you encounter the error "Microsoft Visual C++ 14.0 or greater is required", you have two options:

1. **Install Microsoft C++ Build Tools** (Recommended):
   - Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
   - Install "Desktop development with C++" workload
   - Restart your terminal and retry the installation

2. **Build from source** (Alternative):
   ```powershell
   git clone https://github.com/InterDigitalInc/CompressAI
   cd CompressAI
   pip install -U pip
   pip install -e .
   ```

For more details, see: https://stackoverflow.com/questions/64261546/how-to-solve-error-microsoft-visual-c-14-0-or-greater-is-required-when-inst

### 4. Download and prepare pretrained models

**Linux/macOS:**
```bash
chmod +x download_models.sh
./download_models.sh

python3 CAESAR_compressor.py cpu
python3 CAESAR_hyper_decompressor.py cpu
python3 CAESAR_decompressor.py cpu
```

**Windows:**
```powershell
bash download_models.sh

python CAESAR_compressor.py cpu
python CAESAR_hyper_decompressor.py cpu
python CAESAR_decompressor.py cpu
```

### 5. Configure and build with CMake

#### Linux

```bash
mkdir -p build
cd build

TORCH_PATH=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)")

cmake .. \
  -DCMAKE_PREFIX_PATH="$TORCH_PATH" \
  -DBUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build . --config Release --parallel
```

#### macOS

```bash
mkdir -p build
cd build

TORCH_PATH=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)")

cmake .. \
  -DCMAKE_PREFIX_PATH="$TORCH_PATH" \
  -DBUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build . --config Release --parallel
```

#### Windows

```powershell
New-Item -ItemType Directory -Force -Path build
cd build

$TORCH_PATH = python -c "import torch; print(torch.utils.cmake_prefix_path)"

cmake .. `
  -DCMAKE_PREFIX_PATH="$TORCH_PATH" `
  -DBUILD_TESTS=ON `
  -DCMAKE_BUILD_TYPE=Release

cmake --build . --config Release --parallel
```

#### For Debug builds

Replace `-DCMAKE_BUILD_TYPE=Release` with `-DCMAKE_BUILD_TYPE=Debug` in the cmake command above.

## Dependencies

### Core Dependencies

- **LibTorch** (PyTorch C++ API)
- **CMake** (3.10+)
- **Zstandard (zstd) 1.5+** - Required for compression support
- **Python 3.10+** - For model preparation and preprocessing

### GPU Support (Optional)

- **CUDA Toolkit** - Ensure nvcc is in PATH
- **nvCOMP** - NVIDIA Compression Library

### Installing nvCOMP

Download the CUDA 12 archive:

```bash
wget https://developer.download.nvidia.com/compute/nvcomp/redist/nvcomp/linux-x86_64/nvcomp-linux-x86_64-5.0.0.6_cuda12-archive.tar.xz
```

Extract to a local folder:

```bash
mkdir -p ~/local/nvcomp
tar -xJf nvcomp-linux-x86_64-5.0.0.6_cuda12-archive.tar.xz -C ~/local/nvcomp --strip-components=1
```

Set environment variables:

```bash
export CMAKE_PREFIX_PATH=$HOME/local/nvcomp:$CMAKE_PREFIX_PATH
export LD_LIBRARY_PATH=$HOME/local/nvcomp/lib:$LD_LIBRARY_PATH
```

Build with GPU support:

```bash
cmake .. \
  -DCMAKE_PREFIX_PATH="$TORCH_PATH;$HOME/local/nvcomp" \
  -DCMAKE_CXX_FLAGS="-I$HOME/local/nvcomp/include" \
  -DCMAKE_EXE_LINKER_FLAGS="-L$HOME/local/nvcomp/lib" \
  -DBUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Release
```

Install PyTorch with CUDA support:

```bash
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

## Model Directory Configuration

CAESAR automatically finds model files in the following order:

1. **Custom location** (if set): `export CAESAR_MODEL_DIR=/path/to/your/models`
2. **Development build**: Automatically finds `../exported_model/` relative to executable
3. **After installation**: Automatically finds models at `/usr/local/share/caesar/models`

For most users, no configuration is needed. CAESAR will find the models automatically.

If you have multiple model versions or want to use models from a different location:

```bash
export CAESAR_MODEL_DIR=/path/to/custom/models
./your_program
```

## Installation with ADIOS2

To build with ADIOS2 support:

```bash
cmake .. \
  -DCMAKE_INSTALL_PREFIX=~/Programs/CAESAR_C/install \
  -DTorch_DIR=/path/to/python/site-packages/torch/share/cmake/Torch
```

## Platform-Specific Notes

### Windows
- Visual Studio 2019 or newer recommended
- Microsoft C++ Build Tools required for CompressAI
- Use PowerShell or Git Bash for running scripts
- Some test executables may be in `build/tests/Debug/` or `build/tests/Release/` depending on build configuration

### macOS
- Straightforward build process
- No special configuration needed

### Linux
- Most straightforward platform for building
- Package managers handle dependencies easily

## References

- Original CAESAR repository: https://github.com/Shaw-git/CAESAR
- NVIDIA nvCOMP: https://developer.nvidia.com/nvcomp
- CUDA Toolkit: https://developer.nvidia.com/cuda-toolkit
- PyTorch: https://pytorch.org/
- Zstandard (zstd): https://facebook.github.io/zstd/
- CompressAI: https://github.com/InterDigitalInc/CompressAI
