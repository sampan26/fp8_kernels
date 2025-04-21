## fp8_kernels

# Q8 MatMul CUDA Extension Build Instructions

This repository contains a custom PyTorch CUDA extension for efficient quantized matrix multiplication using CUTLASS.

## 🚀 Quick Rebuild Steps (RunPod Session)

Each time I launch a new RunPod instance, follow these steps to rebuild the extension.

---

### 1. Clone Repo & Install Dependencies

```bash
git clone https://github.com/sampan26/fp8_kernels.git
cd fp8_kernels

# (Optional) Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install required Python packages
pip install -r requirements.txt
pip3 install torch --index-url https://download.pytorch.org/whl/cu118
pip install git+https://github.com/Dao-AILab/fast-hadamard-transform.git@v1.0.4.post1

mkdir -p third_party
cd third_party
git clone https://github.com/NVIDIA/cutlass.git
cd ..

mkdir -p q8_matmul/gemm
touch q8_matmul/__init__.py
touch q8_matmul/gemm/__init__.py

python setup.py build_ext --inplace
