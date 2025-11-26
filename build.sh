#!/bin/bash

# 仅当第一个参数为 "start" 时才启动 start.py
if [ "$1" = "start" ]; then
	python3 ./vllm_het/start.py
fi

git submodule update --init --recursive HMC
cd HMC

python3 -m pip uninstall -y hmc
rm -rf build dist *.egg-info
python3 -m pip install --upgrade build wheel setuptools

mkdir -p build && cd build
cmake .. -DBUILD_PYTHON_MOD=ON -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=$(which python3)
make -j
cd ..

python3 -m build
python3 -m pip install --force-reinstall dist/hmc-*.whl

cp -f vllm_het/vllm_auto_patch.py /usr/local/lib/python3.12/dist-packages/
cp -f vllm_het/p2p_backend.py /usr/local/lib/python3.12/dist-packages/

cat >/usr/local/lib/python3.12/dist-packages/vllm_patch.pth <<'EOF'
import vllm_auto_patch  # noqa: F401
EOF
