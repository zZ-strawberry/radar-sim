#!/usr/bin/env bash
# 在无 sudo 的 CI/容器里可直接退出；本机需 GPU 推理时请在本机执行并输入密码。
set -euo pipefail

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  echo "[OK] NVIDIA 驱动已就绪:"
  nvidia-smi
  exit 0
fi

K=$(uname -r)
echo "[!] nvidia-smi 不可用（常见：有显卡和 nvidia-driver-* 元包，但缺少与当前内核匹配的 kmod）。"
echo "    当前内核: ${K}"
echo

if ! command -v apt-cache >/dev/null 2>&1; then
  echo "未找到 apt-cache，请自行安装与内核版本一致的 linux-modules-nvidia-*-${K}。"
  exit 1
fi

echo "与当前内核匹配的候选包（请与已安装的 nvidia-driver-XXX 大版本一致，优先 580）："
found=0
for series in 580 575-open 575 565-server 535; do
  pkg="linux-modules-nvidia-${series}-${K}"
  if apt-cache show "$pkg" >/dev/null 2>&1; then
    echo "  - ${pkg}"
    found=1
  fi
done
if [[ "$found" -eq 0 ]]; then
  echo "  (apt 缓存中未找到匹配 ${K} 的包，请先 sudo apt update)"
fi

echo
echo "安装示例（二选一，需 sudo）："
echo "  sudo apt update && sudo apt install -y linux-modules-nvidia-580-${K}"
echo "然后执行其一:"
echo "  sudo modprobe nvidia"
echo "  或 reboot"
echo
echo "若 pip 的 torch 仍为 +cpu，再安装 CUDA 版 wheel，例如:"
echo "  /usr/bin/python3 -m pip install -U torch torchvision --index-url https://download.pytorch.org/whl/cu130"
