# Windows 11 WSL2 + WSLg experimental setup

This is an unofficial helper note for running the AI Challenge racing kart
development environment on Windows 11 with WSL2, Docker Engine, NVIDIA
Container Toolkit, WSLg, and Mesa Dozen (`dzn`).

This is not an official supported setup. Use the official documentation for
rules, submission, and baseline setup instructions.

## Scope

This branch keeps the default Linux/Ubuntu workflow unchanged. The WSLg Vulkan
path is opt-in and is enabled only when `docker-compose.dzn.yml` is explicitly
included and `AWSIM_FORCE_VULKAN=1` is set for the simulator service.

The goal is to help Windows users:

- build the normal development image in WSL2 Ubuntu 22.04,
- display RViz through WSLg,
- run AWSIM with Vulkan through Mesa Dozen (`dzn`),
- keep the normal compose workflow available.

## Tested environment

- Windows 11
- WSL2 Ubuntu 22.04
- NVIDIA GPU exposed to WSL2
- Docker Engine inside WSL2
- NVIDIA Container Toolkit
- WSLg
- Mesa 25.0.7 from `ppa:kisak/turtle`

NVIDIA GPU + WSLg + dzn was tested. Other GPUs are not covered by this note.

## What are Vulkan and dzn?

AWSIM is a Unity application. On Linux it can render through graphics APIs such
as OpenGL or Vulkan.

On WSLg, RViz may work through OpenGL while AWSIM OpenGL rendering can still
show a magenta screen. This branch uses Vulkan instead.

`dzn`, also called Dozen, is a Mesa Vulkan driver that translates Vulkan calls
to Direct3D12. In this setup the rendering path is:

```text
AWSIM / Unity
  -> Vulkan
  -> Mesa Dozen (dzn)
  -> Direct3D12
  -> Windows NVIDIA Driver
  -> NVIDIA GPU
```

Dozen may print:

```text
dzn is not a conformant Vulkan implementation, testing use only
```

Treat this as a practical development workaround for Windows/WSL users, not as
a replacement for the officially recommended Linux environment.

## Host WSL setup

Install the normal project prerequisites first, then install a Mesa version that
includes dzn:

```bash
sudo apt install -y ppa-purge software-properties-common
sudo add-apt-repository -y ppa:kisak/turtle
sudo apt update
sudo apt install -y mesa-vulkan-drivers vulkan-tools libvulkan1
```

Check that dzn sees the GPU:

```bash
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/dzn_icd.x86_64.json \
  vulkaninfo --summary
```

Expected output includes:

```text
deviceName = Microsoft Direct3D12 (NVIDIA GeForce RTX 3060)
deviceType = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
driverName = Dozen
```

## Build the dzn development image

Build the normal development image first:

```bash
./docker_build.sh dev
make autoware-build
```

Then build the dzn-derived image:

```bash
docker build -f Dockerfile.dzn -t aichallenge-2025-dev:dzn .
```

## Run dev2 with dzn

```bash
COMPOSE_FILE=docker-compose.yml:docker-compose.gpu.yml:docker-compose.dzn.yml make dev2
```

Check the Unity player log:

```bash
docker exec aichallenge-racingkart-simulator-1 \
  grep -n "Vulkan renderer" /tmp/.config/unity3d/TIERIV/AWSIM/Player.log
