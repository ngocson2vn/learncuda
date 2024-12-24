########################################################################################################################
# Daily commands
########################################################################################################################

# GPU Memory Utilization of GPU0
while true; do nvidia-smi --id=0 --query-gpu=timestamp,memory.total,memory.used,memory.free --format=csv && echo && sleep 5; done

# QUERY GPU METRICS FOR HOST-SIDE LOGGING
nvidia-smi --query-gpu=timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 5


########################################################################################################################
# Uninstall CUDA Driver completely
########################################################################################################################
# Step 1: Run uninstaller
/usr/bin/nvidia-uninstall

# Step 2: Stop fabric manager (H100)
systemctl stop nvidia-fabricmanager.service

# Step 3: Remove kernel modules
lsmod | grep -E 'Module|nvidia'
Module                  Size  Used by
nvidia              56750080  0
# Remove all modules with used by 0

rmmod nvidia

# Step 4: If Used by > 0, then find processes still using /dev/nvidia*
lsof | grep nvidia

########################################################################################################################
# Download cuda
########################################################################################################################

# cuda_11.4
wget https://developer.download.nvidia.com/compute/cuda/11.4.0/local_installers/cuda_11.4.0_470.42.01_linux.run
wget https://developer.download.nvidia.com/compute/cuda/11.4.4/local_installers/cuda_11.4.4_470.82.01_linux.run

# cuda_12.4.0
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run

# cuda 12.4.1
wget https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda_12.4.1_550.54.15_linux.run


# Driver 550.54.15
wget https://us.download.nvidia.com/tesla/550.54.15/NVIDIA-Linux-x86_64-550.54.15.run

# Fabric Manager
https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/nvidia-fabricmanager-550_550.54.15-1_amd64.deb

########################################################################################################################


./deviceQuery Starting...
CUDA Device Query (Runtime API) version (CUDART static linking)
cudaGetDeviceCount returned 802
-> system not yet initialized
Result = FAIL

Please make sure nVidia fabric manager is installed and activated.
1. install nvidia DCGM.
2. terminate the nv-hostengine* first in order to enable fabric manager.
# sudo nv-hostengine -t

3. Download and install Fabric manager
https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/
dpkg -i nvidia-fabricmanager-550_550.54.15-1_amd64.deb

4. Start Fabric manager
sudo systemctl status nvidia-fabricmanager
sudo systemctl start nvidia-fabricmanager

########################################################################################################################
# NVCC
########################################################################################################################
# Generate ptx
nvcc -keep

# Install cccl
```Bash
cd ~/workspace/setup/
wget https://github.com/NVIDIA/cccl/archive/refs/tags/v2.6.1.tar.gz
tar -xzf v2.6.1.tar.gz

sudo rsync -avP ~/workspace/setup/cccl-2.6.1/libcudacxx/include/cuda /usr/local/cuda/include/
sudo rsync -avP /data00/home/son.nguyen/workspace/setup/cccl-2.6.1/libcudacxx/include/nv /usr/local/cuda/include/
sudo rsync -avP /data00/home/son.nguyen/workspace/setup/cccl-2.6.1/thrust/thrust /usr/local/cuda/include/
sudo rsync -avP /data00/home/son.nguyen/workspace/setup/cccl-2.6.1/cub/cub /usr/local/cuda/include/
```
