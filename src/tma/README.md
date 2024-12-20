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
