########################################################################################################################
# Daily commands
########################################################################################################################

# GPU Memory Utilization of GPU0
while true; do nvidia-smi --id=0 --query-gpu=timestamp,memory.total,memory.used,memory.free --format=csv && echo && sleep 5; done

# QUERY GPU METRICS FOR HOST-SIDE LOGGING
nvidia-smi --query-gpu=timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 5




########################################################################################################################
# Download cuda
########################################################################################################################

# cuda_12.4
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run

########################################################################################################################
