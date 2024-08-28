# GPU Memory Utilization of GPU0
while true; do nvidia-smi --id=0 --query-gpu=timestamp,memory.total,memory.used,memory.free --format=csv && echo && sleep 5; done

# QUERY GPU METRICS FOR HOST-SIDE LOGGING
nvidia-smi --query-gpu=timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 5
