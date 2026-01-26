#!/bin/bash

# Cache dir
export TRITON_CACHE_DIR=./tmp

# Autotune
export TRITON_PRINT_AUTOTUNING="1"


python3.11 matmul_tma.py
