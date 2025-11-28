# Debug CuTe
```Bash
# 1. Compile test.cpp with clang++ (version 17)
clang++ -std=c++17 -g -O0 -fno-inline $(ALL_INCLUDE) -o test test.cpp

# 2. lldb-dap
ln -sf /usr/lib/llvm-17/bin/lldb-vscode /usr/bin/lldb-dap

# 3. Launch config
{
  "name": "cute_lldb",
  "type": "lldb-dap",
  "request": "launch",
  "program": "${workspaceFolder}/src/cutlass_examples/hopper/test",
  "args": [],
  "env": {
    "FOO": "1",
  }
}
```
