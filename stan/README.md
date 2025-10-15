# Stan Binaries Directory

This directory contains platform-specific CmdStan binaries for the Prophet forecasting model.

## Directory Structure

```
stan/
├── windows/          # Windows binaries (.exe and .dll files)
├── linux/            # Linux binaries (ELF executables and .so files)
├── macos/            # macOS binaries (Mach-O executables and .dylib files)
└── prophet.stan      # Stan model source code
```

## Platform-Specific Files

### Windows (`windows/`)
- `prophet_model.exe` - Main CmdStan executable
- `*.dll` - TBB (Threading Building Blocks) libraries

### Linux (`linux/`)
- `prophet_model` - Main CmdStan executable
- `*.so`, `*.so.2` - TBB shared libraries

### macOS (`macos/`)
- `prophet_model` - Main CmdStan executable
- `*.dylib` - TBB dynamic libraries

## Building Stan Binaries

### For Windows

1. Install CmdStan on Windows following: https://mc-stan.org/docs/cmdstan-guide/installation.html
2. Compile the prophet.stan model:
   ```powershell
   cd path\to\cmdstan
   make path\to\seer\stan\prophet_model.exe
   ```
3. Copy the generated executable and TBB DLLs to `stan/windows/`

### For Linux

1. Install CmdStan on Linux:
   ```bash
   wget https://github.com/stan-dev/cmdstan/releases/download/vX.XX.X/cmdstan-X.XX.X.tar.gz
   tar -xzf cmdstan-X.XX.X.tar.gz
   cd cmdstan-X.XX.X
   make build
   ```
2. Compile the prophet.stan model:
   ```bash
   make /path/to/seer/stan/prophet_model
   ```
3. Copy the generated executable and .so files to `stan/linux/`

### For macOS

Similar to Linux, but libraries will have `.dylib` extension instead of `.so`

## Environment Variable Override

You can override the model path by setting the `PROPHET_MODEL_PATH` environment variable:

```bash
export PROPHET_MODEL_PATH=/custom/path/to/prophet_model
```

## Notes

- The Rust code automatically detects the platform and loads the appropriate binary
- Binaries must be compiled on the target platform (Windows binaries won't work on Linux)
- TBB libraries are required for parallel execution
- The model source (`prophet.stan`) is the same across all platforms
