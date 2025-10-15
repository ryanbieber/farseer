# macOS CmdStan Binaries

Place macOS-compiled CmdStan binaries here:

- `prophet_model` - Main executable (Mach-O format)
- `*.dylib` or `*.so` files - Threading Building Blocks libraries

## Building for macOS

See the parent README.md for instructions on building CmdStan binaries for macOS.

You can also use the build script:
```bash
./build_stan_binaries.sh
```

## Required Files

After building, you should have files similar to:
- prophet_model (Mach-O executable)
- libtbb.dylib or libtbb.so.2
- libtbbmalloc.dylib or libtbbmalloc.so.2
- libtbbmalloc_proxy.dylib or libtbbmalloc_proxy.so.2

Note: The exact filenames may vary depending on your CmdStan version.

## Architecture Support

The binaries should work on both Intel (x86_64) and Apple Silicon (ARM64) Macs through Rosetta 2, though native ARM64 builds are preferred for Apple Silicon.
