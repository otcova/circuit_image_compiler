# Development Compilation (Dynamic linking)

```bash
# Download LLVM 20.1.x (newer version might also work)
sudo apt install llvm-20

# Fast Build (~5s after first build)
cargo build
```

# Release Compilation (Static linking)

```bash
# Download LLVM with all static libraries.
# - Often some like "libPolly.a" are ommited, so previous "apt install" wont work.
# - Is recommended to use compile llvm: git clone + cmake + ninja (or make)

# Production Build (static)
cargo build --release --features llvm_static --no-default-features
```
