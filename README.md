## Compile for development

- Simplest & fastest way (without llvm circuit compiler):

```bash
# Fast Build (<10s after first build)
cargo build
```

- Fast builds including llvm circuit compiler:

```bash
# Download LLVM 20.1.x (newer version might also work)
sudo apt install llvm-20

# Fast Build (<20s after first build)
# The executable will only work on computers with llvm dynamic libraries
cargo build --features llvm_dyn
```

## Compile to release

```bash
# Download LLVM with all static libraries.
# - Often some like "libPolly.a" are ommited, so previous "apt install" wont work.
# - Is recommended to use compile llvm: git clone + cmake + ninja (or make)

# Production Build (includes llvm with static linking)
cargo build --release --features llvm --no-default-features
```
