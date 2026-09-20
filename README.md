<p align="center">
  <img src="./images/logo.png" width="340" alt="centaur logo">
</p>
<h1 align="center">centaur</h1>
<p align="center">
  SIMD experiments in Rust.
</p>
<p align="center">
  <a href="https://github.com/shayne-fletcher/centaur/actions/workflows/build-and-test-ubuntu.yml">
    <img src="https://github.com/shayne-fletcher/centaur/actions/workflows/build-and-test-ubuntu.yml/badge.svg" alt="Ubuntu CI">
  </a>
  <a href="https://github.com/shayne-fletcher/centaur/actions/workflows/build-and-test-neon.yml">
    <img src="https://github.com/shayne-fletcher/centaur/actions/workflows/build-and-test-neon.yml/badge.svg" alt="ARM64 NEON CI">
  </a>
  <a href="https://shayne-fletcher.github.io/centaur/">
    <img src="https://img.shields.io/badge/docs-github.io-blue" alt="docs">
  </a>
</p>

`centaur` is SIMD experiments in Rust.

The project uses the pinned nightly Rust toolchain recorded in `rust-toolchain`.

## Building

```bash
cargo build                            # build
cargo test                             # the whole suite
```

## Acknowledgments

Edward Kmett's [simd](https://github.com/ekmett/simd) is a source of ideas for this project.

## License

BSD-3-Clause. See [LICENSE](LICENSE).
