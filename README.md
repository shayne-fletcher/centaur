<p align="center">
  <img src="./images/logo.png" width="340" alt="centaur logo">
</p>
<h1 align="center">centaur</h1>
<p align="center">
  SIMD experiments in Rust.
</p>
<p align="center">
  <a href="https://github.com/shayne-fletcher/centaur/actions/workflows/build-and-test-sse.yml">
    <img src="https://github.com/shayne-fletcher/centaur/actions/workflows/build-and-test-sse.yml/badge.svg" alt="x86-64 SSE CI">
  </a>
  <a href="https://github.com/shayne-fletcher/centaur/actions/workflows/build-and-test-scalar.yml">
    <img src="https://github.com/shayne-fletcher/centaur/actions/workflows/build-and-test-scalar.yml/badge.svg" alt="scalar-only CI">
  </a>
  <a href="https://github.com/shayne-fletcher/centaur/actions/workflows/build-and-test-neon.yml">
    <img src="https://github.com/shayne-fletcher/centaur/actions/workflows/build-and-test-neon.yml/badge.svg" alt="ARM64 NEON CI">
  </a>
  <a href="https://shayne-fletcher.github.io/centaur/">
    <img src="https://img.shields.io/badge/docs-github.io-blue" alt="docs">
  </a>
</p>

Centaur is an experimental `no_std` Rust crate for expressing numerical kernels once and running them over scalar or architecture-specific SIMD registers.

Its first kernel is `dot_f32`, with SSE and NEON implementations sharing the same four-register pack structure.

## Usage

```rust
use centaur::dot_f32;

let lhs = [1.0, 2.0, 3.0, 4.0];
let rhs = [5.0, 6.0, 7.0, 8.0];

assert_eq!(dot_f32(&lhs, &rhs), 70.0);
```

`dot_f32` multiplies corresponding elements and returns their sum. If the slices have different lengths, it uses their common prefix.

## Backends

| Target | Default backend |
| --- | --- |
| x86-64 | Four four-lane SSE registers |
| AArch64 with NEON | Four four-lane NEON registers |
| Other targets | Four scalar accumulators |

The `scalar-only` feature forces the scalar backend on every target. Here, scalar describes the source arithmetic: four independent `f32` accumulators. An optimizing compiler may still auto-vectorize that code.

Centaur is at an early experimental stage. Its API and backend structure may change as further kernels establish the design.

## Development

```bash
cargo build                            # build
cargo test                             # the whole suite
cargo test --features scalar-only      # force the scalar backend
cargo doc --workspace --no-deps --document-private-items # build developer docs
```

This repository pins a nightly toolchain for formatting. The library itself builds on stable Rust.

## Acknowledgments

Edward Kmett's [simd](https://github.com/ekmett/simd) is a source of ideas for this project.

## License

BSD-3-Clause. See [LICENSE](LICENSE).
