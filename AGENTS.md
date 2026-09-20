# Writing conventions

Wrap comments in Rust source at 80 columns, including module documentation (`//!`), item documentation (`///`), and ordinary comments (`//`). The preference for one physical line per prose paragraph applies to Markdown and other documentation files, not comments in Rust source. Preserve code examples and URLs when wrapping would change their meaning.

When importing multiple items from one Rust module, use one `use` statement per item so each imported name occupies its own line.

Centaur pins the nightly toolchain because its formatter policy uses `imports_granularity = "Item"`, matching the established Monarch project convention. Keep the formatter policy in `rustfmt.toml` and run `cargo fmt --all --check` before review.
