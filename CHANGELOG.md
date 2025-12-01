# Changelog

All notable changes to Pavlov will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-12-01

### Breaking Changes

- **Package restructuring**: Environments moved to `pavlov.envs` submodule
- **Class renaming**: Removed `Py` prefix (e.g., `PyCartPoleVecEnv` → `CartPoleVecEnv`)
- Old import patterns deprecated (backwards compatible with warnings until v0.4.0)

### Added

- `pavlov.envs` submodule for environment classes (CartPoleVecEnv, MountainCarVecEnv, PendulumVecEnv)
- `pavlov.utils` submodule for utilities (Logger)
- Migration guide in README.md
- Comprehensive CHANGELOG.md

### Changed

- **Recommended import pattern**: `from pavlov.envs import CartPoleVecEnv` (was: `from pavlov import PyCartPoleVecEnv`)
- Cleaner API without `Py` prefix for better Python ergonomics
- Updated all examples and documentation to use new import patterns

### Deprecated

- Root-level imports: `from pavlov import PyCartPoleVecEnv`
  - Use instead: `from pavlov.envs import CartPoleVecEnv`
  - Old imports will be removed in v0.4.0

### Technical Details

- Rust PyO3 submodule registration for `pavlov.envs`
- Python facade layer for clean class name exports (removes `Py` prefix)
- All tests and benchmarks updated to new API
- Backwards compatibility maintained for smooth migration

## [0.1.0] - Previous Release

### Added

- Initial release with three Gymnasium-compatible environments:
  - CartPole-v1 (discrete actions)
  - MountainCar-v0 (discrete actions)
  - Pendulum-v1 (continuous actions)
- SIMD-optimized Rust implementations using AVX2
- Zero-copy numpy array integration via PyO3
- ~600x faster than Gymnasium for vectorized environments
- Auto-reset functionality for seamless episode transitions
- Episode logging and statistics tracking
- Struct-of-Arrays (SoA) memory layout for cache efficiency
- Python bindings with complete type annotations
