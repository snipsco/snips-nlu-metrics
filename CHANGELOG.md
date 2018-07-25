# Changelog
All notable changes to this project will be documented in this file.

## [0.13.0] - 2018-07-25
### Fixed
- Crash while computing metrics when either actual or predicted intent is unknown

### Removed
- APIs depending implicitely on Snips NLU: 
    - `compute_cross_val_nlu_metrics`
    - `compute_train_test_nlu_metrics`
    
### Changed
- Use flexible version specifiers for dependencies


## [0.12.0] - 2018-03-29
### Added
- F1 scores for intent classification and entity extraction
- Confusion matrix
- New option to exclude slot metrics in the output
- Samples


[0.13.0]: https://github.com/snipsco/snips-nlu-metrics/compare/0.12.0...0.13.0
[0.12.0]: https://github.com/snipsco/snips-nlu-metrics/compare/0.11.1...0.12.0