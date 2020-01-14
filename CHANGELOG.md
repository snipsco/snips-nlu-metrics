# Changelog
All notable changes to this project will be documented in this file.

## [0.14.6] - 2020-01-14
### Added
- Support for python3.8

## [0.14.5] - 2019-08-20
### Fixed
- Fix issue with intents filter when dataset has not enough data [#118](https://github.com/snipsco/snips-nlu-metrics/pull/118)

## [0.14.4] - 2019-06-18
### Fixed
- Update dependencies

## [0.14.3] - 2019-05-10
### Added
- Add optional parameter `intents_filter` to metrics APIs [#115](https://github.com/snipsco/snips-nlu-metrics/pull/115)

## [0.14.2] - 2019-03-21
### Added
- Number of exact parsings
- Possibility to provide out-of-domain utterances
- Logging

### Fixed
- Hanging issue when using multiple workers, when one job returns a non-zero exit code

## [0.14.1] - 2019-01-07
### Added
- Support for new NLU output format

### Fixed
- Bug with None intent when computing average metrics

## [0.14.0] - 2018-11-13
### Added
- Possibility to use parallel workers
- Seed parameter for reproducibility
- Average metrics for intent classification and slot filling

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


[0.14.5]: https://github.com/snipsco/snips-nlu-metrics/compare/0.14.4...0.14.5
[0.14.4]: https://github.com/snipsco/snips-nlu-metrics/compare/0.14.3...0.14.4
[0.14.3]: https://github.com/snipsco/snips-nlu-metrics/compare/0.14.2...0.14.3
[0.14.2]: https://github.com/snipsco/snips-nlu-metrics/compare/0.14.1...0.14.2
[0.14.1]: https://github.com/snipsco/snips-nlu-metrics/compare/0.14.0...0.14.1
[0.14.0]: https://github.com/snipsco/snips-nlu-metrics/compare/0.13.0...0.14.0
[0.13.0]: https://github.com/snipsco/snips-nlu-metrics/compare/0.12.0...0.13.0
[0.12.0]: https://github.com/snipsco/snips-nlu-metrics/compare/0.11.1...0.12.0