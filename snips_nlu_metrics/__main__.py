# coding=utf-8
from __future__ import print_function, unicode_literals

from snips_nlu_metrics.cli.utils import PrettyPrintLevel, pretty_print


def main():
    import sys

    import plac
    from snips_nlu_metrics.cli import train_test_split

    commands = {
        "train-test-split": train_test_split,
    }
    if len(sys.argv) == 1:
        pretty_print(', '.join(commands), title="Available commands", exits=1,
                     level=PrettyPrintLevel.INFO)
    command = sys.argv.pop(1)
    sys.argv[0] = 'snips-nlu-metrics %s' % command
    if command in commands:
        plac.call(commands[command], sys.argv[1:])
    else:
        pretty_print("Available: %s" % ', '.join(commands),
                     title="Unknown command: %s" % command, exits=1,
                     level=PrettyPrintLevel.INFO)


if __name__ == "__main__":
    main()
