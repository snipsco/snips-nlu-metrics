from __future__ import unicode_literals

from snips_nlu.constants import TEXT


def input_string_from_chunks(chunks):
    return "".join(chunk[TEXT] for chunk in chunks)
