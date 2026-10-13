def test_names_file_shape(words):
    assert len(words) == 32033
    assert min(map(len, words)) == 2
    assert max(map(len, words)) == 15


def test_names_are_lowercase_ascii_letters(words):
    chars = set("".join(words))
    assert chars == set("abcdefghijklmnopqrstuvwxyz")


def test_vocabulary_size_matches_notebooks(words):
    """Notebooks build a 27-symbol vocabulary: 26 letters + '.' start/end marker."""
    assert len(set("".join(words))) + 1 == 27
