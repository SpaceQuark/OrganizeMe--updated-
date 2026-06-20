from organizeme.extract import read_file_data


def test_reads_plain_text(tmp_path):
    f = tmp_path / 'note.txt'
    f.write_text('hello world')
    assert read_file_data(str(f)) == 'hello world'


def test_unsupported_extension_returns_none(tmp_path):
    f = tmp_path / 'data.xyz'
    f.write_text('whatever')
    assert read_file_data(str(f)) is None


def test_empty_file_returns_none(tmp_path):
    f = tmp_path / 'empty.txt'
    f.write_text('   ')
    assert read_file_data(str(f)) is None
