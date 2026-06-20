import os

from organizeme.metadata import sanitize_filename
from organizeme import organize


def test_sanitize_filename_strips_illegal_and_limits_words():
    assert sanitize_filename('a/b:c*d', max_words=5) == 'abcd'
    assert sanitize_filename('one two three four', max_words=2) == 'one_two'
    assert sanitize_filename('   ', max_words=3) == 'untitled'
    assert sanitize_filename('hello   world', max_words=5) == 'hello_world'


def test_compute_operations_resolves_collisions():
    records = [
        {'file_path': '/src/a.txt', 'foldername': 'docs', 'filename': 'report'},
        {'file_path': '/src/b.txt', 'foldername': 'docs', 'filename': 'report'},
    ]
    ops = organize.compute_operations(records, '/out')
    destinations = [op['destination'] for op in ops]
    assert destinations[0] == os.path.join('/out', 'docs', 'report.txt')
    assert destinations[1] == os.path.join('/out', 'docs', 'report_1.txt')
    assert len(set(destinations)) == 2


def test_compute_operations_skips_already_processed():
    records = [{'file_path': '/src/a.txt', 'foldername': 'docs', 'filename': 'a'}]
    processed = {'/src/a.txt'}
    ops = organize.compute_operations(records, '/out', processed_files=processed)
    assert ops == []


def test_process_files_by_type_groups_by_category(tmp_path):
    txt = tmp_path / 'note.txt'
    img = tmp_path / 'pic.png'
    txt.write_text('hi')
    img.write_bytes(b'\x89PNG')
    ops = organize.process_files_by_type([str(txt), str(img)], str(tmp_path / 'out'))
    dests = {op['source']: op['destination'] for op in ops}
    assert os.path.basename(os.path.dirname(dests[str(txt)])) == 'text'
    assert os.path.basename(os.path.dirname(dests[str(img)])) == 'image'


def test_quarantine_operations_moves_only_duplicates():
    clusters = [{'keep': '/a/orig.txt', 'duplicates': ['/a/copy.txt', '/a/copy2.txt']}]
    ops = organize.quarantine_operations(clusters, '/q')
    assert {op['source'] for op in ops} == {'/a/copy.txt', '/a/copy2.txt'}
    assert all(op['action'] == 'move' for op in ops)
    assert all(op['destination'].startswith('/q') for op in ops)


def test_execute_operations_dry_run_makes_no_changes(tmp_path, capsys):
    src = tmp_path / 'x.txt'
    src.write_text('data')
    dest = tmp_path / 'out' / 'x.txt'
    organize.execute_operations(
        [{'source': str(src), 'destination': str(dest), 'action': 'copy'}], dry_run=True
    )
    assert not dest.exists()
    assert '[dry-run]' in capsys.readouterr().out


def test_execute_operations_apply_copies(tmp_path):
    src = tmp_path / 'x.txt'
    src.write_text('data')
    dest = tmp_path / 'out' / 'x.txt'
    organize.execute_operations(
        [{'source': str(src), 'destination': str(dest), 'action': 'copy'}], dry_run=False
    )
    assert dest.read_text() == 'data'
