from distrainer.ledger import LEDGER_FILENAME, Ledger


def test_defaults_and_done_positions():
    fresh = Ledger()
    assert (fresh.segment, fresh.cursor, fresh.world_size, fresh.pass_idx) == (0, 0, 0, 0)
    assert fresh.done_positions() == 0
    assert Ledger(segment=2, cursor=2, world_size=3).done_positions() == 6


def test_save_load_roundtrip(tmp_path):
    ledger = Ledger(segment=2, cursor=3, world_size=3, pass_idx=1, run_attempt=2)
    path = ledger.save(str(tmp_path))
    assert path.endswith(LEDGER_FILENAME)
    assert Ledger.load(str(tmp_path)) == ledger


def test_json_ignores_unknown_keys_and_coerces_ints():
    ledger = Ledger.from_json('{"segment": "4", "cursor": 1, "world_size": 2, "extra": true}')
    assert ledger == Ledger(segment=4, cursor=1, world_size=2)
    assert '"segment": 4' in ledger.to_json()
