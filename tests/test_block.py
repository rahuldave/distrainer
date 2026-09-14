import pyarrow as pa
import pytest
from conftest import make_table

from distrainer.block import BLOCK_ID_COLUMN, BlockRef, block_locator, read_block, write_block


def test_write_then_read_roundtrip_adds_block_id_column(store):
    fs, root = store
    ref = write_block(fs, root, "b000017", make_table(5), meta={"kind": "toy"})
    assert ref == BlockRef("b000017", "blocks/b000017.parquet", 5, {"kind": "toy"})
    table = read_block(fs, root, ref)
    assert table.num_rows == 5
    assert table.column(BLOCK_ID_COLUMN).to_pylist() == ["b000017"] * 5
    assert table.column("x").to_pylist() == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_existing_block_id_column_is_validated(store):
    fs, root = store
    ok = make_table(2).append_column(BLOCK_ID_COLUMN, pa.array(["b1", "b1"]))
    assert write_block(fs, root, "b1", ok).num_rows == 2
    bad = make_table(2).append_column(BLOCK_ID_COLUMN, pa.array(["b1", "b2"]))
    with pytest.raises(ValueError):
        write_block(fs, root, "b1", bad)


@pytest.mark.parametrize("bad_id", ["", "a/b", ".hidden", 7])
def test_invalid_block_ids_rejected(bad_id):
    with pytest.raises(ValueError):
        block_locator(bad_id)


def test_blockref_dict_roundtrip():
    ref = BlockRef("b", "blocks/b.parquet", 3, {"m": 1})
    assert BlockRef.from_dict(ref.to_dict()) == ref
    assert BlockRef.from_dict({"block_id": "b", "locator": "l", "num_rows": "3"}).meta == {}


@pytest.mark.parametrize("locator", ["../x.parquet", "/abs/x.parquet", "blocks//x.parquet", ""])
def test_read_block_rejects_locators_escaping_the_store(store, locator):
    fs, root = store
    with pytest.raises(ValueError):
        read_block(fs, root, BlockRef("x", locator, 1))


def test_block_num_rows_reads_the_footer(store):
    from distrainer.block import block_num_rows

    fs, root = store
    ref = write_block(fs, root, "n", make_table(7))
    assert block_num_rows(fs, root, ref) == 7
    assert [n for n in __import__("os").listdir(f"{root}/blocks") if ".tmp-" in n] == []
