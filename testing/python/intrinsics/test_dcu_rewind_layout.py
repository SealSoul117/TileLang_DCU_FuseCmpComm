"""Hygon-style AT/BN + AN/BT rewind storage is a physical bijection."""

from tilelang.intrinsics.dcu_lds_rewind import (
    AnBtRewindGeom,
    AtBnRewindGeom,
    an_bt_storage_map,
    at_bn_storage_map,
    shared_to_local_rewind_an_bt,
    shared_to_local_rewind_at_bn,
)


def test_at_bn_storage_is_bijection():
    # 128x32 fp16, 256 threads, 4 MN warps × 32-wide atoms → ds_read_b128 wrap.
    geom = AtBnRewindGeom.try_create(128, 32, 256, 2, 4, 32, 1, 16)
    assert geom is not None
    seen = set()
    for row in range(geom.block_mn):
        for col in range(geom.block_k):
            phys = at_bn_storage_map(row, col, geom)
            assert 0 <= phys[0] < geom.block_mn
            assert 0 <= phys[1] < geom.block_k
            assert phys not in seen
            seen.add(phys)
    assert len(seen) == geom.block_mn * geom.block_k


def test_an_bt_storage_is_bijection():
    geom = AnBtRewindGeom.try_create(32, 128, 256, 2)
    assert geom is not None
    seen = set()
    for k in range(geom.block_k):
        for n in range(geom.block_mn):
            phys = an_bt_storage_map(k, n, geom)
            assert 0 <= phys[0] < geom.block_k
            assert 0 <= phys[1] < geom.block_mn
            assert phys not in seen
            seen.add(phys)
    assert len(seen) == geom.block_k * geom.block_mn


def test_ldmatrix_maps_compose_storage():
    at = AtBnRewindGeom.try_create(128, 32, 256, 2, 4, 32, 1, 16)
    an = AnBtRewindGeom.try_create(32, 128, 256, 2)
    assert at is not None and an is not None
    for tx in (0, 17, 31, 45, 63):
        for lid in range(4):
            r, c = shared_to_local_rewind_at_bn(tx, lid, 0, 0, 1, 4, 0, at)
            assert 0 <= r < 128 and 0 <= c < 32
        for lid in range(8):
            r, c = shared_to_local_rewind_an_bt(tx, lid, 0, 0, 4, 0, an)
            assert 0 <= r < 32 and 0 <= c < 128


if __name__ == "__main__":
    test_at_bn_storage_is_bijection()
    test_an_bt_storage_is_bijection()
    test_ldmatrix_maps_compose_storage()
    print("ok")
