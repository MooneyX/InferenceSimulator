"""块身份直接编码的基础约定。"""

import pytest

from simcore.prefix_hash import encode, stable_derive


def test_encode_uses_request_and_absolute_block_index() -> None:
    assert encode(0, 0) == "chunk:0:0"
    assert encode(12, 0) == "chunk:12:0"
    assert encode(12, 3) == "chunk:12:3"
    assert encode(123, 4) != encode(12, 34)


def test_encode_is_independent_of_previous_blocks() -> None:
    assert encode(7, 3) == encode(7, 3)
    assert encode(7, 3) != encode(7, 2)
    assert encode(7, 3) != encode(8, 3)


@pytest.mark.parametrize("request_id, block_index", [(-1, 0), (0, -1)])
def test_encode_rejects_negative_values(request_id: int, block_index: int) -> None:
    with pytest.raises(ValueError, match="非负整数"):
        encode(request_id, block_index)


@pytest.mark.parametrize("request_id, block_index", [(True, 0), (0, 1.5), ("1", 0)])
def test_encode_rejects_non_integer_values(request_id: object, block_index: object) -> None:
    with pytest.raises(TypeError, match="非负整数"):
        encode(request_id, block_index)  # type: ignore[arg-type]


def test_stable_derive_remains_domain_specific() -> None:
    assert stable_derive(17, "length") == stable_derive(17, "length")
    assert stable_derive(17, "length") != stable_derive(17, "root-arrival")
