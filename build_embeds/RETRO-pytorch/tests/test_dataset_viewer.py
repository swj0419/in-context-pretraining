from typing import Dict

from retro_pytorch.utils import parse_meta


def test_parse_meta():
    entry: Dict = {"id": "myid", "meta": {"pile_set_name": "a_pile"}}
    val = parse_meta(entry)
    assert val["doc_id"] == "myid"
    assert val["pile_set_name"] == "a_pile"
