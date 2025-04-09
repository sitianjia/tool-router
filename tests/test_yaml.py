from toolrouter.yaml_loader import from_yaml


def test_load_yaml(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text(
        "tools:\n  - name: a\n    description: aa\n  - name: b\n    description: bb\n"
    )
    reg = from_yaml(p)
    assert reg.names() == ["a", "b"]
    assert reg.get("a").description == "aa"
