"""config.example.toml と config.toml を比較し、config.toml に無い項目だけを追加する。

既存の値・コメント・並び順は一切変更しない（config.example.toml 側にしかないキー／テーブルのみ追記）。
config.toml が存在しない場合は config.example.toml をそのままコピーして新規作成する。

    venv/bin/python -m scratch.LatentDDPM.sync_config
"""
from __future__ import annotations

import copy
from pathlib import Path

import tomlkit

_DIR = Path(__file__).parent
_EXAMPLE_PATH = _DIR / "config.example.toml"
_CONFIG_PATH = _DIR / "config.toml"


def _merge_missing(src: tomlkit.TOMLDocument, dst: tomlkit.TOMLDocument, prefix: str = "") -> list[str]:
    """src (example) にあって dst (既存) に無いキー／テーブルを dst に追記する。追加したキーのパス一覧を返す。"""
    added: list[str] = []
    for key, val in src.items():
        path = f"{prefix}{key}"
        if key not in dst:
            dst[key] = copy.deepcopy(val)
            added.append(path)
        elif isinstance(val, dict) and isinstance(dst[key], dict):
            added.extend(_merge_missing(val, dst[key], prefix=f"{path}."))
    return added


def main() -> None:
    if not _EXAMPLE_PATH.exists():
        raise FileNotFoundError(f"{_EXAMPLE_PATH} が見つかりません")

    example = tomlkit.parse(_EXAMPLE_PATH.read_text(encoding="utf-8"))

    if not _CONFIG_PATH.exists():
        _CONFIG_PATH.write_text(tomlkit.dumps(example), encoding="utf-8")
        print(f"{_CONFIG_PATH} が無かったため {_EXAMPLE_PATH.name} をコピーして新規作成しました。")
        return

    config = tomlkit.parse(_CONFIG_PATH.read_text(encoding="utf-8"))
    added = _merge_missing(example, config)

    if not added:
        print(f"{_CONFIG_PATH.name} は {_EXAMPLE_PATH.name} と比べて不足している項目はありませんでした。")
        return

    _CONFIG_PATH.write_text(tomlkit.dumps(config), encoding="utf-8")
    print(f"{_CONFIG_PATH.name} に以下の項目を追加しました:")
    for path in added:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
