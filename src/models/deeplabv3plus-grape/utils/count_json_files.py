#!/usr/bin/env python
"""Count JSON files in a directory."""

from __future__ import annotations

from pathlib import Path


def count_json_files(directory: Path, recursive: bool) -> int:
    pattern = "**/*.json" if recursive else "*.json"
    return sum(1 for path in directory.glob(pattern) if path.is_file())


def main() -> None:
    # ====================== 在这里修改你的目录 ======================
    TARGET_DIR = r"D:\Code\grape_new_code\dataset"
    RECURSIVE = True                       # True=递归子文件夹 / False=只统计当前文件夹
    # ==============================================================

    directory = Path(TARGET_DIR).expanduser().resolve()
    if not directory.exists():
        raise SystemExit(f"目录不存在: {directory}")
    if not directory.is_dir():
        raise SystemExit(f"不是文件夹: {directory}")

    count = count_json_files(directory, recursive=RECURSIVE)
    scope = "包含子文件夹" if RECURSIVE else "仅当前文件夹"
    print(f"JSON 文件数量: {count}")
    print(f"扫描目录: {directory}")
    print(f"扫描范围: {scope}")


if __name__ == "__main__":
    main()