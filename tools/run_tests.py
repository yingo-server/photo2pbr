#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""轻量测试跑器（无 pytest 环境也能跑；CI 上仍用 pytest）。"""
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    tests_dir = ROOT / "tests"
    files = sorted(p for p in tests_dir.glob("test_*.py"))
    passed, failed = 0, []
    for f in files:
        import importlib.util

        spec = importlib.util.spec_from_file_location(f.stem, f)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Exception:
            failed.append((f.name, "import", traceback.format_exc()))
            continue
        for name in dir(mod):
            if not name.startswith("test_"):
                continue
            fn = getattr(mod, name)
            if not callable(fn):
                continue
            try:
                fn()
                passed += 1
                print("  OK   %s::%s" % (f.stem, name))
            except Exception:
                failed.append((f.name, name, traceback.format_exc()))
                print("  FAIL %s::%s" % (f.stem, name))
    print("-" * 60)
    print("通过 %d / 失败 %d" % (passed, len(failed)))
    for fname, name, tb in failed:
        print("=" * 60)
        print("FAIL %s::%s" % (fname, name))
        print(tb)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())