"""Quick syntax validator for QRQFL_colab.ipynb."""
import json, ast, pathlib

nb_path = pathlib.Path(__file__).parent / "QRQFL_colab.ipynb"
nb = json.load(open(nb_path, "r", encoding="utf-8"))
print(f"Total cells: {len(nb['cells'])}")

errs = 0
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    src = c["source"]
    if isinstance(src, list):
        src = "".join(src)
    try:
        ast.parse(src)
        print(f"  cell {i:2d} (code, {len(src):4d} chars): OK")
    except SyntaxError as e:
        print(f"  cell {i:2d} (code, {len(src):4d} chars): SYNTAX ERROR")
        print(f"    -> line {e.lineno}: {e.msg}")
        if src:
            lines = src.splitlines()
            if 0 < e.lineno <= len(lines):
                print(f"    -> {lines[e.lineno - 1]!r}")
        errs += 1

print()
print("NOTEBOOK CLEAN" if errs == 0 else f"{errs} CELLS WITH SYNTAX ERRORS")
