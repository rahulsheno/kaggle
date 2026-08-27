import json
import shutil

NB = "Challenge_TheDoseLedger.ipynb"
BACKUP = r"C:\WINDOWS\TEMP\kilo\Challenge_TheDoseLedger_backup.ipynb"
shutil.copyfile(BACKUP, NB)
nb = json.load(open(NB, encoding="utf-8"))


def md(path):
    return {"cell_type": "markdown", "metadata": {},
            "source": open(path, encoding="utf-8").read().splitlines(keepends=True)}


def code(path):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": open(path, encoding="utf-8").read().splitlines(keepends=True)}


new_cells = [
    md("_nb_A_approach.md"),
    code("_nb_B_setup.py"),
    code("_nb_C_denoisers.py"),
    code("_nb_D_classifiers.py"),
    code("_nb_E_eval.py"),
]

cells = nb["cells"]
assert cells[13]["cell_type"] == "code" and "".join(cells[13]["source"]).strip() == ""
assert cells[12]["cell_type"] == "markdown" and "Your work" in "".join(cells[12]["source"])
cells[13:14] = new_cells

sub = [c for c in cells if c["cell_type"] == "code"
       and "write_submission" in "".join(c["source"]) and "SUBMISSION" in "".join(c["source"])]
assert len(sub) == 1
src = "".join(sub[0]["source"])
src = src.replace("# write_submission(restored, digits, ledger)",
                  "write_submission(restored, digits, ledger)")
sub[0]["source"] = src.splitlines(keepends=True)
sub[0]["outputs"] = []
sub[0]["execution_count"] = None

nb["cells"] = cells
json.dump(nb, open(NB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("cells:", len(cells))
for i, c in enumerate(cells):
    t = "".join(c["source"])
    print(i, c["cell_type"], len(t), "|", t.splitlines()[0][:70] if t.strip() else "(empty)")
