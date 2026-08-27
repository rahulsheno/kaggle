import json
import nbformat
from nbclient import NotebookClient

NB = "Challenge_TheDoseLedger.ipynb"
nb = json.load(open(NB, encoding="utf-8"))

src17 = open("_nb_E_eval.py", encoding="utf-8").read()
src19 = open("_nb_F_submit.py", encoding="utf-8").read()
nb["cells"][17]["source"] = src17.splitlines(keepends=True)
nb["cells"][19]["source"] = src19.splitlines(keepends=True)
json.dump(nb, open(NB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("sources patched")

tmp = nbformat.v4.new_notebook()
tmp.cells = [
    nbformat.v4.new_code_cell(
        "import numpy as np\n"
        "_s = np.load('submission.npz')\n"
        "restored = _s['restored']\n"
        "digits = _s['digits']\n"
        "ledger = 'R10:6:0'\n"),
    nbformat.v4.new_code_cell(src19),
]
client = NotebookClient(tmp, timeout=None, kernel_name="python3",
                        resources={"metadata": {"path": "."}})
client.execute()
print("--- cell 19 re-execution output ---")
for o in tmp.cells[1].outputs:
    if "text" in o:
        print("".join(o["text"]))

nb = json.load(open(NB, encoding="utf-8"))
nb["cells"][19]["outputs"] = json.loads(json.dumps(tmp.cells[1].outputs))
nb["cells"][19]["execution_count"] = tmp.cells[1].execution_count
json.dump(nb, open(NB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("cell 19 outputs updated in notebook")
