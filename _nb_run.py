import time
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


class LoggingClient(NotebookClient):
    def execute_cell(self, cell, index, **kw):
        src = "".join(cell.source).splitlines()
        head = src[0][:60] if src else ""
        print("CELL %d START %s | %s" % (index, cell.cell_type, head), flush=True)
        t0 = time.time()
        try:
            out = super().execute_cell(cell, index, **kw)
        except CellExecutionError as e:
            print("CELL %d FAILED after %.0fs" % (index, time.time() - t0), flush=True)
            raise
        tail = []
        for o in cell.get("outputs", []):
            if "text" in o:
                tail += o["text"].splitlines()
        for ln in tail[-3:]:
            print("   >", ln[:110], flush=True)
        print("CELL %d done %.0fs" % (index, time.time() - t0), flush=True)
        return out


nb = nbformat.read("Challenge_TheDoseLedger.ipynb", as_version=4)
client = LoggingClient(nb, timeout=None, kernel_name="python3",
                        resources={"metadata": {"path": "."}})
client.execute()
nbformat.write(nb, "Challenge_TheDoseLedger.ipynb")
print("notebook executed and saved", flush=True)
