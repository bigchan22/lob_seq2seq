import csv
from pathlib import Path
def write_csv(path,rows,fieldnames):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=fieldnames);w.writeheader();w.writerows(rows)
