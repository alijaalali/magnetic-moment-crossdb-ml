"""
Materials Project magnetic-moment query for the magnetic-moment
cross-database transfer study (MP vs. JARVIS-DFT).

Retrieves total_magnetization for all MP computed entries with
energy_above_hull <= 0.1 eV/atom. No element restriction is applied,
since magnetism is not limited to a single stoichiometry. Requires the
MP_API_KEY environment variable. Output: data/mp_magmom_raw.json.
"""
import json
import os
import time
from pathlib import Path

from mp_api.client import MPRester

API_KEY = os.environ["MP_API_KEY"]
import os
PROJ = Path(os.environ.get("MAGMOM_ROOT", str(Path(__file__).resolve().parent)))
DATA = PROJ / "data"
DATA.mkdir(exist_ok=True)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def query_mp():
    with MPRester(API_KEY) as mpr:
        docs = mpr.materials.summary.search(
            energy_above_hull=(0, 0.1),
            fields=[
                "material_id", "formula_pretty", "total_magnetization",
                "energy_above_hull", "nsites", "theoretical", "is_stable",
            ],
            chunk_size=1000,
        )
    return docs


def main():
    t0 = time.time()
    log("Querying Materials Project (energy_above_hull <= 0.1 eV/atom)...")
    docs = query_mp()
    log(f"Retrieved {len(docs)} docs in {time.time()-t0:.1f}s")

    rows = []
    for d in docs:
        rows.append({
            "material_id": str(d.material_id),
            "formula_pretty": d.formula_pretty,
            "total_magnetization": d.total_magnetization,
            "energy_above_hull": d.energy_above_hull,
            "nsites": d.nsites,
            "theoretical": d.theoretical,
            "is_stable": d.is_stable,
        })

    with open(DATA / "mp_magmom_raw.json", "w") as f:
        json.dump(rows, f)
    log(f"Saved {len(rows)} rows to data/mp_magmom_raw.json")


if __name__ == "__main__":
    main()
