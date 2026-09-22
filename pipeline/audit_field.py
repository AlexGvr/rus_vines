"""Audit field partition integrity and reference coverage without loading models."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def digest(path):
    with path.open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def main():
    manifest = ROOT / "data/field/manifest.csv"
    rows = list(csv.DictReader(manifest.open()))
    index = list(csv.DictReader((ROOT / "data/index/clean_mv.csv").open()))
    wines = {w["slug"]: w for w in json.loads(
        (ROOT / "data/catalog/catalog.json").read_text())["wines"]}
    hashes, missing = defaultdict(list), []
    for r in rows:
        path = next((p for base in ("data/field", "dataset/eval/queries")
                     if (p := ROOT / base / r["image"]).exists()), None)
        if path is None:
            missing.append(r["image"])
        else:
            hashes[digest(path)].append(r)
    overlaps = {}
    for key in ("image", "series", "slug"):
        groups = defaultdict(set)
        for r in rows:
            if r[key]:
                groups[r[key]].add(r["split"])
        overlaps[key] = sorted(k for k, parts in groups.items() if len(parts) > 1)
    indexed_slugs = {r["slug"] for r in index if r["slug"]}
    ref_hashes, missing_refs = defaultdict(list), []
    for name in sorted({r["path"] for r in index}):
        p = ROOT / name
        if p.exists():
            ref_hashes[digest(p)].append(name)
        else:
            missing_refs.append(name)
    report = {
        "manifest_sha256": digest(manifest), "n": len(rows),
        "test_role": "development_regression_not_independent",
        "split_overlaps": overlaps, "missing_images": missing,
        "unknown_slugs": sorted({r['slug'] for r in rows if r['slug'] and r['slug'] not in wines}),
        "exact_duplicate_images": [[r['image'] for r in rs] for rs in hashes.values() if len(rs)>1],
        "exact_duplicate_split_overlaps": [[r['image'] for r in rs] for rs in hashes.values()
                                            if len({r['split'] for r in rs})>1],
        "query_reference_overlap": [dict(images=[r['image'] for r in hashes[h]], refs=ref_hashes[h])
                                    for h in hashes.keys() & ref_hashes.keys()],
        "missing_index_files": missing_refs,
        "catalog_without_index": sorted(wines.keys() - indexed_slugs),
        "positive_images_without_index": [r for r in rows if r['slug'] and r['slug'] not in indexed_slugs],
        "limitations": ["Byte hashes detect exact duplicates only; resized and near duplicates require visual review.",
                        "Valid slug does not verify that its label matches the photograph."]}
    out = ROOT / "data/validation/field_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(out)
    print(json.dumps({k: len(v) for k,v in report.items() if isinstance(v,list)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
