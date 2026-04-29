"""VCF parsing utilities for gnomAD-style population allele counts."""

from __future__ import annotations

from dataclasses import dataclass
import gzip
from pathlib import Path
from typing import Iterable, Iterator, Optional, Union

import pandas as pd


DEFAULT_POPULATIONS = ("afr", "eas", "nfe")
REQUIRED_BASE_INFO = ("variant_type",)


@dataclass(frozen=True)
class VariantRecord:
    """A retained VCF variant with parsed INFO fields."""

    chrom: str
    pos: int
    ref: str
    alt: str
    info: dict[str, Union[str, bool]]
    block_id: str
    variant_type: str


def resolve_vcf_path(path: Union[str, Path]) -> Path:
    """Resolve an input VCF path, accepting an implicit .gz suffix fallback."""
    input_path = Path(path)
    if input_path.exists():
        return input_path
    gzip_path = Path(f"{input_path}.gz")
    if gzip_path.exists():
        return gzip_path
    raise FileNotFoundError(f"VCF input not found: {input_path}")


def open_text(path: Union[str, Path]):
    """Open plain-text or gzip-compressed VCF content for reading."""
    resolved = resolve_vcf_path(path)
    if resolved.suffix == ".gz":
        return gzip.open(resolved, "rt")
    return resolved.open("rt")


def parse_info_field(info_text: str) -> dict[str, Union[str, bool]]:
    """Parse the semicolon-delimited VCF INFO column."""
    info: dict[str, Union[str, bool]] = {}
    if info_text in {"", "."}:
        return info
    for item in info_text.split(";"):
        if not item:
            continue
        if "=" not in item:
            info[item] = True
            continue
        key, value = item.split("=", 1)
        info[key] = value
    return info


def _first_value(value: Optional[Union[str, bool]]) -> Optional[str]:
    if value is None or isinstance(value, bool):
        return None
    return value.split(",", 1)[0]


def parse_int_info(info: dict[str, Union[str, bool]], key: str) -> Optional[int]:
    """Parse an integer INFO field, returning None for missing values."""
    value = _first_value(info.get(key))
    if value in {None, "", "."}:
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"INFO field {key} must be an integer; got {value!r}") from exc


def infer_block_id(
    chrom: str,
    pos: int,
    info: dict[str, Union[str, bool]],
    state: dict[str, Optional[Union[int, str]]],
    *,
    max_gap: int = 1_000_000,
) -> str:
    """Use explicit block metadata when present, otherwise infer by genomic gaps."""
    for key in ("block_id", "block", "BLOCK", "region", "locus_block"):
        value = info.get(key)
        if isinstance(value, str) and value:
            return value

    last_chrom = state.get("last_chrom")
    last_pos = state.get("last_pos")
    block_index = int(state.get("block_index") or 0)
    if last_chrom != chrom or last_pos is None or pos - int(last_pos) > max_gap:
        block_index += 1
    state["last_chrom"] = chrom
    state["last_pos"] = pos
    state["block_index"] = block_index
    return f"block_{block_index:04d}"


def iter_vcf_records(
    path: Union[str, Path],
) -> Iterator[tuple[str, int, str, str, dict[str, Union[str, bool]]]]:
    """Yield basic variant records from a VCF or VCF-like file."""
    with open_text(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 8:
                raise ValueError(f"VCF line {line_number} has fewer than 8 columns")
            chrom, pos_text, _id, ref, alt, _qual, _filter, info_text = fields[:8]
            try:
                pos = int(pos_text)
            except ValueError as exc:
                raise ValueError(f"VCF line {line_number} has invalid POS {pos_text!r}") from exc
            yield chrom, pos, ref, alt, parse_info_field(info_text)


def parse_retained_snvs(path: Union[str, Path], *, max_block_gap: int = 1_000_000) -> list[VariantRecord]:
    """Parse VCF records and retain only variants with variant_type == 'snv'."""
    records: list[VariantRecord] = []
    block_state: dict[str, Optional[Union[int, str]]] = {}
    for chrom, pos, ref, alt, info in iter_vcf_records(path):
        variant_type = _first_value(info.get("variant_type"))
        if variant_type != "snv":
            continue
        block_id = infer_block_id(chrom, pos, info, block_state, max_gap=max_block_gap)
        records.append(
            VariantRecord(
                chrom=chrom,
                pos=pos,
                ref=ref,
                alt=alt,
                info=info,
                block_id=block_id,
                variant_type=variant_type,
            )
        )
    return records


def _validate_population_fields(records: Iterable[VariantRecord], populations: Iterable[str]) -> None:
    """Ensure each requested population has at least one AC and AN field in retained SNVs."""
    seen = {population: {"AC": False, "AN": False} for population in populations}
    for record in records:
        for population in seen:
            if f"AC_{population}" in record.info:
                seen[population]["AC"] = True
            if f"AN_{population}" in record.info:
                seen[population]["AN"] = True
    missing = [
        f"{kind}_{population}"
        for population, kinds in seen.items()
        for kind, present in kinds.items()
        if not present
    ]
    if missing:
        raise ValueError(
            "Retained SNVs are missing required population INFO fields: "
            + ", ".join(sorted(missing))
        )


def retained_snv_count(path: Union[str, Path]) -> int:
    """Count records whose INFO field has variant_type == 'snv'."""
    return len(parse_retained_snvs(path))


def total_variant_count(path: Union[str, Path]) -> int:
    """Count non-header VCF records."""
    return sum(1 for _record in iter_vcf_records(path))


def build_population_allele_table(
    path: Union[str, Path],
    populations: Iterable[str] = DEFAULT_POPULATIONS,
) -> tuple[pd.DataFrame, dict[str, int], dict[str, int]]:
    """Build projected, segregating allele-count rows for requested populations.

    The cohort size for each population is the maximum AN across retained SNVs.
    Each SNV/population row is projected to that cohort size and excluded if the
    projected count is monomorphic.
    """
    populations = tuple(population.lower() for population in populations)
    records = parse_retained_snvs(path)
    _validate_population_fields(records, populations)

    n_by_population: dict[str, int] = {}
    valid_raw_by_population: dict[str, int] = {}
    for population in populations:
        ans = [
            an
            for record in records
            if (an := parse_int_info(record.info, f"AN_{population}")) is not None and an > 0
        ]
        n_by_population[population] = max(ans) if ans else 0
        valid_raw_by_population[population] = len(ans)

    rows: list[dict[str, object]] = []
    for record in records:
        for population in populations:
            ac = parse_int_info(record.info, f"AC_{population}")
            an = parse_int_info(record.info, f"AN_{population}")
            n_pop = n_by_population[population]
            if ac is None or an is None or an <= 0 or n_pop <= 1:
                continue
            if ac < 0 or ac > an:
                raise ValueError(
                    f"Invalid AC/AN for {record.chrom}:{record.pos} {population}: AC={ac}, AN={an}"
                )
            af = ac / an
            k_projected = int(af * n_pop + 0.5)
            if not (1 <= k_projected <= n_pop - 1):
                continue
            k_folded = min(k_projected, n_pop - k_projected)
            rows.append(
                {
                    "chrom": record.chrom,
                    "pos": record.pos,
                    "ref": record.ref,
                    "alt": record.alt,
                    "population": population,
                    "ac_raw": ac,
                    "an_raw": an,
                    "af_raw": af,
                    "n_pop": n_pop,
                    "k_projected": k_projected,
                    "k_folded": k_folded,
                    "block_id": record.block_id,
                    "variant_type": record.variant_type,
                }
            )

    columns = [
        "chrom",
        "pos",
        "ref",
        "alt",
        "population",
        "ac_raw",
        "an_raw",
        "af_raw",
        "n_pop",
        "k_projected",
        "k_folded",
        "block_id",
        "variant_type",
    ]
    return pd.DataFrame(rows, columns=columns), n_by_population, valid_raw_by_population
