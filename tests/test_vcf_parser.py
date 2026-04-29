from pathlib import Path

import pytest

from src.io.vcf_parser import build_population_allele_table, parse_info_field, parse_retained_snvs


def write_vcf(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "synthetic.vcf"
    path.write_text(
        "##fileformat=VCFv4.2\n"
        "##INFO=<ID=variant_type,Number=1,Type=String,Description=\"Variant type\">\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        + body
    )
    return path


def test_parse_info_field_handles_key_values_and_flags() -> None:
    info = parse_info_field("AC_afr=1;AN_afr=10;outside_capture;variant_type=snv")
    assert info["AC_afr"] == "1"
    assert info["AN_afr"] == "10"
    assert info["outside_capture"] is True
    assert info["variant_type"] == "snv"


def test_vcf_parser_filters_non_snvs_and_projects_counts(tmp_path: Path) -> None:
    vcf = write_vcf(
        tmp_path,
        "1\t10\t.\tA\tG\t.\tPASS\tAC_afr=1;AN_afr=10;AC_eas=0;AN_eas=8;AC_nfe=7;AN_nfe=8;variant_type=snv\n"
        "1\t11\t.\tA\tAT\t.\tPASS\tAC_afr=1;AN_afr=10;AC_eas=1;AN_eas=8;AC_nfe=1;AN_nfe=8;variant_type=indel\n"
        "1\t12\t.\tC\tT\t.\tPASS\tAC_afr=9;AN_afr=10;AC_eas=8;AN_eas=8;AC_nfe=0;AN_nfe=8;variant_type=snv\n",
    )

    records = parse_retained_snvs(vcf)
    assert len(records) == 2
    assert all(record.variant_type == "snv" for record in records)

    table, n_by_population, _valid = build_population_allele_table(vcf, ["afr", "eas", "nfe"])
    assert n_by_population == {"afr": 10, "eas": 8, "nfe": 8}
    assert set(table["variant_type"]) == {"snv"}
    assert set(table["population"]) == {"afr", "nfe"}
    assert len(table) == 3


def test_missing_population_fields_raise_informative_error(tmp_path: Path) -> None:
    vcf = write_vcf(
        tmp_path,
        "1\t10\t.\tA\tG\t.\tPASS\tAC_afr=1;AN_afr=10;variant_type=snv\n",
    )
    with pytest.raises(ValueError, match="AC_eas"):
        build_population_allele_table(vcf, ["afr", "eas"])

