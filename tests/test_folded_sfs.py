from pathlib import Path

from src.io.vcf_parser import build_population_allele_table
from src.sfs.folded import make_count_folded_sfs
from src.sfs.summary_stats import singleton_count, total_variant_count


def write_vcf(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "folded.vcf"
    path.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        + body
    )
    return path


def test_count_based_folded_sfs_core_edge_cases(tmp_path: Path) -> None:
    vcf = write_vcf(
        tmp_path,
        "1\t1\t.\tA\tC\t.\tPASS\tAC_afr=1;AN_afr=10;variant_type=snv\n"
        "1\t2\t.\tA\tC\t.\tPASS\tAC_afr=9;AN_afr=10;variant_type=snv\n"
        "1\t3\t.\tA\tC\t.\tPASS\tAC_afr=0;AN_afr=10;variant_type=snv\n"
        "1\t4\t.\tA\tC\t.\tPASS\tAC_afr=10;AN_afr=10;variant_type=snv\n"
        "1\t5\t.\tA\tC\t.\tPASS\tAC_afr=1;AN_afr=6;variant_type=snv\n"
        "1\t6\t.\tA\tAC\t.\tPASS\tAC_afr=1;AN_afr=10;variant_type=indel\n",
    )

    table, n_by_population, _valid = build_population_allele_table(vcf, ["afr"])
    assert n_by_population["afr"] == 10
    assert len(table) == 3
    assert table.loc[table["pos"] == 1, "k_projected"].item() == 1
    assert table.loc[table["pos"] == 1, "k_folded"].item() == 1
    assert table.loc[table["pos"] == 2, "k_projected"].item() == 9
    assert table.loc[table["pos"] == 2, "k_folded"].item() == 1
    assert set(table["pos"]) == {1, 2, 5}

    sfs = make_count_folded_sfs(table, "afr")
    assert len(sfs) == 5
    assert singleton_count(sfs) == 2
    assert total_variant_count(sfs) == 3
    assert sfs.loc[sfs["k_folded"] == 1, "count"].item() == 2
    assert sfs.loc[sfs["k_folded"] == 2, "count"].item() == 1


def test_total_retained_counts_match_expectations_for_multiple_populations(tmp_path: Path) -> None:
    vcf = write_vcf(
        tmp_path,
        "1\t1\t.\tA\tC\t.\tPASS\tAC_afr=1;AN_afr=10;AC_eas=1;AN_eas=6;variant_type=snv\n"
        "1\t2\t.\tA\tC\t.\tPASS\tAC_afr=0;AN_afr=10;AC_eas=5;AN_eas=6;variant_type=snv\n"
        "1\t3\t.\tA\tC\t.\tPASS\tAC_afr=4;AN_afr=8;AC_eas=6;AN_eas=6;variant_type=snv\n",
    )
    table, n_by_population, _valid = build_population_allele_table(vcf, ["afr", "eas"])
    assert n_by_population == {"afr": 10, "eas": 6}
    assert (table["population"] == "afr").sum() == 2
    assert (table["population"] == "eas").sum() == 2
    assert len(table) == 4
