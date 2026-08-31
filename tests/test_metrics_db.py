from src.models.metrics_db import get_connection, read_comparison_table, record_comparison_row


def test_record_and_read_comparison_row_roundtrips(tmp_path):
    db_path = tmp_path / "metrics_test.duckdb"
    con = get_connection(db_path)
    try:
        record_comparison_row(con, "demand_mwh", "naive", mean_wape=4.79, mean_mae=12.3, n_splits=5)
        record_comparison_row(
            con, "demand_mwh", "mlp_pytorch", mean_wape=2.9, mean_mae=8.1, n_splits=5,
            model_variant="mlp_pytorch_gelu", latency_ms_per_1k_rows=1.2,
        )
    finally:
        con.close()

    df = read_comparison_table(db_path)
    assert len(df) == 2
    assert set(df["model_family"]) == {"naive", "mlp_pytorch"}
    gelu_row = df[df["model_variant"] == "mlp_pytorch_gelu"].iloc[0]
    assert gelu_row["mean_wape"] == 2.9
    assert gelu_row["latency_ms_per_1k_rows"] == 1.2


def test_get_connection_creates_table_idempotently(tmp_path):
    db_path = tmp_path / "metrics_test2.duckdb"
    con1 = get_connection(db_path)
    con1.close()
    con2 = get_connection(db_path)  # no debe fallar si la tabla ya existe
    con2.close()
