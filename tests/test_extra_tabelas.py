"""
Testes do scripts/22_extra_tabelas.py (v1.0, 26 out 2026): num mini-projeto inventado (o do test_extra_congelar.py),
avalia os dois modelos e a referência em VAL_EX e TEST_BOX com o avaliador, corre as 6 comparações declaradas e
confere a tabela, as comparações, os estratos, as figuras e a linha para o decisoes.md. Nunca leem dados reais.
    python tests/test_extra_tabelas.py          ou          python -m pytest tests/ -q
"""
import importlib.util
import sys
import tempfile
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
from test_evaluate import Saltar, ev, saltar  # noqa: E402
from test_extra_congelar import _montar  # noqa: E402

PARES = (("SAM2T_L12", "SAM2_L12"), ("MEDSAM2_L12", "SAM2T_L12"), ("MEDSAM2_L12", "SAM2_L12"))


def _carregar():
    spec = importlib.util.spec_from_file_location("extra_tabelas", AQUI.parent / "scripts" / "22_extra_tabelas.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _avaliado(tmp):
    try:
        import yaml  # noqa: F401
    except ImportError:
        saltar("falta o PyYAML")
    raiz, conj, pred, cfg, args, ymls = _montar(tmp)
    prev = raiz / "outputs/preds"
    for n in ("SAM2T_L12", "MEDSAM2_L12"):
        for c in ("VAL_EX", "TEST_BOX"):
            assert ev.main(["avaliar", "--metodo", n, "--conjunto", c, "--pred", str(prev / f"pred_{n}_{c}.csv"),
                            "--config", cfg]) == 0
    for a, b in PARES:
        for c in ("VAL_EX", "TEST_BOX"):
            assert ev.main(["comparar", "--conjunto", c, "--a", a, "--b", b, "--config", cfg]) == 0
    base = ["--config", cfg, "--config-extra", str(raiz / "configs/extra.yaml"), "--saida", str(raiz / "out22")]
    return raiz, cfg, base


def test_tabela_comparacoes_e_linha():
    m = _carregar()
    with tempfile.TemporaryDirectory() as tmp:
        raiz, cfg, base = _avaliado(tmp)
        r = m.main(base + ["--sem-figuras"])
        tab = {t["metodo"]: t for t in r["tabela"]}
        assert list(tab) == ["SAM2_L12", "SAM2T_L12", "MEDSAM2_L12"]
        j = ev.ler_config(raiz / "outputs/eval/TEST_BOX/MEDSAM2_L12.json")            # JSON é YAML válido
        assert tab["MEDSAM2_L12"]["R_loc"] == j["por_op"]["OP_F1"]["micro"]["R_loc"]["valor"]
        v = ev.ler_config(raiz / "outputs/eval/VAL_EX/SAM2T_L12.json")
        assert tab["SAM2T_L12"]["F1_por_janela"] == v["por_op"]["OP_F1"]["macro"]["F1"]["valor"]
        assert tab["SAM2T_L12"]["tau_OP_F1"] == v["por_op"]["OP_F1"]["tau"]
        assert len(r["comparacoes"]) == 6
        assert {(x["conjunto"], x["metrica"]) for x in r["comparacoes"]} == {("VAL_EX", "F1"), ("TEST_BOX", "R_loc")}
        est = r["estratos"]
        assert set(est["grupo"][est["estrato"] == "distancia"]) == {"1-3", "4-6", "7-12"}
        assert len(est[(est["estrato"] == "distancia")]) == 3 * 3 * 3          # 3 métodos × (P, R, R_loc) × 3
        texto = r["resumo"].read_text(encoding="utf-8")
        assert "| SAM2 base+ (principal) |" in texto and "Tiny − base+" in texto and "MedSAM2 − Tiny" in texto
        assert "1 execução(ões) registada(s)" in texto
        assert r["linha"].startswith("- ") and "R_loc@0,3 em TEST-BOX" in r["linha"] and "," in r["linha"]
        for f in ("tabela.csv", "comparacoes.csv", "estratos.csv", "resumo.md"):
            assert (raiz / "out22" / f).is_file()


def test_figuras():
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        saltar("falta o matplotlib")
    m = _carregar()
    with tempfile.TemporaryDirectory() as tmp:
        raiz, cfg, base = _avaliado(tmp)
        m.main(base)
        for f in ("fig_extra_comparacoes", "fig_extra_distancia"):
            for ext in ("png", "pdf"):
                p = raiz / "out22" / f"{f}.{ext}"
                assert p.is_file() and p.stat().st_size > 5000, p


def test_recusa_sem_avaliacao_ou_comparacao():
    m = _carregar()
    with tempfile.TemporaryDirectory() as tmp:
        raiz, cfg, base = _avaliado(tmp)
        f = raiz / "outputs/eval/TEST_BOX/comparar_MEDSAM2_L12_vs_SAM2_L12_R_loc_OP_F1.json"
        f.rename(f.with_suffix(".bak"))
        try:
            m.main(base + ["--sem-figuras"])
            raise AssertionError("devia ter parado")
        except ev.ErroAvaliacao as e:
            assert "Falta a comparação" in str(e)
        f.with_suffix(".bak").rename(f)
        g = raiz / "outputs/eval/VAL_EX/SAM2T_L12.json"
        g.rename(g.with_suffix(".bak"))
        try:
            m.main(base + ["--sem-figuras"])
            raise AssertionError("devia ter parado")
        except ev.ErroAvaliacao as e:
            assert "não está avaliado" in str(e)


if __name__ == "__main__":
    testes = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    falhas = 0
    for nome, f in testes:
        try:
            f()
            print(f"  ok      {nome}")
        except Saltar as e:
            print(f"  saltado {nome} ({e})")
        except Exception as e:                                              # noqa: BLE001
            falhas += 1
            print(f"  FALHOU  {nome}: {type(e).__name__}: {e}")
    sys.exit(1 if falhas else 0)
