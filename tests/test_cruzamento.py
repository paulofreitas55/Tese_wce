"""
Testes do script 10 (cruzamento PF v2 × caixas da anotação original, em CAL).
Usam o mini-projeto inventado de test_evaluate.py; nunca leem dados reais.
    python tests/test_cruzamento.py          ou          python -m pytest tests/ -q
"""
import importlib.util
import sys
import tempfile
from pathlib import Path

import pandas as pd

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
from test_evaluate import Saltar, _projeto, perto, saltar  # noqa: E402

_spec = importlib.util.spec_from_file_location("cruzamento", AQUI.parent / "scripts" / "10_cruzamento_pf_caixas.py")
cz = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cz)


def test_cruzamento_no_mini_projeto():
    """Mini-projeto: 3 janelas de CAL; caixas em |offset| ≤ 2; PF v2 visível em |offset| ≤ 3, dúvida em +4;
    sem ficheiro de caixa em −8. Por janela: 4 frames com caixa e PF 1, 2 com PF 1 sem caixa (±3), 12 com PF 0
    sem caixa. Totais: a 12 · b 6 · c 0 · d 36 → caixas confirmadas 100 %, PF v2 com caixa 2/3,
    concordância positiva 0,8, κ 8/11."""
    try:
        import yaml  # noqa: F401
    except ImportError:
        saltar("PyYAML não instalado")
    with tempfile.TemporaryDirectory() as tmp:
        raiz, _, _ = _projeto(tmp)
        r = cz.main(["--config", str(raiz / "configs/avaliacao.yaml"), "--saida", str(raiz / "outputs")])
        assert r["totais"] == {"a": 12, "b": 6, "c": 0, "d": 36, "dc": 0}
        m = r["medidas"]
        assert m["caixas_confirmadas"] == 1 and perto(m["pf_com_caixa"], 2 / 3)
        assert perto(m["concordancia_positiva"], 0.8) and perto(m["kappa"], 8 / 11) and perto(m["concordancia"], 48 / 54)
        assert r["contagens"] == {"janelas": 3, "frames_sem_seed": 60, "duvidas": 3, "sem_ficheiro": 3, "invalidas": 0}
        d = r["distancia"]
        assert (d["1-3"]["a"], d["1-3"]["b"]) == (12, 6) and d["4-6"]["d"] == 15 and d["7-12"]["d"] == 21
        assert d[">12"]["a"] + d[">12"]["b"] + d[">12"]["c"] + d[">12"]["d"] == 0
        assert perto(r["dominante"]["sem_ela"]["kappa"], 8 / 11)
        dis = pd.read_csv(raiz / "outputs/10_cruzamento_pf_caixas_discordancias.csv", dtype={"caso": str})
        assert len(dis) == 6 and set(dis["tipo"]) == {"pf_sem_caixa"} and set(dis["offset"].abs()) == {3}
        jan = pd.read_csv(raiz / "outputs/10_cruzamento_pf_caixas_janelas.csv", dtype={"caso": str})
        assert len(jan) == 3 and jan["caixas"].tolist() == [4, 4, 4] and jan["pf_visiveis"].tolist() == [6, 6, 6]
        texto = (raiz / "outputs/10_cruzamento_pf_caixas_resumo.txt").read_text(encoding="utf-8")
        assert "= 100,0 %" in texto and "= 66,7 %" in texto


def test_caixa_que_a_pf_nao_viu():
    """Uma caixa num frame que a PF v2 marcou como não visível conta como caixa não confirmada (caixa_sem_pf)."""
    try:
        import yaml  # noqa: F401
    except ImportError:
        saltar("PyYAML não instalado")
    with tempfile.TemporaryDirectory() as tmp:
        raiz, _, _ = _projeto(tmp)
        f = raiz / "data/derived/paulo_frame_labels.csv"
        pf = pd.read_csv(f, dtype=str, keep_default_na=False)
        alvo = (pf["caso"] == "0101") & (pf["offset"] == "2")
        assert alvo.sum() == 1
        pf.loc[alvo, "visivel"] = "0"
        pf.to_csv(f, index=False)
        r = cz.main(["--config", str(raiz / "configs/avaliacao.yaml"), "--saida", str(raiz / "outputs")])
        assert r["totais"]["a"] == 11 and r["totais"]["c"] == 1
        assert perto(r["medidas"]["caixas_confirmadas"], 11 / 12)
        dis = r["discordancias"]
        caixa_sem_pf = dis[dis["tipo"] == "caixa_sem_pf"]
        assert len(caixa_sem_pf) == 1 and caixa_sem_pf.iloc[0]["caso"] == "0101" and caixa_sem_pf.iloc[0]["offset"] == 2
        assert r["janelas"].iloc[0]["caso"] == "0101"            # a janela com a caixa não confirmada aparece primeiro


if __name__ == "__main__":
    testes = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    falhas = 0
    for nome, f in testes:
        try:
            f()
            print(f"  ok      {nome}")
        except Saltar as e:
            print(f"  saltado {nome} ({e})")
        except Exception as e:                                          # noqa: BLE001
            falhas += 1
            print(f"  FALHOU  {nome}: {type(e).__name__}: {e}")
    sys.exit(1 if falhas else 0)
