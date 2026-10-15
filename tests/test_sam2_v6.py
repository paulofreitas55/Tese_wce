"""
Testes do V6 (methods/sam2_v6.py, paragem adaptativa) e do diagnóstico dos frames longe do seed
(scripts/11_diagnostico_longe.py): a regra de paragem em sequências pequenas, uma corrida completa num mini-projeto
com previsões inventadas no formato do SAM 2 (N = 1, 3 e 6, avaliador) e o diagnóstico (troço, blocos, frames
visíveis). Não precisam do PyTorch nem do OpenCV. Nunca leem dados reais.
    python tests/test_sam2_v6.py          ou          python -m pytest tests/ -q
"""
import importlib.util
import math
import sys
import tempfile
from pathlib import Path

import pandas as pd

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
from test_evaluate import Saltar, _projeto, ev  # noqa: E402

CAIXA = ["box_x1", "box_y1", "box_x2", "box_y2"]


def _carregar(rel, nome):
    spec = importlib.util.spec_from_file_location(nome, AQUI.parent / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_ponto_de_paragem():
    v6 = _carregar("methods/sam2_v6.py", "sam2_v6")
    seq = [0.9, 0.8, 0.3, 0.9, 0.2, 0.1, 0.05, 0.9]
    assert v6.ponto_de_paragem(seq, 0.5, 1) == 2                            # o 1.º frame abaixo
    assert v6.ponto_de_paragem(seq, 0.5, 3) == 4                            # o 1.º de 3 seguidos abaixo
    assert v6.ponto_de_paragem(seq, 0.5, 6) == 8                            # nunca há 6 seguidos: não pára
    assert v6.ponto_de_paragem([0.9, math.nan, 0.9], 0.5, 1) == 1           # frame sem linha conta como abaixo
    assert v6.ponto_de_paragem([0.2, 0.9], 0.5, 1) == 0 and v6.ponto_de_paragem([], 0.5, 1) == 0
    assert v6.ponto_de_paragem([0.5, 0.49], 0.5, 1) == 1                    # 0,5 ainda conta como visto


def _pred_l100(raiz, conj):
    """Previsões inventadas no formato do sam2_prop.py (L100, sem a linha do seed). Janela 0101, por distância ao
    seed — depois: 0,9 0,8 0,7 | 0,3 | 0,9 | 0,2 0,1 0,1 | 0,95 | 0,05; antes: 0,8 | 0,2 | 0,9 0,9 | 0,05 até ao fim.
    Nas outras: 0,9 até 2 frames do seed e 0,05 depois. Caixa e área só onde o score ≥ 0,5."""
    depois = {1: .9, 2: .8, 3: .7, 4: .3, 5: .9, 6: .2, 7: .1, 8: .1, 9: .95, 10: .05}
    antes = {1: .8, 2: .2, 3: .9, 4: .9}
    linhas = []
    for sp, caso, seed, _ in conj["CAL"]:
        for o in range(-10, 11):
            if o == 0:
                continue
            if caso == "0101":
                s = depois[o] if o > 0 else antes.get(-o, 0.05)
            else:
                s = 0.9 if abs(o) <= 2 else 0.05
            cx = (0.3, 0.3, 0.5, 0.5) if s >= 0.5 else ("",) * 4
            linhas.append({"split": sp, "caso": caso, "seed": seed, "frame": seed + o, "score": s,
                           **dict(zip(CAIXA, cx)), "area_px": 4096 if s >= 0.5 else 0})
    (raiz / "outputs/preds").mkdir(parents=True)
    f = raiz / "outputs/preds/pred_SAM2_L100_CAL.csv"
    pd.DataFrame(linhas).to_csv(f, index=False)
    (raiz / "configs/sam2_v6.yaml").write_text(
        f"versao: 1\nbase: SAM2_L100\nlimiar: 0.5\nn: [1, 3, 6]\nentrada: {raiz / 'outputs/preds'}\n"
        f"saida: {raiz / 'outputs/preds'}\n", encoding="utf-8")
    return f


def _yaml():
    try:
        import yaml  # noqa: F401
    except ImportError:
        raise Saltar("falta o PyYAML")


def test_v6_ponta_a_ponta():
    _yaml()
    v6 = _carregar("methods/sam2_v6.py", "sam2_v6")
    with tempfile.TemporaryDirectory() as tmp:
        raiz, conj, _ = _projeto(tmp)
        base = pd.read_csv(_pred_l100(raiz, conj), dtype={"caso": str})
        cfg = str(raiz / "configs/avaliacao.yaml")
        r = v6.main(["--conjunto", "CAL", "--config-v6", str(raiz / "configs/sam2_v6.yaml")])
        assert sorted(r["csv"]) == [1, 3, 6]
        # (frame, score esperado) na janela 0101 (seed 500); 0 = cortado depois da paragem
        esperado = {1: {503: 0.7, 504: 0, 505: 0, 509: 0, 499: 0.8, 498: 0, 497: 0, 496: 0},
                    3: {503: 0.7, 504: 0.3, 505: 0.9, 506: 0, 509: 0, 499: 0.8, 498: 0.2, 497: 0.9, 496: 0.9, 495: 0},
                    6: {505: 0.9, 506: 0.2, 509: 0.95, 510: 0.05, 497: 0.9, 496: 0.9, 495: 0}}
        for n, casos in esperado.items():
            p = pd.read_csv(r["csv"][n], dtype={"caso": str})
            assert list(p.columns) == list(base.columns) and len(p) == len(base) == 60
            j = p[p["caso"] == "0101"].set_index("frame")
            for f, s in casos.items():
                assert abs(j.loc[f, "score"] - s) < 1e-9, (n, f, j.loc[f, "score"], s)
                if s == 0 and base.set_index(["caso", "frame"]).loc[("0101", f), "score"] >= 0.5:
                    assert pd.isna(j.loc[f, "box_x1"]) and j.loc[f, "area_px"] == 0   # cortado: sem caixa
            k = p[p["caso"] == "0102"].set_index("frame")                   # pára no 3.º frame, com qualquer N
            assert (k.loc[[801, 802, 798, 799], "score"] == 0.9).all() and (k.loc[[803, 797], "score"] == 0).all()
            assert ev.main(["calibrar", "--metodo", f"SAM2_V6n{n}", "--pred", str(r["csv"][n]), "--config", cfg]) == 0
        texto = r["resumo"].read_text(encoding="utf-8")
        assert "N = 3: o seguimento parou em" in texto and "SAM2_V6n6" in texto
        (raiz / "outputs/preds/pred_SAM2_L100_CAL.csv").unlink()
        try:
            v6.main(["--conjunto", "CAL", "--config-v6", str(raiz / "configs/sam2_v6.yaml")])
            raise AssertionError("devia ter parado")
        except ev.ErroAvaliacao as e:
            assert "sam2_prop.py" in str(e)


def test_diagnostico_longe():
    _yaml()
    d11 = _carregar("scripts/11_diagnostico_longe.py", "diag11")
    assert d11.troco_e_blocos([0.9, 0.8, 0.2, 0.6, 0.7, 0.1, 0.9]) == (2, [(3, 4), (6, 6)])
    assert d11.troco_e_blocos([0.2, 0.9]) == (0, [(1, 1)]) and d11.troco_e_blocos([]) == (0, [])
    with tempfile.TemporaryDirectory() as tmp:
        raiz, conj, _ = _projeto(tmp)
        f = _pred_l100(raiz, conj)
        r = d11.main(["--config", str(raiz / "configs/avaliacao.yaml"), "--pred", str(f),
                      "--saida", str(raiz / "outputs")])
        b = r["blocos"].set_index(["caso", "direcao", "inicio"])
        # mini-projeto: visível (PF 1) a ≤ 3 frames do seed, dúvida a 4, não visível depois
        assert len(b) == 3
        assert tuple(b.loc[("0101", "depois", 5), ["comprimento", "pf1", "pf0"]]) == (1, 0, 1)
        assert tuple(b.loc[("0101", "depois", 9), ["pf0", "frames_forte", "pf0_forte"]]) == (1, 1, 1)
        # antes do seed: o −3 é visível e o −4 não (no mini-projeto a dúvida só está no +4)
        assert tuple(b.loc[("0101", "antes", 3), ["comprimento", "pf1", "pf0", "pf1_forte"]]) == (2, 1, 1, 1)
        vis = r["frames"][r["frames"]["pf"] == "1"]
        assert len(vis) == 18 and list(vis["onde"].value_counts().reindex(["troço", "bloco", "nada"])) == [12, 1, 5]
        assert "Troço do seed" in r["texto"] and "precisão" in r["texto"]
        assert (raiz / "outputs/11_diagnostico_longe_blocos.csv").is_file()


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
