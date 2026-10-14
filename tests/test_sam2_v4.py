"""
Testes do A5, V4 (methods/sam2_v4.py): as duas regras em sequências pequenas; a garantia de que o novo score dá, para
qualquer τ, o mesmo que aplicar as regras à presença (score ≥ τ); e uma corrida completa num mini-projeto com previsões
inventadas no formato do SAM 2 (buraco preenchido com a caixa dos vizinhos, ilha removida conforme o k, frame sem
imagem, avaliador). Não precisam do PyTorch nem do OpenCV. Nunca leem dados reais.
    python tests/test_sam2_v4.py          ou          python -m pytest tests/ -q
"""
import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
from test_evaluate import Saltar, _projeto, ev  # noqa: E402

CAIXA = ["box_x1", "box_y1", "box_x2", "box_y2"]


def _carregar():
    spec = importlib.util.spec_from_file_location("sam2_v4", AQUI.parent / "methods" / "sam2_v4.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _seq(posicoes, n=15):
    b = np.zeros(n, dtype=bool)
    b[list(posicoes)] = True
    return b


def test_regras():
    v4 = _carregar()
    # seed na posição 5; falta a imagem 10
    b = _seq([1, 4, 6, 8, 9, 11, 14])
    existe = ~_seq([10])
    # 7 é preenchido (entre 6 e 8); 10 não (sem imagem); troço do seed = 4–9
    # ilhas: 1 (a 3 do troço), 11 (a 2; o vizinho 10 não tem imagem), 14 (a 5)
    assert np.array_equal(v4.regras(b, existe, 5, 3)[0], _seq([1, 4, 5, 6, 7, 8, 9, 11]))
    assert np.array_equal(v4.regras(b, existe, 5, 6)[0], _seq([1, 4, 5, 6, 7, 8, 9, 11, 14]))
    assert np.array_equal(v4.regras(b, existe, 5, 2)[0], _seq([4, 5, 6, 7, 8, 9, 11]))
    # 4 é preenchido porque o seed conta como presente; 8 é ilha a 3 do troço (3–5); 13–14 não são ilhas
    b2 = _seq([3, 8, 13, 14])
    todos = np.ones(15, dtype=bool)
    assert np.array_equal(v4.regras(b2, todos, 5, 3)[0], _seq([3, 4, 5, 8, 13, 14]))
    assert np.array_equal(v4.regras(b2, todos, 5, 2)[0], _seq([3, 4, 5, 13, 14]))
    # várias linhas de uma vez (uma por τ) dão o mesmo que uma de cada vez
    duas = v4.regras(np.stack([b, b2]), todos, 5, 3)
    assert np.array_equal(duas[1], v4.regras(b2, todos, 5, 3)[0])
    assert np.array_equal(duas[0], v4.regras(b, todos, 5, 3)[0])


def test_novo_score_comuta_com_o_limiar():
    v4 = _carregar()
    rng = np.random.default_rng(0)
    grelha = np.arange(1, 51) / 50                                          # a grelha do avaliador, sem o 0
    for _ in range(300):
        n = int(rng.integers(3, 40))
        i_seed = int(rng.integers(0, n))
        s = np.round(rng.random(n) ** 2, 2)                                 # empates e muitos valores baixos
        s[rng.random(n) < 0.3] = 0.0                                        # como os frames para lá de L_max
        existe = rng.random(n) > 0.1
        for k in (0, 1, 3, 6):
            novo = v4.novo_score(s, existe, i_seed, k)
            assert ((novo >= 0) & (novo <= 1)).all()
            assert set(novo[existe]) <= set(s[existe]) | {0.0, 1.0}
            for tau in np.unique(np.concatenate([grelha, s[s > 0]])):
                esperado = v4.regras(s >= tau, existe, i_seed, k)[0]
                assert np.array_equal((novo >= tau)[existe], esperado[existe]), (n, i_seed, k, tau)


def _cx(dx=0.0):
    return (round(0.30 + dx, 6), 0.30, round(0.50 + dx, 6), 0.50)


def _pred_sam2(raiz, conj):
    """Previsões inventadas no formato do sam2_prop.py para as 3 janelas de CAL (sem a linha do seed; falta a imagem
    0102/804). 0101: buraco em +3 (entre +2 e +4) e ilha em +9, a 5 frames do troço do seed (que acaba em +4).
    0102: +3 e +5 presentes à volta da imagem que falta (+4): nada muda."""
    linhas = []
    for sp, caso, seed, _ in conj["CAL"]:
        for o in range(-10, 11):
            if o == 0 or (caso, o) == ("0102", 4):
                continue
            score, caixa, area = 0.05, None, 0
            if abs(o) <= 2:
                score, caixa, area = 0.9, _cx(), 4096
            if caso == "0101" and o == 3:
                score, caixa, area = 0.2, None, 0
            if caso == "0101" and o == 4:
                score, caixa, area = 0.8, _cx(0.04), 3000
            if caso == "0101" and o == 9:
                score, caixa, area = 0.7, (0.6, 0.6, 0.7, 0.7), 500
            if caso == "0102" and o in (3, 5):
                score, caixa, area = 0.9, _cx(), 4096
            linhas.append({"split": sp, "caso": caso, "seed": seed, "frame": seed + o, "score": score,
                           **dict(zip(CAIXA, caixa or ("",) * 4)), "area_px": area})
    (raiz / "outputs/preds").mkdir(parents=True)
    f = raiz / "outputs/preds/pred_SAM2_L12_CAL.csv"
    pd.DataFrame(linhas).to_csv(f, index=False)
    (raiz / "configs/sam2_v4.yaml").write_text(
        f"versao: 1\nbases: [SAM2_L12]\nk: [3, 6]\nentrada: {raiz / 'outputs/preds'}\nsaida: {raiz / 'outputs/preds'}\n",
        encoding="utf-8")
    return f


def test_v4_ponta_a_ponta():
    try:
        import yaml  # noqa: F401
    except ImportError:
        raise Saltar("falta o PyYAML")
    v4 = _carregar()
    with tempfile.TemporaryDirectory() as tmp:
        raiz, conj, _ = _projeto(tmp)
        f = _pred_sam2(raiz, conj)
        cfg = str(raiz / "configs/avaliacao.yaml")
        r = v4.main(["--conjunto", "CAL", "--config-v4", str(raiz / "configs/sam2_v4.yaml")])
        assert sorted(r["csv"]) == [("SAM2_L12", 3), ("SAM2_L12", 6)]
        assert r["contagens"][("SAM2_L12", 3)]["subiu"] == 1 and r["contagens"][("SAM2_L12", 3)]["desceu"] == 1
        assert r["contagens"][("SAM2_L12", 6)]["desceu"] == 0
        base = pd.read_csv(f, dtype={"caso": str})
        for k in (3, 6):
            p = pd.read_csv(r["csv"][("SAM2_L12", k)], dtype={"caso": str})
            assert list(p.columns) == list(base.columns) and len(p) == len(base) == 59
            assert not ((p["caso"] == "0102") & (p["frame"] == 804)).any()          # não cria a linha que falta
            j = p.merge(base, on=["split", "caso", "seed", "frame"], suffixes=("", "_base"))
            mudou = j[j["score"] != j["score_base"]]
            h = p[p["caso"] == "0101"].set_index("frame")
            # buraco em +3: score = o menor dos vizinhos; caixa = média das caixas dos vizinhos; área = a menor
            assert h.loc[503, "score"] == 0.8 and h.loc[503, "area_px"] == 3000
            assert np.allclose(h.loc[503, CAIXA].astype(float), _cx(0.02))
            # ilha em +9 (a 5 frames do troço): sai com k = 3 (fica com o score dos frames à volta), fica com k = 6
            assert h.loc[509, "score"] == (0.05 if k == 3 else 0.7) and h.loc[509, "area_px"] == 500
            assert sorted(mudou["frame"]) == ([503, 509] if k == 3 else [503])
            assert ev.main(["calibrar", "--metodo", f"SAM2_L12_V4k{k}", "--pred", str(r["csv"][("SAM2_L12", k)]),
                            "--config", cfg]) == 0
        assert "SAM2_L12_V4k3" in r["resumo"].read_text(encoding="utf-8")
        # sem o CSV da base, pára com uma mensagem clara
        f.unlink()
        try:
            v4.main(["--conjunto", "CAL", "--config-v4", str(raiz / "configs/sam2_v4.yaml")])
            raise AssertionError("devia ter parado")
        except ev.ErroAvaliacao as e:
            assert "sam2_prop.py" in str(e)


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
