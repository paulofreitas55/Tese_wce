"""
Testes do avaliador único (passo 0.8). Correm de duas maneiras, a partir da pasta do projeto:
    python tests/test_evaluate.py          (não precisa de pytest)
    python -m pytest tests/test_evaluate.py -q
Usam só janelas inventadas, com resultados contados à mão; nunca leem dados reais.
Versão 1.1 (12 out 2026): +2 testes (o avaliar recusa CAL; execuções de teste exigem o código guardado no git).
"""
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "eval"))
import evaluate as ev  # noqa: E402

NAN = float("nan")
OPS = {"OP_F1": {"tau": 0.5, "a_min": 0}, "OP_R90": {"tau": 0.2, "a_min": 0}}
PRINCIPAIS = ["angiectasia", "erosao", "ulcera", "polipo", "eritema", "xantelasma"]


class Saltar(Exception):
    pass


def saltar(motivo):
    if "pytest" in sys.modules:
        import pytest
        pytest.skip(motivo)
    raise Saltar(motivo)


def perto(a, b, tol=1e-6):
    return a is not None and b is not None and abs(a - b) <= tol


def erro(fn, *a, contem=None, **k):
    try:
        fn(*a, **k)
    except ev.ErroAvaliacao as e:
        assert contem is None or contem in str(e), f"a mensagem não diz '{contem}': {e}"
        return str(e)
    raise AssertionError("devia ter dado erro")


def cfg_base(reamostragens=200):
    return {"limiar": {"grelha": {"inicio": 0.0, "fim": 1.0, "passo": 0.02}, "a_min": [0, 25, 100], "recall_op_r90": 0.9},
            "iou": {"loc": 0.3, "limiares": [0.3, 0.5]},
            "bootstrap": {"reamostragens": reamostragens, "semente": 0, "nivel": 0.95},
            "estratos": {"classes_principais": PRINCIPAIS, "distancia": [[1, 3], [4, 6], [7, 12], [13, 100]],
                         "tercis_area_seed": [0.0096, 0.0282]},
            "fp_aparentes": {"amostra": 5, "semente": 0}}


def montar(janelas):
    """Janelas inventadas. Cada uma: caso, seed, n (frames para cada lado), pos / duv (offsets),
    caixas {offset: (x1, y1, x2, y2)}, sem_ficheiro (offsets), categoria."""
    linhas, info = [], []
    for w, j in enumerate(janelas):
        caso, seed, n = j.get("caso", f"{9000 + w}"), j.get("seed", 1000), j.get("n", 10)
        split = j.get("split", "Dataset_Teste")
        cx = j.get("caixas", {})
        for f in range(seed - n, seed + n + 1):
            o = f - seed
            pf = "1" if o == 0 or o in j.get("pos", ()) else ("D" if o in j.get("duv", ()) else "0")
            if o in cx:
                est, b = "caixa", cx[o]
            elif o in j.get("sem_ficheiro", ()):
                est, b = "sem_ficheiro", (NAN,) * 4
            else:
                est, b = "vazio", (NAN,) * 4
            linhas.append({"w": w, "split": split, "caso": caso, "seed": seed, "frame": f, "offset": o, "pf": pf,
                           "caixa": est, "rx1": b[0], "ry1": b[1], "rx2": b[2], "ry2": b[3]})
        cat = j.get("categoria", "angiectasia")
        area = j.get("area_seed", 0.02)
        info.append({"w": w, "split": split, "caso": caso, "seed": seed, "categoria": cat,
                     "classe": cat if cat in PRINCIPAIS else "outras", "area_seed": area,
                     "tercil": "pequena" if area < 0.0096 else "media" if area < 0.0282 else "grande"})
    return pd.DataFrame(linhas), pd.DataFrame(info)


def prev(linhas):
    """linhas: (caso, seed, frame, score[, (x1, y1, x2, y2)])"""
    reg = []
    for l in linhas:
        b = l[4] if len(l) > 4 and l[4] is not None else (NAN,) * 4
        reg.append({"caso": l[0], "seed": l[1], "frame": l[2], "score": l[3],
                    "box_x1": b[0], "box_y1": b[1], "box_x2": b[2], "box_y2": b[3]})
    return pd.DataFrame(reg, columns=["caso", "seed", "frame", "score", "box_x1", "box_y1", "box_x2", "box_y2"])


def correr(janelas, linhas, usa_pf=True, usa_cx=False, ops=OPS, reamostragens=200):
    ref, info = montar(janelas)
    if not usa_cx:
        ref["caixa"] = "nao_usada"
    if not usa_pf:
        ref["pf"] = ""
    d, avisos = ev.juntar(ref, info, ev.ler_previsoes(prev(linhas)))
    return ev.avaliar(d, info, cfg_base(reamostragens), ops, usa_pf, usa_cx, avisos)


def micro(res, op="OP_F1"):
    return res["por_op"][op]["micro"]


# ------------------------------------------------------------------ casos base (o guião pede os três primeiros)
def test_previsoes_perfeitas_dao_F1_1():
    pos = (-4, -3, -2, 2, 3)                                  # 5 positivos conhecidos
    linhas = [("9999", 1000, 1000 + o, 1.0 if o in pos else 0.0) for o in range(-10, 11)]
    r = correr([{"caso": "9999", "pos": pos}], linhas)
    m = micro(r)
    assert m["n_pos"] == 5 and m["P"]["valor"] == 1 and m["R"]["valor"] == 1 and m["F1"]["valor"] == 1
    assert r["por_op"]["OP_F1"]["macro"]["F1"]["valor"] == 1
    assert r["AP"]["valor"] == 1


def test_previsoes_vazias_dao_recall_0():
    r = correr([{"caso": "9999", "pos": (-2, -1, 1, 2, 3)}], [])
    m = micro(r)
    assert m["R"]["valor"] == 0 and m["F1"]["valor"] == 0
    assert m["P"]["valor"] is None                            # sem previsões, a precisão não está definida
    assert any("sem previsão" in a for a in r["avisos"])
    assert r["contagens"]["frames_sem_previsao"] == 20


def test_o_seed_e_sempre_excluido():
    base = [("9999", 1000, 999, 1.0), ("9999", 1000, 1001, 0.9)]
    sem = correr([{"caso": "9999", "pos": (-1,)}], base)
    com1 = correr([{"caso": "9999", "pos": (-1,)}], base + [("9999", 1000, 1000, 1.0)])
    com0 = correr([{"caso": "9999", "pos": (-1,)}], base + [("9999", 1000, 1000, 0.0)])
    for r in (com1, com0):
        assert micro(r)["P"] == micro(sem)["P"] and micro(r)["R"] == micro(sem)["R"]
        assert r["por_janela"][0]["OP_F1"]["ganho"] == 2        # o seed não conta como ganho
    assert sem["contagens"]["seeds_excluidos"] == 1


def test_exemplo_janela_9999():
    """Caso inventado: janela 9999/1000 com positivos em 998 e 999 e previsões em 998, 999 e 1001
    → P 0,67 · R 1,0 · F1 0,80."""
    linhas = [("9999", 1000, f, 1.0) for f in (998, 999, 1001)]
    m = micro(correr([{"caso": "9999", "pos": (-2, -1)}], linhas))
    assert perto(m["P"]["valor"], 2 / 3) and m["R"]["valor"] == 1 and perto(m["F1"]["valor"], 0.8)


# ------------------------------------------------------------------ regras do protocolo
def test_duvidosos_ficam_fora():
    j = [{"caso": "9999", "pos": (-1,), "duv": (2, 3)}]
    a = correr(j, [("9999", 1000, 999, 1.0), ("9999", 1000, 1002, 1.0)])   # D previsto presente: não é FP
    b = correr(j, [("9999", 1000, 999, 1.0)])                              # D ausente: não é FN
    for r in (a, b):
        m = micro(r)
        assert m["P"]["valor"] == 1 and m["R"]["valor"] == 1 and m["n_duvidosos"] == 2
    assert a["contagens"]["duvidosos"] == 2


def test_iou():
    assert perto(ev.iou([0, 0, 1, 1], [0, 0, 1, 1])[0], 1.0)
    assert perto(ev.iou([0, 0, 0.2, 0.2], [0.5, 0.5, 0.9, 0.9])[0], 0.0)
    assert perto(ev.iou([0, 0, 1, 1], [0, 0, 0.5, 1])[0], 0.5)              # metade da outra
    assert perto(ev.iou([0.1, 0.1, 0.1, 0.5], [0.1, 0.1, 0.1, 0.5])[0], 0.0)  # sem área


def test_micro_e_macro():
    """A: 3 positivos, 3 acertos (F1 1). B: 10 positivos, 5 acertos e 5 falsos (F1 0,5).
    Micro: TP 8, FP 5, FN 5 → F1 16/26. Macro: (1 + 0,5)/2 = 0,75."""
    ja = {"caso": "0001", "pos": (1, 2, 3)}
    jb = {"caso": "0002", "n": 30, "pos": tuple(range(1, 11))}
    linhas = [("0001", 1000, 1000 + o, 1.0) for o in (1, 2, 3)]
    linhas += [("0002", 1000, 1000 + o, 1.0) for o in (1, 2, 3, 4, 5, -1, -2, -3, -4, -5)]
    r = correr([ja, jb], linhas)
    assert perto(micro(r)["F1"]["valor"], 16 / 26)
    assert perto(r["por_op"]["OP_F1"]["macro"]["F1"]["valor"], 0.75)
    assert [j["OP_F1"]["F1"] for j in r["por_janela"]] == [1.0, 0.5]


def test_F1_da_janela_sem_positivos():
    """Sem positivos além do seed: nada previsto → F1 1; um frame previsto → F1 0."""
    vazia = correr([{"caso": "0001"}], [])
    fp = correr([{"caso": "0001"}], [("0001", 1000, 1003, 1.0)])
    assert vazia["por_janela"][0]["OP_F1"]["F1"] == 1.0 and fp["por_janela"][0]["OP_F1"]["F1"] == 0.0
    assert vazia["contagens"]["janelas_sem_positivos_alem_do_seed"] == 1
    assert fp["por_op"]["OP_F1"]["macro"]["F1_janelas_com_positivos"]["n_janelas"] == 0


def test_test_box_localizacao_sem_precisao():
    """Caixas do médico em −2, −1, +1, +2; sem ficheiro de caixa em −3. Previsões: −2 caixa igual (IoU 1),
    −1 metade (IoU 0,5), +1 deslocada (IoU ≈ 0,21), +2 nada; +5 (vazio) previsto → FP aparente;
    −3 (sem ficheiro) previsto → ignorado."""
    c = (0.2, 0.2, 0.4, 0.4)
    janela = {"caso": "0007", "caixas": {-2: c, -1: c, 1: c, 2: c}, "sem_ficheiro": (-3,)}
    linhas = [("0007", 1000, 998, 0.9, c), ("0007", 1000, 999, 0.9, (0.2, 0.2, 0.3, 0.4)),
              ("0007", 1000, 1001, 0.9, (0.33, 0.2, 0.53, 0.4)), ("0007", 1000, 1005, 0.9, c),
              ("0007", 1000, 997, 0.9, c)]
    r = correr([janela], linhas, usa_pf=False, usa_cx=True)
    m = micro(r)
    iou3 = ev.iou([0.33, 0.2, 0.53, 0.4], c)[0]
    assert m["n_caixas"] == 4 and perto(m["R_box"]["valor"], 0.75) and perto(m["R_loc"]["valor"], 0.5)
    assert perto(m["mIoU"]["valor"], (1 + 0.5 + iou3) / 3, 1e-5)
    assert perto(m["pct_IoU>=0.5"]["valor"], 2 / 3) and perto(m["pct_IoU>=0.3"]["valor"], 2 / 3)
    assert m["n_fp_aparentes"] == 1 and perto(m["taxa_fp_aparentes"]["valor"], 0.25)
    assert "P" not in m and "F1" not in m and "AP" not in r          # sem precisão em TEST_BOX
    assert r["contagens"]["frames_sem_ficheiro_de_caixa"] == 1


def test_presenca_sem_caixa_prevista_nao_conta_para_R_loc():
    c = (0.2, 0.2, 0.4, 0.4)
    r = correr([{"caso": "0008", "caixas": {1: c, 2: c}}], [("0008", 1000, 1001, 0.9), ("0008", 1000, 1002, 0.9, c)],
               usa_pf=False, usa_cx=True)
    m = micro(r)
    assert m["R_box"]["valor"] == 1 and m["R_loc"]["valor"] == 0.5 and m["n_com_as_duas_caixas"] == 1


def test_estratos_distancia_e_direcao():
    pos = (-2, -5, 8, 20)
    linhas = [("0003", 1000, 998, 1.0), ("0003", 1000, 1020, 1.0)]
    r = correr([{"caso": "0003", "n": 30, "pos": pos}], linhas)
    e = r["por_op"]["OP_F1"]["estratos"]
    rec = {k: v["R"]["valor"] for k, v in e["distancia"].items()}
    assert rec == {"1-3": 1.0, "4-6": 0.0, "7-12": 0.0, ">12": 1.0}
    assert e["direcao"]["antes"]["R"]["valor"] == 0.5 and e["direcao"]["depois"]["R"]["valor"] == 0.5
    assert sum(v["n_frames"] for v in e["distancia"].values()) == 60        # 61 frames menos o seed


def test_estratos_classe_e_tamanho():
    ja = {"caso": "0001", "pos": (1,), "categoria": "angiectasia", "area_seed": 0.005}
    jb = {"caso": "0002", "pos": (1,), "categoria": "sangue", "area_seed": 0.05}
    r = correr([ja, jb], [("0001", 1000, 1001, 1.0)])
    e = r["por_op"]["OP_F1"]["estratos"]
    assert e["classe"]["angiectasia"]["R"]["valor"] == 1 and e["classe"]["outras"]["R"]["valor"] == 0
    assert e["classe"]["ulcera"]["n_janelas"] == 0 and e["classe"]["ulcera"]["R"]["valor"] is None
    assert e["tamanho"]["pequena"]["R"]["valor"] == 1 and e["tamanho"]["grande"]["R"]["valor"] == 0


def test_bootstrap_reprodutivel_e_coerente():
    rng = np.random.default_rng(1)
    js, linhas = [], []
    for i in range(12):
        pos = tuple(int(o) for o in rng.choice([-3, -2, -1, 1, 2, 3, 4], size=3, replace=False))
        js.append({"caso": f"{100 + i:04d}", "pos": pos})
        linhas += [(f"{100 + i:04d}", 1000, 1000 + o, float(rng.random())) for o in range(-10, 11) if o != 0]
    a, b = correr(js, linhas), correr(js, linhas)
    assert json.dumps(ev._limpo(a)) == json.dumps(ev._limpo(b))
    f1 = micro(a)["F1"]
    assert f1["ic95"][0] <= f1["valor"] <= f1["ic95"][1] and f1["ic95"][0] < f1["ic95"][1]


def test_ap_igual_ao_scikit_learn():
    try:
        from sklearn.metrics import average_precision_score
    except ImportError:
        saltar("scikit-learn não instalado")
    rng = np.random.default_rng(3)
    js, linhas, ys, ss = [], [], [], []
    for i in range(5):
        pos = tuple(int(o) for o in rng.choice([o for o in range(-10, 11) if o], size=4, replace=False))
        js.append({"caso": f"{200 + i:04d}", "pos": pos})
        for o in range(-10, 11):
            if o:
                s = float(np.round(rng.random(), 2))                    # com empates
                linhas.append((f"{200 + i:04d}", 1000, 1000 + o, s))
                ys.append(int(o in pos))
                ss.append(s)
    r = correr(js, linhas)
    assert perto(r["AP"]["valor"], average_precision_score(ys, ss), 1e-6)


def test_calibracao_op_f1_e_op_r90():
    """10 positivos: 8 com score 0,9 e 2 com 0,3; 3 negativos com 0,5, o resto com 0,1.
    F1 máximo (16/18) para τ em ]0,5; 0,9] → OP-F1 = 0,90 (maior τ). Recall ≥ 0,9 só com τ ≤ 0,3 → OP-R90 = 0,30."""
    pos = tuple(range(-10, 0))
    sc = {o: 0.9 for o in pos[:8]} | {o: 0.3 for o in pos[8:]} | {1: 0.5, 2: 0.5, 3: 0.5}
    linhas = [("0005", 1000, 1000 + o, sc.get(o, 0.1)) for o in range(-10, 11) if o]
    ref, info = montar([{"caso": "0005", "pos": pos, "split": "Dataset_Treino"}])
    d, _ = ev.juntar(ref, info, ev.ler_previsoes(prev(linhas)))
    ops, T = ev.calibrar_ops(d, cfg_base())
    assert ops["OP_F1"]["tau"] == 0.9 and ops["OP_R90"]["tau"] == 0.3
    assert perto(T.loc[T["tau"] == 0.9, "F1"].iloc[0], 16 / 18)


def test_calibracao_quando_o_recall_nao_chega_a_90():
    pos = tuple(range(-10, 0))
    linhas = [("0006", 1000, 1000 + o, 0.8) for o in pos[:8]]           # 2 positivos sem previsão: R máx. 0,8
    ref, info = montar([{"caso": "0006", "pos": pos, "split": "Dataset_Treino"}])
    d, _ = ev.juntar(ref, info, ev.ler_previsoes(prev(linhas)))
    ops, _ = ev.calibrar_ops(d, cfg_base())
    assert ops["OP_R90"]["tau"] == 0.8 and "não é atingível" in ops["OP_R90"]["regra"]


def test_a_min_com_area():
    pos = (1, 2)
    p = prev([("0009", 1000, 1001, 0.9), ("0009", 1000, 1002, 0.9), ("0009", 1000, 1003, 0.9)])
    p["area_px"] = [200, 300, 10]                                       # o falso positivo é minúsculo
    ref, info = montar([{"caso": "0009", "pos": pos, "split": "Dataset_Treino"}])
    d, _ = ev.juntar(ref, info, ev.ler_previsoes(p))
    ops, _ = ev.calibrar_ops(d, cfg_base())
    assert ops["OP_F1"]["a_min"] == 25 and ops["OP_F1"]["tau"] == 0.9
    r = ev.avaliar(d, info, cfg_base(), ops, True, False)
    assert micro(r)["P"]["valor"] == 1


def test_validacao_das_previsoes():
    ref, info = montar([{"caso": "0004", "pos": (1,)}])
    junta = lambda p: ev.juntar(ref, info, ev.ler_previsoes(p))         # noqa: E731
    erro(junta, prev([("0004", 1000, 1001, 0.5), ("0004", 1000, 1001, 0.7)]), contem="repetidas")
    erro(junta, prev([("0005", 1000, 1001, 0.5)]), contem="não pertencem ao conjunto")
    erro(junta, prev([("0004", 1000, 1050, 0.5)]), contem="não existem nas janelas")
    erro(junta, prev([("0004", 1000, 1001, 1.5)]), contem="fora de [0, 1]")
    erro(junta, prev([("0004", 1000, 1001, NAN)]), contem="sem score")
    erro(junta, prev([("0004", 1000, 1001, 0.5, (10, 10, 50, 50))]), contem="normalizadas")
    erro(junta, prev([("0004", 1000, 1001, 0.5, (0.5, 0.1, 0.2, 0.3))]), contem="x2 < x1")
    p = prev([("0004", 1000, 1001, 0.5, (0.1, 0.1, 0.2, 0.3))])
    p.loc[0, "box_y2"] = NAN
    erro(junta, p, contem="incompletas")
    erro(junta, prev([("0004", 1000, 1001, 0.5)]).drop(columns="score"), contem="faltam as colunas")
    p = prev([("4", 1000, 1001, 0.5)])                                  # caso sem zeros à esquerda: aceite
    assert len(junta(p)[0]) == 21


def test_diferenca_pareada():
    rng = np.random.default_rng(5)
    js, la, lb = [], [], []
    for i in range(15):
        c = f"{300 + i:04d}"
        js.append({"caso": c, "pos": (1, 2, 3, 4)})
        la += [(c, 1000, 1000 + o, 0.9) for o in (1, 2, 3, 4)]               # A acerta tudo
        lb += [(c, 1000, 1000 + o, 0.9) for o in (1, 2) + tuple(int(x) for x in rng.choice([-5, -6, -7], 1))]
    a, b = correr(js, la), correr(js, lb)
    for r, m in ((a, "A"), (b, "B")):
        r.update({"metodo": m, "conjunto": "VAL_EX"})
    cfg = cfg_base()
    igual = ev.diferenca_pareada(ev._limpo(a), ev._limpo(a), "F1", "OP_F1", cfg)
    assert igual["diferenca_media"] == 0 and igual["ic95"] == [0, 0] and not igual["diferentes"]
    dif = ev.diferenca_pareada(ev._limpo(a), ev._limpo(b), "F1", "OP_F1", cfg)
    assert dif["diferentes"] and dif["ic95"][0] > 0 and dif["janelas_a_melhor"] == 15
    b2 = ev._limpo(b)
    b2["por_janela"] = b2["por_janela"][:-1]
    erro(ev.diferenca_pareada, ev._limpo(a), b2, "F1", "OP_F1", cfg, contem="mesmas janelas")


def test_config_do_projeto_bate_com_o_protocolo():
    p = RAIZ / "configs" / "avaliacao.yaml"
    if not p.exists():
        saltar("configs/avaliacao.yaml não existe")
    try:
        cfg = ev.ler_config(p)
    except ev.ErroAvaliacao as e:
        saltar(str(e))
    g = ev.grelha_tau(cfg)
    assert len(g) == 51 and g[0] == 0 and g[-1] == 1 and perto(g[1], 0.02)
    assert cfg["bootstrap"] == {"reamostragens": 1000, "semente": 0, "nivel": 0.95}
    assert cfg["iou"]["loc"] == 0.3 and cfg["limiar"]["recall_op_r90"] == 0.9
    assert cfg["estratos"]["tercis_area_seed"] == [0.0096, 0.0282]
    assert cfg["estratos"]["distancia"] == [[1, 3], [4, 6], [7, 12], [13, 100]]
    assert cfg["conjuntos"]["TEST_BOX"]["referencia"] == ["caixas"] and cfg["conjuntos"]["VAL_EX"]["referencia"] == ["pf"]


# ------------------------------------------------------------------ ponta a ponta, com ficheiros
def _projeto(raiz):
    """Mini-projeto com o formato real dos ficheiros: 3 janelas de treino (CAL), 2 de teste (VAL_EX e TEST_BOX)."""
    import yaml
    raiz = Path(raiz)
    (raiz / "data/derived").mkdir(parents=True)
    (raiz / "sets").mkdir()
    (raiz / "configs").mkdir()
    (raiz / "docs").mkdir()
    janelas = [("Dataset_Treino", "0101", 500, "angiectasia"), ("Dataset_Treino", "0102", 800, "ulcera"),
               ("Dataset_Treino", "0103", 300, "sangue"), ("Dataset_Teste", "0201", 600, "polipo"),
               ("Dataset_Teste", "0202", 900, "erosao")]
    cx = (0.30, 0.30, 0.50, 0.50)
    an, pf, jt = [], [], []
    for sp, caso, seed, cat in janelas:
        for f in range(seed - 10, seed + 11):
            o = f - seed
            tem = abs(o) <= 2                                           # caixas do médico em −2..+2
            src = "medico_seed" if o == 0 else ("medico_vizinho" if tem else ("sem_ficheiro" if o == -8 else "vazio"))
            b = cx if tem else ("", "", "", "")
            # boxc_* diferentes de box_*: o avaliador tem de usar box_* (a caixa da própria janela)
            bc = (0.0, 0.0, 0.1, 0.1) if tem else ("", "", "", "")
            an.append({"split": sp, "caso": caso, "seed": seed, "frame": f, "offset": o, "box_src": src,
                       "box_x1": b[0], "box_y1": b[1], "box_x2": b[2], "box_y2": b[3], "img_w": 320, "img_h": 320,
                       "excluida": 0, "boxc_x1": bc[0], "boxc_y1": bc[1], "boxc_x2": bc[2], "boxc_y2": bc[3],
                       "categoria": cat, "lesao": 1})
            vis = "1" if abs(o) <= 3 else ("duvidoso" if o == 4 else "0")
            pf.append({"split": sp, "caso": caso, "seed": seed, "frame": f, "offset": o, "visivel": vis, "nota": ""})
        jt.append({"split": sp, "caso": caso, "seed": seed, "excluida": 0, "seed_tem_box": 1, "categoria": cat, "lesao": 1})
    pd.DataFrame(an).to_csv(raiz / "data/derived/annotations_v1.csv", index=False)
    pd.DataFrame(pf).to_csv(raiz / "data/derived/paulo_frame_labels.csv", index=False)
    pd.DataFrame(jt).to_csv(raiz / "data/derived/janelas_v1.csv", index=False)
    conj = {"CAL": janelas[:3], "VAL_EX": janelas[3:], "TEST_BOX": janelas[3:]}
    for nome, js in conj.items():
        (raiz / "sets" / f"{nome}.json").write_text(json.dumps(
            {"nome": nome, "janelas": [{"split": s, "caso": c, "seed": e} for s, c, e, _ in js]}))
    cfg = yaml.safe_load((RAIZ / "configs" / "avaliacao.yaml").read_text(encoding="utf-8")) \
        if (RAIZ / "configs" / "avaliacao.yaml").exists() else None
    if cfg is None:
        saltar("configs/avaliacao.yaml não existe")
    cfg["caminhos"] = {k: str(raiz / v) for k, v in cfg["caminhos"].items()}
    cfg["bootstrap"]["reamostragens"] = 200
    (raiz / "configs/avaliacao.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    (raiz / "docs/DATASET_VERSION.txt").write_text("sha256 do conteúdo (…): abc123\n", encoding="utf-8")

    def pred(nome, js, deteta, caixa=cx):
        linhas = []
        for sp, caso, seed, _ in js:
            for o in range(-10, 11):
                s = 0.95 if abs(o) in deteta else 0.05
                b = caixa if s > 0.5 else ("", "", "", "")
                linhas.append({"caso": caso, "seed": seed, "frame": seed + o, "score": s,
                               "box_x1": b[0], "box_y1": b[1], "box_x2": b[2], "box_y2": b[3]})
        p = raiz / f"pred_{nome}.csv"
        pd.DataFrame(linhas).to_csv(p, index=False)
        return str(p)
    return raiz, conj, pred


def test_ponta_a_ponta_com_ficheiros():
    try:
        import yaml  # noqa: F401
    except ImportError:
        saltar("PyYAML não instalado")
    with tempfile.TemporaryDirectory() as tmp:
        raiz, conj, pred = _projeto(tmp)
        cfgp = str(raiz / "configs/avaliacao.yaml")
        corre = lambda *a: ev.main(list(a) + ["--config", cfgp])          # noqa: E731
        # CAL: o método vê |offset| ≤ 2 (a referência PF tem visível até 3) → recall 4/6 por janela
        assert corre("calibrar", "--metodo", "M1", "--pred", pred("M1_CAL", conj["CAL"], {1, 2})) == 0
        cal = json.loads((raiz / "outputs/eval/CAL/M1.json").read_text())
        m = cal["por_op"]["OP_F1"]["micro"]
        # F1 = 0,8 para τ em ]0,05; 0,95] → OP-F1 = 0,94; recall ≥ 0,9 só com τ ≤ 0,05 → OP-R90 = 0,04
        assert cal["pontos_operacao"]["OP_F1"]["tau"] == 0.94 and cal["pontos_operacao"]["OP_R90"]["tau"] == 0.04
        assert perto(m["R"]["valor"], 4 / 6) and m["P"]["valor"] == 1 and m["n_duvidosos"] == 3
        assert m["R_loc"]["valor"] == 1                                 # usa box_* (com boxc_* daria 0)
        assert cal["registo"]["versao_dataset"] == "abc123"
        # VAL_EX: uma vez; a segunda é recusada; com --repetir passa e fica registada
        pv = pred("M1_VAL", conj["VAL_EX"], {1, 2, 3})
        assert corre("avaliar", "--metodo", "M1", "--conjunto", "VAL_EX", "--pred", pv) == 0
        assert corre("avaliar", "--metodo", "M1", "--conjunto", "VAL_EX", "--pred", pv) == 2
        assert corre("avaliar", "--metodo", "M1", "--conjunto", "VAL_EX", "--pred", pv, "--repetir", "teste") == 0
        val = json.loads((raiz / "outputs/eval/VAL_EX/M1.json").read_text())
        assert val["por_op"]["OP_F1"]["micro"]["R"]["valor"] == 1 and "R_loc" not in val["por_op"]["OP_F1"]["micro"]
        # TEST_BOX: sem precisão; FP aparentes exportados
        pt = pred("M1_TB", conj["TEST_BOX"], {1, 2, 5})
        assert corre("avaliar", "--metodo", "M1", "--conjunto", "TEST_BOX", "--pred", pt) == 0
        tb = json.loads((raiz / "outputs/eval/TEST_BOX/M1.json").read_text())
        mt = tb["por_op"]["OP_F1"]["micro"]
        assert mt["R_box"]["valor"] == 1 and mt["R_loc"]["valor"] == 1 and "P" not in mt and mt["n_fp_aparentes"] == 4
        fp = pd.read_csv(raiz / "outputs/eval/TEST_BOX/M1_fp_aparentes.csv")
        assert len(fp) == 4 and fp["amostra"].sum() == 4
        reg = pd.read_csv(raiz / "outputs/test_runs.csv")
        assert len(reg) == 3 and reg["motivo_repeticao"].fillna("").tolist() == ["", "teste", ""]
        # outro método e comparação pareada
        assert corre("calibrar", "--metodo", "M2", "--pred", pred("M2_CAL", conj["CAL"], {1})) == 0
        assert corre("avaliar", "--metodo", "M2", "--conjunto", "VAL_EX", "--pred", pred("M2_VAL", conj["VAL_EX"], {1})) == 0
        assert ev.main(["comparar", "--conjunto", "VAL_EX", "--a", "M1", "--b", "M2", "--config", cfgp]) == 0
        cmp_ = json.loads((raiz / "outputs/eval/VAL_EX/comparar_M1_vs_M2_F1_OP_F1.json").read_text())
        assert cmp_["diferenca_media"] > 0
        res = pd.read_csv(raiz / "outputs/eval/resultados.csv")
        assert {"M1", "M2"} <= set(res["metodo"]) and "F1_micro" in set(res["metrica"])
        # calibração de outro método não serve
        assert corre("avaliar", "--metodo", "M3", "--conjunto", "VAL_EX", "--pred", pv,
                     "--op", str(raiz / "outputs/eval/CAL/M1.json")) == 2
        # previsões de uma janela de teste em CAL → recusado
        assert corre("calibrar", "--metodo", "M4", "--pred", pv) == 2


def test_referencia_incompleta_e_recusada():
    try:
        import yaml  # noqa: F401
    except ImportError:
        saltar("PyYAML não instalado")
    with tempfile.TemporaryDirectory() as tmp:
        raiz, conj, pred = _projeto(tmp)
        f = raiz / "data/derived/paulo_frame_labels.csv"
        pf = pd.read_csv(f, dtype=str)
        pf.drop(index=[3, 4]).to_csv(f, index=False)                    # dois frames sem rótulo
        cfg = ev.ler_config(raiz / "configs/avaliacao.yaml")
        js = ev.ler_conjunto(raiz / "sets/CAL.json")
        erro(ev.carregar_referencia, cfg, "CAL", js, contem="não cobre 2 frames")
        erro(ev.carregar_referencia, cfg, "VAL_EX", js, contem="só pode ter janelas de Dataset_Teste")


# ------------------------------------------------------------------ versão 1.1 (12 out 2026)
def test_avaliar_recusa_o_conjunto_de_calibracao():
    """O avaliar em CAL gravaria por cima da calibração (perdia a grelha e o F1 por janela): é recusado."""
    try:
        import yaml  # noqa: F401
    except ImportError:
        saltar("PyYAML não instalado")
    with tempfile.TemporaryDirectory() as tmp:
        raiz, conj, pred = _projeto(tmp)
        cfgp = str(raiz / "configs/avaliacao.yaml")
        pc = pred("M1_CAL", conj["CAL"], {1, 2})
        assert ev.main(["calibrar", "--metodo", "M1", "--pred", pc, "--config", cfgp]) == 0
        f = raiz / "outputs/eval/CAL/M1.json"
        antes = f.read_bytes()
        assert ev.main(["avaliar", "--metodo", "M1", "--conjunto", "CAL", "--pred", pc, "--config", cfgp]) == 2
        assert f.read_bytes() == antes and "calibracao" in json.loads(antes)


def _git(pasta, *a):
    import subprocess
    r = subprocess.run(["git", "-C", str(pasta), "-c", "user.name=Teste", "-c", "user.email=teste@exemplo.invalid",
                        "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false", *a],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


def test_execucao_de_teste_exige_o_codigo_guardado_no_git():
    """Cada resultado regista o commit e se havia alterações por guardar. Em VAL_EX e TEST_BOX, com alterações
    por guardar, a execução é recusada e nada fica registado; em CAL só avisa."""
    import shutil
    try:
        import yaml  # noqa: F401
    except ImportError:
        saltar("PyYAML não instalado")
    if shutil.which("git") is None:
        saltar("git não instalado")
    with tempfile.TemporaryDirectory() as tmp:
        raiz, conj, pred = _projeto(tmp)
        (raiz / ".gitignore").write_text("outputs/\npred_*.csv\n", encoding="utf-8")
        (raiz / "methods").mkdir()
        (raiz / "methods/metodo.py").write_text("# versão 1\n", encoding="utf-8")
        _git(raiz, "init", "-q")
        _git(raiz, "add", "-A")
        _git(raiz, "commit", "-q", "-m", "inicial")
        cfgp = str(raiz / "configs/avaliacao.yaml")
        corre = lambda *a: ev.main(list(a) + ["--config", cfgp])          # noqa: E731
        # tudo guardado: o registo tem o commit e diz que não havia alterações por guardar
        assert corre("calibrar", "--metodo", "M1", "--pred", pred("M1_CAL", conj["CAL"], {1, 2})) == 0
        reg = json.loads((raiz / "outputs/eval/CAL/M1.json").read_text())["registo"]
        assert reg["commit"] == _git(raiz, "rev-parse", "--short", "HEAD") and reg["alteracoes_por_guardar"] is False
        # código alterado e não guardado: em CAL avisa; em VAL_EX recusa e não regista nada
        (raiz / "methods/metodo.py").write_text("# versão 2\n", encoding="utf-8")
        assert corre("calibrar", "--metodo", "M1", "--pred", pred("M1_CAL", conj["CAL"], {1, 2})) == 0
        cal = json.loads((raiz / "outputs/eval/CAL/M1.json").read_text())
        assert cal["registo"]["alteracoes_por_guardar"] is True
        assert any("alterações por guardar" in a for a in cal["avisos"])
        pv = pred("M1_VAL", conj["VAL_EX"], {1, 2, 3})
        assert corre("avaliar", "--metodo", "M1", "--conjunto", "VAL_EX", "--pred", pv) == 2
        assert not (raiz / "outputs/test_runs.csv").exists() and not (raiz / "outputs/eval/VAL_EX/M1.json").exists()
        # depois do commit, corre, e o registo aponta para o commit novo
        _git(raiz, "commit", "-q", "-am", "versão 2")
        novo = _git(raiz, "rev-parse", "--short", "HEAD")
        assert corre("avaliar", "--metodo", "M1", "--conjunto", "VAL_EX", "--pred", pv) == 0
        val = json.loads((raiz / "outputs/eval/VAL_EX/M1.json").read_text())
        assert val["registo"]["commit"] == novo and val["registo"]["alteracoes_por_guardar"] is False
        assert pd.read_csv(raiz / "outputs/test_runs.csv", dtype=str)["commit"].tolist() == [novo]


# ------------------------------------------------------------------ correr sem pytest
if __name__ == "__main__":
    testes = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    falhas = saltados = 0
    for nome, f in testes:
        try:
            f()
            print(f"  ok      {nome}")
        except Saltar as e:
            saltados += 1
            print(f"  saltado {nome} ({e})")
        except Exception as e:                                          # noqa: BLE001
            falhas += 1
            print(f"  FALHOU  {nome}: {type(e).__name__}: {e}")
    print(f"\n{len(testes) - falhas - saltados} passaram · {falhas} falharam · {saltados} saltados")
    sys.exit(1 if falhas else 0)
