"""
Testes do scripts/17_c_itens.py (Guião C, C1–C2: lista da revisão). Correm de duas maneiras, a partir da pasta do projeto:
    python tests/test_c_itens.py          (não precisa de pytest)
    python -m pytest tests/test_c_itens.py -q
Usam um mini-projeto inventado, com o formato real dos ficheiros (annotations_v1, janelas_v1, expanded_v1, sets e um
dataset falso só com os nomes das imagens); nunca leem dados reais.
Versão 1.0 (23 out 2026).
"""
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("c_itens", RAIZ / "scripts" / "17_c_itens.py")
ci = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ci)
ev = ci.ev

# (split, caso, seed, categoria, papel): "rev" tem caixas novas; "pil" não tem (candidata ao piloto)
JANELAS = [("Dataset_Teste", "0201", 600, "polipo", "rev"), ("Dataset_Teste", "0202", 900, "erosao", "rev"),
           ("Dataset_Teste", "0203", 1200, "angiectasia", "rev"), ("Dataset_Teste", "0204", 450, "sangue", "rev"),
           ("Dataset_Treino", "0101", 500, "angiectasia", "rev"), ("Dataset_Treino", "0102", 800, "ulcera", "rev"),
           ("Dataset_Treino", "0103", 300, "eritema", "rev"), ("Dataset_Treino", "0104", 1500, "xantelasma", "rev"),
           ("Dataset_Teste", "0205", 700, "polipo", "pil"), ("Dataset_Teste", "0206", 650, "erosao", "pil"),
           ("Dataset_Treino", "0105", 400, "angiectasia", "pil"), ("Dataset_Treino", "0106", 950, "ulcera", "pil"),
           ("Dataset_Treino", "0107", 1000, "polipo", "pil"), ("Dataset_Treino", "0108", 1100, "erosao", "pil")]
CAIXA_BAIXA = {4: 0.13, 6: 0.17, 9: 0.01, 10: 0.9, 11: 0.14, 12: 0.2}    # |offset| → lado da caixa (9 minúscula, 10 enorme)
REV = [j for j in JANELAS if j[4] == "rev"]
N_NOVAS = len(REV) * 4 + sum(1 for i in range(len(REV)) if i % 2 == 0)      # −3, 3, 5, 7 e, em metade, −8
N_SEM_FICHEIRO = sum(1 for i in range(len(REV)) if i % 2 == 0)


class Saltar(Exception):
    pass


def saltar(motivo):
    if "pytest" in sys.modules:
        import pytest
        pytest.skip(motivo)
    raise Saltar(motivo)


def erro(fn, *a, contem=None, **k):
    try:
        fn(*a, **k)
    except ev.ErroAvaliacao as e:
        assert contem is None or contem in str(e), f"a mensagem não diz '{contem}': {e}"
        return str(e)
    raise AssertionError("devia ter dado erro")


def _caixa(lado):
    return (round(0.5 - lado / 2, 6), round(0.5 - lado / 2, 6), round(0.5 + lado / 2, 6), round(0.5 + lado / 2, 6))


def _projeto(raiz):
    """Mini-projeto: 8 janelas com caixas novas (−3, 3, 5, 7 e, em metade, −8, sem ficheiro de caixa) e 6 sem
    caixas novas (só alta onde já há caixa original, −2..2). Caixas originais em −2..+2; −8 sem ficheiro."""
    try:
        import yaml
    except ImportError:
        saltar("PyYAML não instalado")
    raiz = Path(raiz)
    for p in ("data/derived", "sets", "configs", "ds"):
        (raiz / p).mkdir(parents=True)
    an, jt, ex = [], [], []
    i_rev = 0
    for n, (sp, caso, seed, cat, papel) in enumerate(JANELAS):
        lado_seed = (0.08, 0.12, 0.2)[n % 3]
        pasta = raiz / "ds" / sp / caso / "GT_com_classes" / str(seed)
        pasta.mkdir(parents=True)
        for f in range(seed - 20, seed + 21):
            o = f - seed
            (pasta / f"{caso}_{f}_{seed}.jpg").write_bytes(b"")
            src = ("medico_seed" if o == 0 else "medico_vizinho" if abs(o) <= 2 else
                   "sem_ficheiro" if o == -8 else "vazio")
            b = _caixa(lado_seed if o == 0 else 0.1) if abs(o) <= 2 else ("", "", "", "")
            an.append({"split": sp, "caso": caso, "seed": seed, "frame": f, "offset": o, "box_src": src,
                       "box_x1": b[0], "box_y1": b[1], "box_x2": b[2], "box_y2": b[3], "img_w": 320, "img_h": 320,
                       "excluida": 0, "boxc_x1": "", "boxc_y1": "", "boxc_x2": "", "boxc_y2": "",
                       "categoria": cat, "lesao": 1})
        jt.append({"split": sp, "caso": caso, "seed": seed, "excluida": 0, "seed_tem_box": 1, "categoria": cat,
                   "lesao": 1, "descricao": f"lesão {cat}"})
        alta = {-2, -1, 1, 2}
        if papel == "rev":
            alta |= {-3, 3, 5, 7} | ({-8} if i_rev % 2 == 0 else set())
            i_rev += 1
        for o in [x for x in range(-12, 13) if x]:
            if o in alta:
                nivel, s, b = "alta", 0.95, _caixa(0.05 + 0.02 * abs(o))
            else:
                nivel, s = "baixa", 0.3
                b = _caixa(CAIXA_BAIXA[abs(o)]) if abs(o) in CAIXA_BAIXA else ("", "", "", "")
            ex.append({"caso": caso, "split": sp, "seed": seed, "frame": seed + o, "score": s,
                       "box_x1": b[0], "box_y1": b[1], "box_x2": b[2], "box_y2": b[3], "nivel": nivel,
                       "metodo": "SAM2_L12", "versao_config": "final_SAM2_L12.yaml sha256 abc", "offset": o,
                       "area_px": 0 if b[0] == "" else 500, "caixa_original": int(abs(o) <= 2)})
    pd.DataFrame(an).to_csv(raiz / "data/derived/annotations_v1.csv", index=False)
    pd.DataFrame(jt).to_csv(raiz / "data/derived/janelas_v1.csv", index=False)
    pd.DataFrame(ex).to_csv(raiz / "data/derived/expanded_v1.csv", index=False)
    conj = {"ALL_SEEDBOX": JANELAS, "CAL": [j for j in JANELAS if j[1] in ("0101", "0102", "0105")],
            "VAL_EX": [j for j in JANELAS if j[1] in ("0201", "0202", "0205")],
            "TEST_BOX": [j for j in JANELAS if j[0] == "Dataset_Teste"]}
    for nome, js in conj.items():
        (raiz / "sets" / f"{nome}.json").write_text(json.dumps(
            {"nome": nome, "janelas": [{"split": s, "caso": c, "seed": e} for s, c, e, _, _ in js]}))
    base = RAIZ / "configs" / "avaliacao.yaml"
    if not base.exists():
        saltar("configs/avaliacao.yaml não existe")
    cfg = yaml.safe_load(base.read_text(encoding="utf-8"))
    cfg["caminhos"] = {k: str(raiz / v) for k, v in cfg["caminhos"].items()}
    (raiz / "configs/avaliacao.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return raiz


def _args(raiz, *extra, saida="guiao_c"):
    raiz = Path(raiz)
    exp = raiz / "data/derived/expanded_v1.csv"
    return ["--ds", str(raiz / "ds"), "--config", str(raiz / "configs/avaliacao.yaml"), "--expandida", str(exp),
            "--saida-dir", str(raiz / "data/derived" / saida), "--resumo", str(raiz / "outputs" / f"{saida}.txt"),
            "--esperadas", str(N_NOVAS), "--sha-expandida", ev._sha256(exp)[:16],
            "--controlos-original", "3", "--controlos-baixa", "3", *extra]


def _ler(raiz, saida="guiao_c"):
    d = Path(raiz) / "data/derived" / saida
    le = lambda n: pd.read_csv(d / n, dtype={"caso": str})                       # noqa: E731
    return le("itens_v1.csv"), le("itens_chave_v1.csv"), le("piloto_v1.csv"), le("piloto_chave_v1.csv")


def _janela(t):
    return list(zip(t["split"], t["caso"], t["seed"]))


# ------------------------------------------------------------------ o que entra na lista e o que fica escondido
def test_lista_so_tem_as_caixas_novas_e_os_controlos():
    with tempfile.TemporaryDirectory() as tmp:
        raiz = _projeto(tmp)
        r = ci.main(_args(raiz))
        itens, chave, _, _ = _ler(raiz)
        # a ferramenta só vê janela, frame e caixa: nada que revele o score, o nível ou a caixa original
        assert list(itens.columns) == ci.COL_ITENS and list(chave.columns) == ci.COL_CHAVE
        assert not {"score", "nivel", "caixa_original", "caixa_ref", "tipo"} & set(itens.columns)
        assert (itens[["item", "split", "caso", "seed", "frame"]] == chave[["item", "split", "caso", "seed", "frame"]]).all().all()
        assert itens["item"].tolist() == list(range(1, len(itens) + 1))
        assert not chave.duplicated(["split", "caso", "seed", "frame"]).any()
        novas = chave[chave["tipo"] == "nova"]
        assert len(novas) == N_NOVAS and (novas["nivel"] == "alta").all() and (novas["caixa_original"] == 0).all()
        assert (novas["caixa_ref"] == "sem_ficheiro").sum() == N_SEM_FICHEIRO
        assert set(novas.loc[novas["caixa_ref"] == "sem_ficheiro", "offset"]) == {-8}
        # controlos: só em janelas com caixas novas, no máximo um de cada tipo por janela
        rev = set(_janela(novas))
        for tipo, n in (("controlo_original", 3), ("controlo_baixa", 3)):
            c = chave[chave["tipo"] == tipo]
            assert len(c) == n and set(_janela(c)) <= rev and not pd.Series(_janela(c)).duplicated().any()
        co, cb = chave[chave["tipo"] == "controlo_original"], chave[chave["tipo"] == "controlo_baixa"]
        assert (co["nivel"] == "alta").all() and (co["caixa_original"] == 1).all()
        lo, hi = r["area_p5_p95"]
        assert (cb["nivel"] == "baixa").all() and cb["area_caixa"].between(lo - 1e-6, hi + 1e-6).all()
        assert set(chave["tipo"]) == {"nova", "controlo_original", "controlo_baixa"}
        # classe, tercil e conjuntos com as regras do avaliador
        assert set(chave["classe"]) <= set(ci.NOMES) and set(chave.loc[chave["caso"] == "0204", "classe"]) == {"outras"}
        assert set(chave.loc[chave["caso"] == "0201", "tercil"]) == {"pequena"}
        assert set(chave.loc[chave["caso"] == "0202", "tercil"]) == {"media"}
        assert (chave.loc[chave["caso"].isin(["0101", "0102"]), "cal"] == 1).all()
        assert (chave.loc[chave["split"] == "Dataset_Teste", "test_box"] == 1).all()
        # ficheiros só de leitura
        for f in (Path(raiz) / "data/derived/guiao_c").glob("*.csv"):
            assert not os.stat(f).st_mode & 0o222, f"{f.name} devia ser só de leitura"


def test_ordem_janelas_baralhadas_e_do_seed_para_fora():
    with tempfile.TemporaryDirectory() as tmp:
        raiz = _projeto(tmp)
        ci.main(_args(raiz))
        _, chave, _, _ = _ler(raiz)
        js = _janela(chave)
        blocos = [js[0]] + [b for a, b in zip(js, js[1:]) if a != b]
        assert len(blocos) == len(set(blocos)), "as caixas de cada janela têm de vir seguidas"
        for k in blocos:
            off = chave.loc[[j == k for j in js], "offset"].tolist()
            neg, pos = [o for o in off if o < 0], [o for o in off if o > 0]
            assert off == neg + pos and neg == sorted(neg, reverse=True) and pos == sorted(pos)
        # a mesma semente dá exatamente os mesmos ficheiros; outra semente, outra ordem de janelas
        ci.main(_args(raiz, saida="outra_vez"))
        ci.main(_args(raiz, "--semente", "1", saida="semente1"))
        d = Path(raiz) / "data/derived"
        for n in ("itens_v1.csv", "itens_chave_v1.csv", "piloto_v1.csv", "piloto_chave_v1.csv"):
            assert (d / "guiao_c" / n).read_bytes() == (d / "outra_vez" / n).read_bytes()
        _, ch1, _, _ = _ler(raiz, "semente1")
        b1 = [k for i, k in enumerate(_janela(ch1)) if i == 0 or _janela(ch1)[i - 1] != k]
        assert b1 != blocos and sorted(b1) == sorted(blocos)


def test_sem_controlos_as_caixas_novas_ficam_na_mesma_ordem():
    with tempfile.TemporaryDirectory() as tmp:
        raiz = _projeto(tmp)
        ci.main(_args(raiz))
        ci.main(_args(raiz, "--sem-controlos", saida="sem"))
        _, com, _, _ = _ler(raiz)
        _, sem, _, _ = _ler(raiz, "sem")
        assert set(sem["tipo"]) == {"nova"} and len(sem) == N_NOVAS
        k = ["split", "caso", "seed", "frame"]
        assert com.loc[com["tipo"] == "nova", k].values.tolist() == sem[k].values.tolist()


def test_lista_congela_e_nao_e_reescrita():
    with tempfile.TemporaryDirectory() as tmp:
        raiz = _projeto(tmp)
        r1 = ci.main(_args(raiz))
        assert r1["estados"] == ["gravado"] * 4
        f = Path(raiz) / "data/derived/guiao_c/itens_v1.csv"
        antes = f.read_bytes()
        r2 = ci.main(_args(raiz))
        assert all(e.startswith("igual") for e in r2["estados"]) and r2["sha256"] == r1["sha256"]
        erro(ci.main, _args(raiz, "--semente", "1"), contem="já existe com outro conteúdo")
        assert f.read_bytes() == antes                                          # tudo ou nada: nada mudou
        r3 = ci.main(_args(raiz, "--semente", "1", "--forcar"))
        assert r3["estados"][0].startswith("substituído") and f.read_bytes() != antes


def test_piloto_fica_fora_da_revisao():
    with tempfile.TemporaryDirectory() as tmp:
        raiz = _projeto(tmp)
        r = ci.main(_args(raiz))
        itens, chave, pil, pch = _ler(raiz)
        rev = set(_janela(chave))
        assert len(r["janelas_piloto"]) == 5 and not set(r["janelas_piloto"]) & rev
        assert set(_janela(pch)) == set(r["janelas_piloto"])
        assert set(pch["tipo"]) == {"piloto_alta", "piloto_baixa"} and len(pil) == 50
        assert pch.groupby(["split", "caso", "seed"]).size().max() <= 10
        assert list(pil.columns) == ci.COL_ITENS


def test_verificacoes_antes_de_gravar():
    with tempfile.TemporaryDirectory() as tmp:
        raiz = _projeto(tmp)
        a = _args(raiz)
        i = a.index("--sha-expandida")
        erro(ci.main, a[:i + 1] + ["0000"] + a[i + 2:], contem="não é o do A8")
        j = a.index("--esperadas")
        erro(ci.main, a[:j + 1] + [str(N_NOVAS + 1)] + a[j + 2:], contem="esperava")
        # imagem em falta
        img = Path(raiz) / "ds/Dataset_Teste/0201/GT_com_classes/600/0201_603_600.jpg"
        img.unlink()
        erro(ci.main, _args(raiz), contem="Faltam 1 imagens")
        img.write_bytes(b"")
        # caixa_original que não bate com a anotação original
        p = Path(raiz) / "data/derived/expanded_v1.csv"
        e = pd.read_csv(p, dtype=str, keep_default_na=False)
        e.loc[(e["caso"] == "0201") & (e["offset"] == "3"), "caixa_original"] = "1"
        e.to_csv(p, index=False)
        erro(ci.main, _args(raiz), contem="não bate com annotations_v1.csv")
        # janelas que não são as de ALL_SEEDBOX
        _projeto(Path(tmp) / "b")
        s = Path(tmp) / "b/sets/ALL_SEEDBOX.json"
        d = json.loads(s.read_text())
        d["janelas"] = d["janelas"][1:]
        s.write_text(json.dumps(d))
        erro(ci.main, _args(Path(tmp) / "b"), contem="não são as de ALL_SEEDBOX")
        assert not (Path(raiz) / "data/derived/guiao_c").exists()                # nada foi gravado


def test_resumo_tem_a_linha_para_o_decisoes():
    with tempfile.TemporaryDirectory() as tmp:
        raiz = _projeto(tmp)
        r = ci.main(_args(raiz))
        txt = (Path(raiz) / "outputs/guiao_c.txt").read_text(encoding="utf-8")
        assert "Linha para o docs/decisoes.md:" in txt and r["linha"] in txt
        assert f"itens_v1.csv sha256 {r['sha256']['itens'][:16]}" in r["linha"]
        assert f"{N_NOVAS} caixas novas" in r["linha"] and f"{N_SEM_FICHEIRO} delas em frames sem ficheiro" in r["linha"]
        assert "3 controlos às cegas com caixa original e 3 abaixo do limiar" in r["linha"]
        assert "NÃO abras o itens_chave_v1.csv" in txt


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
