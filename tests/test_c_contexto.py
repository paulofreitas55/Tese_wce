"""
Testes do scripts/17b_c_contexto.py (contexto da ferramenta de revisão v1.3). Correm de duas maneiras, a partir da pasta
do projeto:
    python tests/test_c_contexto.py          (não precisa de pytest)
    python -m pytest tests/test_c_contexto.py -q
Usam o mini-projeto inventado do tests/test_c_itens.py e a lista gravada pelo script 17; nunca leem dados reais.
Versão 1.0 (24 out 2026).
"""
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]


def _modulo(nome, caminho):
    spec = importlib.util.spec_from_file_location(nome, caminho)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


cx = _modulo("c_contexto", RAIZ / "scripts" / "17b_c_contexto.py")
ti = _modulo("t_c_itens_para_17b", RAIZ / "tests" / "test_c_itens.py")      # o mini-projeto do script 17
ci, ev = ti.ci, cx.ev
Saltar, saltar, erro = ti.Saltar, ti.saltar, ti.erro
K3, K4 = ["split", "caso", "seed"], ["split", "caso", "seed", "frame"]


def _montar(tmp):
    raiz = ti._projeto(tmp)
    ci.main(ti._args(raiz))                                              # a lista da revisão e o piloto
    return raiz


def _args(raiz, *extra, itens="itens_v1.csv", saida="contexto_v1.csv"):
    raiz = Path(raiz)
    g, exp = raiz / "data/derived/guiao_c", raiz / "data/derived/expanded_v1.csv"
    return ["--config", str(raiz / "configs/avaliacao.yaml"), "--itens", str(g / itens), "--expandida", str(exp),
            "--saida", str(g / saida), "--resumo", str(raiz / "outputs" / f"{saida}.txt"),
            "--sha-expandida", ev._sha256(exp)[:16], *extra]


def _ler(raiz, nome):
    return pd.read_csv(Path(raiz) / "data/derived/guiao_c" / nome, dtype={"caso": str})


def test_contexto_so_tem_caixas_dos_outros_frames():
    with tempfile.TemporaryDirectory() as tmp:
        raiz = _montar(tmp)
        r = cx.main(_args(raiz))
        t, itens, chave = _ler(raiz, "contexto_v1.csv"), _ler(raiz, "itens_v1.csv"), _ler(raiz, "itens_chave_v1.csv")
        f = Path(raiz) / "data/derived/guiao_c/contexto_v1.csv"
        assert list(t.columns) == cx.COL_CONTEXTO and not os.stat(f).st_mode & 0o222
        assert set(t["fonte"]) == {"original", "modelo"}
        # nada dos frames da lista, nem do seed; só as janelas da lista e |offset| ≤ 12
        assert t.merge(itens[K4], on=K4).empty
        off = t["frame"] - t["seed"]
        assert ((off.abs() >= 1) & (off.abs() <= 12)).all()
        assert set(map(tuple, t[K3].values)) == set(map(tuple, itens[K3].values))
        # caixas do modelo: todas as dos outros frames, de qualquer nível (o expanded tem alta e baixa)
        e = pd.read_csv(Path(raiz) / "data/derived/expanded_v1.csv", dtype={"caso": str}, keep_default_na=False)
        e = e[e["box_x1"] != ""]
        e = e.merge(itens[K3].drop_duplicates(), on=K3).merge(itens[K4], on=K4, how="left", indicator=True)
        e = e[e["_merge"] == "left_only"]
        m = t[t["fonte"] == "modelo"]
        assert len(m) == len(e) and set(e["nivel"]) == {"alta", "baixa"}
        x = m.merge(e[K4 + ["box_x1"]].astype({"box_x1": float}), on=K4, suffixes=("", "_e"))
        assert len(x) == len(m) and ((x["box_x1"] - x["box_x1_e"]).abs() < 1e-6).all()
        # caixas originais: −2..2 do mini-projeto, menos os frames da lista (os controlos com caixa original ficam escondidos)
        o = t[t["fonte"] == "original"]
        assert set((o["frame"] - o["seed"]).abs()) <= {1, 2}
        co = chave[chave["tipo"] == "controlo_original"]
        assert len(co) == 3 and o.merge(co[K4], on=K4).empty and r["escondidas"] == 3
        janelas = itens[K3].drop_duplicates()
        assert len(o) == 4 * len(janelas) - 3                              # 4 vizinhos com caixa por janela − 3 controlos
        # sem score, nível nem tipo
        txt = f.read_text(encoding="utf-8")
        assert not {"score", "nivel", "tipo"} & set(txt.splitlines()[0].split(","))
        assert "Linha para o docs/decisoes.md:" in r["texto"] and "nenhum frame da lista entra" in r["linha"]


def test_congela_e_serve_para_o_piloto():
    with tempfile.TemporaryDirectory() as tmp:
        raiz = _montar(tmp)
        assert cx.main(_args(raiz))["estado"] == "gravado"
        assert cx.main(_args(raiz))["estado"].startswith("igual")
        erro(cx.main, _args(raiz, "--alcance", "6"), contem="já existe com outro conteúdo")
        assert cx.main(_args(raiz, "--alcance", "6", "--forcar"))["estado"].startswith("substituído")
        # o piloto tem o seu contexto, sem os frames do piloto
        cx.main(_args(raiz, itens="piloto_v1.csv", saida="contexto_piloto_v1.csv"))
        t, pil = _ler(raiz, "contexto_piloto_v1.csv"), _ler(raiz, "piloto_v1.csv")
        assert not t.empty and t.merge(pil[K4], on=K4).empty
        assert set(map(tuple, t[K3].values)) == set(map(tuple, pil[K3].values))


def test_verificacoes():
    with tempfile.TemporaryDirectory() as tmp:
        raiz = _montar(tmp)
        erro(cx.main, _args(raiz, itens="nao_existe.csv"), contem="Não encontrei a lista")
        a = _args(raiz)
        i = a.index("--sha-expandida")
        erro(cx.main, a[:i + 1] + ["0000"] + a[i + 2:], contem="não é o do A8")
        erro(cx.main, _args(raiz, itens="itens_chave_v1.csv", saida="x.csv"), contem="tem de ser uma lista do script 17")
        assert not (Path(raiz) / "data/derived/guiao_c/contexto_v1.csv").exists()


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
