"""
Testes do A8 (scripts/15_a8_expandida.py): anotação expandida a partir de previsões inventadas num mini-projeto, com as
contas feitas à mão (níveis, frames novos, caixas antes e depois, classes, splits) e os erros (janela sem previsões,
substituir um expanded_v1.csv diferente). Nunca leem dados reais.
    python tests/test_a8.py          ou          python -m pytest tests/ -q
"""
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
from test_evaluate import Saltar, _projeto, ev  # noqa: E402

CX = (0.30, 0.30, 0.50, 0.50)


def _carregar():
    spec = importlib.util.spec_from_file_location("a8", AQUI.parent / "scripts" / "15_a8_expandida.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _previsoes(raiz, janelas, sem=None):
    """|offset| ≤ 3: 0,95 com caixa; na 0101 o +4 tem 0,9 sem caixa (alta sem localização); o resto 0,5 com caixa; a
    0202 fica toda a 0,1 (sem nenhum frame alta). As caixas originais do mini-projeto estão em −2..+2."""
    linhas = []
    for sp, caso, seed, _ in janelas:
        if caso == sem:
            continue
        for o in range(-10, 11):
            if o == 0:
                continue
            s, cx = (0.95, CX) if abs(o) <= 3 else (0.5, CX)
            if caso == "0101" and o == 4:
                s, cx = 0.9, ("", "", "", "")
            if caso == "0202":
                s = 0.1
            linhas.append({"split": sp, "caso": caso, "seed": seed, "frame": seed + o, "score": s,
                           **dict(zip(["box_x1", "box_y1", "box_x2", "box_y2"], cx)), "area_px": 400})
    f = raiz / "outputs/preds/pred_SAM2_L12_ALL_SEEDBOX.csv"
    f.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(linhas).to_csv(f, index=False)
    return f


def test_a8_expandida():
    try:
        import yaml  # noqa: F401
    except ImportError:
        raise Saltar("falta o PyYAML")
    a8 = _carregar()
    with tempfile.TemporaryDirectory() as tmp:
        raiz, conj, _ = _projeto(tmp)
        todas = conj["CAL"] + conj["VAL_EX"]                                     # as 5 janelas do mini-projeto
        (raiz / "sets/ALL_SEEDBOX.json").write_text(json.dumps(
            {"nome": "ALL_SEEDBOX", "janelas": [{"split": s, "caso": c, "seed": e} for s, c, e, _ in todas]}))
        (raiz / "configs/final_SAM2_L12.yaml").write_text(
            "metodo: SAM2_L12\nparametros: {prompt: caixa, l_max: 5}\n"
            "pontos_operacao: {OP_F1: {tau: 0.86, a_min: 0.0}, OP_R90: {tau: 0.0, a_min: 0.0}}\n"
            "decisoes_A6: {criterio2_op: OP_F1, niveis_A8: {alta: {score_minimo: 0.86}, media: null}}\n", encoding="utf-8")
        pred = _previsoes(raiz, todas)
        (raiz / "outputs/masks/ALL_SEEDBOX").mkdir(parents=True)
        for sp, caso, seed, _ in todas[:4]:                                      # falta a máscara de uma janela
            (raiz / f"outputs/masks/ALL_SEEDBOX/{sp}__{caso}__{seed}.json.gz").write_bytes(b"")
        base = ["--config", str(raiz / "configs/avaliacao.yaml"), "--final", str(raiz / "configs/final_SAM2_L12.yaml"),
                "--pred", str(pred), "--mascaras", str(raiz / "outputs/masks/ALL_SEEDBOX"),
                "--saida", str(raiz / "data/derived/expanded_v1.csv"), "--saida-resumo", str(raiz / "outputs")]

        r = a8.main(base)
        x = pd.read_csv(raiz / "data/derived/expanded_v1.csv", dtype={"caso": str})
        assert list(x.columns[:12]) == ["caso", "split", "seed", "frame", "score", "box_x1", "box_y1", "box_x2", "box_y2",
                                        "nivel", "metodo", "versao_config"]
        assert len(x) == 5 * 10 and x["offset"].abs().max() == 5 and (x["offset"] != 0).all()   # L_max 5, sem o seed
        assert (x["nivel"] == "alta").sum() == 25 and set(x["nivel"]) == {"alta", "baixa"}        # 6 × 4 janelas + o +4
        assert x["versao_config"].iloc[0].startswith("final_SAM2_L12.yaml sha256 ")
        assert (x.loc[x["caixa_original"] == 1, "offset"].abs() <= 2).all() and x["caixa_original"].sum() == 5 * 4

        pj = r["por_janela"]
        assert pj["alta"].sum() == 25 and pj["novos"].sum() == 9                  # 0101: ±3 e +4; as outras: ±3
        assert (pj["antes"] == 5).all() and pj["depois"].sum() == 25 + 9          # seed + 4 originais por janela
        pc = r["por_classe"]
        assert pc.loc["angiectasia", "novos"] == 3 and pc.loc["outras", "novos"] == 2 and pc.loc["erosao", "alta"] == 0
        t = r["texto"]
        assert "16 já tinham caixa original · 9 são novos · 1 sem caixa" in t
        assert "antes 25, depois 34 (×1,36)" in t and "janelas sem nenhum frame alta: 1" in t
        assert "Dataset_Treino: 3 janelas · alta 19 · novos 7" in t and "Dataset_Teste: 2 janelas · alta 6 · novos 2" in t
        assert "Máscaras: 4 ficheiros" in t and "não batem" in t
        assert (raiz / "outputs/15_a8_resumo.txt").is_file() and (raiz / "outputs/15_a8_por_classe.csv").is_file()

        assert "igual ao que já existia" in a8.main(base)["texto"]               # correr outra vez não muda nada
        p = pd.read_csv(pred, dtype={"caso": str})                                 # outro conteúdo: recusa sem --forcar
        p.loc[(p["caso"] == "0102") & (p["frame"] == 801), "score"] = 0.99       # um frame seguido (offset +1)
        p.to_csv(pred, index=False)
        for args, msg in ((base, "já existe com outro conteúdo"), (None, None)):
            if args is None:
                assert a8.main(base + ["--forcar"])["sha256"] != r["sha256"]
                continue
            try:
                a8.main(args)
                raise AssertionError("devia ter parado")
            except ev.ErroAvaliacao as e:
                assert msg in str(e)
        _previsoes(raiz, todas, sem="0103")                                       # uma janela sem previsões
        try:
            a8.main(base + ["--forcar"])
            raise AssertionError("devia ter parado")
        except ev.ErroAvaliacao as e:
            assert "sem previsões" in str(e) and "0103" in str(e)


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
