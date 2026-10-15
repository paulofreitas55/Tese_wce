"""
Testes do A6 (scripts/12_a6_congelar.py): calibrações verdadeiras do avaliador num mini-projeto, YAML final por método
(pontos de operação, sha256 das previsões, decisões no principal), estabilidade com a sensibilidade à janela dominante,
tabela do nível "média" e os erros (previsões em falta, nível "média" fora da faixa). Nunca leem dados reais.
    python tests/test_a6.py          ou          python -m pytest tests/ -q
"""
import importlib.util
import sys
import tempfile
from pathlib import Path

import pandas as pd

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
from test_evaluate import Saltar, _projeto, ev  # noqa: E402

CAIXA = ["box_x1", "box_y1", "box_x2", "box_y2"]


def _carregar():
    spec = importlib.util.spec_from_file_location("a6", AQUI.parent / "scripts" / "12_a6_congelar.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _pred_principal(raiz, conj):
    """CAL: 0,95 a ≤ 2 frames do seed; 0,7 a 3 (menos na 0102, onde fica 0,05); 0,6 a 5; 0,05 no resto. No mini-
    projeto a lesão está visível a ≤ 3 frames, em dúvida no +4 e não visível depois. Logo: OP-F1 τ = 0,70 (F1 32/34) e
    a faixa [0,5; 0,7) só tem os frames a 5, todos errados."""
    linhas = []
    for sp, caso, seed, _ in conj["CAL"]:
        for o in range(-10, 11):
            if o == 0:
                continue
            s = 0.95 if abs(o) <= 2 else (0.7 if abs(o) == 3 and caso != "0102" else 0.6 if abs(o) == 5 else 0.05)
            cx = (0.3, 0.3, 0.5, 0.5) if s >= 0.5 else ("",) * 4
            linhas.append({"split": sp, "caso": caso, "seed": seed, "frame": seed + o, "score": s, **dict(zip(CAIXA, cx))})
    f = raiz / "pred_principal_CAL.csv"
    pd.DataFrame(linhas).to_csv(f, index=False)
    return f


def _a6_yaml(raiz, nivel_media="null"):
    (raiz / "configs/a6.yaml").write_text(
        "versao: 1\nprincipal: SAM2_L12\ncriterio2_op: OP_F1\n"
        f"nivel_media: {nivel_media}\nlimites_media: [0.5, 0.6, 0.8]\n"
        "janela_dominante: [Dataset_Treino, \"0102\", 800]\n"
        f"previsoes: {raiz / 'outputs/preds'}\nconjuntos: [VAL_EX, TEST_BOX]\nmetodos:\n"
        "  - {metodo: SAM2_L12, linha: SAM2 box, script: methods/sam2_prop.py, config: configs/sam2.yaml, "
        "parametros: {prompt: caixa, l_max: 12}}\n"
        "  - {metodo: SAM2_multi_L12, linha: SAM2 multi-prompt, script: methods/sam2_prop.py, config: configs/sam2.yaml, "
        "parametros: {prompt: multi, l_max: 12}, conjuntos: [VAL_EX]}\n", encoding="utf-8")
    return str(raiz / "configs/a6.yaml")


def test_a6_ponta_a_ponta():
    try:
        import yaml
    except ImportError:
        raise Saltar("falta o PyYAML")
    a6 = _carregar()
    with tempfile.TemporaryDirectory() as tmp:
        raiz, conj, pred = _projeto(tmp)
        cfg = str(raiz / "configs/avaliacao.yaml")
        assert ev.main(["calibrar", "--metodo", "SAM2_L12", "--pred", str(_pred_principal(raiz, conj)), "--config", cfg]) == 0
        assert ev.main(["calibrar", "--metodo", "SAM2_multi_L12", "--pred", pred("multi", conj["CAL"], (1, 2)),
                        "--config", cfg]) == 0
        (raiz / "outputs/preds").mkdir(parents=True)
        for nome in ("SAM2_L12_VAL_EX", "SAM2_L12_TEST_BOX", "SAM2_multi_L12_VAL_EX"):
            (raiz / f"outputs/preds/pred_{nome}.csv").write_text(f"previsões de teste de {nome}\n", encoding="utf-8")
        base = ["--config", cfg, "--saida-configs", str(raiz / "finais"), "--saida", str(raiz / "out")]

        r = a6.main(base + ["--config-a6", _a6_yaml(raiz)])
        f = yaml.safe_load((raiz / "finais/final_SAM2_L12.yaml").read_text(encoding="utf-8"))
        assert f["principal"] is True and f["linha_tabela_mestre"] == "SAM2 box"
        assert f["pontos_operacao"]["OP_F1"]["tau"] == 0.7 and f["pontos_operacao"]["OP_R90"]["tau"] >= 0
        assert f["previsoes"]["TEST_BOX"]["sha256"] == ev._sha256(raiz / "outputs/preds/pred_SAM2_L12_TEST_BOX.csv")
        assert f["gerado_por"]["script_sha256"] == ev._sha256(AQUI.parent / "methods/sam2_prop.py")
        assert f["decisoes_A6"] == {"criterio2_op": "OP_F1", "niveis_A8": {"alta": {"score_minimo": 0.7}, "media": None}}
        m = yaml.safe_load((raiz / "finais/final_SAM2_multi_L12.yaml").read_text(encoding="utf-8"))
        assert m["principal"] is False and m["conjuntos_A7"] == ["VAL_EX"] and "decisoes_A6" not in m

        est = r["estabilidade"].set_index("metodo")
        assert sorted(est.index) == ["SAM2_L12", "SAM2_multi_L12"]
        e = est.loc["SAM2_L12"]                                             # janela dominante: a 0102 (2 FN)
        assert abs(e["F1_micro"] - 32 / 34) < 1e-9 and e["F1_micro_sem_dominante"] == 1.0
        assert (e["positivos"], e["positivos_dominante"], e["dominante_FN"]) == (18, 6, 2)

        md = r["media"].set_index("tau_inferior")
        assert sorted(md.index) == [0.5, 0.6]                               # 0,8 não fica abaixo do τ do OP-F1
        assert tuple(md.loc[0.5, ["frames", "certos", "errados"]]) == (6, 0, 6) and md.loc[0.5, "precisao"] == 0
        assert abs(md.loc[0.5, "recall_so_alta"] - 16 / 18) < 1e-9
        texto = (raiz / "out/12_a6_resumo.txt").read_text(encoding="utf-8")
        assert "Nível \"média\" do SAM2_L12" in texto and "git tag a6-congelado" in texto

        a6.main(base + ["--config-a6", _a6_yaml(raiz, "0.5")])              # com nível "média"
        f = yaml.safe_load((raiz / "finais/final_SAM2_L12.yaml").read_text(encoding="utf-8"))
        assert f["decisoes_A6"]["niveis_A8"]["media"] == {"score_minimo": 0.5, "score_abaixo_de": 0.7}
        for nm, falta, msg in (("0.8", None, "nivel_media"), ("null", "SAM2_L12_TEST_BOX", "Faltam as previsões")):
            if falta:
                (raiz / f"outputs/preds/pred_{falta}.csv").unlink()
            try:
                a6.main(base + ["--config-a6", _a6_yaml(raiz, nm)])
                raise AssertionError("devia ter parado")
            except ev.ErroAvaliacao as erro:
                assert msg in str(erro)


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
