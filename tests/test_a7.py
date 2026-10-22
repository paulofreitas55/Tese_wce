"""
Testes do A7: tabelas e figuras (scripts/13_a7_tabelas.py) e inspeção dos FP aparentes (scripts/14_fp_inspecao.py),
num mini-projeto avaliado com o avaliador verdadeiro (calibrar, avaliar, comparar). Nunca leem dados reais.
    python tests/test_a7.py          ou          python -m pytest tests/ -q
"""
import gzip
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI.parent / "methods"))
from test_evaluate import Saltar, _projeto, ev  # noqa: E402


def _carregar(nome, ficheiro):
    spec = importlib.util.spec_from_file_location(nome, AQUI.parent / "scripts" / ficheiro)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _montar(tmp):
    """Mini-projeto com três métodos avaliados como no A7. SAM2_L12 (principal) vê |offset| ≤ 3 com a caixa certa;
    B1 vê só |offset| = 1 e não dá caixas; SAM2_multi_L12 só em VAL_EX. Na referência PF a lesão está visível a ≤ 3
    frames; em TEST_BOX há caixas em −2..+2 e "vazio" no resto, por isso os frames a ±3 do SAM2 são FP aparentes."""
    try:
        import yaml  # noqa: F401
    except ImportError:
        raise Saltar("falta o PyYAML")
    raiz, conj, pred = _projeto(tmp)
    cfg = str(raiz / "configs/avaliacao.yaml")
    corre = lambda *a: ev.main(list(a) + ["--config", cfg])                # noqa: E731
    sem = ("", "", "", "")
    deteta = {"SAM2_L12": ({1, 2, 3}, (0.30, 0.30, 0.50, 0.50)), "B1_endofm_cls": ({1}, sem),
              "SAM2_multi_L12": ({1, 2}, (0.30, 0.30, 0.50, 0.50))}
    prev = {}
    for m, (d, cx) in deteta.items():
        assert corre("calibrar", "--metodo", m, "--pred", pred(f"{m}_CAL", conj["CAL"], d, cx)) == 0
        for c in ("VAL_EX",) if m == "SAM2_multi_L12" else ("VAL_EX", "TEST_BOX"):
            prev[(m, c)] = pred(f"{m}_{c}", conj[c], d, cx)
            assert corre("avaliar", "--metodo", m, "--conjunto", c, "--pred", prev[(m, c)]) == 0
    assert corre("comparar", "--conjunto", "VAL_EX", "--a", "SAM2_L12", "--b", "B1_endofm_cls") == 0
    assert corre("comparar", "--conjunto", "TEST_BOX", "--a", "SAM2_L12", "--b", "B1_endofm_cls") == 0
    (raiz / "configs/a6.yaml").write_text(
        "versao: 1\nprincipal: SAM2_L12\ncriterio2_op: OP_F1\nconjuntos: [VAL_EX, TEST_BOX]\nmetodos:\n"
        "  - {metodo: B1_endofm_cls, linha: B1 global EndoFM}\n"
        "  - {metodo: SAM2_L12, linha: SAM2 box}\n"
        "  - {metodo: SAM2_multi_L12, linha: SAM2 multi-prompt, conjuntos: [VAL_EX]}\n", encoding="utf-8")
    return raiz, conj, cfg, prev


def _a7_yaml(raiz, prev, r_loc_minimo=0.70):
    p = raiz / "configs/a7.yaml"
    p.write_text(
        "versao: 1\nfiguras:\n  metodos: [SAM2_L12, B1_endofm_cls]\n"
        f"go_no_go: {{b1: B1_endofm_cls, r_loc_minimo: {r_loc_minimo}, precisao_minima: 0.60}}\n"
        f"fp_inspecao: {{metodo: SAM2_L12, previsoes: {prev[('SAM2_L12', 'TEST_BOX')]}, "
        f"mascaras: {raiz / 'outputs/masks/TEST_BOX'}, saida: {raiz / 'outputs/14_fp_inspecao'}}}\n", encoding="utf-8")
    return str(p)


def _args13(raiz, cfg, a7):
    return ["--config", cfg, "--config-a6", str(raiz / "configs/a6.yaml"), "--config-a7", a7,
            "--saida", str(raiz / "outputs/13_a7")]


def test_a7_tabelas_ponta_a_ponta():
    t13 = _carregar("a7", "13_a7_tabelas.py")
    with tempfile.TemporaryDirectory() as tmp:
        raiz, conj, cfg, prev = _montar(tmp)
        r = t13.main(_args13(raiz, cfg, _a7_yaml(raiz, prev)))
        out = raiz / "outputs/13_a7"

        m = r["mestre"].set_index(["metodo", "conjunto", "op", "metrica"])["valor"]
        val = json.loads((raiz / "outputs/eval/VAL_EX/SAM2_L12.json").read_text())
        assert m[("SAM2_L12", "VAL_EX", "OP_F1", "P_micro")] == 1.0                  # vê exatamente os visíveis
        assert m[("SAM2_L12", "VAL_EX", "-", "AP")] == val["AP"]["valor"]
        assert m[("SAM2_L12", "TEST_BOX", "OP_F1", "R_loc")] == 1.0                  # caixas certas em −2..+2
        assert abs(m[("B1_endofm_cls", "VAL_EX", "OP_F1", "R_micro")] - 1 / 3) < 1e-6    # 2 dos 6 visíveis por janela
        assert ("SAM2_multi_L12", "TEST_BOX") not in {(a, b) for a, b, _, _ in m.index}

        c = r["comparacoes"].set_index("conjunto")
        assert c.loc["VAL_EX", "linha_a"] == "SAM2 box" and c.loc["VAL_EX", "linha_b"] == "B1 global EndoFM"
        assert abs(c.loc["VAL_EX", "diferenca"] - 0.5) < 1e-6 and c.loc["VAL_EX", "diferentes"]   # F1 1 contra 0,5
        assert c.loc["TEST_BOX", "metrica"] == "R_loc"

        assert [ok for _, _, ok in r["criterios"]] == [True, True, True]
        assert "Resultado pelos critérios: go" in r["texto"] and "+0,500 [+0,500; +0,500] → cumpre" in r["texto"]
        e = r["estratos"]
        d = e[(e["metodo"] == "SAM2_L12") & (e["conjunto"] == "VAL_EX") & (e["op"] == "OP_F1")
              & (e["tipo"] == "distancia") & (e["metrica"] == "R")].set_index("estrato")["valor"]
        assert d["1-3"] == 1.0 and pd.isna(d["4-6"])                     # o ±4 é dúvida e não há visíveis além dele
        assert "#### SAM2 box por distância ao seed (frames) (OP-F1)" in r["texto"]
        assert "Ainda não" in r["texto"] and r["fp"] is None                       # sem folha de FP
        f = pd.read_csv(out / "fontes.csv")
        assert len(f) == 5 + 2 + 2 and f["sha256"].str.len().eq(64).all()           # 5 avaliar, 2 CAL, 2 comparar
        for nome in ("a7_resumo.md", "tabela_mestre.csv", "comparacoes.csv", "estratos.csv"):
            assert (out / nome).is_file()
        try:
            import matplotlib  # noqa: F401
            assert {p.name for p in r["figuras"]} == {f"{n}.{e}" for n in ("fig_pr", "fig_distancia", "fig_classe_tamanho")
                                                       for e in ("png", "pdf")}
        except ImportError:
            assert "falta o matplotlib" in r["texto"]

        # critério 2 com um limite que o SAM2 não atinge → no-go
        r2 = t13.main(_args13(raiz, cfg, _a7_yaml(raiz, prev, 1.01)) + ["--sem-figuras"])
        assert [ok for _, _, ok in r2["criterios"]] == [True, False, True]
        assert "no-go (falha o critério 2)" in r2["texto"]


def test_a7_fp_inspecao_e_resumo():
    try:
        import cv2
    except ImportError:
        raise Saltar("falta o OpenCV")
    t13, t14 = _carregar("a7", "13_a7_tabelas.py"), _carregar("fp", "14_fp_inspecao.py")
    from sam2_prop import rle
    with tempfile.TemporaryDirectory() as tmp:
        raiz, conj, cfg, prev = _montar(tmp)
        a7 = _a7_yaml(raiz, prev)
        ds = raiz / "ds"
        rng = np.random.default_rng(0)
        for sp, caso, seed, _ in conj["TEST_BOX"]:
            pasta = ds / sp / caso / "GT_com_classes" / str(seed)
            pasta.mkdir(parents=True)
            for f in range(seed - 10, seed + 11):
                cv2.imwrite(str(pasta / f"{caso}_{f}_{seed}.jpg"), rng.integers(0, 255, (64, 80, 3), dtype=np.uint8))
        sp, caso, seed, _ = conj["TEST_BOX"][0]                                    # máscara só no frame +3 da 1.ª
        mk = np.zeros((64, 80), dtype=bool)
        mk[20:32, 24:40] = True
        (raiz / "outputs/masks/TEST_BOX").mkdir(parents=True)
        (raiz / f"outputs/masks/TEST_BOX/{sp}__{caso}__{seed}.json.gz").write_bytes(gzip.compress(json.dumps(
            {"tamanho": [64, 80], "frames": {str(seed + 3): rle(mk)}}).encode("utf-8")))
        base = ["--ds", str(ds), "--config", cfg, "--config-a7", a7, "--lado", "120"]

        r = t14.main(base)
        folha = pd.read_csv(r["folha"], dtype=str, keep_default_na=False)
        assert len(folha) == 4 and set(folha["offset"]) == {"-3", "3"}           # ±3 nas duas janelas
        assert (folha["classificacao"] == "").all() and len(r["imagens"]) == 4 and not r["sem_imagem"]
        img = cv2.imread(str(r["imagens"][0]))
        assert img.shape == (30 + 2 * (26 + 120) + 6, 2 * 120 + 6, 3)
        assert t14.recorte((10, 10, 20, 20), 80, 64) == (0, 0, 48)               # quadrado mínimo, dentro da imagem
        assert t14.ler_mascara(raiz / "outputs/masks/TEST_BOX", sp, caso, seed, seed + 3).sum() == mk.sum()

        folha["classificacao"] = ["lesao_nao_anotada", "lesao_nao_anotada", "fundo_mucosa", ""]
        folha.to_csv(r["folha"], index=False)
        assert t14.main(base)["reescrita"] is False                               # não apaga o trabalho feito
        assert pd.read_csv(r["folha"], dtype=str, keep_default_na=False)["classificacao"].iloc[0] == "lesao_nao_anotada"

        res = t13.main(_args13(raiz, cfg, a7) + ["--sem-figuras"])
        fp = res["fp"].set_index("classe")
        assert fp.loc["lesao_nao_anotada", "n"] == 2 and fp.loc["fundo_mucosa", "n"] == 1
        assert abs(fp.loc["lesao_nao_anotada", "fracao"] - 2 / 3) < 1e-9
        lo, hi = t13.wilson(2, 3)
        assert abs(lo - 0.2077) < 1e-3 and abs(hi - 0.9385) < 1e-3                 # Wilson 95 % para 2/3
        assert "3 de 4 FP classificados" in res["texto"]

        folha.loc[3, "classificacao"] = "bolha"
        folha.to_csv(r["folha"], index=False)
        try:
            t13.main(_args13(raiz, cfg, a7) + ["--sem-figuras"])
            raise AssertionError("devia ter parado")
        except ev.ErroAvaliacao as e:
            assert "bolha" in str(e)

        Path(prev[("SAM2_L12", "TEST_BOX")]).write_text("mudou\n", encoding="utf-8")   # previsões trocadas
        try:
            t14.main(base)
            raise AssertionError("devia ter parado")
        except ev.ErroAvaliacao as e:
            assert "não é o ficheiro avaliado" in str(e)


def test_a7_exemplos():
    """16_a7_exemplos.py: obrigatórios (par e caso inteiro), nota dos FP, escolhas automáticas nas janelas de teste."""
    try:
        import cv2
        import matplotlib  # noqa: F401
        import yaml  # noqa: F401
    except ImportError:
        raise Saltar("falta o OpenCV, o matplotlib ou o PyYAML")
    t16 = _carregar("ex", "16_a7_exemplos.py")
    from sam2_prop import rle
    with tempfile.TemporaryDirectory() as tmp:
        raiz, conj, _ = _projeto(tmp)
        todas = conj["CAL"] + conj["VAL_EX"]
        (raiz / "sets/ALL_SEEDBOX.json").write_text(json.dumps(
            {"nome": "ALL_SEEDBOX", "janelas": [{"split": s, "caso": c, "seed": e} for s, c, e, _ in todas]}))
        (raiz / "configs/final_SAM2_L12.yaml").write_text(
            "metodo: SAM2_L12\nparametros: {prompt: caixa, l_max: 12}\n"
            "decisoes_A6: {niveis_A8: {alta: {score_minimo: 0.86}, media: null}}\n", encoding="utf-8")
        certa, errada, linhas = (0.30, 0.30, 0.50, 0.50), (0.60, 0.60, 0.80, 0.80), []
        for sp, caso, seed, _ in todas:                          # 0202 (erosão) com a caixa no sítio errado
            for o in range(-10, 11):
                if o:
                    s, b = (0.95, errada if caso == "0202" else certa) if abs(o) <= 3 else (0.4, certa)
                    linhas.append({"split": sp, "caso": caso, "seed": seed, "frame": seed + o, "score": s,
                                   **dict(zip(["box_x1", "box_y1", "box_x2", "box_y2"], b)), "area_px": 300})
        pred = raiz / "outputs/preds/pred_SAM2_L12_ALL_SEEDBOX.csv"
        pred.parent.mkdir(parents=True)
        pd.DataFrame(linhas).to_csv(pred, index=False)
        ds, rng = raiz / "ds", np.random.default_rng(0)
        for sp, caso, seed, _ in todas:
            pasta = ds / sp / caso / "GT_com_classes" / str(seed)
            pasta.mkdir(parents=True)
            for f in range(seed - 10, seed + 11):
                cv2.imwrite(str(pasta / f"{caso}_{f}_{seed}.jpg"), rng.integers(0, 255, (64, 80, 3), dtype=np.uint8))
        mk = np.zeros((64, 80), dtype=bool)
        mk[20:32, 24:40] = True
        (raiz / "outputs/masks/ALL_SEEDBOX").mkdir(parents=True)
        (raiz / "outputs/masks/ALL_SEEDBOX/Dataset_Teste__0201__600.json.gz").write_bytes(gzip.compress(json.dumps(
            {"tamanho": [64, 80], "frames": {"601": rle(mk)}}).encode("utf-8")))
        (raiz / "outputs/14_fp_inspecao").mkdir(parents=True)
        pd.DataFrame([{"n": 1, "imagem": "x.png", "split": "Dataset_Teste", "caso": "0201", "seed": 600, "frame": 603,
                       "offset": 3, "score": 0.95, "classificacao": "lesao_nao_anotada", "nota": " exemplo_bom"}]
                     ).to_csv(raiz / "outputs/14_fp_inspecao/classificacao.csv", index=False)
        a7 = raiz / "configs/a7.yaml"
        a7.write_text(
            f"fp_inspecao: {{saida: {raiz / 'outputs/14_fp_inspecao'}}}\n"
            "exemplos:\n  casos: [[\"0201\", 600], [\"9999\", 1], \"0101\"]\n"
            "  offsets: [-12, -8, -4, -2, 0, 2, 4, 8, 12]\n"
            f"  previsoes: {pred}\n  mascaras: {raiz / 'outputs/masks/ALL_SEEDBOX'}\n"
            f"  saida: {raiz / 'outputs/16_a7_exemplos'}\n", encoding="utf-8")

        r = t16.main(["--ds", str(ds), "--config", str(raiz / "configs/avaliacao.yaml"), "--config-a7", str(a7),
                      "--final", str(raiz / "configs/final_SAM2_L12.yaml")])
        ind = r["indice"].set_index("caso")
        assert list(r["indice"]["caso"]) == ["0201", "0101", "0202"] and r["fora"] == ["9999/1"]
        assert ind.loc["0201", "motivos"] == "obrigatório · marcado nos FP · acerto"
        assert ind.loc["0202", "motivos"] == "falha · falha em lesão plana"
        assert ind.loc["0201", "R_loc_janela"] == 1.0 and ind.loc["0202", "R_loc_janela"] == 0.0
        assert ind.loc["0101", "R_loc_janela"] == ""                       # treino: sem escolha automática
        assert "+3" in ind.loc["0201", "offsets"] and "+2" not in ind.loc["0201", "offsets"]   # o frame do FP entra
        for f in ind["ficheiro"]:
            assert (raiz / "outputs/16_a7_exemplos" / f).is_file()
            assert (raiz / "outputs/16_a7_exemplos" / f).with_suffix(".pdf").is_file()


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
