"""
Testes do scripts/18_c_analise.py (Guião C, C9: análise da revisão). Correm de duas maneiras, a partir da pasta do projeto:
    python tests/test_c_analise.py          (não precisa de pytest)
    python -m pytest tests/test_c_analise.py -q
Usam o mini-projeto inventado do tests/test_c_itens.py, a lista gravada pelo script 17 e respostas, PF v2, folha dos 50
FP e resultado do avaliador inventados, com as contas feitas à mão; nunca leem dados reais.
Versão 1.0 (24 out 2026).
"""
import importlib.util
import json
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]


def _modulo(nome, caminho):
    spec = importlib.util.spec_from_file_location(nome, caminho)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ca = _modulo("c_analise", RAIZ / "scripts" / "18_c_analise.py")
ti = _modulo("t_c_itens_para_18", RAIZ / "tests" / "test_c_itens.py")      # o mini-projeto do script 17
ci, ev = ti.ci, ca.ev
Saltar, saltar, erro = ti.Saltar, ti.saltar, ti.erro

COL_DEC = ["item", "split", "caso", "seed", "frame", "decisao", "nota", "box_aj_x1", "box_aj_y1", "box_aj_x2", "box_aj_y2",
           "modo", "dur_ms", "t_decisao", "sessao", "revisor"]
# descrições com mais de uma lesão (e uma com linfangiectasia sozinha, que não é angiectasia)
MISTAS = {"0202": "Erosões_e_pólipo_séssil", "0203": "Angiectasia | Sangue_em_pequena_quantidade",
          "0101": "Linfangiectasia_e_angiectasia", "0103": "Linfangiectasia"}
# perguntas pela ordem da folha (0202, 0203, 0101): «9» é inválido e repete; 2 = pólipo; Enter = não é claro; 2 = angiectasia
RESPOSTAS_CLASSES = ["9", "2", "", "2"]
CASOS = sorted({j[1] for j in ti.JANELAS})


def _entrada(respostas):
    fila = list(respostas)

    def f(_prompt=""):
        if not fila:
            raise KeyboardInterrupt
        return fila.pop(0)
    return f


def _nao_pergunta(_prompt=""):
    raise AssertionError("não devia perguntar")


def _decisao(r):
    """Respostas inventadas: −3, 3 e −8 aceites; 5 rejeitada; 7 sem decisão; a janela 0104 toda N com «Seed duvidoso».
    Controlos com caixa original: aceites só com IoU ≥ 0,5 (|offset| = 2); abaixo do limiar: rejeitados."""
    if r["tipo"] == "nova":
        if r["caso"] == "0104":
            return "nao_avaliavel", "Seed duvidoso"
        return {-3: "aceitar", 3: "aceitar", -8: "aceitar", 5: "rejeitar", 7: "nao_avaliavel"}[r["offset"]], ""
    if r["tipo"] == "controlo_original":
        return ("aceitar" if abs(r["offset"]) == 2 else "rejeitar"), ""
    return "rejeitar", ""


def _respostas(raiz, nome="decisoes_paulo.csv", mudar=None):
    g = Path(raiz) / "data/derived/guiao_c"
    ch = pd.read_csv(g / "itens_chave_v1.csv", dtype={"caso": str})
    linhas = []
    for r in ch.to_dict("records"):
        dec, nota = _decisao(r)
        dur = "" if r["item"] == 1 else 300000 if r["item"] == 2 else 800 + 37 * r["item"]
        linhas.append([r["item"], r["split"], r["caso"], r["seed"], r["frame"], dec, nota, "", "", "", "", "marcar", dur,
                       "2026-10-24T10:00:00", "s1", "paulo"])
    t = pd.DataFrame(linhas, columns=COL_DEC)
    if mudar:
        t = mudar(t, ch)
    t.to_csv(g / nome, index=False)
    return t


def _pf(raiz):
    """PF v2 inventada nas janelas de CAL e VAL-EX: visível até 3 frames do seed e em −8; D em +7; o resto não visível,
    menos o +3 da 0201 (não visível: a revisão aceitou-o, é a única discordância)."""
    linhas = []
    for sp, caso, seed, _, _ in ti.JANELAS:
        if caso not in ("0101", "0102", "0105", "0201", "0202", "0205"):
            continue
        for o in [x for x in range(-12, 13) if x]:
            v = "1" if abs(o) <= 3 or o == -8 else "duvidoso" if o == 7 else "0"
            if (caso, o) == ("0201", 3):
                v = "0"
            linhas.append({"split": sp, "caso": caso, "seed": seed, "frame": seed + o, "visivel": v})
    pd.DataFrame(linhas).to_csv(Path(raiz) / "data/derived/paulo_frame_labels.csv", index=False)


def _fp(raiz):
    seeds = {j[1]: (j[0], j[2]) for j in ti.JANELAS}
    fp = [("0201", -3, "lesao_nao_anotada"), ("0201", 5, "fundo_mucosa"), ("0202", 3, "fundo_mucosa"),
          ("0203", -3, "lesao_nao_anotada"), ("0203", 7, "nao_avaliavel"), ("0205", 4, "fundo_mucosa")]
    linhas = [{"n": i, "imagem": f"{i:02d}.png", "split": seeds[c][0], "caso": c, "seed": seeds[c][1],
               "frame": seeds[c][1] + o, "offset": o, "score": 0.95, "classificacao": cl, "nota": ""}
              for i, (c, o, cl) in enumerate(fp, 1)]
    p = Path(raiz) / "outputs/14_fp_inspecao/classificacao.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(linhas).to_csv(p, index=False)


def _avaliador(raiz):
    """Resultado do avaliador em TEST-BOX com previsões inventadas: na i-ésima janela, localiza i % 4 + 1 das 4 caixas
    originais (0201: 1, 0202: 2, 0203: 3, 0204: 4, 0205: 1, 0206: 2)."""
    raiz = Path(raiz)
    cfg = ev.ler_config(raiz / "configs/avaliacao.yaml")
    janelas = ev.ler_conjunto(raiz / "sets/TEST_BOX.json")
    ref, info = ev.carregar_referencia(cfg, "TEST_BOX", janelas)
    linhas = []
    for i, (sp, caso, seed) in enumerate(janelas):
        certos = [-2, -1, 1, 2][:i % 4 + 1]
        for o in [x for x in range(-12, 13) if x]:
            b = ti._caixa(0.1) if o in certos else (0.0, 0.0, 0.05, 0.05)
            linhas.append({"split": sp, "caso": caso, "seed": seed, "frame": seed + o, "score": 0.95,
                           "box_x1": b[0], "box_y1": b[1], "box_x2": b[2], "box_y2": b[3]})
    d, avisos = ev.juntar(ref, info, ev.ler_previsoes(pd.DataFrame(linhas)))
    res = {"metodo": "SAM2_L12", "conjunto": "TEST_BOX"}
    res.update(ev.avaliar(d, info, cfg, {"OP_F1": {"tau": 0.86, "a_min": 0}, "OP_R90": {"tau": 0.0, "a_min": 0}},
                          False, True, avisos))
    ev._gravar_json(raiz / "outputs/eval/TEST_BOX/SAM2_L12.json", res)
    ev._gravar_json(raiz / "outputs/eval/VAL_EX/SAM2_L12.json",
                    {"por_op": {"OP_F1": {"micro": {"P": {"valor": 0.906, "ic95": [0.838, 0.959]}}}}})


def _imagens(raiz):
    """Imagens a sério (pequenas) no seed das janelas com duas lesões e duas imagens do relatório; sem Pillow, nada."""
    try:
        from PIL import Image
    except ImportError:
        return False
    for sp, caso, seed, _, _ in ti.JANELAS:
        if caso in ("0202", "0203", "0101"):
            gt = Path(raiz) / "ds" / sp / caso / "GT_com_classes"
            Image.new("RGB", (64, 64), (150, 60, 50)).save(gt / str(seed) / f"{caso}_{seed}_{seed}.jpg")
            if caso != "0203":                                            # a 0203 fica sem imagem do relatório
                Image.new("RGB", (48, 48), (90, 40, 40)).save(gt / f"imagem1_1match_at_videoframe{seed}_0.9.jpg")
    return True


def _montar(tmp, responder=True):
    raiz = ti._projeto(tmp)
    p = raiz / "data/derived/janelas_v1.csv"
    jt = pd.read_csv(p, dtype=str, keep_default_na=False)
    for caso, txt in MISTAS.items():
        jt.loc[jt["caso"] == caso, "descricao"] = txt
    jt.to_csv(p, index=False)
    ci.main(ti._args(raiz))                                              # a lista da revisão (script 17)
    _respostas(raiz)
    _pf(raiz)
    _fp(raiz)
    _avaliador(raiz)
    _imagens(raiz)
    if responder:
        ca.main(_args18(raiz, "--preparar-classes"), entrada=_entrada(RESPOSTAS_CLASSES))
    return raiz


def _args18(raiz, *extra, saida="18_c_analise", respostas="decisoes_paulo.csv", classes="classes_mistas.csv"):
    raiz, g = Path(raiz), Path(raiz) / "data/derived/guiao_c"
    return ["--config", str(raiz / "configs/avaliacao.yaml"), "--itens", str(g / "itens_v1.csv"),
            "--chave", str(g / "itens_chave_v1.csv"), "--respostas", str(g / respostas), "--classes", str(g / classes),
            "--fp", str(raiz / "outputs/14_fp_inspecao/classificacao.csv"),
            "--eval-testbox", str(raiz / "outputs/eval/TEST_BOX/SAM2_L12.json"), "--saida", str(raiz / "outputs" / saida),
            "--ds", str(raiz / "ds"), *extra]


def _v(r):
    return r["precisao"]["valor"]


# ------------------------------------------------------------------ 1) classes das janelas com duas lesões
def test_lesoes_no_texto_pela_ordem_da_primeira_mencao():
    assert ca.lesoes_no_texto("Úlcera_e_erosões") == ["ulcera", "erosao"]
    assert ca.lesoes_no_texto("Linfangiectasia") == ["linfangiectasia"]
    assert ca.lesoes_no_texto("Linfangiectasia_e_angiectasia") == ["linfangiectasia", "angiectasia"]
    assert ca.lesoes_no_texto("Angiodisplasia | coágulo") == ["angiectasia", "sangue"]
    assert ca.lesoes_no_texto("Pólipo") == ["polipo"] and ca.lesoes_no_texto("") == []


def test_preparar_classes_folha_imagem_e_perguntas():
    with tempfile.TemporaryDirectory() as tmp:
        raiz = _montar(tmp, responder=False)
        folha = raiz / "data/derived/guiao_c/classes_mistas.csv"
        # interrompido depois da primeira resposta: fica gravada e só se pergunta o resto
        try:
            ca.main(_args18(raiz, "--preparar-classes"), entrada=_entrada(["2"]))
            raise AssertionError("devia ter parado")
        except KeyboardInterrupt:
            pass
        t = pd.read_csv(folha, dtype=str, keep_default_na=False)
        assert list(t.columns) == ca.COL_CLASSES and t["caso"].tolist() == ["0202", "0203", "0101"]
        assert t["opcoes"].tolist() == ["erosao|polipo", "angiectasia|sangue", "linfangiectasia|angiectasia"]
        assert t["primeira"].tolist() == ["erosao", "angiectasia", "linfangiectasia"]
        assert t["classe_caixa"].tolist() == ["polipo", "", ""]
        r = ca.main(_args18(raiz, "--preparar-classes"), entrada=_entrada(["", "x", "2"]))
        assert r["tabela"]["classe_caixa"].tolist() == ["polipo", "primeira", "angiectasia"]
        assert r["contagens"] == {"janelas": 3, "escolhidas": 2, "nao_claras": 1, "por_responder": 0, "mudam": 1}
        # tudo respondido: não pergunta nem reescreve
        antes = folha.read_bytes()
        ca.main(_args18(raiz, "--preparar-classes"), entrada=_nao_pergunta)
        assert folha.read_bytes() == antes
        over, _ = ca.ler_classes(folha, ev.ler_config(raiz / "configs/avaliacao.yaml"))
        assert over == {("Dataset_Teste", "0202", 900): "polipo", ("Dataset_Teste", "0203", 1200): "angiectasia",
                        ("Dataset_Treino", "0101", 500): "angiectasia"}
        if r["figura"] is not None:
            from PIL import Image
            assert Image.open(r["figura"]).size[0] > 500
        # folha com outras janelas: recusa
        pd.read_csv(folha, dtype=str, keep_default_na=False).iloc[:2].to_csv(folha, index=False)
        erro(ca.main, _args18(raiz, "--preparar-classes"), contem="não tem as janelas")


# ------------------------------------------------------------------ 2) análise
def test_precisao_com_as_contas_feitas_a_mao():
    with tempfile.TemporaryDirectory() as tmp:
        raiz = _montar(tmp)
        r = ca.main(_args18(raiz))
        R = r["R"]
        # teste: 0201 (A3 R1 N1), 0202 (A2 R1 N1), 0203 (A3 R1 N1), 0204 (A2 R1 N1)
        t = R[("teste", "todas")]
        assert (t["aceitar"], t["rejeitar"], t["nao_decidir"], t["janelas"]) == (10, 4, 4, 4)
        assert _v(t) == round(10 / 14, 6) and t["precisao_n_rejeitadas"]["valor"] == round(10 / 18, 6)
        lo, hi = t["precisao"]["ic95"]
        assert lo <= _v(t) <= hi
        # treino sem CAL: 0103 (A3 R1 N1) e 0104 (toda N); total
        assert _v(R[("treino_sem_cal", "todas")]) == 0.75
        assert R[("treino_sem_cal", "todas")]["precisao_n_rejeitadas"]["valor"] == round(3 / 9, 6)
        assert _v(R[("total", "todas")]) == 0.72                                       # 18 / (18 + 7)
        # estratos: distância, classe (com a classe nova da 0202), tamanho e frames sem ficheiro de caixa
        assert _v(R[("teste", "distancia 1-3")]) == 1.0 and _v(R[("teste", "distancia 4-6")]) == 0.0
        assert _v(R[("teste", "distancia 7-12")]) == 1.0                               # −8 aceites; +7 sem decisão
        assert _v(R[("teste", "classe polipo")]) == round(5 / 7, 6)                    # 0201 + 0202
        assert _v(R[("teste", "classe erosao")]) is None and _v(R[("teste", "classe outras")]) == round(2 / 3, 6)
        assert _v(R[("total", "classe angiectasia")]) == 0.75 and _v(R[("total", "classe xantelasma")]) is None
        assert _v(R[("teste", "tamanho pequena")]) == round(5 / 7, 6)                  # 0201 + 0204
        assert _v(R[("total", "sem ficheiro de caixa")]) == 1.0
        assert R[("total", "sem ficheiro de caixa")]["aceitar"] == ti.N_SEM_FICHEIRO
        # o bootstrap é reprodutível
        ca.main(_args18(raiz, saida="outra_vez"))
        assert (raiz / "outputs/18_c_analise/precisao.csv").read_bytes() == (raiz / "outputs/outra_vez/precisao.csv").read_bytes()


def test_controlos_e_concordancias():
    with tempfile.TemporaryDirectory() as tmp:
        raiz = _montar(tmp)
        r = ca.main(_args18(raiz))
        c = r["controlos"].set_index("tipo")
        assert c.loc["controlo_baixa", "taxa_aceitacao"] == 0.0 and c.loc["controlo_baixa", "caixas"] == 3
        ch = pd.read_csv(raiz / "data/derived/guiao_c/itens_chave_v1.csv", dtype={"caso": str})
        co = ch[ch["tipo"] == "controlo_original"]
        assert c.loc["controlo_original", "aceitar"] == int((co["offset"].abs() == 2).sum())
        k = r["concordancias"].set_index("comparacao")
        # IoU da caixa do SAM 2 com a original: 0,49 a 1 frame do seed, 0,81 a 2; aceites só a 2 → concordância total
        x = k.loc["controlos com caixa original: aceitar × IoU ≥ 0,5"]
        assert x["n"] == 3 and x["concordancia"] == 1.0
        # PF v2: 14 caixas comparáveis, 13 concordantes (κ = 72/86)
        x = k.loc["caixas novas: revisão × PF v2"]
        assert (x["n"], x["a_sim_b_sim"], x["a_sim_b_nao"], x["a_nao_b_nao"]) == (14, 9, 1, 4)
        assert abs(x["concordancia"] - 13 / 14) < 1e-12 and abs(x["kappa"] - 72 / 86) < 1e-12
        # 50 FP: 5 na lista, 4 comparáveis, 3 concordantes, κ 0,5
        x = k.loc["caixas novas: revisão × 50 FP (20–22 out)"]
        assert x["n"] == 4 and x["concordancia"] == 0.75 and abs(x["kappa"] - 0.5) < 1e-12
        assert "5 dos 6 FP estão na lista" in r["texto"]


def test_ganho_tempo_e_seed_duvidoso():
    with tempfile.TemporaryDirectory() as tmp:
        raiz = _montar(tmp)
        r = ca.main(_args18(raiz))
        g = r["ganho"].set_index(["grupo", "grupo_valor"])
        # antes = seed + 4 caixas originais (−2..2) em cada uma das 14 janelas; aceites = 18
        assert tuple(g.loc[("total", "total"), ["janelas", "antes", "propostas", "aceites", "depois"]]) == (14, 70, 36, 18, 88)
        assert abs(g.loc[("total", "total"), "pct"] - 100 * 18 / 70) < 1e-9
        assert tuple(g.loc[("classe", "polipo"), ["janelas", "antes", "aceites"]]) == (4, 20, 5)      # com a 0202
        assert tuple(g.loc[("classe", "erosao"), ["janelas", "antes", "aceites"]]) == (2, 10, 0)
        assert g.loc[("split", "Dataset_Teste"), "aceites"] == 10
        # tempo: sem as pausas (> 120 s) nem as caixas sem tempo
        d = _respostas(raiz)
        s = pd.to_numeric(d["dur_ms"], errors="coerce")
        s = s[s <= 120000] / 1000
        t0 = r["tempo"].iloc[0]
        assert t0["caixas"] == len(s) and abs(t0["mediana_s"] - s.median()) < 1e-9
        assert abs(t0["p90_s"] - s.quantile(0.9)) < 1e-9 and abs(t0["total_min"] - s.sum() / 60) < 1e-9
        assert (r["pausas"], r["sem_tempo"]) == (1, 1)
        assert r["seed_duvidoso"] == 1 and r["incoerentes"] == 0

        # uma caixa de uma janela «seed duvidoso» que não ficou em N: avisa
        def uma_aceite(t, ch):
            i = t.index[(t["caso"] == "0104") & (t["nota"] == "Seed duvidoso")][0]
            t.loc[i, "decisao"] = "aceitar"
            return t
        _respostas(raiz, "decisoes_v2.csv", uma_aceite)
        r2 = ca.main(_args18(raiz, respostas="decisoes_v2.csv", saida="v2"))
        assert r2["incoerentes"] == 1 and "1 caixa nova dessas janelas não está em N" in r2["texto"]


def test_guiao_a_por_classe_com_a_classe_nova():
    with tempfile.TemporaryDirectory() as tmp:
        raiz = _montar(tmp)
        r = ca.main(_args18(raiz))
        ga = r["guiao_a"].set_index("classe")
        assert "1 das 6 janelas de TEST-BOX muda de classe" in r["guiao_a_txt"]
        assert "confere com a do avaliador" in r["guiao_a_txt"] and "NÃO" not in r["guiao_a_txt"]
        # localizadas: 0201 1, 0202 2, 0203 3, 0204 4, 0205 1, 0206 2 (de 4 caixas cada)
        assert ga.loc["polipo", "rloc_antes"] == 0.25 and ga.loc["polipo", "rloc_agora"] == round(4 / 12, 6)
        assert (ga.loc["polipo", "caixas_antes"], ga.loc["polipo", "caixas_agora"]) == (8, 12)
        assert ga.loc["erosao", "rloc_antes"] == 0.5 and ga.loc["erosao", "rloc_agora"] == 0.5
        assert ga.loc["erosao", "janelas_agora"] == 1 and ga.loc["outras", "rloc_agora"] == 1.0
        assert pd.isna(ga.loc["ulcera", "rloc_agora"])
        # o mesmo ficheiro com um estrato mexido: avisa que não confere
        p = raiz / "outputs/eval/TEST_BOX/SAM2_L12.json"
        res = json.loads(p.read_text(encoding="utf-8"))
        res["por_op"]["OP_F1"]["estratos"]["classe"]["polipo"]["R_loc"]["valor"] = 0.3
        p.write_text(json.dumps(res), encoding="utf-8")
        assert "NÃO confere" in ca.main(_args18(raiz, saida="mexido"))["guiao_a_txt"]


def test_recusa_respostas_e_classes_erradas():
    with tempfile.TemporaryDirectory() as tmp:
        raiz = _montar(tmp)
        erro(ca.main, _args18(raiz, respostas="nao_existe.csv"), contem="consolida primeiro")
        _respostas(raiz, "falta.csv", lambda t, ch: t.iloc[1:])
        erro(ca.main, _args18(raiz, respostas="falta.csv"), contem="exatamente uma resposta por caixa")

        def outro_frame(t, ch):
            t.loc[0, "frame"] = int(t.loc[0, "frame"]) + 100
            return t
        _respostas(raiz, "outra_lista.csv", outro_frame)
        erro(ca.main, _args18(raiz, respostas="outra_lista.csv"), contem="não batem com a chave")

        def talvez(t, ch):
            t.loc[0, "decisao"] = "talvez"
            return t
        _respostas(raiz, "talvez.csv", talvez)
        erro(ca.main, _args18(raiz, respostas="talvez.csv"), contem="respostas desconhecidas")
        erro(ca.main, _args18(raiz, classes="nao_existe.csv"), contem="--preparar-classes")
        g = raiz / "data/derived/guiao_c"
        t = pd.read_csv(g / "classes_mistas.csv", dtype=str, keep_default_na=False)
        t.loc[0, "classe_caixa"] = "úlcera"
        t.to_csv(g / "classes_erradas.csv", index=False)
        erro(ca.main, _args18(raiz, classes="classes_erradas.csv"), contem="não é uma das lesões do texto")
        t.loc[0, "classe_caixa"] = "Pólipo"                           # com acento e maiúscula também serve
        t.to_csv(g / "classes_acento.csv", index=False)
        over, _ = ca.ler_classes(g / "classes_acento.csv", ev.ler_config(raiz / "configs/avaliacao.yaml"))
        assert over[("Dataset_Teste", "0202", 900)] == "polipo"


def test_resumo_ficheiros_e_linha_para_o_decisoes():
    with tempfile.TemporaryDirectory() as tmp:
        raiz = _montar(tmp)
        r = ca.main(_args18(raiz))
        s = raiz / "outputs/18_c_analise"
        for f in ("resumo.md", "precisao.csv", "controlos.csv", "concordancias.csv", "ganho.csv", "tempo.csv",
                  "guiao_a_rloc_classe.csv"):
            assert (s / f).is_file(), f
        if importlib.util.find_spec("matplotlib"):
            assert all((s / f"{n}.{e}").is_file() for n in ("fig_c_precisao", "fig_c_ganho") for e in ("png", "pdf"))
        txt = (s / "resumo.md").read_text(encoding="utf-8")
        assert "## Linha para o docs/decisoes.md" in txt and r["linha"] in txt
        assert "Teste (número principal): 0,714 [" in txt and "70 → 88 (+25,7 %)" in txt
        assert "no teste 0,714 [" in r["linha"] and "Seed duvidoso em 1 janela." in r["linha"]
        assert "2 pela lesão da caixa do seed, 1 pela primeira mencionada" in r["linha"]
        assert "2 com a lesão da caixa do seed, 1 não clara (fica a primeira mencionada), 0 por responder; 1 muda" in txt
        assert "ficam fora 1 pausa de mais de 120 s e 1 caixa sem tempo" in txt
        # só números agregados: nenhum caso aparece no resumo
        for caso in CASOS:
            assert not re.search(rf"\b{caso}\b", txt), f"o caso {caso} aparece no resumo"
        pr = pd.read_csv(s / "precisao.csv")
        assert set(pr["subconjunto"]) == {"teste", "treino_sem_cal", "total"}
        assert np.isclose(pr.loc[(pr["subconjunto"] == "teste") & (pr["estrato"] == "todas"), "precisao"].iloc[0], 10 / 14,
                          atol=1e-6)


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
