"""
Testes do scripts/19_c_dataset_v2.py (Guião C, C10: Dataset WCE v2). Correm de duas maneiras, a partir da pasta do projeto:
    python tests/test_c_dataset_v2.py          (não precisa de pytest)
    python -m pytest tests/test_c_dataset_v2.py -q
Usam o mini-projeto inventado do tests/test_c_itens.py, com a revisão inventada do tests/test_c_analise.py (e a análise C9
corrida nele), mais duas janelas fora de ALL-SEEDBOX; as contas estão feitas à mão. Nunca leem dados reais.
Versão 1.0 (25 out 2026).
"""
import importlib.util
import os
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


cd = _modulo("c_dataset_v2", RAIZ / "scripts" / "19_c_dataset_v2.py")
ta = _modulo("t_c_analise_para_19", RAIZ / "tests" / "test_c_analise.py")     # mini-projeto, revisão e análise C9
ti, ca, ev = ta.ti, ta.ca, cd.ev
Saltar, saltar, erro = ti.Saltar, ti.saltar, ti.erro
K4 = ["split", "caso", "seed", "frame"]
CASOS = ta.CASOS + ["0109", "0110"]
# contas à mão nas 14 janelas de ALL-SEEDBOX (41 frames cada, caixas originais em −2..2, −8 sem ficheiro de caixa):
# caixas novas em −3, 3, 5, 7 e, em metade das janelas de revisão, −8; aceites −3, 3 e −8; rejeitada 5; N em 7; 0104 toda N
ESPERADO_A = {"frames": 574, "original": 70, "nova_aceite": 18, "nova_rejeitada": 7, "nova_nao_avaliavel": 11,
              "nova_mal_posta": 0, "sem_ficheiro": 10, "sem_caixa": 458, "invalida": 0}


def _mais_janelas(raiz):
    """Duas janelas fora de ALL-SEEDBOX: 0109, de não-lesão, na janelas_v1 (caixas originais em −2..1 e uma inválida em
    +2); 0110, de pólipo, que não está na janelas_v1 (só a caixa do seed)."""
    p = Path(raiz) / "data/derived/annotations_v1.csv"
    an = pd.read_csv(p, dtype=str, keep_default_na=False)
    linhas = []
    for caso, seed, cat, les in (("0109", 2000, "nao_lesao", 0), ("0110", 2500, "polipo", 1)):
        for f in range(seed - 20, seed + 21):
            o = f - seed
            perto = abs(o) <= 2 if caso == "0109" else o == 0
            src = ("medico_seed" if o == 0 else "medico_vizinho") if perto else "sem_ficheiro" if o == -8 else "vazio"
            b = ti._caixa(0.1) if perto and (caso, o) != ("0109", 2) else ("", "", "", "")
            linhas.append({"split": "Dataset_Treino", "caso": caso, "seed": seed, "frame": f, "offset": o, "box_src": src,
                           "box_x1": b[0], "box_y1": b[1], "box_x2": b[2], "box_y2": b[3], "img_w": 320, "img_h": 320,
                           "excluida": 0, "boxc_x1": "", "boxc_y1": "", "boxc_x2": "", "boxc_y2": "",
                           "categoria": cat, "lesao": les})
    pd.concat([an, pd.DataFrame(linhas).astype(str)], ignore_index=True)[an.columns].to_csv(p, index=False)
    pj = Path(raiz) / "data/derived/janelas_v1.csv"
    jt = pd.read_csv(pj, dtype=str, keep_default_na=False)
    extra = {"split": "Dataset_Treino", "caso": "0109", "seed": "2000", "excluida": "0", "seed_tem_box": "1",
             "categoria": "nao_lesao", "lesao": "0", "descricao": "Bolhas"}
    pd.concat([jt, pd.DataFrame([extra])], ignore_index=True)[jt.columns].to_csv(pj, index=False)


def _montar(tmp):
    raiz = ta._montar(tmp)                    # lista (17), respostas, PF v2, 50 FP, avaliador e classes das janelas mistas
    ca.main(ta._args18(raiz))                 # análise C9: outputs/18_c_analise/ganho.csv
    _mais_janelas(raiz)
    return raiz


def _args19(raiz, *extra, saida="Dataset_WCE_v2", respostas="decisoes_paulo.csv", classes="classes_mistas.csv",
            ganho="18_c_analise/ganho.csv", sha=None):
    raiz, g = Path(raiz), Path(raiz) / "data/derived/guiao_c"
    exp, resp = raiz / "data/derived/expanded_v1.csv", g / respostas
    sha = (ev._sha256(resp)[:16] if resp.exists() else "") if sha is None else sha
    return ["--config", str(raiz / "configs/avaliacao.yaml"), "--itens", str(g / "itens_v1.csv"),
            "--chave", str(g / "itens_chave_v1.csv"), "--respostas", str(resp), "--classes", str(g / classes),
            "--expandida", str(exp), "--sha-expandida", ev._sha256(exp)[:16], "--sha-respostas", sha,
            "--ganho", str(raiz / "outputs" / ganho), "--saida-dir", str(raiz / "data/derived" / saida),
            "--resumo", str(raiz / "outputs" / f"{saida}.txt"), *extra]


def _ler(raiz, saida="Dataset_WCE_v2"):
    d = Path(raiz) / "data/derived" / saida
    a = pd.read_csv(d / "annotations_v2.csv", dtype=str, keep_default_na=False)
    c = pd.read_csv(d / "caixas_v2.csv", dtype={"caso": str}, keep_default_na=False)
    return a, c


def _chave(raiz):
    ch = pd.read_csv(Path(raiz) / "data/derived/guiao_c/itens_chave_v1.csv", dtype={"caso": str})
    it = pd.read_csv(Path(raiz) / "data/derived/guiao_c/itens_v1.csv", dtype={"caso": str})
    return ch.merge(it[["item", "box_x1", "box_y1", "box_x2", "box_y2"]], on="item")


def _linhas(a, t):
    """As linhas da annotations_v2 (texto) dos frames da tabela t, pela ordem de t."""
    k = a.assign(seed=a["seed"].astype(int), frame=a["frame"].astype(int))
    return t[K4].merge(k, on=K4, how="left")


# ------------------------------------------------------------------ o que entra e o que não entra
def test_dataset_v2_com_as_contas_feitas_a_mao():
    with tempfile.TemporaryDirectory() as tmp:
        raiz = _montar(tmp)
        r = cd.main(_args19(raiz))
        A, O = r["A"], r["O"]
        assert len(A) == 14 and len(O) == 2
        assert {k: int(A[k].sum()) for k in ESPERADO_A} == ESPERADO_A
        assert int(A["aceite_sem_ficheiro"].sum()) == ti.N_SEM_FICHEIRO          # as −8 aceites eram desconhecidas
        assert {k: int(O[k].sum()) for k in ("original", "invalida", "sem_ficheiro", "sem_caixa")} == \
            {"original": 5, "invalida": 1, "sem_ficheiro": 2, "sem_caixa": 74}
        assert "confere com as regras do avaliador (70 caixas originais em 14 janelas)" in r["conferencia_avaliador"]
        assert r["conferencia_c9"] == "confere com a análise C9 (ganho.csv: 70 → 88)"
        a, c = _ler(raiz)
        assert len(a) == 574 + 82 and (a["estado"].value_counts().sum() == len(a))
        # caixas novas: aceites com a caixa da lista; rejeitadas, N e controlos sem caixa
        ch = _chave(raiz)
        nova = ch[ch["tipo"] == "nova"]
        resp = pd.read_csv(Path(raiz) / "data/derived/guiao_c/decisoes_paulo.csv", dtype={"caso": str})
        nova = nova.merge(resp[["item", "decisao"]], on="item")
        x = _linhas(a, nova)
        assert (x["estado"].to_numpy() == nova["decisao"].map(cd.DE_RESPOSTA).to_numpy()).all()
        ac = (nova["decisao"] == "aceitar").to_numpy()
        v2 = x[cd.V2].replace("", np.nan).astype(float).to_numpy()
        assert np.allclose(v2[ac], nova.loc[ac, ["box_x1", "box_y1", "box_x2", "box_y2"]].to_numpy(float), atol=1e-9)
        assert np.isnan(v2[~ac]).all()
        assert (x["revisor"] == "paulo").all() and (x["data_revisao"] == "2026-10-24").all()
        co = ch[ch["tipo"] == "controlo_original"]                 # fica a caixa original, não a do SAM 2
        y = _linhas(a, co)
        assert (y["estado"] == "original").all() and (y["revisor"] == "").all()
        assert np.allclose(y[cd.V2].astype(float).to_numpy(), ti._caixa(0.1))
        assert not np.allclose(co[["box_x1", "box_y1", "box_x2", "box_y2"]].to_numpy(float), ti._caixa(0.1))
        cb = ch[ch["tipo"] == "controlo_baixa"]                    # nunca entram
        assert (_linhas(a, cb)["estado"] == "sem_caixa").all() and (_linhas(a, cb)["box_v2_x1"] == "").all()
        # score e nível só a 1–12 do seed das janelas de ALL-SEEDBOX
        off = a["offset"].astype(int).abs()
        com_score = a["nivel"] != ""
        assert com_score.sum() == 14 * 24 and (off[com_score].between(1, 12)).all()
        assert (a.loc[com_score, "all_seedbox"] == "1").all()
        # classe: 0202 passa a pólipo (lesão da caixa do seed); sangue fica em outras; não-lesão sem classe; 0110 sem
        # linha na janelas_v1 fica com a categoria da annotations_v1
        cl = a.drop_duplicates("caso").set_index("caso")
        assert tuple(cl.loc["0202", ["categoria", "categoria_v2", "classe_v2"]]) == ("erosao", "polipo", "polipo")
        assert tuple(cl.loc["0204", ["categoria_v2", "classe_v2"]]) == ("sangue", "outras")
        assert tuple(cl.loc["0109", ["categoria_v2", "classe_v2", "all_seedbox"]]) == ("nao_lesao", "", "0")
        assert tuple(cl.loc["0110", ["categoria_v2", "classe_v2", "all_seedbox"]]) == ("polipo", "polipo", "0")
        # caixas_v2: as linhas com caixa, nem mais nem menos
        assert len(c) == 70 + 18 + 5 and list(c.columns) == cd.COL_CAIXAS
        assert set(c["estado"]) == {"original", "nova_aceite"} and (c["estado"] == "nova_aceite").sum() == 18
        com = a[a["estado"].isin(cd.COM_CAIXA)]
        assert sorted(zip(com["caso"], com["frame"].astype(int))) == sorted(zip(c["caso"], c["frame"]))
        assert (a.loc[~a["estado"].isin(cd.COM_CAIXA), cd.V2] == "").all().all()
        assert c.loc[c["caso"] == "0109", "lesao"].astype(str).eq("0").all()


def test_colunas_da_v1_ficam_iguais_e_ficheiros_congelados():
    with tempfile.TemporaryDirectory() as tmp:
        raiz = _montar(tmp)
        r = cd.main(_args19(raiz))
        a, _ = _ler(raiz)
        v1 = pd.read_csv(Path(raiz) / "data/derived/annotations_v1.csv", dtype=str, keep_default_na=False)
        assert list(a.columns) == list(v1.columns) + cd.COL_V2
        assert a[list(v1.columns)].equals(v1)                         # o texto da v1, linha a linha e pela mesma ordem
        pasta = Path(raiz) / "data/derived/Dataset_WCE_v2"
        assert sorted(f.name for f in pasta.iterdir()) == ["README.md", "SHA256SUMS", "annotations_v2.csv", "caixas_v2.csv"]
        for f in pasta.iterdir():
            assert not os.stat(f).st_mode & 0o222, f"{f.name} devia ser só de leitura"
        sums = dict(reversed(x.split("  ")) for x in (pasta / "SHA256SUMS").read_text(encoding="utf-8").splitlines())
        assert set(sums) == {"annotations_v2.csv", "caixas_v2.csv", "README.md"}
        assert all(ev._sha256(pasta / n) == s for n, s in sums.items())
        assert r["estados"] == ["gravado"] * 4
        # correr outra vez com os mesmos ficheiros dá exatamente o mesmo (o README não tem data nem commit)
        assert cd.main(_args19(raiz))["estados"] == ["igual ao que já existia"] * 4
        # outras respostas: recusa sem --forcar e não mexe em nada; com --forcar substitui
        antes = {f.name: ev._sha256(f) for f in pasta.iterdir()}

        def uma_rejeitada(t, ch):
            i = t.index[t["decisao"] == "aceitar"][0]
            t.loc[i, "decisao"] = "rejeitar"
            return t
        ta._respostas(raiz, "decisoes_v2.csv", uma_rejeitada)
        args = _args19(raiz, respostas="decisoes_v2.csv", ganho="nao_existe.csv")
        erro(cd.main, args, contem="já existe com outro conteúdo")
        assert {f.name: ev._sha256(f) for f in pasta.iterdir()} == antes
        assert not [f for f in pasta.iterdir() if f.name.endswith(".tmp")]
        r2 = cd.main(args + ["--forcar"])
        assert r2["estados"] == ["substituído (--forcar)"] * 4 and int(r2["A"]["nova_aceite"].sum()) == 17
        assert r2["conferencia_c9"].startswith("não conferido com a análise C9")
        assert "sem conferir com a análise C9" in r2["linha"]


def test_controlos_nunca_entram_e_mal_posta_fica_sem_caixa():
    with tempfile.TemporaryDirectory() as tmp:
        raiz = _montar(tmp)
        cd.main(_args19(raiz))

        def controlos_aceites(t, ch):                              # aceitar todos os controlos não muda nada
            t.loc[ch["tipo"].str.startswith("controlo").to_numpy(), "decisao"] = "aceitar"
            return t
        ta._respostas(raiz, "decisoes_ctrl.csv", controlos_aceites)
        r = cd.main(_args19(raiz, respostas="decisoes_ctrl.csv", saida="v2_ctrl"))
        assert r["conferencia_c9"].startswith("confere")
        for n in ("annotations_v2.csv", "caixas_v2.csv"):
            assert ev._sha256(Path(raiz) / "data/derived/Dataset_WCE_v2" / n) == \
                ev._sha256(Path(raiz) / "data/derived/v2_ctrl" / n), n

        def uma_mal_posta(t, ch):                                  # presença confirmada, sem caixa
            i = t.index[(ch["tipo"] == "nova").to_numpy() & (t["decisao"] == "aceitar").to_numpy()][0]
            t.loc[i, "decisao"] = "mal_posta"
            return t
        d = ta._respostas(raiz, "decisoes_mp.csv", uma_mal_posta)
        ca.main(ta._args18(raiz, respostas="decisoes_mp.csv", saida="18_mp"))       # a análise C9 com estas respostas
        r = cd.main(_args19(raiz, respostas="decisoes_mp.csv", saida="v2_mp", ganho="18_mp/ganho.csv"))
        assert r["conferencia_c9"].startswith("confere com a análise C9")
        assert int(r["A"]["nova_mal_posta"].sum()) == 1 and int(r["A"]["nova_aceite"].sum()) == 17
        a, c = _ler(raiz, "v2_mp")
        mp = d[d["decisao"] == "mal_posta"]
        x = _linhas(a, mp.assign(seed=mp["seed"].astype(int), frame=mp["frame"].astype(int)))
        assert (x["estado"] == "nova_mal_posta").all() and (x[cd.V2] == "").all().all()
        assert len(c) == 70 + 17 + 5


def test_recusa_entradas_erradas():
    with tempfile.TemporaryDirectory() as tmp:
        raiz = _montar(tmp)
        pasta = Path(raiz) / "data/derived/Dataset_WCE_v2"
        erro(cd.main, _args19(raiz, sha="77f51cca98f9baa8"), contem="não é o da análise C9")
        # folha das classes com uma janela por responder
        g = Path(raiz) / "data/derived/guiao_c"
        t = pd.read_csv(g / "classes_mistas.csv", dtype=str, keep_default_na=False)
        t.loc[0, "classe_caixa"] = ""
        t.to_csv(g / "classes_falta.csv", index=False)
        erro(cd.main, _args19(raiz, classes="classes_falta.csv"), contem="ainda sem classe")
        # totais diferentes dos da análise C9: não grava nada
        gp = Path(raiz) / "outputs/18_c_analise/ganho.csv"
        gn = pd.read_csv(gp)
        gn.loc[gn["grupo"] == "total", "aceites"] += 1
        gn.to_csv(Path(raiz) / "outputs/ganho_mexido.csv", index=False)
        erro(cd.main, _args19(raiz, ganho="ganho_mexido.csv"), contem="não confere com a análise C9")
        assert not pasta.exists() or not any(pasta.iterdir())
        # uma caixa nova num frame que afinal tem caixa original
        p = Path(raiz) / "data/derived/annotations_v1.csv"
        an = pd.read_csv(p, dtype=str, keep_default_na=False)
        ch = _chave(raiz)
        x = ch[ch["tipo"] == "nova"].iloc[0]
        i = an.index[(an["caso"] == x["caso"]) & (an["frame"].astype(int) == int(x["frame"]))][0]
        an.loc[i, ["box_src", "box_x1", "box_y1", "box_x2", "box_y2"]] = ["medico_vizinho", "0.4", "0.4", "0.6", "0.6"]
        an.to_csv(p, index=False)
        erro(cd.main, _args19(raiz), contem="já têm caixa original")
        assert not pasta.exists() or not any(pasta.iterdir())


def test_readme_resumo_e_linha_para_o_decisoes():
    with tempfile.TemporaryDirectory() as tmp:
        raiz = _montar(tmp)
        r = cd.main(_args19(raiz))
        txt = (Path(raiz) / "data/derived/Dataset_WCE_v2/README.md").read_text(encoding="utf-8")
        for s in ("# Dataset WCE v2 — anotação v2.0", "## Regras", "## Origem", "## Limitações", "## Colunas da v2",
                  "as rejeitadas e as não avaliáveis não entram", "nunca entram", "um frame sem caixa não é um negativo",
                  "as caixas passam de 70 para 88 (+25,7 %)", "| **total** | 14 | 70 | 36 | 18 | 88 | +25,7 % |",
                  "| pólipo | 4 | 20 | 9 | 5 | 25 | +25,0 % |", "`nova_rejeitada` (7)",
                  "2 escolhidas pelo revisor, 1 não clara, com a primeira mencionada; 1 muda de classe",
                  "(Guião C, 24 out 2026)", "aceitou 2 de 3 caixas do SAM 2 em frames com caixa original e 0 de 3 abaixo",
                  "metodo = SAM2_L12; versao_config = final_SAM2_L12.yaml sha256 abc"):
            assert s in txt, s
        assert ev._sha256(Path(raiz) / "data/derived/guiao_c/decisoes_paulo.csv") in txt
        res = (Path(raiz) / "outputs/Dataset_WCE_v2.txt").read_text(encoding="utf-8")
        assert r["linha"] in res and "Linha para o docs/decisoes.md:" in res
        assert "ALL-SEEDBOX (14 janelas): caixas 70 → 88 (+25,7 %)" in res
        assert "aceites 18 (entram; 4 em frames sem ficheiro de caixa)" in res
        assert "1 janela sem linha na janelas_v1.csv" in res
        assert "as 18 caixas novas aceites" in r["linha"] and "igual à análise C9" in r["linha"]
        assert r["sha256"]["anot"][:16] in r["linha"]
        # só números agregados: nenhum caso aparece no README nem no resumo
        for caso in CASOS:
            assert not re.search(rf"\b{caso}\b", txt), f"o caso {caso} aparece no README"
            assert not re.search(rf"\b{caso}\b", res), f"o caso {caso} aparece no resumo"


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
