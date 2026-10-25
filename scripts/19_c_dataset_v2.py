#!/usr/bin/env python3
"""
19 — Guião C, C10: Dataset WCE v2 (anotação v2.0), com a regra fixada antes da revisão (docs/decisoes.md, 24 out 2026):
entram as caixas da anotação original e as caixas novas aceites na revisão; as mal postas ficam como presença confirmada,
sem caixa; as rejeitadas e as não avaliáveis não entram; os controlos às cegas nunca entram (serviram só a análise).
A classe das 14 janelas com mais de uma lesão no texto é a da lesão da caixa do seed (classes_mistas.csv), como na análise.

Lê (os caminhos da anotação original e das janelas vêm do configs/avaliacao.yaml):
  data/derived/annotations_v1.csv                         a anotação original, uma linha por frame
  data/derived/janelas_v1.csv                             categoria e lesão de cada janela
  data/derived/expanded_v1.csv                            score e nível do SAM 2 (confirma o sha256 do A8)
  data/derived/guiao_c/itens_v1.csv, itens_chave_v1.csv   a lista da revisão e a chave (scripts/17_c_itens.py)
  data/derived/guiao_c/decisoes_paulo.csv                 a revisão consolidada (confirma o sha256 da análise C9)
  data/derived/guiao_c/classes_mistas.csv                 a classe das janelas com mais de uma lesão no texto
  outputs/18_c_analise/ganho.csv                          (se existir) para conferir os totais com a análise C9
Grava, só de leitura, em data/derived/Dataset_WCE_v2/ (fora do git):
  annotations_v2.csv   uma linha por frame, as mesmas da annotations_v1.csv: todas as colunas da v1, sem mudanças, mais
                       estado, box_v2_x1..box_v2_y2, score, nivel, revisor, data_revisao, categoria_v2, classe_v2, all_seedbox
  caixas_v2.csv        uma linha por caixa da v2 (estado original ou nova_aceite), só com as colunas para um detetor
  README.md            o que é, colunas, regras, como foi feito, contagens, origem (sha256) e limitações
  SHA256SUMS           sha256 dos três; dentro da pasta, «sha256sum -c SHA256SUMS» confirma que não mudaram
e o resumo em outputs/19_c_dataset_v2_resumo.txt (só números agregados: pode colar-se no chat), com a linha para o
docs/decisoes.md. Antes de gravar, confere: todas as caixas novas da lista estão na annotations_v1.csv e em frames sem
caixa original; as caixas originais de ALL-SEEDBOX são as do avaliador (eval/evaluate.py); os totais por classe e por
split são os da análise C9. O Dataset v2 congela-se uma vez: se já existir com outro conteúdo, não é reescrito
(--forcar substitui-o, com linha no docs/decisoes.md e versão nova).

Uso, a partir da pasta do projeto, depois do source …/inicio.sh:
    python scripts/19_c_dataset_v2.py
Versão 1.0 (25 out 2026).
"""
import argparse
import datetime
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]


def _modulo(nome, ficheiro):
    spec = importlib.util.spec_from_file_location(nome, RAIZ / "scripts" / ficheiro)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ci = _modulo("c_itens_17", "17_c_itens.py")          # leitura da anotação expandida e o sha256 do A8
ca = _modulo("c_analise_18", "18_c_analise.py")      # leitura da revisão e das classes, igual à da análise C9
ev = ci.ev

VERSAO = "1.0"
VERSAO_DATASET = "2.0"
SHA_C9 = "77f51cca98f9baa8"          # decisoes_paulo.csv da análise C9 (25 out 2026)
K3 = ["split", "caso", "seed"]
K4 = K3 + ["frame"]
CAIXA = ["box_x1", "box_y1", "box_x2", "box_y2"]
REF = ["rx1", "ry1", "rx2", "ry2"]
V2 = ["box_v2_x1", "box_v2_y1", "box_v2_x2", "box_v2_y2"]
RV = ["rv_x1", "rv_y1", "rv_x2", "rv_y2"]
COL_V2 = ["estado"] + V2 + ["score", "nivel", "revisor", "data_revisao", "categoria_v2", "classe_v2", "all_seedbox"]
COL_CAIXAS = K4 + ["offset", "estado", "categoria_v2", "classe_v2", "lesao", "excluida", "all_seedbox"] + CAIXA + \
    ["img_w", "img_h"]
PRECISAS = K4 + ["offset", "box_src"] + CAIXA
ESTADOS = {
    "original": "tem caixa da anotação original (desenhada pelo Rodrigo: box_src medico_seed ou medico_vizinho, com "
                "caixa válida). A caixa da v2 é essa.",
    "nova_aceite": "não tinha caixa original; o SAM 2 propôs uma caixa (nível alta) e a revisão aceitou-a. A caixa da v2 "
                   "é a do SAM 2, tal como foi revista.",
    "nova_mal_posta": "a revisão confirmou a lesão do seed, mas a caixa proposta estava mal posta: presença confirmada, "
                      "sem caixa.",
    "nova_rejeitada": "a caixa proposta não marcava a lesão do seed. Sem caixa. **Não é um negativo**: a lesão pode estar "
                      "noutro sítio do frame.",
    "nova_nao_avaliavel": "o revisor não conseguiu decidir (imagem má, lesão do seed que não reconhece, ou dúvida se é a "
                          "mesma). Sem caixa.",
    "sem_caixa": "sem caixa original (box_src vazio) e sem caixa nova revista. Não é um negativo: a anotação original é "
                 "incompleta longe do seed.",
    "sem_ficheiro": "o ficheiro da caixa original não foi exportado (offsets −70, −39 e −8): não se sabe se tinha caixa.",
    "invalida": "caixa original inválida (sem coordenadas ou de área nula) e sem caixa nova aceite.",
}
DE_RESPOSTA = {"aceitar": "nova_aceite", "mal_posta": "nova_mal_posta", "rejeitar": "nova_rejeitada",
               "nao_avaliavel": "nova_nao_avaliavel"}
COM_CAIXA = ("original", "nova_aceite")
NOVAS = [e for e in ESTADOS if e.startswith("nova_")]
BLOCO = 200_000
TESTE, TREINO = "Dataset_Teste", "Dataset_Treino"
_n, _p, _pl, NOMES = ci._n, ca._p, ca._pl, ca.NOMES


# ============================================================================ leitura
def _inteiros(s, nome):
    v = pd.to_numeric(s, errors="coerce")
    if v.isna().any() or (v != v.round()).any():
        raise ev.ErroAvaliacao(f"A coluna {nome} da annotations_v1.csv tem valores que não são números inteiros.")
    return v.astype(int).to_numpy()


def ler_revisao(args):
    """As caixas novas revistas (tipo nova), com a resposta, e o número de controlos às cegas. Lê como a análise C9."""
    itens = ca.ler_itens(args.itens)
    resp = ca.ler_respostas(args.respostas, itens)
    for c in ("revisor", "t_decisao"):
        if c not in resp.columns:
            resp[c] = ""
    d = ca.juntar(resp, ca.ler_chave(args.chave), itens)
    estranhos = sorted(set(d["tipo"]) - {"nova", "controlo_original", "controlo_baixa"})
    if estranhos:
        raise ev.ErroAvaliacao(f"A chave da lista tem tipos de item desconhecidos: {estranhos}.")
    novas = d[d["tipo"] == "nova"]
    if novas.empty:
        raise ev.ErroAvaliacao("A lista não tem caixas novas (tipo nova).")
    rev = novas[K4].copy()
    for c, r in zip(CAIXA, RV):
        rev[r] = novas[c].astype(float).to_numpy()
    rev["estado_rev"] = novas["decisao"].map(DE_RESPOSTA).to_numpy()
    rev["revisor"] = novas["revisor"].astype(str).str.strip().to_numpy()
    rev["data_revisao"] = novas["t_decisao"].astype(str).str.strip().str[:10].to_numpy()
    ctrl = {}
    for tipo in ("controlo_original", "controlo_baixa"):
        x = d.loc[d["tipo"] == tipo, "decisao"]
        ctrl[tipo] = {"caixas": len(x), "aceitar": int((x == "aceitar").sum()),
                      "com_resposta": int(x.isin(["aceitar", "rejeitar", "mal_posta"]).sum())}
    return rev.reset_index(drop=True), ctrl, sorted({t[:10] for t in d["t_decisao"].astype(str).str.strip() if len(t) >= 10})


def ler_janelas(cfg):
    """{(split, caso, seed): (categoria, lesao, excluida)}, da janelas_v1.csv (a fonte das classes do avaliador)."""
    p = Path(cfg["caminhos"]["janelas"])
    if not p.is_file():
        raise ev.ErroAvaliacao(f"Não encontrei {p}.")
    jt = pd.read_csv(p, dtype=str, keep_default_na=False)
    if "categoria" not in jt.columns:
        raise ev.ErroAvaliacao(f"{p} não tem a coluna 'categoria'.")
    return {ev.chave(r["split"], r["caso"], r["seed"]): (r["categoria"].strip(), r.get("lesao", "1").strip(),
                                                         r.get("excluida", "0").strip())
            for r in jt.to_dict("records")}


def ler_mistas(caminho, cfg):
    """Categoria das janelas com mais de uma lesão no texto: a lesão da caixa do seed (ou, se não for clara, a primeira
    mencionada), com a validação da análise C9 (scripts/18_c_analise.py, ler_classes)."""
    over, cont = ca.ler_classes(caminho, cfg)
    if cont["por_responder"]:
        raise ev.ErroAvaliacao(f"{caminho}: {_pl(cont['por_responder'], 'janela', 'janelas')} ainda sem classe. Corre "
                               "primeiro: python scripts/18_c_analise.py --preparar-classes")
    t = pd.read_csv(caminho, dtype=str, keep_default_na=False)
    cat = {}
    for r in t.to_dict("records"):
        c = ca._tira(r["classe_caixa"]).strip()
        cat[ev.chave(r["split"], r["caso"], r["seed"])] = r["primeira"] if c in ("", "primeira") else c
    return cat, over, cont


# ============================================================================ a v2, bloco a bloco
def atributos(b, k, cache, jan, mistas, alvo, principais):
    """Categoria, classe e marcas de cada janela do bloco (tabela para juntar por janela)."""
    primeiras = k.drop_duplicates(K3)
    for pos, (s, c, e) in zip(primeiras.index, zip(primeiras["split"], primeiras["caso"], primeiras["seed"])):
        kk = (s, c, int(e))
        if kk in cache:
            continue
        if kk in jan:
            cat, les, exc = jan[kk]
        else:                                           # janela que não está na janelas_v1: fica a da annotations_v1
            linha = b.iloc[pos]
            cat, les, exc = (str(linha.get(x, "")).strip() for x in ("categoria", "lesao", "excluida"))
        cat2 = mistas.get(kk, cat)
        cache[kk] = {"categoria_v2": cat2, "classe_v2": (cat2 if cat2 in principais else "outras") if les == "1" else "",
                     "lesao": les, "excluida": exc, "all_seedbox": int(kk in alvo), "fora_janelas_v1": kk not in jan}
    ks = list(dict.fromkeys(zip(k["split"], k["caso"], k["seed"])))
    return pd.DataFrame([{"split": x[0], "caso": x[1], "seed": x[2], **cache[x]} for x in ks])


def bloco_v2(b, rev, exp, attrs):
    """Um bloco da annotations_v1.csv (tudo em texto, que fica igual) com as colunas da v2. Devolve também as caixas da v2,
    as contagens por janela, as caixas originais das janelas de ALL-SEEDBOX e quantas caixas novas da lista apanhou."""
    n = len(b)
    k = pd.DataFrame({"split": b["split"].astype(str).str.strip().to_numpy(),
                      "caso": b["caso"].astype(str).str.strip().str.zfill(4).to_numpy(),
                      "seed": _inteiros(b["seed"], "seed"), "frame": _inteiros(b["frame"], "frame")})
    off = _inteiros(b["offset"], "offset")
    v = b[CAIXA].apply(pd.to_numeric, errors="coerce")
    src = b["box_src"].astype(str).str.strip().to_numpy()
    medico = np.isin(src, ["medico_seed", "medico_vizinho"])
    valida = (medico & v.notna().all(axis=1).to_numpy() & (v["box_x2"] > v["box_x1"]).to_numpy()
              & (v["box_y2"] > v["box_y1"]).to_numpy())             # as regras do avaliador (carregar_referencia)
    estado = np.select([valida, medico, src == "vazio", src == "sem_ficheiro"],
                       ["original", "invalida", "sem_caixa", "sem_ficheiro"], "invalida").astype(object)
    r = k.merge(rev, on=K4, how="left", validate="many_to_one")
    e = k.merge(exp, on=K4, how="left", validate="many_to_one")
    w = k[K3].merge(attrs, on=K3, how="left", validate="many_to_one")
    revisto = r["estado_rev"].notna().to_numpy()
    if (revisto & valida).any():
        x = k[revisto & valida].iloc[0]
        raise ev.ErroAvaliacao(f"{int((revisto & valida).sum())} caixas novas da lista estão em frames que já têm caixa "
                               f"original (ex.: {ev._txt(tuple(x))}). A lista não é desta annotations_v1.csv? Não gravei nada.")
    estado[revisto] = r["estado_rev"].to_numpy()[revisto]
    o, a = estado == "original", estado == "nova_aceite"
    caixa = np.full((n, 4), np.nan)
    caixa[o] = v.to_numpy(dtype=float)[o]
    caixa[a] = r[RV].to_numpy(dtype=float)[a]

    out = b                                             # o bloco não volta a ser usado: sem cópia (a v1 é grande)
    out["estado"] = estado
    for i, c in enumerate(V2):
        out[c] = caixa[:, i]
    out["score"] = e["score"].to_numpy(dtype=float)
    out["nivel"] = e["nivel"].fillna("").to_numpy()
    out["revisor"] = np.where(revisto, r["revisor"].fillna("").astype(str).to_numpy(), "")
    out["data_revisao"] = np.where(revisto, r["data_revisao"].fillna("").astype(str).to_numpy(), "")
    for c in ("categoria_v2", "classe_v2", "all_seedbox"):
        out[c] = w[c].to_numpy()

    m = np.isin(estado, COM_CAIXA)
    cx = k[m].reset_index(drop=True)
    cx["offset"] = off[m]
    cx["estado"] = estado[m]
    for c in ("categoria_v2", "classe_v2", "lesao", "excluida", "all_seedbox"):
        cx[c] = w[c].to_numpy()[m]
    for i, c in enumerate(CAIXA):
        cx[c] = caixa[m, i]
    for c in ("img_w", "img_h"):
        cx[c] = out[c].to_numpy()[m] if c in out.columns else ""

    t = k[K3].copy()
    for s in ESTADOS:
        t[s] = (estado == s).astype(int)
    t["frames"] = 1
    t["aceite_sem_ficheiro"] = (a & (src == "sem_ficheiro")).astype(int)
    t["seed_ok"] = ((off == 0) & o).astype(int)
    pj = t.groupby(K3, sort=False).sum()

    alv = o & (w["all_seedbox"].to_numpy() == 1)
    orig = k[alv].reset_index(drop=True)
    for i, c in enumerate(REF):
        orig[c] = caixa[alv, i]
    return out, cx, pj, orig, int(revisto.sum())


# ============================================================================ conferências
def conferir_avaliador(cfg, todas, origs, cache, over):
    """As caixas originais de ALL-SEEDBOX e a classe das janelas têm de ser as do avaliador (eval/evaluate.py)."""
    ref, info = ca.referencia(cfg, todas)
    rc = ref.loc[ref["caixa"] == "caixa", K4 + REF]
    m = rc.merge(origs, on=K4, how="outer", suffixes=("_ref", "_v2"), indicator=True)
    so_ref, so_v2 = int((m["_merge"] == "left_only").sum()), int((m["_merge"] == "right_only").sum())
    ab = m[m["_merge"] == "both"]
    caixas_iguais = np.allclose(ab[[c + "_ref" for c in REF]].to_numpy(float), ab[[c + "_v2" for c in REF]].to_numpy(float),
                                rtol=0, atol=1e-12)
    jcl = dict(zip(zip(info["split"], info["caso"], info["seed"]), info["classe"]))
    cl_dif = [k for k in todas if cache[k]["classe_v2"] != over.get(k, jcl[k])]
    if so_ref or so_v2 or not caixas_iguais or cl_dif:
        raise ev.ErroAvaliacao(
            "A v2 não confere com as regras do avaliador em ALL-SEEDBOX: "
            f"{so_ref} caixas originais só no avaliador, {so_v2} só na v2"
            + ("" if caixas_iguais else ", coordenadas diferentes")
            + (f", {len(cl_dif)} janelas com outra classe (ex.: {ev._txt(cl_dif[0])})" if cl_dif else "")
            + ". Não gravei nada.")
    return f"confere com as regras do avaliador ({_n(len(rc))} caixas originais em {_n(len(todas))} janelas)"


def ganho_c9(A):
    """Contas da análise C9 (scripts/18_c_analise.py, ganho) a partir das contagens da v2 por janela de ALL-SEEDBOX:
    antes = 1 (o seed) + caixas originais fora do seed; depois = antes + novas aceites."""
    A = A.assign(antes=A["original"] - A["seed_ok"] + 1, propostas=A[NOVAS].sum(axis=1), aceites=A["nova_aceite"])
    A["depois"] = A["antes"] + A["aceites"]
    campos = ["antes", "propostas", "aceites", "depois"]
    out = {}
    for grupo, col in (("classe", "classe_v2"), ("split", "split")):
        for val, t in A.groupby(col):
            out[(grupo, val)] = {"janelas": len(t), **{c: int(t[c].sum()) for c in campos}}
    out[("total", "total")] = {"janelas": len(A), **{c: int(A[c].sum()) for c in campos}}
    return out


def conferir_c9(caminho, A):
    p = Path(caminho)
    if not p.is_file():
        return f"não conferido com a análise C9: falta {p} (corre primeiro o scripts/18_c_analise.py)"
    g = pd.read_csv(p, dtype=str, keep_default_na=False)
    meu, dif, vistos = ganho_c9(A), [], set()
    for r in g.to_dict("records"):
        kk = (r["grupo"], r["grupo_valor"])
        vistos.add(kk)
        if kk not in meu:
            dif.append(f"{NOMES.get(kk[1], kk[1])} não existe na v2")
            continue
        for c in ("janelas", "antes", "propostas", "aceites", "depois"):
            if int(float(r[c])) != meu[kk][c]:
                dif.append(f"{NOMES.get(kk[1], kk[1])}: {c} {meu[kk][c]} na v2, {int(float(r[c]))} na análise")
    dif += [f"{NOMES.get(kk[1], kk[1])} não está em {p.name}" for kk in meu if kk not in vistos]
    if dif:
        raise ev.ErroAvaliacao(f"A v2 não confere com a análise C9 ({p}): " + "; ".join(dif[:6]) + ". São as mesmas "
                               "respostas e a mesma folha das classes? Não gravei nada.")
    t = meu[("total", "total")]
    return f"confere com a análise C9 ({p.name}: {_n(t['antes'])} → {_n(t['depois'])})"


# ============================================================================ gravar
def _temporario(pasta, nome):
    f = tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=pasta, delete=False, prefix=f".{nome}.",
                                    suffix=".tmp")
    return f, Path(f.name)


def congelar(pares, forcar):
    """pares: [(destino, temporário)]. Tudo ou nada: se algum destino já existir com outro conteúdo (e sem --forcar),
    não grava nenhum. Os ficheiros ficam só de leitura."""
    estados = []
    for dst, tmp in pares:
        if dst.exists():
            if ev._sha256(dst) == ev._sha256(tmp):
                estados.append("igual ao que já existia")
                continue
            if not forcar:
                raise ev.ErroAvaliacao(
                    f"{dst} já existe com outro conteúdo. O Dataset WCE v2 congela-se uma vez: não o reescrevo. Se for "
                    "mesmo para o substituir, corre com --forcar, muda a versão e regista o motivo no docs/decisoes.md.")
            estados.append("substituído (--forcar)")
        else:
            estados.append("gravado")
    for (dst, tmp), est in zip(pares, estados):
        if est.startswith("igual"):
            tmp.unlink()
            continue
        os.chmod(tmp, 0o444)
        os.replace(tmp, dst)
    return estados


# ============================================================================ textos
MESES = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]


def _pct(a, b):
    return f"+{_p(100 * a / b, 1)} %" if b else "—"


def _datas(dias):
    """['2026-10-24', '2026-10-25'] → «24 a 25 out 2026», como na tese."""
    try:
        ds = [datetime.date.fromisoformat(x) for x in dias]
    except ValueError:
        return "datas por registar"
    if not ds:
        return "datas por registar"
    a, b = ds[0], ds[-1]
    fim = f"{b.day} {MESES[b.month - 1]} {b.year}"
    if a == b:
        return fim
    if (a.year, a.month) == (b.year, b.month):
        return f"{a.day} a {fim}"
    return f"{a.day} {MESES[a.month - 1]}" + (f" {a.year}" if a.year != b.year else "") + f" a {fim}"


def tabela_ganho(A, classes):
    """Linhas (rótulo, janelas, originais, revistas, aceites, depois) por classe, por split e no total."""
    linhas = []
    for grupo, col, valores in (("classe", "classe_v2", classes), ("split", "split", [TESTE, TREINO])):
        for val in valores:
            t = A[A[col] == val]
            if len(t):
                linhas.append((NOMES.get(val, val) if grupo == "classe" else f"**{NOMES[val]}**", len(t),
                               int(t["original"].sum()), int(t[NOVAS].to_numpy().sum()), int(t["nova_aceite"].sum())))
    linhas.append(("**total**", len(A), int(A["original"].sum()), int(A[NOVAS].to_numpy().sum()),
                   int(A["nova_aceite"].sum())))
    return linhas


def readme(c):
    """Texto do README.md. Sem data nem commit, para sair igual sempre que se volta a correr com os mesmos ficheiros."""
    A, O, W = c["A"], c["O"], c["W"]
    tot = lambda t, col: int(t[col].sum())                                   # noqa: E731
    n_rev = int(A[NOVAS].to_numpy().sum())
    L = ["# Dataset WCE v2 — anotação v" + VERSAO_DATASET, "",
         "A anotação do Dataset WCE com as caixas da anotação original e as caixas novas da anotação expandida v1 "
         "(SAM 2) confirmadas numa revisão humana (Guião C da dissertação). As imagens são as do Dataset WCE, sem "
         "mudanças (`raw/Dataset_WCE/<split>/<caso>/GT_com_classes/<seed>/<caso>_<frame>_<seed>.jpg`): esta pasta só "
         "tem anotação.", "",
         f"Gerada por `scripts/19_c_dataset_v2.py` v{VERSAO} (sha256 `{c['sha_script']}`), a partir dos ficheiros da "
         "secção «Origem». Os ficheiros são só de leitura; dentro desta pasta, `sha256sum -c SHA256SUMS` confirma que "
         "não mudaram.", "",
         "## Em números", "",
         "| Grupo de janelas | Janelas | Frames | Caixas da anotação original | Caixas novas aceites | Caixas na v2 |",
         "| --- | --- | --- | --- | --- | --- |"]
    for rot, t in (("ALL-SEEDBOX (janelas de lesão com caixa no seed)", A), ("outras (ficam como na v1)", O),
                   ("todas", W)):
        L.append(f"| {rot} | {_n(len(t))} | {_n(tot(t, 'frames'))} | {_n(tot(t, 'original'))} | "
                 f"{_n(tot(t, 'nova_aceite'))} | {_n(tot(t, 'original') + tot(t, 'nova_aceite'))} |")
    L += ["", f"Em ALL-SEEDBOX, as caixas passam de {_n(tot(A, 'original'))} para "
          f"{_n(tot(A, 'original') + tot(A, 'nova_aceite'))} ({_pct(tot(A, 'nova_aceite'), tot(A, 'original'))}). "
          f"Revistas: {_n(n_rev)} caixas novas — aceites {_n(tot(A, 'nova_aceite'))} (entram; "
          f"{_n(tot(A, 'aceite_sem_ficheiro'))} em frames sem ficheiro de caixa), rejeitadas {_n(tot(A, 'nova_rejeitada'))}, "
          f"não dá para decidir {_n(tot(A, 'nova_nao_avaliavel'))}, mal postas {_n(tot(A, 'nova_mal_posta'))}. "
          f"Os {_n(c['controlos'])} controlos às cegas da revisão não entram.", "",
          "| ALL-SEEDBOX | Janelas | Caixas originais | Caixas novas revistas | Aceites | Caixas na v2 | Ganho |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
    for rot, nj, orig, rev, ac in tabela_ganho(A, c["classes"]):
        L.append(f"| {rot} | {_n(nj)} | {_n(orig)} | {_n(rev)} | {_n(ac)} | {_n(orig + ac)} | {_pct(ac, orig)} |")
    m_ = c["mistas"]
    L += ["", "Classes: angiectasia, erosão, úlcera, pólipo, eritema, xantelasma e outras (as restantes categorias de lesão)."
          f" Classe das {m_['janelas']} janelas com mais de uma lesão no texto: a da lesão da caixa do seed "
          f"({_pl(m_['escolhidas'], 'escolhida', 'escolhidas')} pelo revisor, {_pl(m_['nao_claras'], 'não clara', 'não claras')}"
          f", com a primeira mencionada; {_pl(m_['mudam'], 'muda', 'mudam')} de classe).", "",
          "## Ficheiros", "",
          "| Ficheiro | O que tem |", "| --- | --- |",
          f"| `annotations_v2.csv` | Uma linha por frame de cada janela ({_n(tot(W, 'frames'))} frames de {_n(len(W))} "
          "janelas, as mesmas da annotations_v1.csv, pela mesma ordem). Todas as colunas da annotations_v1.csv, sem "
          "mudanças, mais as colunas da v2 (abaixo). |",
          f"| `caixas_v2.csv` | Uma linha por caixa da v2 ({_n(c['n_caixas'])}): as linhas da annotations_v2.csv com "
          "caixa, só com as colunas precisas para treinar ou avaliar um detetor. |",
          "| `README.md` | Este ficheiro. |", "| `SHA256SUMS` | sha256 dos três ficheiros acima. |", "",
          "## Colunas da v2", "",
          "- `estado`: o que se sabe do frame na v2."]
    for e_, txt in ESTADOS.items():
        L.append(f"  - `{e_}` ({_n(tot(W, e_))}): {txt}")
    L += ["- `box_v2_x1`, `box_v2_y1`, `box_v2_x2`, `box_v2_y2`: a caixa da v2, normalizada 0–1, em xyxy; só nos estados "
          "`original` e `nova_aceite`. As colunas `box_x1..box_y2` e `box_src` continuam a ser as da anotação original; "
          "as `boxc_*` (caixas copiadas de janelas sobrepostas) nunca entram na v2.",
          "- `score`, `nivel`: score do SAM 2 (SAM2_L12) e nível (alta: score ≥ 0,86; baixa: abaixo) nos frames a 1–12 do "
          "seed das janelas de ALL-SEEDBOX; vazios nos outros. Só informativos: as caixas do nível baixa não entram.",
          "- `revisor`, `data_revisao`: quem reviu a caixa nova e em que dia (estados `nova_*`).",
          "- `categoria_v2`: a categoria da janela (a da janelas_v1.csv, menos nas janelas com mais de uma lesão no texto, "
          "em que é a lesão da caixa do seed). `classe_v2`: a classe dos estratos (angiectasia, erosao, ulcera, polipo, "
          "eritema, xantelasma ou outras); vazia nas janelas que não são de lesão.",
          "- `all_seedbox`: 1 se a janela é de ALL-SEEDBOX, 0 se não.", "",
          "`caixas_v2.csv`: split, caso, seed, frame, offset, estado (original ou nova_aceite), categoria_v2, classe_v2, "
          "lesao, excluida, all_seedbox, box_x1..box_y2 (a caixa da v2, 0–1, xyxy), img_w e img_h.", "",
          "## Regras", "",
          "Fixadas antes da revisão (docs/decisoes.md, 24 out 2026):", "",
          "- entram as caixas da anotação original e as caixas novas aceites;",
          "- as mal postas ficam como presença confirmada, sem caixa;",
          "- as rejeitadas e as não avaliáveis não entram;",
          "- os controlos às cegas da revisão (frames com caixa original e frames abaixo do limiar, misturados com as caixas "
          "novas) nunca entram: serviram só a análise.", "",
          "## Como foi feita", "",
          "1. **Anotação original (v1).** O relatório (imagens e descrições) é da médica; as caixas foram desenhadas por "
          "outro aluno (Rodrigo), uma caixa por frame, em janelas de ±100 frames à volta de cada imagem do relatório "
          "(o seed). Tabela única: annotations_v1.csv (Etapa 0).",
          "2. **Anotação expandida v1** (A8, 16 out 2026). O SAM 2.1 Hiera base+ (SAM2_L12) propaga a caixa do seed até 12 "
          "frames para cada lado, em todas as janelas de ALL-SEEDBOX. Nível alta = score ≥ 0,86, o τ do OP-F1 escolhido em "
          "CAL. Os frames do nível alta sem caixa original são as caixas novas."
          + (f" Na expanded_v1.csv: {c['metodo']}." if c["metodo"] else ""),
          f"3. **Revisão** (Guião C, {c['datas']}). Um revisor viu as {_n(n_rev)} caixas novas, uma a uma, em "
          "tools/rever.py v1.3.1, às cegas ao score, ao nível, à caixa original e ao tipo de item, com "
          f"{_n(c['controlos'])} controlos às cegas misturados. Respostas: aceitar (a caixa marca a lesão do seed, a que "
          "está dentro da caixa do seed, mais ou menos como o revisor a desenharia), rejeitar (não a marca) e não dá para "
          "decidir. Instruções: docs/instrucoes_revisao.md v3. Análise: scripts/18_c_analise.py (outputs/18_c_analise/)."
          + (f" Nos controlos, o revisor aceitou {_n(c['ctrl']['controlo_original']['aceitar'])} de "
             f"{_n(c['ctrl']['controlo_original']['com_resposta'])} caixas do SAM 2 em frames com caixa original e "
             f"{_n(c['ctrl']['controlo_baixa']['aceitar'])} de {_n(c['ctrl']['controlo_baixa']['com_resposta'])} abaixo do "
             "limiar (só as respostas aceitar ou rejeitar)." if c["controlos"] else ""),
          "4. **Esta pasta**: as regras acima, aplicadas por script e conferidas com o avaliador (eval/evaluate.py) e com "
          "a análise C9.", "",
          "## Origem", "", "| Ficheiro | sha256 |", "| --- | --- |"]
    L += [f"| `{nome}` | `{sha}` |" for nome, sha in c["entradas"]]
    L += ["", "## Limitações", "",
          "- **Um só revisor, não especialista** (aluno de mestrado), sem segunda leitura. As concordâncias medidas (com a "
          "anotação PF v2 e com a classificação dos 50 FP) são do mesmo revisor, noutras alturas. As caixas da anotação "
          "original também não foram desenhadas por um médico e não foram revistas.",
          "- **A anotação continua incompleta: um frame sem caixa não é um negativo.** Só se propuseram caixas até 12 frames "
          "do seed e com score ≥ 0,86; longe do seed e abaixo do limiar ficou tudo como estava. Uma caixa rejeitada também "
          "não faz do frame um negativo.",
          "- As caixas novas são as do SAM 2 (a caixa da maior componente da máscara), sem ajuste: aceitou-se uma caixa "
          "«mais ou menos como a desenharias», por isso podem ser um pouco maiores ou mais pequenas do que as originais.",
          "- Cada janela segue só a lesão da caixa do seed. Os frames entre dois seeds aparecem em mais de uma janela, de "
          "cada vez com o seu alvo: para treinar um detetor, juntar as caixas por frame físico (split, caso, frame).",
          "- As janelas sem caixa no seed (fora de ALL-SEEDBOX) não têm caixas novas.",
          "- **Uso:** só dentro do trabalho da dissertação. Os casos são de um dataset privado: não redistribuir sem "
          "confirmar as condições de uso do dataset original.", "",
          "## Como referir", "",
          f"Dataset WCE v2 (anotação v{VERSAO_DATASET}), Guião C da dissertação de mestrado de Paulo Freitas (Engenharia "
          "Biomédica, UTAD)."]
    return "\n".join(L) + "\n"


# ============================================================================ principal
def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--config", default="configs/avaliacao.yaml")
    a.add_argument("--itens", default="data/derived/guiao_c/itens_v1.csv")
    a.add_argument("--chave", default="data/derived/guiao_c/itens_chave_v1.csv")
    a.add_argument("--respostas", default="data/derived/guiao_c/decisoes_paulo.csv")
    a.add_argument("--classes", default="data/derived/guiao_c/classes_mistas.csv")
    a.add_argument("--expandida", default="data/derived/expanded_v1.csv")
    a.add_argument("--ganho", default="outputs/18_c_analise/ganho.csv", help="ganho da análise C9, para conferir os totais")
    a.add_argument("--saida-dir", default="data/derived/Dataset_WCE_v2")
    a.add_argument("--resumo", default="outputs/19_c_dataset_v2_resumo.txt")
    a.add_argument("--sha-expandida", default=ci.SHA_A8, help="início do sha256 esperado do expanded_v1.csv ('' não confirma)")
    a.add_argument("--sha-respostas", default=SHA_C9,
                   help="início do sha256 esperado das respostas consolidadas, as da análise C9 ('' não confirma)")
    a.add_argument("--forcar", action="store_true", help="substitui um Dataset v2 já gravado com outro conteúdo")
    args = a.parse_args(argv)
    cfg = ev.ler_config(args.config)
    principais = list(cfg["estratos"]["classes_principais"])

    # 1. entradas
    if not Path(args.respostas).is_file():
        raise ev.ErroAvaliacao(f"Não encontrei {args.respostas}: consolida primeiro a revisão.")
    sha_r = ev._sha256(args.respostas)
    if args.sha_respostas and not sha_r.startswith(args.sha_respostas):
        raise ev.ErroAvaliacao(f"{args.respostas} não é o da análise C9: sha256 {sha_r[:16]}…, esperado "
                               f"{args.sha_respostas}…. O Dataset v2 tem de sair das respostas analisadas.")
    rev, ctrl, dias = ler_revisao(args)
    n_ctrl = sum(x["caixas"] for x in ctrl.values())
    mistas, over, cont_m = ler_mistas(args.classes, cfg)
    sha_e = ev._sha256(args.expandida) if Path(args.expandida).is_file() else ""
    e = ci.ler_expandida(args.expandida)
    if args.sha_expandida and not sha_e.startswith(args.sha_expandida):
        raise ev.ErroAvaliacao(f"{args.expandida} não é o do A8: sha256 {sha_e[:16]}…, esperado {args.sha_expandida}….")
    exp = e[K4 + ["score", "nivel"]]
    metodo = "; ".join(f"{col} = {', '.join(sorted(set(e[col].astype(str))))}" for col in ("metodo", "versao_config")
                       if col in e.columns)
    del e
    jan = ler_janelas(cfg)
    conj = Path(cfg["caminhos"]["conjuntos"])
    todas = ev.ler_conjunto(conj / "ALL_SEEDBOX.json")
    alvo = set(todas)
    anot = Path(cfg["caminhos"]["anotacoes"])
    if not anot.is_file():
        raise ev.ErroAvaliacao(f"Não encontrei {anot}.")

    # 2. a v2, bloco a bloco (a annotations_v1.csv é grande)
    pasta = Path(args.saida_dir)
    pasta.mkdir(parents=True, exist_ok=True)
    nomes = {"anot": pasta / "annotations_v2.csv", "caixas": pasta / "caixas_v2.csv", "readme": pasta / "README.md",
             "sums": pasta / "SHA256SUMS"}
    temps = {}
    try:
        f, temps["anot"] = _temporario(pasta, nomes["anot"].name)
        cache, cxs, pjs, origs, apanhadas, colunas = {}, [], [], [], 0, None
        with f:
            for i, b in enumerate(pd.read_csv(anot, dtype=str, keep_default_na=False, chunksize=BLOCO)):
                if i == 0:
                    colunas = list(b.columns)
                    falta = [x for x in PRECISAS if x not in colunas]
                    if falta:
                        raise ev.ErroAvaliacao(f"{anot}: faltam as colunas {falta}.")
                    repetidas = [x for x in COL_V2 if x in colunas]
                    if repetidas:
                        raise ev.ErroAvaliacao(f"{anot} já tem as colunas {repetidas}: é mesmo a annotations_v1.csv?")
                k = pd.DataFrame({"split": b["split"].astype(str).str.strip().to_numpy(),
                                  "caso": b["caso"].astype(str).str.strip().str.zfill(4).to_numpy(),
                                  "seed": _inteiros(b["seed"], "seed")})
                attrs = atributos(b, k, cache, jan, mistas, alvo, principais)
                out, cx, pj, orig, n_ap = bloco_v2(b, rev, exp, attrs)
                out.to_csv(f, header=(i == 0), index=False, float_format="%.6f")
                cxs.append(cx)
                pjs.append(pj)
                origs.append(orig)
                apanhadas += n_ap
        if apanhadas != len(rev):
            raise ev.ErroAvaliacao(f"{_n(len(rev) - apanhadas)} das {_n(len(rev))} caixas novas da lista não estão em {anot}. "
                                   "Não gravei nada.")
        sem_jan = [kk for kk in todas if kk not in cache]
        if sem_jan:
            raise ev.ErroAvaliacao(f"{len(sem_jan)} janelas de ALL-SEEDBOX não estão em {anot} (ex.: {ev._txt(sem_jan[0])}).")

        # 3. contagens por janela e conferências
        W = pd.concat(pjs).groupby(level=list(range(3)), sort=False).sum()
        W.index.names = K3
        W = W.reset_index()
        at = pd.DataFrame([{"split": kk[0], "caso": kk[1], "seed": kk[2], **v} for kk, v in cache.items()])
        W = W.merge(at, on=K3, how="left", validate="one_to_one")
        A, O = W[W["all_seedbox"] == 1], W[W["all_seedbox"] == 0]
        seeds_sem = int((A["seed_ok"] == 0).sum())
        txt_av = conferir_avaliador(cfg, todas, pd.concat(origs, ignore_index=True), cache, over)
        txt_c9 = conferir_c9(args.ganho, A)

        # 4. caixas, README e SHA256SUMS
        cx = pd.concat(cxs, ignore_index=True).sort_values(K4, kind="mergesort").reset_index(drop=True)[COL_CAIXAS]
        f, temps["caixas"] = _temporario(pasta, nomes["caixas"].name)
        with f:
            cx.to_csv(f, index=False, float_format="%.6f")
        entradas = [(Path(x).name, ev._sha256(x)) for x in
                    (anot, cfg["caminhos"]["janelas"], args.expandida, args.itens, args.chave, args.respostas, args.classes,
                     conj / "ALL_SEEDBOX.json", args.config)]
        c = {"A": A, "O": O, "W": W, "controlos": n_ctrl, "ctrl": ctrl, "classes": principais + ["outras"],
             "mistas": cont_m, "n_caixas": len(cx), "sha_script": ev._sha256(__file__), "entradas": entradas,
             "datas": _datas(dias), "metodo": metodo}
        f, temps["readme"] = _temporario(pasta, nomes["readme"].name)
        with f:
            f.write(readme(c))
        sha = {x: ev._sha256(temps[x]) for x in ("anot", "caixas", "readme")}
        f, temps["sums"] = _temporario(pasta, nomes["sums"].name)
        with f:
            f.write("".join(f"{sha[x]}  {nomes[x].name}\n" for x in ("anot", "caixas", "readme")))
        estados = congelar([(nomes[x], temps[x]) for x in ("anot", "caixas", "readme", "sums")], args.forcar)
    finally:
        for t in temps.values():
            if t.exists():
                t.unlink()

    # 5. resumo (só números agregados)
    git = ev._estado_git(RAIZ)
    commit = git["commit"] or "sem git"
    tot = lambda t, col: int(t[col].sum())                                   # noqa: E731
    oa, na = tot(A, "original"), tot(A, "nova_aceite")
    n_rev = int(A[NOVAS].to_numpy().sum())
    fora = [kk for kk, v in cache.items() if v["fora_janelas_v1"]]
    L = [f"19 — Guião C, C10: Dataset WCE v2 · {datetime.datetime.now():%Y-%m-%d %H:%M} · "
         + (f"commit {git['commit']}" if git["commit"] else "sem git")
         + (" (com alterações por guardar!)" if git["alteracoes_por_guardar"] else ""),
         f"19_c_dataset_v2.py v{VERSAO} · anotação v{VERSAO_DATASET} · {Path(args.respostas).name} sha256 {sha_r[:16]} · "
         f"{Path(args.expandida).name} sha256 {sha_e[:16]}",
         "",
         "Regra (docs/decisoes.md, 24 out): entram as caixas originais e as caixas novas aceites; mal postas = presença "
         "sem caixa; rejeitadas, não avaliáveis e controlos às cegas não entram.",
         f"Caixas novas revistas: {_n(n_rev)} · aceites {_n(na)} (entram; {_n(tot(A, 'aceite_sem_ficheiro'))} em frames sem "
         f"ficheiro de caixa) · rejeitadas {_n(tot(A, 'nova_rejeitada'))} · não dá para decidir "
         f"{_n(tot(A, 'nova_nao_avaliavel'))} · mal postas {_n(tot(A, 'nova_mal_posta'))}",
         f"Controlos às cegas: {_n(n_ctrl)} (não entram)",
         "",
         f"ALL-SEEDBOX ({_n(len(A))} janelas): caixas {_n(oa)} → {_n(oa + na)} ({_pct(na, oa)})"]
    for rot, nj, orig, _, ac in tabela_ganho(A, principais + ["outras"]):
        L.append(f"  {rot.replace('**', '')}: {_pl(nj, 'janela', 'janelas')} · {_n(orig)} → {_n(orig + ac)} "
                 f"({_pct(ac, orig)})")
    L += [f"  {txt_av}", f"  {txt_c9}"]
    if seeds_sem:
        L.append(f"  atenção: {_pl(seeds_sem, 'janela', 'janelas')} de ALL-SEEDBOX sem caixa original válida no seed")
    L += [f"Outras janelas ({_n(len(O))}): ficam como na v1 ({_n(tot(O, 'original'))} caixas originais)",
          f"Todas: {_n(len(W))} janelas, {_n(tot(W, 'frames'))} frames, {_n(len(cx))} caixas na v2"
          + (f" · {_pl(len(fora), 'janela', 'janelas')} sem linha na janelas_v1.csv (fica a categoria da annotations_v1)"
             if fora else ""),
          "  estados: " + " · ".join(f"{e_} {_n(tot(W, e_))}" for e_ in ESTADOS),
          "",
          f"Ficheiros em {pasta}/ (só de leitura):"]
    for x, est in zip(("anot", "caixas", "readme", "sums"), estados):
        L.append(f"  {nomes[x].name}  {est}" + (f" · sha256 {sha[x]}" if x in sha else ""))
    L += [f"  Para confirmar mais tarde: (cd {pasta} && sha256sum -c SHA256SUMS)", ""]
    linha = (f"- {datetime.date.today().isoformat()}: Guião C, C10 Dataset WCE v2 (anotação v{VERSAO_DATASET}; "
             f"scripts/19_c_dataset_v2.py v{VERSAO}, commit {commit}): {pasta}/ com annotations_v2.csv (sha256 "
             f"{sha['anot'][:16]}; uma linha por frame: as colunas da annotations_v1 sem mudanças, mais o estado e a caixa "
             f"da v2), caixas_v2.csv (sha256 {sha['caixas'][:16]}; {_n(len(cx))} caixas), README.md (sha256 "
             f"{sha['readme'][:16]}; origem, regras e limitações) e SHA256SUMS. Regra de 24 out: caixas originais + as "
             f"{_n(na)} caixas novas aceites; as rejeitadas ({_n(tot(A, 'nova_rejeitada'))}), as não avaliáveis "
             f"({_n(tot(A, 'nova_nao_avaliavel'))}) e os {_n(n_ctrl)} controlos às cegas não entram. ALL-SEEDBOX: "
             f"{_n(oa)} → {_n(oa + na)} caixas ({_pct(na, oa)}), "
             + ("igual à análise C9" if txt_c9.startswith("confere") else "sem conferir com a análise C9")
             + f". A partir de {Path(args.respostas).name} sha256 {sha_r[:16]} e {Path(args.classes).name} sha256 "
             f"{ev._sha256(args.classes)[:16]}.")
    L += ["Linha para o docs/decisoes.md:", linha]
    texto = "\n".join(L)
    rs = Path(args.resumo)
    rs.parent.mkdir(parents=True, exist_ok=True)
    rs.write_text(texto + "\n", encoding="utf-8")
    print(texto)
    return {"W": W, "A": A, "O": O, "caixas": cx, "estados": estados, "sha256": sha, "texto": texto, "linha": linha,
            "conferencia_avaliador": txt_av, "conferencia_c9": txt_c9, "pasta": pasta, "controlos": ctrl}


if __name__ == "__main__":
    try:
        main()
    except ev.ErroAvaliacao as e:
        print(f"ERRO: {e}", file=sys.stderr)
        sys.exit(2)
