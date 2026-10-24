#!/usr/bin/env python3
"""
17 — Guião C, passos C1 e C2: lista congelada das caixas a rever na validação humana da anotação expandida.

Não há amostra (decidido a 20 out): entram todas as caixas novas da anotação expandida v1 (A8), isto é, os frames de
expanded_v1.csv com nivel == "alta" e caixa_original == 0 (2 269). Com elas entram, salvo --sem-controlos, controlos às
cegas, sorteados só nas janelas que já têm caixas novas (para não haver seeds novos para estudar) e no máximo um de cada
tipo por janela:
  controlo_original  frames do nível alta que já tinham caixa original; mostra-se a caixa do SAM 2, como nos outros
  controlo_baixa     frames abaixo do limiar (nível baixa) com caixa do SAM 2, com a área da caixa entre os percentis
                     5 e 95 das caixas novas, para não se distinguirem pelo tamanho
O revisor não sabe quais são. Servem para medir se se aceita de mais e se o «aceitar» corresponde a IoU ≥ 0,5 com a
caixa original. Nunca entram no Dataset WCE v2.

Ordem (guião, C2): janelas baralhadas com a semente; dentro de cada janela, a partir do seed, primeiro para trás
(−1, −2, …) e depois para a frente (+1, +2, …), para se seguir a lesão. A ordem das janelas e a das caixas novas não
dependem dos controlos (cada sorteio usa a sua sequência de números aleatórios).

Piloto (C4): janelas de ALL_SEEDBOX sem caixas novas (ficam fora da revisão), de preferência com 3 ou mais frames do
nível alta, com até 10 caixas do SAM 2 cada, do nível alta e abaixo do limiar, misturadas. Não conta para os resultados.

Lê:
  data/derived/expanded_v1.csv     a anotação expandida do A8 (confirma o sha256 e as 2 269 caixas novas)
  data/derived/annotations_v1.csv  estado da anotação original em cada frame (confirma a coluna caixa_original)
  data/derived/janelas_v1.csv      classe de cada janela; tamanho da caixa do seed (tercis do protocolo)
  sets/ALL_SEEDBOX.json, CAL.json, VAL_EX.json, TEST_BOX.json
  a pasta do dataset ($DS)         só para confirmar que existem as imagens do seed e de cada frame da lista

Grava em data/derived/guiao_c/ (fora do git), só de leitura:
  itens_v1.csv          o que a ferramenta de revisão lê: item, split, caso, seed, frame, box_x1..box_y2 (0–1). Mais nada.
  itens_chave_v1.csv    o que fica escondido do revisor: tipo (nova, controlo_original, controlo_baixa), offset, faixa,
                        score, nivel, caixa_original, caixa_ref (anotação original no frame: vazio, sem_ficheiro, invalida
                        ou caixa), area_px, area_caixa, classe, categoria, tercil, cal, val_ex, test_box.
                        Não se abre antes de consolidar a revisão.
  piloto_v1.csv e piloto_chave_v1.csv   o mesmo para o piloto (tipo piloto_alta ou piloto_baixa)
e o resumo em outputs/17_c_itens_resumo.txt, com os sha256 e a linha para o docs/decisoes.md.
Se os ficheiros já existirem com outro conteúdo, não são reescritos: a lista congela-se uma vez (--forcar substitui-a,
só antes de a revisão começar).

Uso, a partir da pasta do projeto, depois do source …/inicio.sh e do export DS=…:
    python scripts/17_c_itens.py
    python scripts/17_c_itens.py --sem-controlos          (sem os controlos às cegas)
Versão 1.0 (23 out 2026).
"""
import argparse
import datetime
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "eval"))
import evaluate as ev  # noqa: E402

VERSAO = "1.0"
K3 = ["split", "caso", "seed"]
CAIXA = ["box_x1", "box_y1", "box_x2", "box_y2"]
COL_ITENS = ["item", "split", "caso", "seed", "frame"] + CAIXA
COL_CHAVE = ["item", "tipo", "split", "caso", "seed", "frame", "offset", "faixa", "score", "nivel", "caixa_original",
             "caixa_ref", "area_px", "area_caixa", "classe", "categoria", "tercil", "cal", "val_ex", "test_box"]
SHA_A8 = "7fafb425be068423"          # expanded_v1.csv do A8 (16 out 2026, commit c4ab30f)
N_NOVAS_A8 = 2269
CONJUNTOS = {"cal": "CAL", "val_ex": "VAL_EX", "test_box": "TEST_BOX"}
NOMES = {"angiectasia": "angiectasia", "erosao": "erosão", "ulcera": "úlcera", "polipo": "pólipo", "eritema": "eritema",
         "xantelasma": "xantelasma", "outras": "outras", "pequena": "pequena", "media": "média", "grande": "grande",
         "sem_caixa": "sem caixa", "Dataset_Teste": "teste", "Dataset_Treino": "treino"}


def _n(x):
    """2269 → «2 269», como na tese."""
    return f"{int(x):,}".replace(",", " ")


def ler_expandida(caminho):
    p = Path(caminho)
    if not p.is_file():
        raise ev.ErroAvaliacao(f"Não encontrei {p} (é gravado pelo scripts/15_a8_expandida.py).")
    e = pd.read_csv(p, dtype={"caso": str, "split": str, "nivel": str}, keep_default_na=False)
    precisas = ["caso", "split", "seed", "frame", "score", "nivel", "offset", "area_px", "caixa_original"] + CAIXA
    falta = [c for c in precisas if c not in e.columns]
    if falta:
        raise ev.ErroAvaliacao(f"{p}: faltam as colunas {falta}.")
    e["caso"] = e["caso"].str.strip().str.zfill(4)
    for c in ("seed", "frame", "offset", "caixa_original"):
        v = pd.to_numeric(e[c], errors="coerce")
        if v.isna().any() or (v != v.round()).any():
            raise ev.ErroAvaliacao(f"{p}: a coluna {c} tem valores que não são números inteiros.")
        e[c] = v.astype(int)
    if ((e["frame"] - e["seed"]) != e["offset"]).any():
        raise ev.ErroAvaliacao(f"{p}: há linhas em que offset ≠ frame − seed.")
    if e.duplicated(K3 + ["frame"]).any():
        raise ev.ErroAvaliacao(f"{p}: há frames repetidos dentro da mesma janela.")
    e["score"] = pd.to_numeric(e["score"], errors="coerce")
    e["area_px"] = pd.to_numeric(e["area_px"], errors="coerce").fillna(0).round().astype(int)
    v = e[CAIXA].apply(pd.to_numeric, errors="coerce")
    for c in CAIXA:
        e[c] = v[c]
    e["tem_caixa"] = (v.notna().all(axis=1) & (v["box_x2"] > v["box_x1"]) & (v["box_y2"] > v["box_y1"])).to_numpy()
    e["area_caixa"] = ((v["box_x2"] - v["box_x1"]) * (v["box_y2"] - v["box_y1"])).where(e["tem_caixa"])
    niveis = set(e["nivel"])
    if not niveis <= {"alta", "baixa"}:
        raise ev.ErroAvaliacao(f"{p}: níveis inesperados {sorted(niveis - {'alta', 'baixa'})} (o A6 decidiu: só alta e baixa).")
    return e.sort_values(K3 + ["frame"]).reset_index(drop=True)


def referencia(cfg, janelas):
    """Estado da anotação original em cada frame e classe/tercil de cada janela, com as regras do avaliador
    (eval/evaluate.py, carregar_referencia), num conjunto só com a referência das caixas."""
    cfg2 = dict(cfg)
    cfg2["conjuntos"] = dict(cfg["conjuntos"])
    cfg2["conjuntos"]["_GUIAO_C"] = {"referencia": ["caixas"]}
    return ev.carregar_referencia(cfg2, "_GUIAO_C", janelas)


def faixa(offset, faixas):
    d = abs(int(offset))
    for lo, hi in faixas:
        if lo <= d <= hi:
            return f">{lo - 1}" if hi >= 100 else f"{lo}-{hi}"
    return ""


def um_por_janela(pool, n, rng):
    """n frames sorteados do pool (já ordenado), no máximo um por janela."""
    if n <= 0 or pool.empty:
        return pool.iloc[0:0]
    escolhidos, vistas = [], set()
    for i in rng.permutation(len(pool)):
        k = (pool["split"].iat[i], pool["caso"].iat[i], pool["seed"].iat[i])
        if k in vistas:
            continue
        vistas.add(k)
        escolhidos.append(i)
        if len(escolhidos) == n:
            break
    return pool.iloc[sorted(escolhidos)]


def ordenar(itens, rng):
    """Janelas baralhadas; dentro de cada uma, do seed para trás (−1, −2, …) e depois para a frente (+1, +2, …)."""
    janelas = sorted(set(zip(itens["split"], itens["caso"], itens["seed"])))
    perm = rng.permutation(len(janelas))
    pos = {janelas[p]: i for i, p in enumerate(perm)}
    t = itens.assign(_j=[pos[k] for k in zip(itens["split"], itens["caso"], itens["seed"])],
                     _lado=(itens["offset"] > 0).astype(int), _d=itens["offset"].abs())
    t = t.sort_values(["_j", "_lado", "_d"], kind="mergesort").drop(columns=["_j", "_lado", "_d"]).reset_index(drop=True)
    t["item"] = np.arange(1, len(t) + 1)
    return t


def piloto(e, janelas_todas, janelas_rev, n_janelas, por_janela, elegivel_baixa, rng):
    alta = e[(e["nivel"] == "alta") & e["tem_caixa"]]
    n_alta = alta.groupby(K3).size().to_dict()
    fora = sorted(k for k in janelas_todas if k not in janelas_rev)
    cand = [k for k in fora if n_alta.get(k, 0) >= 3]
    if len(cand) < n_janelas:
        cand = [k for k in fora if n_alta.get(k, 0) >= 1]
    if len(cand) < n_janelas:
        cand = fora
    if not cand or n_janelas <= 0:
        return e.iloc[0:0].assign(tipo=pd.Series(dtype=str)), []
    escolha = [cand[i] for i in sorted(rng.choice(len(cand), size=min(n_janelas, len(cand)), replace=False))]
    partes = []
    for k in escolha:
        m = (e["split"] == k[0]) & (e["caso"] == k[1]) & (e["seed"] == k[2])
        a = e[m & (e["nivel"] == "alta") & e["tem_caixa"]]
        b = e[m & (e["nivel"] == "baixa") & elegivel_baixa]
        na = min(len(a), por_janela // 2)
        nb = min(len(b), por_janela - na)
        na = min(len(a), por_janela - nb)
        if na:
            partes.append(a.iloc[sorted(rng.choice(len(a), size=na, replace=False))].assign(tipo="piloto_alta"))
        if nb:
            partes.append(b.iloc[sorted(rng.choice(len(b), size=nb, replace=False))].assign(tipo="piloto_baixa"))
    if not partes:
        return e.iloc[0:0].assign(tipo=pd.Series(dtype=str)), escolha
    return pd.concat(partes, ignore_index=True), escolha


def imagens_em_falta(ds, itens):
    """Imagens do seed e de cada frame da lista que não existem no dataset (mesma regra do tools/anotar.py)."""
    falta = []
    for (sp, caso, seed), g in itens.groupby(K3, sort=True):
        pasta = Path(ds) / sp / caso / "GT_com_classes" / str(seed)
        rx = re.compile(rf"^{re.escape(caso)}_(\d+)_{seed}\.jpe?g$", re.I)
        existem = set()
        if pasta.is_dir():
            with os.scandir(pasta) as it:
                for f in it:
                    m = rx.match(f.name)
                    if m:
                        existem.add(int(m.group(1)))
        for fr in [int(seed)] + sorted(int(x) for x in g["frame"]):
            if fr not in existem:
                falta.append(f"{sp}/{caso}/{seed}, frame {fr}" + (" (o seed)" if fr == int(seed) else ""))
    return falta


def texto_csv(df, colunas):
    t = df[colunas].copy()
    return t.to_csv(index=False, float_format="%.6f")


def gravar_congelados(ficheiros, forcar):
    """ficheiros: [(caminho, texto)]. Tudo ou nada: se algum já existir com outro conteúdo (e sem --forcar), não grava
    nenhum. Os ficheiros ficam só de leitura."""
    estados = []
    for p, txt in ficheiros:
        if p.exists():
            if p.read_text(encoding="utf-8") == txt:
                estados.append("igual ao que já existia")
                continue
            if not forcar:
                raise ev.ErroAvaliacao(
                    f"{p} já existe com outro conteúdo. A lista da revisão congela-se uma vez: não a reescrevo. Se a "
                    "revisão ainda não começou e é mesmo para a substituir, corre com --forcar e regista o motivo no "
                    "docs/decisoes.md.")
            estados.append("substituído (--forcar)")
        else:
            estados.append("gravado")
    for (p, txt), est in zip(ficheiros, estados):
        if est.startswith("igual"):
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=p.parent, delete=False,
                                         prefix=f".{p.name}.", suffix=".tmp") as t:
            t.write(txt)
            t.flush()
            os.fsync(t.fileno())
        os.chmod(t.name, 0o444)
        os.replace(t.name, p)
    return estados


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--ds", default=os.environ.get("DS"), help="pasta do dataset (por omissão, $DS)")
    a.add_argument("--config", default="configs/avaliacao.yaml")
    a.add_argument("--expandida", default="data/derived/expanded_v1.csv")
    a.add_argument("--saida-dir", default="data/derived/guiao_c")
    a.add_argument("--resumo", default="outputs/17_c_itens_resumo.txt")
    a.add_argument("--semente", type=int, default=0)
    a.add_argument("--controlos-original", type=int, default=100)
    a.add_argument("--controlos-baixa", type=int, default=100)
    a.add_argument("--sem-controlos", action="store_true", help="sem os controlos às cegas")
    a.add_argument("--piloto-janelas", type=int, default=5)
    a.add_argument("--piloto-por-janela", type=int, default=10)
    a.add_argument("--esperadas", type=int, default=N_NOVAS_A8, help="caixas novas esperadas (A8: 2 269)")
    a.add_argument("--sha-expandida", default=SHA_A8, help="início do sha256 esperado do expanded_v1.csv ('' não confirma)")
    a.add_argument("--forcar", action="store_true", help="substitui uma lista já gravada com outro conteúdo")
    args = a.parse_args(argv)
    if not args.ds or not Path(args.ds).is_dir():
        raise ev.ErroAvaliacao('Falta a pasta do dataset: faz o export DS=…/raw/Dataset_WCE (ou usa --ds "$DS").')
    if args.sem_controlos:
        args.controlos_original = args.controlos_baixa = 0
    cfg = ev.ler_config(args.config)
    est = cfg["estratos"]

    # 1. anotação expandida do A8 e janelas
    sha_e = ev._sha256(args.expandida) if Path(args.expandida).is_file() else ""
    e = ler_expandida(args.expandida)
    if args.sha_expandida and not sha_e.startswith(args.sha_expandida):
        raise ev.ErroAvaliacao(f"{args.expandida} não é o do A8: sha256 {sha_e[:16]}…, esperado {args.sha_expandida}…. "
                               "A revisão tem de partir da anotação expandida v1 congelada.")
    conj = Path(cfg["caminhos"]["conjuntos"])
    todas = ev.ler_conjunto(conj / "ALL_SEEDBOX.json")
    em_e = set(zip(e["split"], e["caso"], e["seed"]))
    if em_e != set(todas):
        raise ev.ErroAvaliacao(f"As janelas de {args.expandida} ({len(em_e)}) não são as de ALL_SEEDBOX ({len(todas)}).")
    marca = {nome: set(ev.ler_conjunto(conj / f"{f}.json")) for nome, f in CONJUNTOS.items()}

    # 2. anotação original e janelas (regras do avaliador)
    ref, info = referencia(cfg, todas)
    r = ref[K3 + ["frame", "caixa"]].rename(columns={"caixa": "caixa_ref"})
    e = e.merge(r, on=K3 + ["frame"], how="left")
    if e["caixa_ref"].isna().any():
        x = e[e["caixa_ref"].isna()].iloc[0]
        raise ev.ErroAvaliacao(f"{int(e['caixa_ref'].isna().sum())} frames de {args.expandida} não estão em "
                               f"annotations_v1.csv (ex.: {x['split']}/{x['caso']}/{x['seed']}, frame {x['frame']}).")
    mal = (e["caixa_original"] == 1) != (e["caixa_ref"] == "caixa")
    if mal.any():
        x = e[mal].iloc[0]
        raise ev.ErroAvaliacao(f"A coluna caixa_original de {args.expandida} não bate com annotations_v1.csv em "
                               f"{int(mal.sum())} frames (ex.: {x['split']}/{x['caso']}/{x['seed']}, frame {x['frame']}).")
    jinfo = info.set_index(K3)[["categoria", "classe", "tercil"]]
    e = e.join(jinfo, on=K3)
    e["faixa"] = [faixa(o, est["distancia"]) for o in e["offset"]]
    chaves = list(zip(e["split"], e["caso"], e["seed"]))
    for nome, ks in marca.items():
        e[nome] = np.array([k in ks for k in chaves], dtype=int)

    # 3. caixas novas, controlos e piloto
    alta = (e["nivel"] == "alta").to_numpy()
    novas_todas = e[alta & (e["caixa_original"] == 0).to_numpy()]
    if len(novas_todas) != args.esperadas:
        raise ev.ErroAvaliacao(f"Encontrei {_n(len(novas_todas))} caixas novas (nível alta, sem caixa original) e "
                               f"esperava {_n(args.esperadas)}, as do A8. Não gravei nada.")
    sem_caixa = novas_todas[~novas_todas["tem_caixa"]]
    novas = novas_todas[novas_todas["tem_caixa"]].assign(tipo="nova")
    if novas.empty:
        raise ev.ErroAvaliacao("Não há caixas novas com caixa para rever.")
    janelas_rev = set(zip(novas["split"], novas["caso"], novas["seed"]))
    lo, hi = np.percentile(novas["area_caixa"].to_numpy(dtype=float), [5, 95])
    elegivel_baixa = (e["tem_caixa"] & (e["area_caixa"] >= lo) & (e["area_caixa"] <= hi)).to_numpy()
    na_rev = np.array([k in janelas_rev for k in chaves])
    pool_o = e[na_rev & alta & (e["caixa_original"] == 1).to_numpy() & e["tem_caixa"].to_numpy()]
    pool_b = e[na_rev & (e["nivel"] == "baixa").to_numpy() & elegivel_baixa]
    c_o = um_por_janela(pool_o, args.controlos_original, np.random.default_rng([args.semente, 2])).assign(tipo="controlo_original")
    c_b = um_por_janela(pool_b, args.controlos_baixa, np.random.default_rng([args.semente, 3])).assign(tipo="controlo_baixa")
    itens = ordenar(pd.concat([novas, c_o, c_b], ignore_index=True), np.random.default_rng([args.semente, 1]))
    pil, jan_pil = piloto(e, todas, janelas_rev, args.piloto_janelas, args.piloto_por_janela, elegivel_baixa,
                          np.random.default_rng([args.semente, 4]))
    pil = ordenar(pil, np.random.default_rng([args.semente, 5])) if len(pil) else pd.DataFrame(columns=COL_CHAVE)

    # 4. imagens
    falta = imagens_em_falta(args.ds, pd.concat([itens[K3 + ["frame"]], pil[K3 + ["frame"]]], ignore_index=True))
    if falta:
        raise ev.ErroAvaliacao(f"Faltam {len(falta)} imagens no dataset ({args.ds}), por exemplo: " + "; ".join(falta[:5])
                               + ". Confirma o $DS (tem de ser …/raw/Dataset_WCE). Não gravei nada.")

    # 5. gravar (tudo ou nada; só de leitura)
    for t in (itens, pil):
        for c in ("item", "seed", "frame", "offset", "caixa_original", "area_px", "cal", "val_ex", "test_box"):
            if c in t.columns and len(t):
                t[c] = t[c].astype(int)
    sd = Path(args.saida_dir)
    nomes = {"itens": sd / "itens_v1.csv", "chave": sd / "itens_chave_v1.csv",
             "piloto": sd / "piloto_v1.csv", "piloto_chave": sd / "piloto_chave_v1.csv"}
    textos = [(nomes["itens"], texto_csv(itens, COL_ITENS)), (nomes["chave"], texto_csv(itens, COL_CHAVE)),
              (nomes["piloto"], texto_csv(pil, COL_ITENS)), (nomes["piloto_chave"], texto_csv(pil, COL_CHAVE))]
    estados = gravar_congelados(textos, args.forcar)
    sha = {k: ev._sha256(p) for k, p in nomes.items()}

    # 6. resumo
    git = ev._estado_git(RAIZ)
    commit = git["commit"] or "sem git"
    hoje = datetime.date.today().isoformat()
    W = len(janelas_rev)
    por_j = itens.groupby(K3).size()
    classes = list(est["classes_principais"]) + ["outras"]
    faixas = [f">{lo_ - 1}" if hi_ >= 100 else f"{lo_}-{hi_}" for lo_, hi_ in est["distancia"]]
    sf = novas[novas["caixa_ref"] == "sem_ficheiro"]
    inval = novas[novas["caixa_ref"] == "invalida"]
    off_sf = " ".join(f"{o:+d}".replace("-", "−") for o in sorted(set(sf["offset"])))
    n_o, n_b = len(c_o), len(c_b)
    L = [f"17 — Guião C, C1–C2: lista da revisão · {datetime.datetime.now():%Y-%m-%d %H:%M} · "
         + (f"commit {git['commit']}" if git["commit"] else "sem git")
         + (" (com alterações por guardar!)" if git["alteracoes_por_guardar"] else ""),
         f"17_c_itens.py v{VERSAO} · sha256 {ev._sha256(__file__)[:16]} · {args.expandida} sha256 {sha_e[:16]}"
         + (" (o do A8)" if sha_e.startswith(SHA_A8) else ""),
         "",
         f"Caixas novas (nível alta, sem caixa original): {_n(len(novas_todas))}"
         + (f" · {_n(len(sem_caixa))} sem caixa ficam de fora" if len(sem_caixa) else " · todas com caixa"),
         "  por split: " + " · ".join(f"{NOMES[s]} {_n((novas['split'] == s).sum())}"
                                      for s in ("Dataset_Teste", "Dataset_Treino")),
         "  por distância ao seed: " + " · ".join(f"{f.replace('-', '–')} {_n((novas['faixa'] == f).sum())}"
                                                  for f in faixas if (novas["faixa"] == f).any()),
         "  por classe: " + " · ".join(f"{NOMES.get(c, c)} {_n((novas['classe'] == c).sum())}" for c in classes),
         "  por tamanho da caixa do seed: " + " · ".join(f"{NOMES[t]} {_n((novas['tercil'] == t).sum())}"
                                                         for t in ("pequena", "media", "grande", "sem_caixa")
                                                         if (novas["tercil"] == t).any()),
         f"  nas 60 janelas da PF v2: CAL {_n(novas['cal'].sum())} · VAL-EX {_n(novas['val_ex'].sum())}"
         f" · em TEST-BOX {_n(novas['test_box'].sum())}",
         f"  em frames sem ficheiro de caixa ({off_sf or '—'}; falha de exportação, não se sabe se tinham caixa original): "
         f"{_n(len(sf))}" + (f" · com caixa original inválida: {_n(len(inval))}" if len(inval) else ""),
         "",
         (f"Controlos às cegas: {_n(n_o)} com caixa original + {_n(n_b)} abaixo do limiar (área da caixa entre "
          f"{lo:.4f} e {hi:.4f} do frame, P5–P95 das novas), no máximo um de cada tipo por janela"
          if n_o or n_b else "Controlos às cegas: nenhum (--sem-controlos)").replace(".", ","),
         f"Lista: {_n(len(itens))} itens em {_n(W)} janelas · itens por janela: mediana {por_j.median():g}, "
         f"máximo {por_j.max()}".replace(".", ","),
         "  ordem: janelas baralhadas (semente " + str(args.semente) + "); dentro de cada uma, do seed para trás e "
         "depois para a frente",
         f"Piloto: {_n(len(pil))} itens em {len(jan_pil)} janelas sem caixas novas (fora da revisão; "
         f"{_n(len(todas) - W)} janelas de ALL_SEEDBOX não têm caixas novas)",
         "",
         "Ficheiros (só de leitura):"]
    for k, p in nomes.items():
        L.append(f"  {p}  {estados[list(nomes).index(k)]} · sha256 {sha[k]}")
    L += ["  NÃO abras o itens_chave_v1.csv nem o expanded_v1.csv até consolidares a revisão: têm o score, o tipo de "
          "item e a caixa original.", ""]
    if n_o or n_b:
        ctrl = (f"mais {_n(n_o)} controlos às cegas com caixa original e {_n(n_b)} abaixo do limiar (nível baixa; área "
                f"da caixa entre P5 e P95 das novas), sorteados (semente {args.semente}) nas janelas com caixas novas, no "
                "máximo um de cada tipo por janela, que não entram no Dataset WCE v2")
    else:
        ctrl = "sem controlos às cegas"
    sf_txt = (f"; {_n(len(sf))} delas em frames sem ficheiro de caixa, onde não se sabe se havia caixa original"
              if len(sf) else "")
    linha = (f"- {hoje}: Guião C, C1–C2 (scripts/17_c_itens.py v{VERSAO}, commit {commit}): lista da revisão congelada. "
             f"Sem amostra: as {_n(len(novas_todas))} caixas novas da anotação expandida v1 (expanded_v1.csv sha256 "
             f"{sha_e[:16]}; nível alta e sem caixa original{sf_txt}), {ctrl}; {_n(len(itens))} itens em {_n(W)} janelas. Ordem: janelas "
             f"baralhadas (semente {args.semente}); dentro de cada uma, do seed para trás e depois para a frente. Piloto "
             f"(C4): {_n(len(pil))} itens em {len(jan_pil)} janelas sem caixas novas, fora da revisão. "
             f"data/derived/guiao_c/itens_v1.csv sha256 {sha['itens'][:16]}; itens_chave_v1.csv sha256 "
             f"{sha['chave'][:16]} (só se abre depois de consolidar a revisão).")
    L += ["Linha para o docs/decisoes.md:", linha]
    texto = "\n".join(L)
    rs = Path(args.resumo)
    rs.parent.mkdir(parents=True, exist_ok=True)
    rs.write_text(texto + "\n", encoding="utf-8")
    print(texto)
    return {"itens": itens, "piloto": pil, "janelas_piloto": jan_pil, "sha256": sha, "estados": estados,
            "texto": texto, "linha": linha, "area_p5_p95": (lo, hi)}


if __name__ == "__main__":
    try:
        main()
    except ev.ErroAvaliacao as e:
        print(f"ERRO: {e}", file=sys.stderr)
        sys.exit(2)
