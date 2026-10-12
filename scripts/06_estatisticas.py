#!/usr/bin/env python3
"""
Passo 0.5 — Estatísticas do dataset (só lê as tabelas derivadas; não toca no dataset).

Calcula, para o capítulo de dados e para o protocolo:
  1. tamanho das caixas do médico por categoria (área em % do frame e em px², largura, altura) e os
     tercis de tamanho — o estrato "tamanho" do protocolo (pequena / média / grande);
  2. sequências de frames seguidos com caixa em cada janela (quantas, comprimento, buracos entre elas,
     distância máxima caixa–seed) e o valor inicial de L_max (até onde a propagação vai a partir do seed);
  3. distância ao seed dos frames com caixa, pelos estratos do guião (1–3, 4–6, 7–12, > 12 frames);
  4. categorias, janelas com caixa e janelas com caixa no seed, por split.

Regras:
  - só janelas não excluídas; as contas principais são das janelas de lesão (lesao = 1);
  - caixas de cada janela tal como estão (box_*), sem a consolidação do passo 0.4;
  - tamanhos contados uma vez por caixa: um frame repetido em duas janelas com a mesma caixa conta 1 vez;
  - frames sem ficheiro de caixa (os buracos de exportação nos offsets −70/−39/−8) não partem uma
    sequência: não se sabe o que lá estava.

Uso, a partir da pasta do projeto:
    python scripts/06_estatisticas.py
Lê:    data/derived/annotations_v1.csv · data/derived/janelas_v1.csv   (com categoria e lesao, do passo 0.3)
Gera:  outputs/stats/stats_dataset.xlsx · outputs/stats/fig_*.png
       outputs/06_estatisticas_resumo.txt   ← é este que se lê
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

K3 = ["split", "caso", "seed"]
COM_CAIXA = ["medico_seed", "medico_vizinho"]
SEM_INFO = ["sem_ficheiro", "formato_invalido"]
CATEGORIAS = ["angiectasia", "erosao", "ulcera", "polipo", "eritema", "xantelasma", "linfangiectasia", "sangue",
              "hiperplasia_nodular", "estenose", "subepitelial", "outra_patologia", "nao_lesao", "qualidade"]
PRINCIPAIS = CATEGORIAS[:6]
ESTRATOS = [(1, 3, "1–3"), (4, 6, "4–6"), (7, 12, "7–12"), (13, 10_000, "> 12")]
PCT = [50, 90, 95]

# figuras: uma cor (azul) para uma só série; texto e grelha em cinzentos neutros
AZUL, TINTA, TINTA2, GRELHA, FUNDO = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"


def pct(s, q):
    s = pd.Series(s).dropna()
    return float(np.percentile(s, q)) if len(s) else float("nan")


def ler(args):
    for f in (args.tabela, args.janelas):
        if not Path(f).is_file():
            sys.exit(f"Não encontrei {f}.")
    jan = pd.read_csv(args.janelas, dtype=str, keep_default_na=False)
    if "categoria" not in jan.columns:
        sys.exit("As janelas ainda não têm categoria: corre primeiro o 05_rotulos.py --aplicar.")
    for c in ["n_box", "seed_tem_box", "excluida"]:
        jan[c] = pd.to_numeric(jan[c], errors="coerce").fillna(0).astype(int)
    cols = ["split", "caso", "seed", "frame", "offset", "box_src", "box_x1", "box_y1", "box_x2", "box_y2",
            "img_w", "img_h", "excluida", "categoria", "lesao"]
    print("A ler a tabela dos frames…")
    t = pd.read_csv(args.tabela, usecols=cols, keep_default_na=False, na_values=[""],
                    dtype={"split": str, "caso": str, "seed": str, "box_src": str, "categoria": str, "lesao": str})
    t = t[t["excluida"].fillna(0).astype(int) == 0].copy()
    t["categoria"] = t["categoria"].fillna("")
    t["lesao"] = t["lesao"].fillna("").astype(str).str.replace(".0", "", regex=False)
    t["frame"] = t["frame"].astype(int)
    t["offset"] = t["offset"].astype(int)
    return jan[jan["excluida"] == 0].copy(), t


def caixas(t):
    """Uma linha por caixa (frames com caixa), com tamanhos; e a versão sem repetições entre janelas."""
    c = t[t["box_src"].isin(COM_CAIXA)].copy()
    c["larg_px"] = (c["box_x2"] - c["box_x1"]) * c["img_w"]
    c["alt_px"] = (c["box_y2"] - c["box_y1"]) * c["img_h"]
    c["area_px"] = c["larg_px"] * c["alt_px"]
    c["area_pct"] = 100 * (c["box_x2"] - c["box_x1"]) * (c["box_y2"] - c["box_y1"])
    c["dist"] = c["offset"].abs()
    c = c[c["area_pct"].notna() & (c["area_pct"] >= 0)]       # sem coordenadas ou sem tamanho de imagem: fora
    chave = [c["split"], c["caso"], c["frame"]] + [c[k].round(4) for k in ["box_x1", "box_y1", "box_x2", "box_y2"]]
    c["_k"] = pd.Series(list(zip(*chave)), index=c.index)
    unicas = c.sort_values("dist").drop_duplicates("_k")      # a janela do seed mais próximo fica com a caixa
    return c.drop(columns="_k"), unicas.drop(columns="_k")


def sequencias(t):
    """Por janela com caixas: sequências de frames seguidos com caixa (os frames sem informação não partem)."""
    linhas, runs, gaps = [], [], []
    sub = t[t["box_src"].isin(COM_CAIXA + ["vazio"])].sort_values(K3 + ["frame"])
    tem = set(map(tuple, t.loc[t["box_src"].isin(COM_CAIXA), K3].drop_duplicates().to_numpy()))
    for k, g in sub.groupby(K3, sort=False):
        if k not in tem:
            continue
        fr = g["frame"].to_numpy()
        cx = g["box_src"].isin(COM_CAIXA).to_numpy()
        seed = int(k[2])
        corridas, ini = [], None
        for i in range(len(fr)):
            if cx[i] and ini is None:
                ini = i
            if ini is not None and (not cx[i] or i == len(fr) - 1):
                fim = i if cx[i] else i - 1
                corridas.append((int(fr[ini]), int(fr[fim])))
                ini = None
        comp = [b - a + 1 for a, b in corridas]
        bur = [corridas[j + 1][0] - corridas[j][1] - 1 for j in range(len(corridas) - 1)]
        off = fr[cx] - seed
        do_seed = next(((a, b) for a, b in corridas if a <= seed <= b), None)
        linhas.append(dict(split=k[0], caso=k[1], seed=k[2], categoria=g["categoria"].iloc[0],
                           lesao=g["lesao"].iloc[0], n_caixas=int(cx.sum()), n_sequencias=len(corridas),
                           comprimentos=" ".join(map(str, comp)), maior_sequencia=max(comp),
                           buracos=" ".join(map(str, bur)), dist_max=int(np.abs(off).max()),
                           dist_max_antes=int(-off[off < 0].min()) if (off < 0).any() else 0,
                           dist_max_depois=int(off[off > 0].max()) if (off > 0).any() else 0,
                           seed_com_caixa=int(0 in off),
                           seq_seed_antes=seed - do_seed[0] if do_seed else np.nan,
                           seq_seed_depois=do_seed[1] - seed if do_seed else np.nan))
        les = g["lesao"].iloc[0]
        runs += [(les, x) for x in comp]
        gaps += [(les, x) for x in bur]
    return pd.DataFrame(linhas), pd.DataFrame(runs, columns=["lesao", "comprimento"]), \
        pd.DataFrame(gaps, columns=["lesao", "buraco"])


def estrato(d):
    for a, b, nome in ESTRATOS:
        if a <= d <= b:
            return nome
    return "seed"


def figuras(unicas_les, seq_sb, runs_les, lmax, pasta):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except ImportError:
        return ["(sem figuras: falta o matplotlib — pip install matplotlib)"]
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRELHA, "axes.labelcolor": TINTA2, "xtick.color": TINTA2,
                         "ytick.color": TINTA2, "axes.titlecolor": TINTA, "figure.facecolor": FUNDO,
                         "axes.facecolor": FUNDO, "savefig.facecolor": FUNDO})
    feitas = []

    def eixo(ax, grelha="x"):
        ax.grid(axis=grelha, color=GRELHA, linewidth=0.6)
        if grelha == "y":
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_axisbelow(True)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)

    # 1. área das caixas por categoria (caixas sem repetição, janelas de lesão)
    cats = [c for c in CATEGORIAS if c in set(unicas_les["categoria"])]
    cats.sort(key=lambda c: unicas_les.loc[unicas_les["categoria"] == c, "area_pct"].median())
    dados = [unicas_les.loc[unicas_les["categoria"] == c, "area_pct"].clip(lower=0.01) for c in cats]
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(cats) + 1.2))
    estilo = dict(widths=0.5, showfliers=False, patch_artist=True,
                  boxprops=dict(facecolor="#b7d3f6", edgecolor=AZUL, linewidth=1),
                  medianprops=dict(color="#104281", linewidth=2), whiskerprops=dict(color=AZUL, linewidth=1),
                  capprops=dict(color=AZUL, linewidth=1))
    try:
        ax.boxplot(dados, orientation="horizontal", **estilo)       # matplotlib ≥ 3.10
    except TypeError:
        ax.boxplot(dados, vert=False, **estilo)                     # versões mais antigas
    ax.set_yticks(range(1, len(cats) + 1), [f"{c} (n={len(d)})" for c, d in zip(cats, dados)])
    ax.set_xscale("log")
    ax.set_xlabel("área da caixa (% do frame, escala logarítmica)")
    ax.set_title("Tamanho das caixas do médico por categoria", loc="left")
    eixo(ax)
    fig.tight_layout()
    fig.savefig(pasta / "fig_area_caixas_categoria.png", dpi=200)
    plt.close(fig)
    feitas.append(str(pasta / "fig_area_caixas_categoria.png"))

    # 2. comprimento das sequências de frames com caixa
    fig, ax = plt.subplots(figsize=(7, 3.2))
    v = runs_les["comprimento"]
    ax.hist(v, bins=np.arange(0.5, min(v.max(), 80) + 1.5, 1), color=AZUL, edgecolor=FUNDO, linewidth=0.5)
    for q, alt in zip(PCT, [0.94, 0.84, 0.74]):              # rótulos a alturas diferentes: não se tapam
        x = pct(v, q)
        ax.axvline(x, color=TINTA2, linewidth=1, linestyle=(0, (3, 2)))
        ax.annotate(f"P{q} = {x:.0f}", (x, ax.get_ylim()[1] * alt), xytext=(3, 0), textcoords="offset points",
                    color=TINTA2, fontsize=8)
    ax.set_xlabel("frames seguidos com caixa")
    ax.set_ylabel("sequências")
    ax.set_title("Comprimento das sequências de frames com caixa (janelas de lesão)", loc="left")
    eixo(ax, "y")
    fig.tight_layout()
    fig.savefig(pasta / "fig_sequencias.png", dpi=200)
    plt.close(fig)
    feitas.append(str(pasta / "fig_sequencias.png"))

    # 3. distância máxima caixa–seed por janela (ALL-SEEDBOX), com L_max
    fig, ax = plt.subplots(figsize=(7, 3.2))
    v = seq_sb["dist_max"]
    ax.hist(v, bins=np.arange(-0.5, v.max() + 1.5, 1), color=AZUL, edgecolor=FUNDO, linewidth=0.5)
    ax.axvline(lmax, color=TINTA2, linewidth=1, linestyle=(0, (3, 2)))
    direita = lmax > 0.6 * ax.get_xlim()[1]                  # perto da borda: o rótulo vai para a esquerda da linha
    ax.annotate(f"P95 = {lmax} frames (L_max inicial)", (lmax, ax.get_ylim()[1] * 0.92),
                xytext=(-4 if direita else 4, 0), textcoords="offset points", color=TINTA2, fontsize=8,
                ha="right" if direita else "left")
    ax.set_xlabel("distância do seed ao frame com caixa mais afastado (frames)")
    ax.set_ylabel("janelas")
    ax.set_title("Até onde chegam as caixas do médico a partir do seed (janelas ALL-SEEDBOX)", loc="left")
    eixo(ax, "y")
    fig.tight_layout()
    fig.savefig(pasta / "fig_distancia_seed.png", dpi=200)
    plt.close(fig)
    feitas.append(str(pasta / "fig_distancia_seed.png"))
    return feitas


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tabela", default="data/derived/annotations_v1.csv")
    ap.add_argument("--janelas", default="data/derived/janelas_v1.csv")
    ap.add_argument("--saida", default="outputs/stats")
    ap.add_argument("--resumo", default="outputs/06_estatisticas_resumo.txt")
    args = ap.parse_args()
    pasta = Path(args.saida)
    pasta.mkdir(parents=True, exist_ok=True)

    jan, t = ler(args)
    n_cx_total = int(t["box_src"].isin(COM_CAIXA).sum())
    todas, unicas = caixas(t)
    les_u = unicas[unicas["lesao"] == "1"]
    print("A calcular as sequências por janela…")
    seq, runs, gaps = sequencias(t)
    sb = jan[(jan["lesao"] == "1") & (jan["seed_tem_box"] == 1)]
    chaves_sb = set(zip(sb["split"], sb["caso"], sb["seed"]))
    seq_sb = seq[[k in chaves_sb for k in zip(seq["split"], seq["caso"], seq["seed"])]]
    seq_les = seq[seq["lesao"] == "1"]
    runs_les, gaps_les = runs[runs["lesao"] == "1"], gaps[gaps["lesao"] == "1"]
    seeds = todas[(todas["box_src"] == "medico_seed") & (todas["lesao"] == "1")]

    # tercis: caixa a caixa (para as métricas de localização) e caixa do seed (para estratificar janelas)
    if len(les_u) < 3 or len(seeds) < 3:
        sys.exit("Caixas com tamanho válido a menos para calcular tercis — confirma a tabela dos frames.")
    t_cx = (pct(les_u["area_pct"], 100 / 3), pct(les_u["area_pct"], 200 / 3))
    t_sd = (pct(seeds["area_pct"], 100 / 3), pct(seeds["area_pct"], 200 / 3))
    lmax = int(np.ceil(pct(seq_sb["dist_max"], 95)))

    # ---------- tabelas ----------
    def resumo_tamanho(df, por):
        g = df.groupby(por)
        r = pd.DataFrame({"n_caixas": g.size(),
                          "area_pct_mediana": g["area_pct"].median(), "area_pct_P25": g["area_pct"].quantile(.25),
                          "area_pct_P75": g["area_pct"].quantile(.75), "area_pct_P10": g["area_pct"].quantile(.10),
                          "area_pct_P90": g["area_pct"].quantile(.90), "area_px_mediana": g["area_px"].median(),
                          "largura_px_mediana": g["larg_px"].median(), "altura_px_mediana": g["alt_px"].median()})
        return r.reindex([c for c in CATEGORIAS if c in r.index]).round(3)

    tab_cx = resumo_tamanho(unicas, "categoria")
    tab_sd = resumo_tamanho(seeds, "categoria")
    les_u = les_u.assign(tercil=pd.cut(les_u["area_pct"], [-np.inf, t_cx[0], t_cx[1], np.inf],
                                       labels=["pequena", "média", "grande"]))
    tab_tercis = les_u.groupby(["categoria", "tercil"], observed=False).size().unstack(fill_value=0)
    tab_tercis = tab_tercis.reindex([c for c in CATEGORIAS if c in tab_tercis.index])
    viz = todas[(todas["lesao"] == "1") & (todas["offset"] != 0)].copy()
    viz["estrato"] = viz["dist"].map(estrato)
    viz["direcao"] = np.where(viz["offset"] < 0, "antes do seed", "depois do seed")
    tab_dist = viz.groupby(["estrato", "direcao"]).size().unstack(fill_value=0).reindex([e[2] for e in ESTRATOS])
    us = jan.copy()
    tab_split = pd.DataFrame({
        (sp, nome): [int(((us["split"] == sp) & (us["categoria"] == c) & cond(us)).sum()) for c in CATEGORIAS]
        for sp in sorted(us["split"].unique())
        for nome, cond in [("janelas", lambda d: d["n_box"] >= 0), ("c/ caixa", lambda d: d["n_box"] > 0),
                           ("c/ caixa no seed", lambda d: d["seed_tem_box"] == 1)]}, index=CATEGORIAS)
    tab_runs = pd.DataFrame({
        "medida": ["comprimento das sequências (frames)", "buracos entre sequências (frames)",
                   "sequências por janela", "distância máxima caixa–seed (janelas de lesão com caixa)",
                   "distância máxima caixa–seed (ALL-SEEDBOX)", "sequência do seed: frames antes do seed",
                   "sequência do seed: frames depois do seed"],
        **{f"P{q}": [pct(runs_les["comprimento"], q), pct(gaps_les["buraco"], q), pct(seq_les["n_sequencias"], q),
                      pct(seq_les["dist_max"], q), pct(seq_sb["dist_max"], q), pct(seq_sb["seq_seed_antes"], q),
                      pct(seq_sb["seq_seed_depois"], q)] for q in PCT},
        "n": [len(runs_les), len(gaps_les), len(seq_les), len(seq_les), len(seq_sb),
              int(seq_sb["seq_seed_antes"].notna().sum()), int(seq_sb["seq_seed_depois"].notna().sum())]})

    # ---------- Excel ----------
    xlsx = pasta / "stats_dataset.xlsx"
    resumo = pd.DataFrame({"medida": [
        "janelas não excluídas", "janelas de lesão", "ALL-SEEDBOX (lesão com caixa no seed)",
        "caixas do médico (por janela, com repetições)", "caixas sem repetição", "caixas sem repetição, janelas de lesão",
        "caixas degeneradas (largura ou altura < 2 px)", "tercis da área, caixa a caixa (% do frame)",
        "tercis da área, caixa do seed (% do frame)", "L_max inicial (P95 da distância máxima caixa–seed)"],
        "valor": [len(jan), int((jan["lesao"] == "1").sum()), len(sb), len(todas), len(unicas), len(les_u),
                  int(((unicas["larg_px"] < 2) | (unicas["alt_px"] < 2)).sum()),
                  f"{t_cx[0]:.2f} | {t_cx[1]:.2f}", f"{t_sd[0]:.2f} | {t_sd[1]:.2f}", lmax]})
    try:
        with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
            resumo.to_excel(xw, sheet_name="resumo", index=False)
            tab_cx.to_excel(xw, sheet_name="caixas_por_categoria")
            tab_sd.to_excel(xw, sheet_name="caixa_do_seed")
            tab_tercis.to_excel(xw, sheet_name="tercis")
            tab_runs.round(1).to_excel(xw, sheet_name="sequencias_percentis", index=False)
            seq.to_excel(xw, sheet_name="sequencias_por_janela", index=False)
            tab_dist.to_excel(xw, sheet_name="distancia_ao_seed")
            tab_split.to_excel(xw, sheet_name="categorias_por_split")
        xlsx_txt = str(xlsx)
    except ImportError:
        xlsx_txt = "(sem Excel: falta o openpyxl — pip install openpyxl)"
    figs = figuras(les_u, seq_sb, runs_les, lmax, pasta)

    # ---------- resumo em texto ----------
    L = ["Estatísticas do dataset (passo 0.5)"]
    sec = lambda titulo: L.append(f"\n== {titulo} " + "=" * max(0, 64 - len(titulo)))
    sec("Base")
    L.append(f"  janelas não excluídas: {len(jan)} · de lesão: {int((jan['lesao'] == '1').sum())} · "
             f"ALL-SEEDBOX: {len(sb)}")
    L.append(f"  caixas do médico: {len(todas)} por janela (com repetições) · {len(unicas)} sem repetição · "
             f"{len(les_u)} sem repetição em janelas de lesão")
    L.append(f"  caixas degeneradas (largura ou altura < 2 px): {int(((unicas['larg_px'] < 2) | (unicas['alt_px'] < 2)).sum())}"
             f" · caixas sem tamanho válido (deixadas de fora): {n_cx_total - len(todas)}")
    sec("Tamanho das caixas (caixas sem repetição)")
    L.append(f"  {'categoria':<20}{'n':>6}{'área % mediana':>16}{'P25–P75':>14}{'px² mediana':>13}{'larg×alt px':>13}")
    for c, r in tab_cx.iterrows():
        L.append(f"  {c:<20}{int(r['n_caixas']):>6}{r['area_pct_mediana']:>16.2f}"
                 f"{r['area_pct_P25']:>7.2f}–{r['area_pct_P75']:<6.2f}{r['area_px_mediana']:>13.0f}"
                 f"{r['largura_px_mediana']:>7.0f}×{r['altura_px_mediana']:<5.0f}")
    sec("Tercis de tamanho (estrato do protocolo)")
    L.append(f"  caixa a caixa, janelas de lesão: pequena < {t_cx[0]:.2f}% ≤ média < {t_cx[1]:.2f}% ≤ grande"
             f"   (em px, num frame 320×320: {t_cx[0] / 100 * 320 * 320:.0f} e {t_cx[1] / 100 * 320 * 320:.0f} px²)")
    L.append(f"  caixa do seed, ALL-SEEDBOX:      pequena < {t_sd[0]:.2f}% ≤ média < {t_sd[1]:.2f}% ≤ grande")
    L.append("  caixas por tercil (caixa a caixa):")
    for c, r in tab_tercis.iterrows():
        L.append(f"    {c:<20}" + "  ".join(f"{k}: {int(v)}" for k, v in r.items()))
    sec("Sequências de frames seguidos com caixa (janelas de lesão)")
    for r in tab_runs.itertuples(index=False):
        L.append(f"  {r.medida:<58} P50 {r.P50:>5.1f} · P90 {r.P90:>5.1f} · P95 {r.P95:>5.1f}  (n = {r.n})")
    L.append(f"  janelas com uma só sequência: {int((seq_les['n_sequencias'] == 1).sum())} de {len(seq_les)}"
             f" · com caixa só no seed: {int((seq_les['n_caixas'] == 1).sum() if len(seq_les) else 0)}")
    sec("L_max inicial (até onde propagar a partir do seed)")
    L.append(f"  L_max = {lmax} frames: em 95 % das janelas ALL-SEEDBOX, todas as caixas do médico estão a ≤ {lmax}"
             f" frames do seed.")
    L.append(f"  (leitura literal do guião, P95 do comprimento das sequências: {pct(runs_les['comprimento'], 95):.0f} frames)")
    sec("Distância ao seed dos frames com caixa (sem o seed; janelas de lesão)")
    tot = int(tab_dist.to_numpy().sum())
    for e, r in tab_dist.iterrows():
        n = int(r.sum())
        L.append(f"  {e:>5} frames: {n:>6} ({100 * n / tot:4.1f} %)  · " + " · ".join(f"{k}: {int(v)}" for k, v in r.items()))
    sec("Categorias por split (janelas · com caixa · com caixa no seed)")
    for c in CATEGORIAS:
        partes = [f"{sp.replace('Dataset_', '')}: {tab_split.loc[c, (sp, 'janelas')]} · "
                  f"{tab_split.loc[c, (sp, 'c/ caixa')]} · {tab_split.loc[c, (sp, 'c/ caixa no seed')]}"
                  for sp in sorted(us["split"].unique())]
        L.append(f"  {c:<20}" + "   ".join(partes))
    sec("Ficheiros")
    L.append(f"  {xlsx_txt}")
    for f in figs:
        L.append(f"  {f}")
    texto = "\n".join(L)
    print(texto)
    Path(args.resumo).parent.mkdir(parents=True, exist_ok=True)
    Path(args.resumo).write_text(texto + "\n", encoding="utf-8")
    print(f"\nResumo gravado em {args.resumo}")


if __name__ == "__main__":
    main()
