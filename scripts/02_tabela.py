#!/usr/bin/env python3
"""
Passo 0.2 — Tabela única de anotação (só lê o dataset, não altera nada).

Junta, para cada frame: a caixa do médico (se houver), a descrição clínica da seed, o valor de
"Similariedade" do emparelhamento e o tamanho real da imagem. Não infere nada: um frame sem caixa
fica "vazio" ou "sem_ficheiro", nunca "negativo".

Regras aplicadas (confirmadas no inventário e nas verificações de 04/10/2026):
  1. Cantos: os dois cantos vêm pela ordem em que a caixa foi desenhada (aparecem as 4 combinações
     de trocas). Ordenam-se por eixo: x1 = mín(x), x2 = máx(x), y1 = mín(y), y2 = máx(y).
  2. Coordenadas normalizadas (0–1) pelo tamanho REAL de cada imagem (há imagens de 256×256).
  3. Origem da caixa (box_src): medico_seed · medico_vizinho · vazio · sem_ficheiro · formato_invalido.
  4. Descrições: limpas (pontos e espaços no fim; vazias e "só extensão" ignoradas); várias → " | ".
  5. Janelas de docs/janelas_excluidas.csv: marcadas (excluida = 1 + motivo), nunca apagadas.

Uso, a partir da pasta do projeto:
    python scripts/02_tabela.py --dataset "<caminho>/Dataset_WCE"

Lê:    data/derived/manifest.csv (passo 0.1) · docs/janelas_excluidas.csv (opcional)
Gera:  data/derived/annotations_v1.csv   uma linha por frame
       data/derived/janelas_v1.csv       uma linha por janela
       outputs/02_tabela_resumo.txt      contagens e verificações  ← é este que se lê
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from PIL import Image
except ImportError:
    sys.exit("Falta o Pillow neste ambiente (pip install pillow).")

K3 = ["split", "caso", "seed"]
SO_EXTENSAO = {"", "png", "jpg", "jpeg"}
COM_CAIXA = ["medico_seed", "medico_vizinho"]


def limpar_desc(d):
    d = str(d).strip(" .")
    return "" if d.lower() in SO_EXTENSAO else d


def tamanho(caminho):
    try:
        with Image.open(caminho) as im:
            return im.size
    except Exception:
        return (np.nan, np.nan)


def unicas(s):
    return [d for d in dict.fromkeys(s) if d]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="pasta Dataset_WCE (cópia de trabalho)")
    ap.add_argument("--manifest", default="data/derived/manifest.csv")
    ap.add_argument("--excluidas", default="docs/janelas_excluidas.csv")
    ap.add_argument("--saida", default="data/derived/annotations_v1.csv")
    ap.add_argument("--janelas", default="data/derived/janelas_v1.csv")
    ap.add_argument("--resumo", default="outputs/02_tabela_resumo.txt")
    args = ap.parse_args()

    ds = Path(args.dataset)
    if not Path(args.manifest).is_file():
        sys.exit(f"Não encontrei {args.manifest}: corre primeiro o 01_inventario.py.")
    for p in (args.saida, args.janelas, args.resumo):
        Path(p).parent.mkdir(parents=True, exist_ok=True)

    print("A ler o manifest…")
    m = pd.read_csv(args.manifest, dtype=str, keep_default_na=False)

    # ---------- 1. frames + caixas ----------
    fr = m.loc[m.tipo == "frame", K3 + ["frame", "caminho"]].rename(columns={"caminho": "ficheiro_img"})
    bx = m.loc[m.tipo == "box", K3 + ["frame", "caminho", "box_estado", "box_valores"]] \
          .rename(columns={"caminho": "ficheiro_box"})
    t = fr.merge(bx, on=K3 + ["frame"], how="left")
    t["box_estado"] = t["box_estado"].fillna("sem_ficheiro")
    t["box_valores"] = t["box_valores"].fillna("")
    t["ficheiro_box"] = t["ficheiro_box"].fillna("")
    t["offset"] = t["frame"].astype(int) - t["seed"].astype(int)

    coord = t["box_estado"] == "coords"
    t["box_src"] = "formato_invalido"
    t.loc[t["box_estado"].isin(["vazio", "so_espacos"]), "box_src"] = "vazio"
    t.loc[t["box_estado"] == "sem_ficheiro", "box_src"] = "sem_ficheiro"
    t.loc[coord & (t["offset"] == 0), "box_src"] = "medico_seed"
    t.loc[coord & (t["offset"] != 0), "box_src"] = "medico_vizinho"

    # ---------- 2. descrição e semelhança por seed ----------
    mt = m.loc[m.tipo == "relatorio_match", K3 + ["imagem_n", "similaridade"]]
    de = m.loc[m.tipo == "relatorio_descricao", ["split", "caso", "imagem_n", "descricao"]].copy()
    de["descricao"] = de["descricao"].map(limpar_desc)
    md = mt.merge(de, on=["split", "caso", "imagem_n"], how="left")
    md["descricao"] = md["descricao"].fillna("")
    md["n_img"] = md["imagem_n"].astype(int)
    md = md.sort_values(K3 + ["n_img"])
    sd = md.groupby(K3, sort=False).agg(
        descricao=("descricao", lambda s: " | ".join(unicas(s))),
        n_descricoes=("descricao", lambda s: len(unicas(s))),
        n_match=("imagem_n", "size"),
        similaridade=("similaridade", lambda s: next((x for x in s if x), "")),
    ).reset_index()
    t = t.merge(sd, on=K3, how="left")
    t["descricao"] = t["descricao"].fillna("")
    t["n_descricoes"] = t["n_descricoes"].fillna(0).astype(int)
    t["n_match"] = t["n_match"].fillna(0).astype(int)
    t["similaridade"] = t["similaridade"].fillna("")
    n_frames_manifest = int((m.tipo == "frame").sum())
    del m, fr, bx, mt, de, md, sd           # liberta memória (a máquina tem 16 GB)

    # ---------- 3. tamanho real das imagens ----------
    ordem = t.assign(d=t["offset"].abs()).sort_values(K3 + ["d"])
    candidatos = ordem.groupby(K3, sort=False)["ficheiro_img"].apply(lambda s: list(s.iloc[:30]))
    print(f"A ler o tamanho de 1 imagem por janela ({len(candidatos)} janelas)…")
    tam_janela = {}
    for i, (chave, lista) in enumerate(candidatos.items(), 1):
        w = h = np.nan
        for f in lista:                       # a primeira legível, a partir do seed
            w, h = tamanho(ds / f)
            if not np.isnan(w):
                break
        tam_janela[chave] = (w, h)
        if i % 500 == 0:
            print(f"  {i}/{len(candidatos)}")
    chaves = list(zip(t["split"], t["caso"], t["seed"]))
    t["img_w"] = [tam_janela[c][0] for c in chaves]
    t["img_h"] = [tam_janela[c][1] for c in chaves]

    sel = t["box_src"].isin(COM_CAIXA)
    print(f"A ler o tamanho das {int(sel.sum())} imagens com caixa…")
    tam_box = [tamanho(ds / f) for f in t.loc[sel, "ficheiro_img"]]
    wb = np.array([x[0] for x in tam_box], dtype=float)
    hb = np.array([x[1] for x in tam_box], dtype=float)
    wj, hj = t.loc[sel, "img_w"].to_numpy(float), t.loc[sel, "img_h"].to_numpy(float)
    ambos = ~np.isnan(wb) & ~np.isnan(wj)
    dif_tamanho = int(np.sum(ambos & ((wb != wj) | (hb != hj))))
    t.loc[sel, "img_w"] = wb
    t.loc[sel, "img_h"] = hb

    # ---------- 4. caixas: ordenar cantos e normalizar ----------
    for c in ["box_x1", "box_y1", "box_x2", "box_y2"]:
        t[c] = np.nan
    t["cantos_trocados"] = ""
    t["box_fora"] = 0
    if sel.any():
        v = t.loc[sel, "box_valores"].str.split(expand=True).astype(float).to_numpy()
        a, b, c, d = v[:, 0], v[:, 1], v[:, 2], v[:, 3]
        x1, x2, y1, y2 = np.minimum(a, c), np.maximum(a, c), np.minimum(b, d), np.maximum(b, d)
        W, H = t.loc[sel, "img_w"].to_numpy(float), t.loc[sel, "img_h"].to_numpy(float)
        t.loc[sel, "box_x1"] = np.round(x1 / W, 6)
        t.loc[sel, "box_x2"] = np.round(x2 / W, 6)
        t.loc[sel, "box_y1"] = np.round(y1 / H, 6)
        t.loc[sel, "box_y2"] = np.round(y2 / H, 6)
        t.loc[sel, "cantos_trocados"] = np.where((c < a) & (d < b), "xy",
                                         np.where(c < a, "x", np.where(d < b, "y", "")))
        t.loc[sel, "box_fora"] = ((x1 < 0) | (y1 < 0) | (x2 > W) | (y2 > H)).astype(int)
    t = t.rename(columns={"box_valores": "box_original"})

    # ---------- 5. janelas excluídas ----------
    t["excluida"] = 0
    t["motivo_exclusao"] = ""
    exc = pd.DataFrame(columns=K3 + ["motivo"])
    if Path(args.excluidas).is_file():
        exc = pd.read_csv(args.excluidas, dtype=str, keep_default_na=False)
        e = exc[K3 + ["motivo"]].rename(columns={"motivo": "_motivo"})
        t = t.merge(e, on=K3, how="left")
        marcada = t["_motivo"].notna()
        t.loc[marcada, "excluida"] = 1
        t.loc[marcada, "motivo_exclusao"] = t.loc[marcada, "_motivo"]
        t = t.drop(columns="_motivo")

    # ---------- 6. gravar ----------
    colunas = ["split", "caso", "seed", "frame", "offset", "box_src", "box_x1", "box_y1", "box_x2", "box_y2",
               "cantos_trocados", "box_fora", "box_original", "img_w", "img_h", "descricao", "n_descricoes",
               "similaridade", "n_match", "excluida", "motivo_exclusao", "ficheiro_img", "ficheiro_box"]
    t = t.assign(_c=t["caso"].astype(int), _s=t["seed"].astype(int), _f=t["frame"].astype(int)) \
         .sort_values(["split", "_c", "_s", "_f"])[colunas]
    for c in ["img_w", "img_h"]:
        t[c] = t[c].astype("Int64")
    t.to_csv(args.saida, index=False)

    g = t.groupby(K3, sort=False)
    jan = g.agg(n_frames=("frame", "size"),
                frame_min=("frame", lambda s: int(s.astype(int).min())),
                frame_max=("frame", lambda s: int(s.astype(int).max())),
                n_box=("box_src", lambda s: int(s.isin(COM_CAIXA).sum())),
                n_sem_ficheiro=("box_src", lambda s: int((s == "sem_ficheiro").sum())),
                seed_tem_box=("box_src", lambda s: int((s == "medico_seed").any())),
                descricao=("descricao", "first"), n_descricoes=("n_descricoes", "first"),
                similaridade=("similaridade", "first"), n_match=("n_match", "first"),
                excluida=("excluida", "first"), motivo_exclusao=("motivo_exclusao", "first")).reset_index()
    jan["img_w"] = [tam_janela[k][0] for k in zip(jan["split"], jan["caso"], jan["seed"])]
    jan["img_h"] = [tam_janela[k][1] for k in zip(jan["split"], jan["caso"], jan["seed"])]
    for c in ["img_w", "img_h"]:
        jan[c] = jan[c].astype("Int64")
    jan.to_csv(args.janelas, index=False)

    # ---------- 7. resumo ----------
    L = []
    sec = lambda titulo: L.append(f"\n== {titulo} " + "=" * max(0, 64 - len(titulo)))
    L.append(f"Tabela única — {args.saida}")
    L.append(f"Frames (linhas): {len(t)} · frames no manifest: {n_frames_manifest}"
             f" · janelas: {len(jan)} · excluídas: {int(jan['excluida'].sum())}")

    sec("Caixas por origem")
    for k, n in t["box_src"].value_counts().items():
        L.append(f"  {k:<18} {n:>9}")
    L.append(f"  caixas do médico (seed + vizinhos): {int(t['box_src'].isin(COM_CAIXA).sum())} (inventário: 9615)")

    cx = t[t["box_src"].isin(COM_CAIXA)]
    sec("Cantos trocados (antes de ordenar)")
    for k, n in cx["cantos_trocados"].replace("", "nenhum").value_counts().items():
        L.append(f"  {k:<10} {n:>6}")
    L.append(f"  caixas fora da imagem: {int(cx['box_fora'].sum())}"
             + (" · ex.: " + "; ".join(cx.loc[cx["box_fora"] == 1, "ficheiro_box"].head(3)) if cx["box_fora"].any() else ""))
    larg = (cx["box_x2"] - cx["box_x1"]) * cx["img_w"]
    alt = (cx["box_y2"] - cx["box_y1"]) * cx["img_h"]
    L.append(f"  degeneradas (largura ou altura < 2 px): {int(((larg < 2) | (alt < 2)).sum())}")

    sec("Tamanho das imagens (por janela)")
    tj = jan.apply(lambda r: "ilegível" if pd.isna(r["img_w"]) else f"{int(r['img_w'])}×{int(r['img_h'])}", axis=1)
    for k, n in tj.value_counts().items():
        L.append(f"  {k:<10} {n:>6} janelas")
    for k in tj.unique():
        if k not in ("320×320", "ilegível"):
            casos = sorted(jan.loc[tj == k, "caso"].unique())
            L.append(f"  casos com {k}: {', '.join(casos[:20])}" + (" …" if len(casos) > 20 else ""))
    L.append(f"  imagens com caixa de tamanho diferente do da sua janela: {dif_tamanho}")

    usaveis = jan[jan["excluida"] == 0]
    sec("Janelas com caixa (sem as excluídas)")
    L.append(f"  {'split':<16}{'janelas':>9}{'c/ ≥1 caixa':>13}{'c/ caixa no seed':>18}")
    for sp, gj in usaveis.groupby("split"):
        L.append(f"  {sp:<16}{len(gj):>9}{int((gj['n_box'] > 0).sum()):>13}{int(gj['seed_tem_box'].sum()):>18}")
    L.append("  (guião: TEST-BOX ≈ 459 janelas de teste com caixa · ALL-SEEDBOX ≈ 1 346 janelas com caixa no seed)")

    sem = t[t["box_src"] == "sem_ficheiro"]
    sec("Frames sem ficheiro de caixa: onde ficam na janela")
    L.append(f"  {len(sem)} frames em {sem.groupby(K3).ngroups} janelas")
    L.append("  offsets mais frequentes: " + ", ".join(f"{o:+d} ({n})" for o, n in sem["offset"].value_counts().head(10).items()))
    L.append(f"  nos extremos da janela (|offset| ≥ 95): {int((sem['offset'].abs() >= 95).sum())}")

    sec("Descrições")
    L.append(f"  janelas sem descrição: {int((jan['n_descricoes'] == 0).sum())} · com 2 ou mais: {int((jan['n_descricoes'] >= 2).sum())}")
    for _, r in jan[jan["n_descricoes"] >= 2].iterrows():
        L.append(f"    {r['split']}/{r['caso']} seed {r['seed']}: {r['descricao']}")
    for _, r in jan[jan["n_descricoes"] == 0].head(10).iterrows():
        L.append(f"    sem descrição: {r['split']}/{r['caso']} seed {r['seed']}")
    L.append(f"  janelas sem valor de Similariedade: {int((jan['similaridade'] == '').sum())}")

    sec("Exemplos (frames com caixa)")
    for _, r in cx.head(3).iterrows():
        L.append(f"  {r['split']}/{r['caso']} seed {r['seed']} frame {r['frame']} (offset {r['offset']:+d}): "
                 f"original '{r['box_original']}' → ({r['box_x1']:.3f}, {r['box_y1']:.3f}, {r['box_x2']:.3f}, {r['box_y2']:.3f})"
                 f" · {int(r['img_w'])}×{int(r['img_h'])} · {r['descricao']}")

    sec("Ficheiros gerados")
    L.append(f"  {args.saida} ({len(t)} linhas)")
    L.append(f"  {args.janelas} ({len(jan)} linhas)")

    texto = "\n".join(L)
    print(texto)
    Path(args.resumo).write_text(texto + "\n", encoding="utf-8")
    print(f"\nResumo gravado em {args.resumo}")


if __name__ == "__main__":
    main()
