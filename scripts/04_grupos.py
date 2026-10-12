#!/usr/bin/env python3
"""
Passo 0.4 — Janelas sobrepostas e conflitos entre cópias (só lê o dataset; atualiza as tabelas derivadas).

Quando duas seeds do mesmo caso estão a menos de ~200 frames, as janelas partilham frames e cada janela
tem a sua cópia (e o seu ficheiro de caixa). Este passo:
  1. junta as janelas que partilham frames em GRUPOS (group_id) — um grupo nunca é partido entre
     conjuntos e o sorteio nunca escolhe duas janelas do mesmo grupo;
  2. consolida a caixa de cada frame entre cópias (colunas boxc_*; as colunas originais box_* não mudam):
       - caixa numa cópia e nenhuma noutra  → a caixa prevalece em todas as cópias;
       - caixas diferentes em cópias diferentes → fica a da janela cujo seed está mais perto do frame;
         a outra fica registada em data/derived/conflitos.csv, com o IoU entre as duas.

Uso, a partir da pasta do projeto:
    python scripts/04_grupos.py --anotadas "$NOVA/annotations"
Atualiza: data/derived/annotations_v1.csv (+ group_id, boxc_*) · data/derived/janelas_v1.csv (+ group_id, …)
Gera:     data/derived/conflitos.csv · outputs/04_grupos_resumo.txt  ← é este que se lê
"""
import argparse
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

K3 = ["split", "caso", "seed"]
KF = ["split", "caso", "frame"]
COM_CAIXA = ["medico_seed", "medico_vizinho"]
BOX = ["box_x1", "box_y1", "box_x2", "box_y2"]
BOXC = ["boxc_x1", "boxc_y1", "boxc_x2", "boxc_y2"]
NOVAS = ["group_id"] + BOXC + ["boxc_src", "boxc_janela"]


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    uniao = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / uniao if uniao > 0 else 0.0


class UniaoPesquisa:
    def __init__(self):
        self.pai = {}

    def raiz(self, x):
        self.pai.setdefault(x, x)
        while self.pai[x] != x:
            self.pai[x] = self.pai[self.pai[x]]
            x = self.pai[x]
        return x

    def unir(self, a, b):
        ra, rb = self.raiz(a), self.raiz(b)
        if ra != rb:
            self.pai[max(ra, rb)] = min(ra, rb)


def gravar(df, caminho):
    tmp = f"{caminho}.tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, caminho)          # só substitui o ficheiro quando a escrita terminou


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tabela", default="data/derived/annotations_v1.csv")
    ap.add_argument("--janelas", default="data/derived/janelas_v1.csv")
    ap.add_argument("--conflitos", default="data/derived/conflitos.csv")
    ap.add_argument("--anotadas", default="", help="pasta com os CSV antigos (só os nomes são lidos)")
    ap.add_argument("--resumo", default="outputs/04_grupos_resumo.txt")
    args = ap.parse_args()

    print("A ler a tabela…")
    t = pd.read_csv(args.tabela, dtype={"caso": str, "seed": str, "frame": str}, keep_default_na=False, na_values=[""])
    t = t.drop(columns=[c for c in NOVAS if c in t.columns])          # permite voltar a correr
    for c in ["box_src", "cantos_trocados", "box_original", "descricao", "similaridade", "motivo_exclusao",
              "ficheiro_img", "ficheiro_box"]:
        if c in t.columns:
            t[c] = t[c].fillna("")
    jan = pd.read_csv(args.janelas, dtype={"caso": str, "seed": str, "similaridade": str}, keep_default_na=False)
    jan = jan.drop(columns=[c for c in ["group_id", "n_janelas_grupo", "n_box_consolidado"] if c in jan.columns])

    # ---------- 1. grupos ----------
    up = UniaoPesquisa()
    for k in zip(jan["split"], jan["caso"], jan["seed"]):
        up.raiz(k)
    partilhados = t[t.duplicated(KF, keep=False)]
    for (sp, ca, _), seeds in partilhados.groupby(KF)["seed"]:
        s = list(dict.fromkeys(seeds))
        for x in s[1:]:
            up.unir((sp, ca, s[0]), (sp, ca, x))
    raizes = {k: up.raiz(k) for k in up.pai}
    nomes, cont = {}, defaultdict(int)
    for r in sorted(set(raizes.values()), key=lambda k: (k[0], k[1], int(k[2]))):
        cont[(r[0], r[1])] += 1
        nomes[r] = f"{r[1]}-{cont[(r[0], r[1])]:02d}"
    jan["group_id"] = [nomes[raizes[k]] for k in zip(jan["split"], jan["caso"], jan["seed"])]
    jan["n_janelas_grupo"] = jan.groupby("group_id")["seed"].transform("size")
    t = t.merge(jan[K3 + ["group_id"]], on=K3, how="left")

    # ---------- 2. consolidar caixas entre cópias ----------
    tem = t["box_src"].isin(COM_CAIXA)
    for c, o in zip(BOXC, BOX):
        t[c] = t[o].where(tem)
    t["boxc_src"] = np.where(tem, "medico", t["box_src"])
    t["boxc_janela"] = np.where(tem, t["seed"], "")
    chaves = t.loc[tem, KF].drop_duplicates()
    copias = t.reset_index().merge(chaves, on=KF)                  # todas as cópias de frames com caixa nalguma cópia
    copias = copias[copias.groupby(KF)["seed"].transform("nunique") > 1]
    n_box_vazio = n_box_semfich = n_iguais = 0
    conflitos = []
    for (sp, ca, fr), g in copias.groupby(KF):
        com = g[g["box_src"].isin(COM_CAIXA)].copy()
        com["dist"] = (com["frame"].astype(int) - com["seed"].astype(int)).abs()
        venc = com.sort_values(["dist", "seed"], key=lambda s: s.astype(int) if s.name == "seed" else s).iloc[0]
        bv = [venc[c] for c in BOX]
        sem = g[~g["box_src"].isin(COM_CAIXA)]
        if len(sem):
            n_box_vazio += int((sem["box_src"] == "vazio").sum())
            n_box_semfich += int((sem["box_src"] == "sem_ficheiro").sum())
            for i in sem["index"]:
                t.loc[i, BOXC] = bv
                t.loc[i, "boxc_src"] = "medico_outra_janela"
                t.loc[i, "boxc_janela"] = venc["seed"]
        for _, r in com.iterrows():
            if r["seed"] == venc["seed"]:
                continue
            bo = [r[c] for c in BOX]
            if np.allclose(bo, bv, atol=1e-6):
                n_iguais += 1
                continue
            conflitos.append({"split": sp, "caso": ca, "frame": fr, "seed_mantida": venc["seed"], "seed_substituida": r["seed"],
                              "iou": round(iou(bv, bo), 4), "box_mantida": " ".join(f"{x:.4f}" for x in bv),
                              "box_substituida": " ".join(f"{x:.4f}" for x in bo)})
            t.loc[r["index"], BOXC] = bv
            t.loc[r["index"], "boxc_src"] = "medico_conflito"
            t.loc[r["index"], "boxc_janela"] = venc["seed"]
    conf = pd.DataFrame(conflitos, columns=["split", "caso", "frame", "seed_mantida", "seed_substituida", "iou",
                                            "box_mantida", "box_substituida"])

    nb = t[t["boxc_src"].str.startswith("medico")].groupby(K3).size().rename("n_box_consolidado")
    jan = jan.merge(nb, on=K3, how="left")
    jan["n_box_consolidado"] = jan["n_box_consolidado"].fillna(0).astype(int)

    # ---------- 3. gravar ----------
    for c in ["img_w", "img_h"]:
        if c in t.columns:
            t[c] = t[c].astype("Int64")
        if c in jan.columns:
            jan[c] = pd.to_numeric(jan[c], errors="coerce").astype("Int64")
    gravar(t, args.tabela)
    gravar(jan, args.janelas)
    conf.to_csv(args.conflitos, index=False)

    # ---------- 4. resumo ----------
    L = []
    sec = lambda titulo: L.append(f"\n== {titulo} " + "=" * max(0, 64 - len(titulo)))
    L.append("Janelas sobrepostas e conflitos (passo 0.4)")
    sec("Grupos")
    tam = jan.groupby("group_id").size()
    L.append(f"  {len(jan)} janelas → {tam.size} grupos   (guião: 2 593 → ≈ 1 796)")
    L.append("  janelas por grupo: " + ", ".join(f"{k}: {v} grupos" for k, v in tam.value_counts().sort_index().items()))
    L.append(f"  janelas que partilham frames com outra: {int((jan['n_janelas_grupo'] > 1).sum())}")
    L.append(f"  frames partilhados (cópias em mais de uma janela): {partilhados[KF].drop_duplicates().shape[0]}")
    sec("Consolidação das caixas")
    L.append(f"  cópias sem caixa que recebem a caixa de outra cópia: {n_box_vazio + n_box_semfich}"
             f"   (ficheiro vazio: {n_box_vazio} · sem ficheiro: {n_box_semfich})   (guião: ≈ 2 604)")
    L.append(f"  caixas iguais em várias cópias (sem conflito): {n_iguais}")
    L.append(f"  caixas diferentes resolvidas pela seed mais próxima: {len(conf)}   (guião: ≈ 181)")
    if len(conf):
        L.append(f"  IoU entre as duas caixas: mediana {conf['iou'].median():.2f} · "
                 f"IoU < 0,5: {int((conf['iou'] < 0.5).sum())} · IoU = 0: {int((conf['iou'] == 0).sum())}")
        for _, r in conf.sort_values("iou").head(5).iterrows():
            L.append(f"    {r['split']}/{r['caso']} frame {r['frame']}: seeds {r['seed_mantida']} vs {r['seed_substituida']} · IoU {r['iou']}")
    L.append(f"  frames com caixa do médico: antes {int(tem.sum())} · depois de consolidar "
             f"{int(t['boxc_src'].str.startswith('medico').sum())}")
    if args.anotadas:
        ja = set()
        for f in Path(args.anotadas).glob("*.csv"):                   # só os nomes
            p = f.stem.split("__")
            if len(p) == 3:
                ja.add(tuple(p))
        sel = jan[[k in ja for k in zip(jan["split"], jan["caso"], jan["seed"])]]
        sec(f"As {len(ja)} janelas já anotadas (só pelos nomes)")
        rep = sel.groupby("group_id").size()
        L.append(f"  encontradas na tabela: {len(sel)} · em grupos com outras janelas: {int((sel['n_janelas_grupo'] > 1).sum())}")
        L.append(f"  pares de janelas anotadas no mesmo grupo: {int((rep > 1).sum())}"
                 + (" → " + ", ".join(rep[rep > 1].index) if (rep > 1).any() else ""))
        L.append(f"  sem caixa no seed: {int((sel['seed_tem_box'] == 0).sum())}"
                 + (" → " + ", ".join(f"{a}/{b}/{c}" for a, b, c in zip(*[sel.loc[sel['seed_tem_box'] == 0, k] for k in K3]))
                    if (sel['seed_tem_box'] == 0).any() else ""))
    sec("Ficheiros")
    L.append(f"  {args.tabela} (+ group_id, boxc_x1…boxc_y2, boxc_src, boxc_janela)")
    L.append(f"  {args.janelas} (+ group_id, n_janelas_grupo, n_box_consolidado)")
    L.append(f"  {args.conflitos} ({len(conf)} linhas)")
    texto = "\n".join(L)
    print(texto)
    Path(args.resumo).parent.mkdir(parents=True, exist_ok=True)
    Path(args.resumo).write_text(texto + "\n", encoding="utf-8")
    print(f"\nResumo gravado em {args.resumo}")


if __name__ == "__main__":
    main()
