#!/usr/bin/env python3
"""
16 — A7: exemplos qualitativos (Guião A). Faz uma figura por janela com os frames à volta do seed, para escolheres 4 a 6
para o capítulo. Não avalia nada: só desenha o que o SAM2_L12 propôs na anotação expandida (A8).

Janelas escolhidas (configs/a7.yaml, secção exemplos), por esta ordem e sem repetir (os motivos juntam-se):
  1. obrigatórias: os casos da Tabela 2 do diagnóstico, que o guião pede no A7, e as outras da lista (um par
     [caso, seed] é essa janela; um caso sozinho são todas as janelas desse caso em ALL_SEEDBOX). Uma janela pedida
     que não está em ALL_SEEDBOX (por exemplo, seed sem caixa) fica fora do âmbito do método: desenha-se só com as
     caixas originais, com o motivo (lido de janelas_v1.csv), e entram as janelas do mesmo grupo (sobrepostas) que
     estão em ALL_SEEDBOX;
  2. as que marcaste na folha dos 50 FP (outputs/14_fp_inspecao/classificacao.csv) com uma nota que tenha "exemplo";
     o frame do FP entra na figura;
  3. automáticas, só em janelas de teste (TEST-BOX) e a partir das caixas originais: acerto (R_loc da janela máximo),
     acerto longe do seed (mais frames certos a 7–12 frames), falha (R_loc mínimo), falha em lesão plana (erosão ou
     eritema) e falha em lesão pequena; em cada uma, a janela com mais caixas originais.
     R_loc da janela = frames com caixa original (sem o seed, até L_max) em que o SAM 2 está no nível alta com
     IoU ≥ 0,3, a dividir pelos frames com caixa original.

Cada figura tem 9 quadros (os offsets da configuração, com o seed ao centro):
  verde     caixa original do frame (médico/Rodrigo), quando existe
  magenta   caixa do SAM 2: contínua no nível alta (score ≥ τ do OP-F1), tracejada abaixo dele
  amarelo   contorno da máscara do SAM 2, quando foi gravada
e em cada quadro o offset e o score.

Uso, a partir da pasta do projeto, depois do source …/inicio.sh:
    python scripts/16_a7_exemplos.py --ds "$DS"
Grava em outputs/16_a7_exemplos/ (configs/a7.yaml): NN_<motivo>_<caso>_<seed>.png e .pdf, e indice.csv.
Versões: 1.0 (22 out 2026) · 1.1 (22 out 2026): janelas obrigatórias fora de ALL_SEEDBOX e o seu grupo.
"""
import argparse
import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "eval"))
sys.path.insert(0, str(RAIZ / "methods"))
import evaluate as ev  # noqa: E402
from sam2_prop import rle_para_mascara  # noqa: E402

VERSAO = "1.1"
CAIXA = ["box_x1", "box_y1", "box_x2", "box_y2"]
PLANAS = {"erosao", "eritema"}
ROTULOS = {"angiectasia": "angiectasia", "erosao": "erosão", "ulcera": "úlcera", "polipo": "pólipo",
           "eritema": "eritema", "xantelasma": "xantelasma"}
MOTIVOS = {"obrigatorio": "obrigatório", "nota_fp": "marcado nos FP", "acerto": "acerto", "acerto_longe":
           "acerto longe do seed", "falha": "falha", "falha_plana": "falha em lesão plana",
           "falha_pequena": "falha em lesão pequena", "fora_ambito": "fora do âmbito",
           "mesmo_grupo": "mesmo grupo de janelas"}
VERDE, MAGENTA, AMARELO = "#00d000", "#ff00ff", "#ffd400"


def caminho_imagem(ds, sp, caso, seed, frame):
    return Path(ds) / sp / caso / "GT_com_classes" / str(seed) / f"{caso}_{frame}_{seed}.jpg"


def ler_anotacoes(cfg, janelas):
    """Frames de cada janela e caixas originais válidas (box_src medico_seed ou medico_vizinho; regra do avaliador)."""
    alvo, casos, partes = set(janelas), {k[1] for k in janelas}, []
    for b in pd.read_csv(cfg["caminhos"]["anotacoes"], dtype=str, keep_default_na=False,
                         usecols=["split", "caso", "seed", "frame", "offset", "box_src"] + CAIXA, chunksize=200_000):
        b = b[b["caso"].isin(casos)]
        if len(b):
            b = b[[k in alvo for k in zip(b["split"], b["caso"], b["seed"].astype(int))]]
            if len(b):
                partes.append(b)
    if not partes:
        raise ev.ErroAvaliacao("Nenhuma das janelas pedidas está em annotations_v1.csv.")
    a = pd.concat(partes, ignore_index=True)
    for c in ("seed", "frame", "offset"):
        a[c] = a[c].astype(int)
    v = a[CAIXA].apply(pd.to_numeric, errors="coerce")
    ok = (a["box_src"].isin(["medico_seed", "medico_vizinho"]) & v.notna().all(axis=1)
          & (v["box_x2"] > v["box_x1"]) & (v["box_y2"] > v["box_y1"]))
    a["orig"] = [tuple(r) if o else None for r, o in zip(v.to_numpy(), ok)]
    return a[["split", "caso", "seed", "frame", "offset", "orig"]]


def ler_previsoes(caminho, janelas):
    p = pd.read_csv(caminho, dtype={"caso": str, "split": str}, keep_default_na=False)
    p["caso"] = p["caso"].str.zfill(4)
    p["seed"], p["frame"] = p["seed"].astype(int), p["frame"].astype(int)
    p = p[[k in set(janelas) for k in zip(p["split"], p["caso"], p["seed"])]].copy()
    p["score"] = pd.to_numeric(p["score"], errors="coerce").fillna(0.0)
    p["prev"] = [None if any(str(x) == "" for x in r) else tuple(float(x) for x in r) for r in p[CAIXA].to_numpy()]
    return p[["split", "caso", "seed", "frame", "score", "prev"]]


def metricas_janelas(fr, l_max, tau):
    """Por janela de teste: caixas originais (sem o seed, até L_max), certas no nível alta (IoU ≥ 0,3) e certas longe."""
    d = fr[(fr["offset"] != 0) & (fr["offset"].abs() <= l_max) & fr["orig"].notna()].copy()
    if d.empty:
        return pd.DataFrame(columns=["n_caixas", "certas", "certas_longe", "R_loc"])
    tem = d["prev"].notna() & (d["score"] >= tau)
    iou = np.zeros(len(d))
    if tem.any():
        iou[tem.to_numpy()] = ev.iou(np.array(d.loc[tem, "prev"].tolist()), np.array(d.loc[tem, "orig"].tolist()))
    d["certa"] = tem.to_numpy() & (iou >= 0.3 - 1e-9)
    d["certa_longe"] = d["certa"] & (d["offset"].abs() >= 7)
    m = d.groupby(["split", "caso", "seed"]).agg(n_caixas=("certa", "size"), certas=("certa", "sum"),
                                                 certas_longe=("certa_longe", "sum"))
    m["R_loc"] = m["certas"] / m["n_caixas"]
    return m


def automaticas(m, info):
    """Uma janela por motivo: a com mais caixas originais entre as melhores (ou piores) no critério do motivo."""
    if m.empty:
        return {}
    t = m.join(info)
    escolhas = {}

    def melhor(sel, chave, maior):
        s = t[sel]
        if s.empty or (chave == "certas_longe" and s[chave].max() == 0):
            return None
        alvo = s[chave].max() if maior else s[chave].min()
        s = s[s[chave] == alvo].sort_values("n_caixas", ascending=False, kind="mergesort")
        return s.index[0]

    todas = pd.Series(True, index=t.index)
    for motivo, sel, chave, maior in (("acerto", todas, "R_loc", True), ("acerto_longe", todas, "certas_longe", True),
                                      ("falha", todas, "R_loc", False),
                                      ("falha_plana", t["categoria"].isin(PLANAS), "R_loc", False),
                                      ("falha_pequena", t["tercil"] == "pequena", "R_loc", False)):
        k = melhor(sel, chave, maior)
        if k is not None:
            escolhas[motivo] = k
    return escolhas


def info_janelas(cfg, janelas, fr):
    jt = pd.read_csv(cfg["caminhos"]["janelas"], dtype=str, keep_default_na=False)
    cat = {ev.chave(s, c, e): x for s, c, e, x in zip(jt["split"], jt["caso"], jt["seed"], jt["categoria"])}
    t1, t2 = cfg["estratos"]["tercis_area_seed"]
    sd = fr[(fr["offset"] == 0) & fr["orig"].notna()]
    area = {(s, c, e): (b[2] - b[0]) * (b[3] - b[1]) for s, c, e, b in zip(sd["split"], sd["caso"], sd["seed"], sd["orig"])}
    linhas = []
    for k in janelas:
        a = area.get(k, np.nan)
        linhas.append({"split": k[0], "caso": k[1], "seed": k[2], "categoria": cat.get(k, ""),
                       "tercil": "sem_caixa" if not np.isfinite(a) else "pequena" if a < t1 else "media" if a < t2
                       else "grande"})
    return pd.DataFrame(linhas).set_index(["split", "caso", "seed"])


def ler_mascaras(pasta, k):
    f = Path(pasta) / f"{k[0]}__{k[1]}__{k[2]}.json.gz"
    if not f.is_file():
        return {}
    return json.loads(gzip.decompress(f.read_bytes()).decode("utf-8")).get("frames", {})


def figura(plt, Rectangle, k, ex, fr_j, pr_j, mascaras, ds, offsets, tau, titulo, destino):
    import cv2
    fig, axs = plt.subplots(3, 3, figsize=(6.3, 6.9))
    por_frame = fr_j.set_index("frame")
    prev = pr_j.set_index("frame")
    for ax, o in zip(axs.ravel(), offsets):
        f = k[2] + o
        ax.set_xticks([])
        ax.set_yticks([])
        img = cv2.imread(str(caminho_imagem(ds, k[0], k[1], k[2], f))) if f in por_frame.index else None
        if img is None:
            ax.set_facecolor("#e9e9e6")
            ax.text(0.5, 0.5, "fora da janela" if f not in por_frame.index else "sem imagem", ha="center",
                    va="center", fontsize=7, color="#52514e", transform=ax.transAxes)
            ax.set_title(f"{o:+d}".replace("-", "−"), fontsize=8)
            continue
        h, w = img.shape[:2]
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        orig = por_frame.loc[f, "orig"]
        if orig is not None:
            ax.add_patch(Rectangle((orig[0] * w, orig[1] * h), (orig[2] - orig[0]) * w, (orig[3] - orig[1]) * h,
                                   fill=False, ec=VERDE, lw=1.4))
        if o == 0:
            ax.set_title("seed", fontsize=8, fontweight="bold")
        elif f in prev.index:
            s, b = float(prev.loc[f, "score"]), prev.loc[f, "prev"]
            r = mascaras.get(str(f))
            if r is not None:
                mk = rle_para_mascara(r)
                if mk.shape != (h, w):
                    mk = cv2.resize(mk.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                ax.contour(mk.astype(float), levels=[0.5], colors=[AMARELO], linewidths=0.6)
            if b is not None:
                ax.add_patch(Rectangle((b[0] * w, b[1] * h), (b[2] - b[0]) * w, (b[3] - b[1]) * h, fill=False,
                                       ec=MAGENTA, lw=1.4 if s >= tau else 1.0, ls="-" if s >= tau else "--"))
            marca = " · FP" if o in ex["offsets_fp"] else ""
            ax.set_title(f"{o:+d} · {s:.2f}{marca}".replace(".", ",").replace("-", "−"), fontsize=8,
                         fontweight="bold" if s >= tau else "normal")
        else:
            ax.set_title(f"{o:+d} · sem previsão".replace("-", "−"), fontsize=8)
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
    fig.suptitle(titulo, fontsize=9, x=0.02, ha="left")
    fig.text(0.02, 0.012, f"verde: caixa original · magenta: SAM 2 (contínua: score ≥ {tau:.2f}; tracejada: abaixo) · "
             "amarelo: máscara".replace(".", ","), fontsize=6.8, color="#52514e")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    for ext in ("png", "pdf"):
        fig.savefig(destino.with_suffix(f".{ext}"), dpi=150, facecolor="white")
    plt.close(fig)


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--ds", default=os.environ.get("DS"), help="pasta do dataset (por omissão, $DS)")
    a.add_argument("--config", default="configs/avaliacao.yaml")
    a.add_argument("--config-a7", default="configs/a7.yaml")
    a.add_argument("--final", default="configs/final_SAM2_L12.yaml")
    args = a.parse_args(argv)
    if not args.ds or not Path(args.ds).is_dir():
        raise ev.ErroAvaliacao('Falta a pasta do dataset: corre primeiro o source …/inicio.sh e usa --ds "$DS".')
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError:
        raise ev.ErroAvaliacao("Falta o matplotlib neste ambiente (pip install matplotlib).")
    cfg, a7, fin = ev.ler_config(args.config), ev.ler_config(args.config_a7), ev.ler_config(args.final)
    conf = a7["exemplos"]
    tau = float(fin["decisoes_A6"]["niveis_A8"]["alta"]["score_minimo"])
    l_max = int(fin["parametros"]["l_max"])
    offsets = [int(o) for o in conf["offsets"]]
    if len(offsets) != 9 or 0 not in offsets:
        raise ev.ErroAvaliacao("exemplos.offsets (configs/a7.yaml) tem de ter 9 valores, incluindo o 0 (o seed).")
    todas = ev.ler_conjunto(Path(cfg["caminhos"]["conjuntos"]) / "ALL_SEEDBOX.json")
    por_caso = {}
    for k in todas:
        por_caso.setdefault(k[1], []).append(k)

    # 1. obrigatórias e 2. notas dos FP
    jt = pd.read_csv(cfg["caminhos"]["janelas"], dtype=str, keep_default_na=False)
    jdict = {ev.chave(r["split"], r["caso"], r["seed"]): r for r in jt.to_dict("records")}
    grupos = {}
    for k, r in jdict.items():
        if r.get("group_id"):
            grupos.setdefault(r["group_id"], []).append(k)
    em_all, escolha, fora = set(todas), {}, []

    def junta(k, chave, texto=None, offset_fp=None):
        e = escolha.setdefault(k, {"motivos": [], "textos": [], "offsets_fp": []})
        if chave not in e["motivos"]:
            e["motivos"].append(chave)
            e["textos"].append(texto or MOTIVOS[chave])
        if offset_fp is not None:
            e["offsets_fp"].append(int(offset_fp))

    def razao(k):
        r = jdict[k]
        if r.get("excluida", "0") == "1":
            return "janela excluída"
        if r.get("lesao", "1") != "1":
            return "não é janela de lesão"
        if r.get("seed_tem_box", "1") != "1":
            return "o seed não tem caixa"
        return "fora de ALL_SEEDBOX"

    for item in conf.get("casos", []):
        if isinstance(item, (list, tuple)):                              # uma janela: [caso, seed]
            caso, seed = str(item[0]).zfill(4), int(item[1])
            ks = sorted(k for k in jdict if k[1] == caso and k[2] == seed)
            if not ks:
                fora.append(f"{caso}/{seed}")
            for k in ks:
                if k in em_all:
                    junta(k, "obrigatorio")
                    continue
                # fora do âmbito do método (o SAM 2 parte da caixa do seed): desenha-se só com as caixas originais,
                # e entram as janelas do mesmo grupo (sobrepostas) que estão em ALL_SEEDBOX
                junta(k, "fora_ambito", f"obrigatório, fora do âmbito: {razao(k)}")
                for k2 in sorted(grupos.get(jdict[k].get("group_id", ""), [])):
                    if k2 != k and k2 in em_all:
                        junta(k2, "mesmo_grupo", f"mesmo grupo de janelas que {caso}/{seed}")
        else:                                                            # um caso: todas as janelas em ALL_SEEDBOX
            caso = str(item).zfill(4)
            ks = por_caso.get(caso, [])
            if not ks:
                fora.append(caso)
            for k in sorted(ks):
                junta(k, "obrigatorio")
    folha = Path(a7["fp_inspecao"]["saida"]) / "classificacao.csv"
    if folha.is_file():
        fp = pd.read_csv(folha, dtype=str, keep_default_na=False)
        for r in fp[fp["nota"].str.contains("exemplo", case=False)].itertuples(index=False):
            k = (r.split, r.caso.zfill(4), int(r.seed))
            if k in set(todas):
                junta(k, "nota_fp", offset_fp=r.offset)

    # 3. automáticas (janelas de teste, a partir das caixas originais)
    teste = [k for k in todas if k[0] == cfg["conjuntos"]["TEST_BOX"]["split"]]
    precisas = sorted(set(teste) | set(escolha))
    fr = ler_anotacoes(cfg, precisas)
    pr = ler_previsoes(conf["previsoes"], precisas)
    fr = fr.merge(pr, on=["split", "caso", "seed", "frame"], how="left")
    fr["score"] = fr["score"].fillna(0.0)
    fr["prev"] = fr["prev"].where(fr["prev"].notna(), None)
    info = info_janelas(cfg, precisas, fr)
    m = metricas_janelas(fr[[k in set(teste) for k in zip(fr["split"], fr["caso"], fr["seed"])]], l_max, tau)
    for motivo, k in automaticas(m, info).items():
        junta(k, motivo)

    saida = Path(conf["saida"])
    saida.mkdir(parents=True, exist_ok=True)
    linhas = []
    for n, (k, ex) in enumerate(escolha.items(), 1):
        offs = list(offsets)
        for o in ex["offsets_fp"]:                                        # o frame do FP entra no lugar do mais perto
            if o not in offs:
                j = min((i for i, x in enumerate(offs) if x != 0 and (x > 0) == (o > 0)),
                        key=lambda i: abs(offs[i] - o), default=None)
                if j is not None:
                    offs[j] = o
        offs = sorted(offs)
        cat = info.loc[k, "categoria"] if k in info.index else ""
        mot = " · ".join(ex["textos"])
        destino = saida / f"{n:02d}_{ex['motivos'][0]}_{k[1]}_{k[2]}"
        fr_j = fr[(fr["split"] == k[0]) & (fr["caso"] == k[1]) & (fr["seed"] == k[2])]
        pr_j = pr[(pr["split"] == k[0]) & (pr["caso"] == k[1]) & (pr["seed"] == k[2])]
        titulo = f"Caso {k[1]} · seed {k[2]} · {ROTULOS.get(cat, cat or 'sem classe')} · {mot}"
        figura(plt, Rectangle, k, ex, fr_j, pr_j, ler_mascaras(conf["mascaras"], k), args.ds, offs, tau, titulo, destino)
        mj = m.loc[k] if k in m.index else None
        linhas.append({"n": n, "motivos": mot, "split": k[0], "caso": k[1], "seed": k[2], "categoria": cat,
                       "tercil": info.loc[k, "tercil"] if k in info.index else "",
                       "caixas_originais": "" if mj is None else int(mj["n_caixas"]),
                       "R_loc_janela": "" if mj is None else round(float(mj["R_loc"]), 3),
                       "offsets": " ".join(f"{o:+d}" for o in offs), "ficheiro": destino.name + ".png"})
    ind = pd.DataFrame(linhas)
    ind.to_csv(saida / "indice.csv", index=False)
    L = [f"16 — exemplos qualitativos · 16_a7_exemplos.py v{VERSAO} · SAM2_L12, nível alta: score ≥ {tau:g}",
         f"{len(ind)} figuras em {saida}/ (png e pdf) · índice em {saida}/indice.csv"]
    if fora:
        L.append(f"não encontradas (sem figura): {', '.join(fora)}")
    L.append("")
    for r in ind.itertuples(index=False):
        extra = "" if r.R_loc_janela == "" else f" · R_loc da janela {r.R_loc_janela} ({r.caixas_originais} caixas)"
        L.append(f"  {r.n:02d}  {r.caso}/{r.seed}  {ROTULOS.get(r.categoria, r.categoria):<12} {r.motivos}{extra}")
    L += ["", "Escolhe 4 a 6 para o capítulo: os obrigatórios do guião, um acerto, uma falha e um caso que explique "
              "a falha (lesão plana, pequena ou salto para outra lesão)."]
    print("\n".join(L))
    return {"indice": ind, "fora": fora}


if __name__ == "__main__":
    try:
        main()
    except ev.ErroAvaliacao as e:
        print(f"ERRO: {e}", file=sys.stderr)
        sys.exit(2)
