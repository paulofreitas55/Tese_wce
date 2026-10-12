#!/usr/bin/env python3
"""
Folhas de miniaturas — verificação visual da tabela única (só lê, não altera nada).

Para cada janela escolhida gera uma imagem com:
  - no topo: a imagem do relatório e o frame do seed, lado a lado, e a descrição clínica;
  - por baixo: todos os frames da janela em miniatura, por ordem, com o nº do frame;
    caixa do médico a VERDE no seed e a AMARELO nos vizinhos; moldura VERMELHA no seed.

Escolha por omissão (semente 0), sem janelas excluídas nem as que já anotaste:
  - 20 janelas ao acaso com caixa (10 treino + 10 teste);
  - as 5 janelas com o valor de "Similariedade" mais baixo e as 5 com o mais alto
    (para perceber de que lado ficam os emparelhamentos relatório→vídeo maus).

Uso, a partir da pasta do projeto:
    python scripts/03_folhas.py --dataset "<caminho>/Dataset_WCE" --anotadas "<pasta dos 30 CSV antigos>"
    python scripts/03_folhas.py --dataset "<...>" --caso 0011 --seed 2652      # uma janela concreta
Gera:
    outputs/folhas/<split>__<caso>__<seed>.jpg
    outputs/folhas/revisao.csv   ← preenche as colunas ok/problema enquanto vês as folhas
"""
import argparse
import random
import unicodedata
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

MINI = 96          # tamanho das miniaturas (px)
COLS = 15          # miniaturas por linha
TOPO = 256         # tamanho das duas imagens do topo
TXT = 14           # altura da faixa do nº do frame
FONTES = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
          "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"]


def fonte(tam):
    for f in FONTES:
        try:
            return ImageFont.truetype(f, tam), True
        except Exception:
            pass
    return ImageFont.load_default(), False


F13, UNICODE = fonte(13)
F10, _ = fonte(10)


def tx(s):
    """Sem fonte com acentos, troca-os por letras simples (para não aparecerem quadrados)."""
    s = str(s)
    if UNICODE:
        return s
    s = s.replace("–", "-").replace("·", "|")
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def abrir(p, tam):
    try:
        im = Image.open(p).convert("RGB")
        return im.resize((tam, tam))
    except Exception:
        im = Image.new("RGB", (tam, tam), (90, 90, 90))
        ImageDraw.Draw(im).text((tam // 2 - 4, tam // 2 - 6), "X", fill=(255, 255, 255))
        return im


def folha(ds, jan, fr, relatorio, saida):
    fr = fr.sort_values("frame_i")
    n = len(fr)
    linhas = (n + COLS - 1) // COLS
    larg = COLS * MINI
    cab = 70
    alt = cab + TOPO + 24 + linhas * (MINI + TXT)
    F = Image.new("RGB", (larg, alt), "white")
    d = ImageDraw.Draw(F)
    d.text((8, 6), tx(f"{jan.split}/{jan.caso}  seed {jan.seed}  ·  frames {jan.frame_min}–{jan.frame_max} ({jan.n_frames})"
                      f"  ·  caixas: {jan.n_box}  ·  Similariedade: {jan.similaridade or '-'}"), fill=(0, 0, 0), font=F13)
    d.text((8, 26), tx(f"descrição: {jan.descricao or '(sem descrição)'}"), fill=(0, 0, 0), font=F13)
    d.text((8, 46), tx("verde = caixa no seed · amarelo = caixas nos vizinhos · moldura vermelha = seed"),
           fill=(90, 90, 90), font=F13)
    # topo: relatório | seed
    y0 = cab
    rel = abrir(ds / relatorio, TOPO) if relatorio else Image.new("RGB", (TOPO, TOPO), (220, 220, 220))
    F.paste(rel, (8, y0))
    d.text((8, y0 + TOPO + 4), tx("imagem do relatório"), fill=(0, 0, 0), font=F13)
    s = fr[fr["offset"] == 0]
    if len(s):
        r = s.iloc[0]
        im = abrir(ds / r["ficheiro_img"], TOPO)
        if r["box_src"] == "medico_seed":
            ImageDraw.Draw(im).rectangle([r["box_x1"] * TOPO, r["box_y1"] * TOPO, r["box_x2"] * TOPO, r["box_y2"] * TOPO],
                                         outline=(0, 255, 0), width=3)
        F.paste(im, (8 + TOPO + 16, y0))
        d.text((8 + TOPO + 16, y0 + TOPO + 4), f"seed (frame {jan.seed})", fill=(0, 0, 0), font=F13)
    # grelha
    y0 = cab + TOPO + 24
    for k, (_, r) in enumerate(fr.iterrows()):
        x, y = (k % COLS) * MINI, y0 + (k // COLS) * (MINI + TXT)
        im = abrir(ds / r["ficheiro_img"], MINI)
        dd = ImageDraw.Draw(im)
        if r["box_src"] in ("medico_seed", "medico_vizinho"):
            cor = (0, 255, 0) if r["box_src"] == "medico_seed" else (255, 220, 0)
            dd.rectangle([r["box_x1"] * MINI, r["box_y1"] * MINI, r["box_x2"] * MINI, r["box_y2"] * MINI], outline=cor, width=2)
        if r["offset"] == 0:
            dd.rectangle([0, 0, MINI - 1, MINI - 1], outline=(255, 0, 0), width=3)
        F.paste(im, (x, y))
        d.text((x + 2, y + MINI + 1), str(r["frame"]), fill=(0, 0, 0), font=F10)
    F.save(saida, quality=85)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--tabela", default="data/derived/annotations_v1.csv")
    ap.add_argument("--janelas", default="data/derived/janelas_v1.csv")
    ap.add_argument("--manifest", default="data/derived/manifest.csv")
    ap.add_argument("--anotadas", default="", help="pasta com os CSV antigos (Dataset_X__caso__seed.csv), para os excluir")
    ap.add_argument("--aleatorias", type=int, default=20)
    ap.add_argument("--extremos", type=int, default=5, help="nº de janelas com Similariedade mais baixa e mais alta")
    ap.add_argument("--caso", default="")
    ap.add_argument("--seed", default="")
    ap.add_argument("--semente", type=int, default=0)
    ap.add_argument("--saida", default="outputs/folhas")
    args = ap.parse_args()
    ds, out = Path(args.dataset), Path(args.saida)
    out.mkdir(parents=True, exist_ok=True)

    jan = pd.read_csv(args.janelas, dtype={"caso": str, "seed": str, "similaridade": str, "descricao": str},
                      keep_default_na=False)
    jan["sim"] = pd.to_numeric(jan["similaridade"], errors="coerce")

    if args.caso:
        escolha = jan[(jan["caso"] == args.caso) & ((jan["seed"] == args.seed) if args.seed else True)].assign(motivo="pedida")
    else:
        base = jan[jan["excluida"] == 0]
        if args.anotadas:
            ja = set()
            for f in Path(args.anotadas).glob("*.csv"):           # só os nomes, nunca o conteúdo
                partes = f.stem.split("__")
                if len(partes) == 3:
                    ja.add(tuple(partes))
            base = base[[(a, b, c) not in ja for a, b, c in zip(base["split"], base["caso"], base["seed"])]]
            print(f"Excluídas {len(ja)} janelas já anotadas (pelos nomes dos ficheiros).")
        rng = random.Random(args.semente)
        partes = []
        com_caixa = base[base["n_box"] > 0]
        por_split = max(1, args.aleatorias // 2)
        for sp in sorted(com_caixa["split"].unique()):
            idx = list(com_caixa.index[com_caixa["split"] == sp])
            rng.shuffle(idx)
            partes.append(com_caixa.loc[idx[:por_split]].assign(motivo="aleatória (com caixa)"))
        com_sim = base.dropna(subset=["sim"]).sort_values("sim")
        partes.append(com_sim.head(args.extremos).assign(motivo="Similariedade mais baixa"))
        partes.append(com_sim.tail(args.extremos).assign(motivo="Similariedade mais alta"))
        escolha = pd.concat(partes).drop_duplicates(subset=["split", "caso", "seed"])

    if escolha.empty:
        print("Nenhuma janela escolhida.")
        return
    chaves = set(zip(escolha["split"], escolha["caso"], escolha["seed"]))
    print(f"A ler a tabela para {len(chaves)} janelas…")
    t = pd.read_csv(args.tabela, dtype={"caso": str, "seed": str, "frame": str}, keep_default_na=False,
                    usecols=["split", "caso", "seed", "frame", "offset", "box_src", "box_x1", "box_y1", "box_x2", "box_y2",
                             "ficheiro_img"])
    t = t[[k in chaves for k in zip(t["split"], t["caso"], t["seed"])]].copy()
    t["frame_i"] = t["frame"].astype(int)
    for c in ["box_x1", "box_y1", "box_x2", "box_y2"]:
        t[c] = pd.to_numeric(t[c], errors="coerce")
    m = pd.read_csv(args.manifest, dtype=str, keep_default_na=False, usecols=["caminho", "tipo", "split", "caso", "seed", "imagem_n"])
    m = m[m["tipo"] == "relatorio_match"].copy()
    m["n"] = m["imagem_n"].astype(int)
    relat = m.sort_values("n").drop_duplicates(subset=["split", "caso", "seed"]).set_index(["split", "caso", "seed"])["caminho"]

    linhas = []
    for i, j in enumerate(escolha.itertuples(index=False), 1):
        nome = f"{j.split}__{j.caso}__{j.seed}.jpg"
        fr = t[(t["split"] == j.split) & (t["caso"] == j.caso) & (t["seed"] == j.seed)]
        folha(ds, j, fr, relat.get((j.split, j.caso, j.seed), ""), out / nome)
        linhas.append({"ficheiro": nome, "split": j.split, "caso": j.caso, "seed": j.seed, "motivo": j.motivo,
                       "similaridade": j.similaridade, "n_box": j.n_box, "descricao": j.descricao, "ok": "", "problema": ""})
        print(f"  [{i}/{len(escolha)}] {nome}  ({j.motivo})")
    rev = out / "revisao.csv"
    if rev.exists() and not args.caso:
        print(f"\nAVISO: {rev} já existia e não foi substituído (pode ter as tuas notas).")
    elif not args.caso:
        pd.DataFrame(linhas).to_csv(rev, index=False)
        print(f"\nLista para revisão: {rev}  (preenche ok = sim/não e problema)")
    print(f"Folhas em {out}/")


if __name__ == "__main__":
    main()
