#!/usr/bin/env python3
"""
Verificação à vista das caixas do médico (antes do passo 0.2). Só lê, não altera nada.

Escolhe caixas ao acaso de cada um dos 4 sentidos em que a caixa pode ter sido desenhada
e desenha-as nas respetivas imagens:
  - retângulo VERDE: a caixa depois de ordenar os cantos (x menor = esquerda, y menor = topo)
  - ponto AMARELO "1": o primeiro canto guardado no ficheiro (onde o desenho começou)
  - ponto VERMELHO "2": o segundo canto guardado (onde o desenho acabou)
Se o retângulo verde estiver em cima da lesão em todos os exemplos, a regra de ordenar está certa.

Uso, a partir da pasta do projeto:
    python scripts/02a_ver_caixas.py --dataset "<caminho>/Dataset_WCE"
Gera:
    outputs/verificacao_caixas.jpg   (abre no VS Code)
"""
import argparse
import random
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

SENTIDOS = {  # (x trocado, y trocado) -> nome
    (False, False): "direita e para baixo (nada trocado)",
    (True, False): "esquerda e para baixo (x trocado)",
    (False, True): "direita e para cima (y trocado)",
    (True, True): "esquerda e para cima (ambos trocados)",
}
ESCALA = 2          # imagens ampliadas 2x para se ver melhor
LEGENDA = 44        # altura da faixa de texto por cima de cada imagem


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--manifest", default="data/derived/manifest.csv")
    ap.add_argument("--por-sentido", type=int, default=2)
    ap.add_argument("--semente", type=int, default=0)
    ap.add_argument("--saida", default="outputs/verificacao_caixas.jpg")
    args = ap.parse_args()
    ds = Path(args.dataset)

    m = pd.read_csv(args.manifest, dtype=str, keep_default_na=False,
                    usecols=["caminho", "split", "caso", "seed", "frame", "box_estado", "box_valores"])
    b = m[m["box_estado"] == "coords"].copy()
    v = b["box_valores"].str.split(expand=True).astype(float)
    b["x1"], b["y1"], b["x2"], b["y2"] = v[0], v[1], v[2], v[3]
    b["sentido"] = list(zip(b["x2"] < b["x1"], b["y2"] < b["y1"]))

    rng = random.Random(args.semente)
    escolhidos = []
    for chave, nome in SENTIDOS.items():
        grupo = b[b["sentido"] == chave]
        idx = list(grupo.index)
        rng.shuffle(idx)
        n = 0
        for i in idx:                                   # salta imagens que não abrem
            r = grupo.loc[i]
            caixa = Path(r["caminho"])
            img_path = ds / caixa.parent.parent / (caixa.stem + ".jpg")
            try:
                im = Image.open(img_path).convert("RGB")
            except Exception:
                continue
            escolhidos.append((nome, r, im))
            n += 1
            if n == args.por_sentido:
                break
        print(f"{nome}: {len(grupo)} caixas no dataset · {n} desenhadas")

    tiles = []
    for nome, r, im in escolhidos:
        w, h = im.size
        im = im.resize((w * ESCALA, h * ESCALA), Image.NEAREST)
        tile = Image.new("RGB", (im.width, im.height + LEGENDA), "white")
        tile.paste(im, (0, LEGENDA))
        d = ImageDraw.Draw(tile)
        x1, y1, x2, y2 = (r["x1"], r["y1"], r["x2"], r["y2"])
        sx = lambda x: x * ESCALA
        sy = lambda y: y * ESCALA + LEGENDA
        d.rectangle([sx(min(x1, x2)), sy(min(y1, y2)), sx(max(x1, x2)), sy(max(y1, y2))],
                    outline=(0, 255, 0), width=3)
        for (px, py), cor, rot in [((x1, y1), (255, 255, 0), "1"), ((x2, y2), (255, 0, 0), "2")]:
            d.ellipse([sx(px) - 6, sy(py) - 6, sx(px) + 6, sy(py) + 6], fill=cor, outline=(0, 0, 0))
            d.text((sx(px) + 8, sy(py) - 8), rot, fill=cor)
        d.text((6, 4), f"{r['split']}/{r['caso']}  seed {r['seed']}  frame {r['frame']}  ({w}x{h})", fill=(0, 0, 0))
        d.text((6, 22), f"ficheiro: {r['box_valores']}  |  {nome}", fill=(0, 0, 0))
        tiles.append(tile)
        print(f"  {r['split']}/{r['caso']} seed {r['seed']} frame {r['frame']}: '{r['box_valores']}' ({w}x{h})")

    if not tiles:
        print("Nenhuma imagem desenhada.")
        return
    colunas = 2
    tw = max(t.width for t in tiles)
    th = max(t.height for t in tiles)
    linhas = (len(tiles) + colunas - 1) // colunas
    folha = Image.new("RGB", (colunas * tw + (colunas + 1) * 10, linhas * th + (linhas + 1) * 10), (200, 200, 200))
    for k, t in enumerate(tiles):
        folha.paste(t, (10 + (k % colunas) * (tw + 10), 10 + (k // colunas) * (th + 10)))
    Path(args.saida).parent.mkdir(parents=True, exist_ok=True)
    folha.save(args.saida, quality=90)
    print(f"\nGravado: {args.saida}  — abre no VS Code e confirma que o retângulo verde está em cima da lesão.")


if __name__ == "__main__":
    main()
