#!/usr/bin/env python3
"""
Passo 0.7 (preparação) — lista das janelas antigas a reanotar às cegas. Só lê NOMES de ficheiros.

As 30 janelas anotadas antes (PF v1) estão em CSV com o nome <split>__<caso>__<seed>.csv. Este
script lê APENAS os nomes desses ficheiros (nunca o conteúdo), cruza-os com a tabela de janelas e
fica com as que podem ser reanotadas pelo critério PF v2: janelas de lesão, com caixa do médico no
seed e não excluídas. Grava-as por ordem aleatória (semente 0), que é a ordem de anotação.

Uso, a partir da pasta do projeto:
    python scripts/07_reanotar_lista.py --anotadas "$NOVA/annotations"

Lê:    os nomes dos CSV em --anotadas · data/derived/janelas_v1.csv
Gera:  docs/janelas_reanotar.csv (ordem, split, caso, seed, categoria)
       outputs/07_reanotar_lista_resumo.txt   ← é este que se lê
"""
import argparse
import csv
import random
import sys
from pathlib import Path

K3 = ("split", "caso", "seed")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--anotadas", required=True, help="pasta dos CSV antigos (só os nomes são lidos)")
    ap.add_argument("--janelas", default="data/derived/janelas_v1.csv")
    ap.add_argument("--saida", default="docs/janelas_reanotar.csv")
    ap.add_argument("--resumo", default="outputs/07_reanotar_lista_resumo.txt")
    ap.add_argument("--semente", type=int, default=0)
    args = ap.parse_args()

    pasta = Path(args.anotadas)
    if not pasta.is_dir():
        sys.exit(f"Não encontrei a pasta {pasta}")
    if not Path(args.janelas).is_file():
        sys.exit(f"Não encontrei {args.janelas}")

    antigas = []
    for f in sorted(pasta.glob("*.csv")):                       # só os nomes, nunca o conteúdo
        p = f.stem.split("__")
        if len(p) == 3:
            antigas.append(tuple(p))

    with open(args.janelas, encoding="utf-8", newline="") as fh:
        jan = {tuple(r[k] for k in K3): r for r in csv.DictReader(fh)}
    for col in ("excluida", "lesao", "seed_tem_box", "categoria", "group_id"):
        if col not in next(iter(jan.values())):
            sys.exit(f"A tabela {args.janelas} não tem a coluna '{col}' (falta correr o 04 ou o 05?).")

    boas, fora = [], []
    for k in antigas:
        r = jan.get(k)
        if r is None:
            fora.append((k, "não está na tabela de janelas"))
        elif r["excluida"] == "1":
            fora.append((k, "janela excluída (imagens vazias)"))
        elif r["lesao"] != "1":
            fora.append((k, f"não é de lesão ({r['categoria'] or 'sem categoria'})"))
        elif r["seed_tem_box"] != "1":
            fora.append((k, "sem caixa do médico no seed"))
        else:
            boas.append(r)

    random.Random(args.semente).shuffle(boas)
    Path(args.saida).parent.mkdir(parents=True, exist_ok=True)
    with open(args.saida, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ordem", "split", "caso", "seed", "categoria"])
        for i, r in enumerate(boas, 1):
            w.writerow([i, r["split"], r["caso"], r["seed"], r["categoria"]])

    n_tr = sum(r["split"].endswith("Treino") for r in boas)
    grupos = {r["group_id"] for r in boas}
    L = ["Passo 0.7 (preparação) — janelas antigas a reanotar às cegas (só pelos nomes dos ficheiros)"]
    L.append(f"\n  CSV antigos encontrados: {len(antigas)}")
    L.append(f"  a reanotar: {len(boas)} (treino {n_tr} · teste {len(boas) - n_tr})   (verificação de 05/10: 22 — 14 · 8)")
    L.append(f"  grupos diferentes entre elas: {len(grupos)} de {len(boas)}")
    L.append(f"  ficam de fora: {len(fora)}")
    for (s, c, sd), motivo in fora:
        L.append(f"    {s}/{c}/{sd} · {motivo}")
    L.append(f"\n  ordem de anotação (semente {args.semente}) em {args.saida}")
    texto = "\n".join(L)
    print(texto)
    Path(args.resumo).parent.mkdir(parents=True, exist_ok=True)
    Path(args.resumo).write_text(texto + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
