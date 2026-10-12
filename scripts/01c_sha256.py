#!/usr/bin/env python3
"""
Passo 0.1 (o que faltava) — impressão digital de cada ficheiro e seeds sem janela. Só lê o dataset.

O guião pede no passo 0.1:
  - um manifesto com o sha256 de cada ficheiro e um DATASET_VERSION.txt. Até aqui a versão do dataset
    era o sha256 da lista de ficheiros (caminho + tamanho); isto acrescenta a impressão digital do
    CONTEÚDO de cada ficheiro, para se poder provar que nenhuma imagem nem caixa mudou;
  - os seeds dos casos sem frames registados como excluídos.

Uso, a partir da pasta do projeto (lê ~1 milhão de ficheiros: corre no tmux, ~10–30 min):
    python scripts/01c_sha256.py --dataset "$DS"
Se for interrompido, volta a correr o mesmo comando: continua de onde parou.

Lê:    outputs/lista_dataset.tsv · data/derived/manifest.csv · data/derived/janelas_v1.csv
Gera:  data/derived/manifest_sha256.csv (caminho, tamanho, sha256) · docs/DATASET_VERSION.txt
       docs/seeds_sem_janela.csv · outputs/01c_sha256_resumo.txt   ← é este que se lê
"""
import argparse
import datetime
import hashlib
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

BLOCO = 20_000


def ler_lista(caminho):
    with open(caminho, encoding="utf-8", errors="replace") as f:
        for linha in f:
            linha = linha.rstrip("\n")
            if linha:
                cam, _, tam = linha.rpartition("\t")
                yield cam, int(tam)


def sha256(ds, cam):
    h = hashlib.sha256()
    try:
        with open(ds / cam, "rb") as f:
            n = 0
            for parte in iter(lambda: f.read(1 << 20), b""):
                h.update(parte)
                n += len(parte)
        return cam, n, h.hexdigest()
    except OSError as e:
        return cam, -1, f"ERRO {type(e).__name__}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="pasta Dataset_WCE (cópia de trabalho)")
    ap.add_argument("--lista", default="outputs/lista_dataset.tsv")
    ap.add_argument("--manifest", default="data/derived/manifest.csv")
    ap.add_argument("--janelas", default="data/derived/janelas_v1.csv")
    ap.add_argument("--saida", default="data/derived/manifest_sha256.csv")
    ap.add_argument("--versao", default="docs/DATASET_VERSION.txt")
    ap.add_argument("--versao-antiga", default="docs/versao_dataset.txt")
    ap.add_argument("--seeds", default="docs/seeds_sem_janela.csv")
    ap.add_argument("--resumo", default="outputs/01c_sha256_resumo.txt")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    ds = Path(args.dataset)
    for f in (args.lista, args.manifest, args.janelas):
        if not Path(f).is_file():
            sys.exit(f"Não encontrei {f}.")
    if not ds.is_dir():
        sys.exit(f"Não encontrei a pasta do dataset: {ds}")
    for f in (args.saida, args.versao, args.seeds, args.resumo):
        Path(f).parent.mkdir(parents=True, exist_ok=True)

    # ---------- 1. sha256 de cada ficheiro (retoma se tiver sido interrompido) ----------
    lista = dict(ler_lista(args.lista))
    parcial = Path(f"{args.saida}.parcial")
    feitos = set()
    if parcial.exists():
        with open(parcial, encoding="utf-8") as f:
            feitos = {l.split("\t", 1)[0] for l in f if l.strip()}
        print(f"A continuar: {len(feitos)} ficheiros já tinham sha256.")
    falta = [c for c in lista if c not in feitos]
    print(f"Ficheiros na lista: {len(lista)} · por fazer: {len(falta)}")
    t0 = time.time()
    with open(parcial, "a", encoding="utf-8") as out, ThreadPoolExecutor(args.threads) as ex:
        for i in range(0, len(falta), BLOCO):
            for cam, n, h in ex.map(lambda c: sha256(ds, c), falta[i:i + BLOCO]):
                out.write(f"{cam}\t{n}\t{h}\n")
            out.flush()
            os.fsync(out.fileno())
            feito = min(i + BLOCO, len(falta))
            ritmo = feito / max(time.time() - t0, 1e-9)
            print(f"  {feito}/{len(falta)} · {ritmo:.0f} ficheiros/s · faltam ~{(len(falta) - feito) / max(ritmo, 1e-9) / 60:.0f} min")

    m = pd.read_csv(parcial, sep="\t", names=["caminho", "tamanho_lido", "sha256"], dtype=str, keep_default_na=False,
                    quoting=3)
    m = m.drop_duplicates("caminho", keep="last")
    m = m[m["caminho"].isin(lista)].sort_values("caminho")
    m["tamanho"] = m["caminho"].map(lista)
    erros = m[m["sha256"].str.startswith("ERRO")]
    dif_tam = m[(~m["sha256"].str.startswith("ERRO")) & (m["tamanho_lido"].astype(int) != m["tamanho"])]
    tmp = f"{args.saida}.tmp"
    m[["caminho", "tamanho", "sha256"]].to_csv(tmp, index=False)
    os.replace(tmp, args.saida)

    h = hashlib.sha256()
    for cam, dig in zip(m["caminho"], m["sha256"]):
        h.update(f"{cam}\t{dig}\n".encode("utf-8"))
    versao = h.hexdigest()
    antiga = Path(args.versao_antiga).read_text(encoding="utf-8").strip() if Path(args.versao_antiga).exists() else "?"
    total = int(m["tamanho"].sum())
    Path(args.versao).write_text(
        "Dataset WCE — versão congelada (passo 0.1)\n"
        f"data: {datetime.date.today():%d/%m/%Y}\n"
        f"ficheiros: {len(m)} · tamanho total: {total / 1e9:.2f} GB\n"
        f"sha256 do conteúdo (sha256 das linhas 'caminho<TAB>sha256', por ordem de caminho): {versao}\n"
        f"sha256 da lista de ficheiros (caminho + tamanho, versão anterior): {antiga}\n"
        f"manifesto por ficheiro: {args.saida}\n", encoding="utf-8")

    # ---------- 2. seeds sem janela ----------
    man = pd.read_csv(args.manifest, dtype=str, keep_default_na=False,
                      usecols=["tipo", "split", "caso", "seed", "imagem_n", "similaridade"])
    jan = pd.read_csv(args.janelas, dtype=str, keep_default_na=False, usecols=["split", "caso", "seed"])
    com_janela = set(zip(jan["split"], jan["caso"], jan["seed"]))
    casos_com_frames = set(zip(*[man.loc[man["tipo"] == "frame", c] for c in ["split", "caso"]]))
    mt = man[man["tipo"] == "relatorio_match"].drop_duplicates(["split", "caso", "seed"])
    sem = mt[[k not in com_janela for k in zip(mt["split"], mt["caso"], mt["seed"])]].copy()
    sem["motivo"] = ["caso sem frames" if (s, c) not in casos_com_frames else "seed sem pasta de frames"
                     for s, c in zip(sem["split"], sem["caso"])]
    sem = sem[["split", "caso", "seed", "imagem_n", "similaridade", "motivo"]].sort_values(["split", "caso", "seed"])
    sem.to_csv(args.seeds, index=False)

    # ---------- 3. resumo ----------
    L = ["Passo 0.1 (complemento) — sha256 por ficheiro e seeds sem janela"]
    L.append(f"\n  ficheiros com sha256: {len(m)} de {len(lista)} da lista · tamanho total {total / 1e9:.2f} GB")
    L.append(f"  ficheiros que não abriram: {len(erros)}" + (" → " + "; ".join(erros['caminho'].head(5)) if len(erros) else ""))
    L.append(f"  ficheiros com tamanho diferente do da lista: {len(dif_tam)}"
             + (" → " + "; ".join(dif_tam['caminho'].head(5)) if len(dif_tam) else "   (0 = nada mudou desde a lista)"))
    img = m["caminho"].str.lower().str.endswith((".jpg", ".jpeg", ".png"))
    L.append(f"  imagens vazias (0 bytes): {int((img & (m['tamanho'] == 0)).sum())}   (esperado: 139, nas 2 janelas excluídas)")
    L.append(f"\n  versão do dataset (sha256 do conteúdo): {versao}")
    L.append(f"  versão anterior (sha256 da lista):      {antiga}")
    L.append(f"\n  seeds com imagem do relatório mas sem janela de frames: {len(sem)}")
    for r in sem.itertuples(index=False):
        L.append(f"    {r.split}/{r.caso}/{r.seed} · imagem {r.imagem_n} · Similariedade {r.similaridade or '-'} · {r.motivo}")
    L.append(f"\n  ficheiros: {args.saida} · {args.versao} · {args.seeds}")
    texto = "\n".join(L)
    print(texto)
    Path(args.resumo).write_text(texto + "\n", encoding="utf-8")
    if not len(erros):
        parcial.unlink(missing_ok=True)
    print(f"\nResumo gravado em {args.resumo}")


if __name__ == "__main__":
    main()
