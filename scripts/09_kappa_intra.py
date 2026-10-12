#!/usr/bin/env python3
"""
Passo 0.7 (fecho) — concordância intra-anotador: anotação antiga (PF v1) × reanotação às cegas (PF v2),
nas 22 janelas antigas de lesão com caixa no seed (docs/janelas_reanotar.csv).

Só corre com a referência PF v2 congelada: confirma primeiro que data/derived/paulo_frame_labels.csv tem o
sha256 registado em data/derived/paulo_frame_labels.sha256. As anotações antigas só são lidas aqui.

Comparação frame a frame, sem o seed. A PF v1 tem dois estados (Contains_Lesion = 1 / 0). Os frames D da
PF v2 ficam fora do κ, como nas métricas, e são contados à parte com o rótulo que tinham na v1.
Resultados: κ de Cohen com IC 95 % por bootstrap sobre janelas (1 000 reamostragens, semente 0),
concordância observada e concordância positiva, 2a / (2a + b + c), que diz mais do que o κ quando há
tão poucos positivos.

Uso, a partir da pasta do projeto:
    python scripts/09_kappa_intra.py --antigas "$NOVA/annotations"

Lê:    <antigas>/<split>__<caso>__<seed>.csv (só as janelas da lista) · docs/janelas_reanotar.csv ·
       data/derived/paulo_frame_labels.csv e .sha256
Gera:  outputs/09_kappa_intra_janelas.csv (uma linha por janela)
       outputs/09_kappa_intra_resumo.txt   ← é este que se lê
"""
import argparse
import csv
import datetime
import hashlib
import sys
from pathlib import Path

import numpy as np

COLUNAS_JANELA = ["split", "caso", "seed", "categoria", "comparados", "v1_pos", "v2_pos", "ambos_pos",
                  "so_v1", "so_v2", "concordancia_pct", "v2_duvidas", "duvidas_eram_1_na_v1",
                  "sem_rotulo_v1"]


def sha256(caminho):
    h = hashlib.sha256()
    with open(caminho, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def ler_csv(caminho):
    with open(caminho, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def chave(split, caso, seed):
    return (str(split).strip(), str(caso).strip().zfill(4), str(int(float(seed))))


def kappa(a, b, c, d):
    """κ de Cohen numa tabela 2×2: a = 1/1, b = só v1, c = só v2, d = 0/0."""
    n = a + b + c + d
    if n == 0:
        return float("nan")
    po = (a + d) / n
    pe = ((a + b) * (a + c) + (c + d) * (b + d)) / n ** 2
    return float("nan") if pe >= 1 else (po - pe) / (1 - pe)


def escala(k):
    """Leitura de Landis e Koch (1977)."""
    if not np.isfinite(k):
        return "indefinido"
    if k < 0:
        return "pior do que o acaso"
    for lim, nome in ((0.20, "ligeira"), (0.40, "razoável"), (0.60, "moderada"), (0.80, "substancial")):
        if k <= lim:
            return nome
    return "quase perfeita"


def pt(x, casas=2):
    return "—" if x is None or not np.isfinite(x) else f"{x:.{casas}f}".replace(".", ",")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--antigas", required=True, help="pasta dos CSV antigos (PF v1), só leitura")
    ap.add_argument("--lista", default="docs/janelas_reanotar.csv")
    ap.add_argument("--referencia", default="data/derived/paulo_frame_labels.csv")
    ap.add_argument("--sha", default="data/derived/paulo_frame_labels.sha256")
    ap.add_argument("--saida", default="outputs/09_kappa_intra_janelas.csv")
    ap.add_argument("--resumo", default="outputs/09_kappa_intra_resumo.txt")
    ap.add_argument("--reamostragens", type=int, default=1000)
    ap.add_argument("--semente", type=int, default=0)
    args = ap.parse_args()

    # 1. a referência tem de ser a congelada
    ref, fsha = Path(args.referencia), Path(args.sha)
    for p in (ref, fsha, Path(args.lista)):
        if not p.is_file():
            sys.exit(f"Não encontrei {p}.")
    esperado = fsha.read_text(encoding="utf-8").split()[0].lower()
    atual = sha256(ref)
    if atual != esperado:
        sys.exit(f"O sha256 de {ref} ({atual[:8]}…) não é o congelado ({esperado[:8]}…). Não comparo nada.")
    pasta = Path(args.antigas)
    if not pasta.is_dir():
        sys.exit(f"Não encontrei a pasta {pasta}.")

    # 2. janelas a comparar e rótulos da v2
    lista = [(chave(r["split"], r["caso"], r["seed"]), r.get("categoria", "")) for r in ler_csv(args.lista)]
    alvo = {k for k, _ in lista}
    v2 = {}
    for r in ler_csv(ref):
        k = chave(r["split"], r["caso"], r["seed"])
        if k in alvo:
            v2.setdefault(k, {})[int(float(r["frame"]))] = r["visivel"].strip()

    # 3. comparação janela a janela
    linhas, matrizes, avisos, hashes = [], [], [], []
    tot_d = {"1": 0, "0": 0}
    for k, categoria in lista:
        nome = f"{k[0]}__{k[1]}__{k[2]}.csv"
        f = pasta / nome
        if k not in v2:
            avisos.append(f"{'/'.join(k)}: não está na referência PF v2")
            continue
        if not f.is_file():
            avisos.append(f"{'/'.join(k)}: falta o CSV antigo {nome}")
            continue
        antigas = ler_csv(f)
        if not antigas or not {"Frame_Number", "Contains_Lesion"} <= set(antigas[0]):
            avisos.append(f"{'/'.join(k)}: o CSV antigo não tem as colunas Frame_Number e Contains_Lesion")
            continue
        hashes.append(f"{sha256(f)}  {nome}")
        v1 = {int(float(r["Frame_Number"])): r["Contains_Lesion"].strip() for r in antigas}
        seed = int(k[2])
        a = b = c = d = sem = 0
        dv = {"1": 0, "0": 0}
        for fr, l2 in v2[k].items():
            if fr == seed:
                continue
            l1 = v1.get(fr)
            if l1 not in ("0", "1"):
                sem += 1
                continue
            if l2 == "duvidoso":
                dv[l1] += 1
                continue
            if l1 == "1" and l2 == "1":
                a += 1
            elif l1 == "1":
                b += 1
            elif l2 == "1":
                c += 1
            else:
                d += 1
        if sem:
            avisos.append(f"{'/'.join(k)}: {sem} frames da janela sem rótulo 0/1 na v1 (ficam fora)")
        so_v1 = sorted(set(v1) - set(v2[k]))
        if so_v1:
            avisos.append(f"{'/'.join(k)}: a v1 tem {len(so_v1)} frames que não existem na janela (ignorados)")
        for x in dv:
            tot_d[x] += dv[x]
        n = a + b + c + d
        matrizes.append((a, b, c, d))
        linhas.append([k[0], k[1], k[2], categoria, n, a + b, a + c, a, b, c,
                       round(100 * (a + d) / n, 1) if n else "", dv["1"] + dv["0"], dv["1"], sem])

    if not matrizes:
        sys.exit("Nenhuma janela comparada:\n  " + "\n  ".join(avisos))

    # 4. κ global e IC por bootstrap sobre janelas
    M = np.array(matrizes, dtype=float)
    a, b, c, d = M.sum(axis=0)
    n = a + b + c + d
    k_glob = kappa(a, b, c, d)
    rng = np.random.default_rng(args.semente)
    boot = []
    for _ in range(args.reamostragens):
        s = M[rng.integers(0, len(M), len(M))].sum(axis=0)
        boot.append(kappa(*s))
    boot = np.array(boot)
    boot = boot[np.isfinite(boot)]
    ic = np.percentile(boot, [2.5, 97.5]) if len(boot) else [np.nan, np.nan]
    pos = 2 * a / (2 * a + b + c) if (2 * a + b + c) else float("nan")
    neg = 2 * d / (2 * d + b + c) if (2 * d + b + c) else float("nan")
    discordes = sum(1 for m in matrizes if m[1] + m[2] > 0)

    # 5. saídas
    Path(args.saida).parent.mkdir(parents=True, exist_ok=True)
    with open(args.saida, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(COLUNAS_JANELA)
        w.writerows(linhas)
    conj = hashlib.sha256("\n".join(sorted(hashes)).encode()).hexdigest()
    n_tr = sum(r[0].endswith("Treino") for r in linhas)
    L = ["Passo 0.7 (fecho) — concordância intra-anotador: PF v1 (antiga) × PF v2 (reanotação às cegas)",
         f"  {datetime.datetime.now().isoformat(timespec='seconds')}",
         "",
         f"  referência PF v2: {ref} · sha256 {atual[:8]}…{atual[-4:]} (confere com o congelado)",
         f"  anotações antigas: {pasta} · {len(hashes)} CSV lidos · sha256 do conjunto {conj[:8]}…{conj[-4:]}",
         f"  janelas comparadas: {len(linhas)} de {len(lista)} (treino {n_tr} · teste {len(linhas) - n_tr})",
         "",
         f"  frames comparados (sem o seed e sem os D da v2): {int(n)}",
         "                    v2 = 1    v2 = 0",
         f"      v1 = 1     {int(a):8d}  {int(b):8d}",
         f"      v1 = 0     {int(c):8d}  {int(d):8d}",
         "",
         f"  concordância observada: {pt(100 * (a + d) / n, 1)} %",
         f"  κ de Cohen: {pt(k_glob)} (IC 95 % {pt(ic[0])} a {pt(ic[1])}; bootstrap por janela, "
         f"{args.reamostragens} reamostragens, semente {args.semente}) · {escala(k_glob)} (Landis e Koch)",
         f"  concordância positiva: {pt(pos)} · concordância negativa: {pt(neg)}",
         f"  positivos: {int(a + b)} na v1 · {int(a + c)} na v2 · janelas com alguma discordância: {discordes} de {len(linhas)}",
         f"  frames D na v2: {tot_d['1'] + tot_d['0']} (na v1 estavam 1: {tot_d['1']} · 0: {tot_d['0']})",
         "",
         f"  por janela: {args.saida}"]
    if avisos:
        L += ["", "  AVISOS:"] + ["    " + x for x in avisos]
    texto = "\n".join(L)
    print(texto)
    Path(args.resumo).parent.mkdir(parents=True, exist_ok=True)
    Path(args.resumo).write_text(texto + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
