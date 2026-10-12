#!/usr/bin/env python3
"""
Passo 0.1 — Inventário do dataset (só leitura: não altera nada no dataset).

Parte da lista de ficheiros criada na Parte 1 (caminho relativo + tamanho em bytes) e classifica cada
ficheiro como: frame, box, relatorio_match, relatorio_descricao ou outro.
Só abre os ficheiros de box que não estão vazios (~9 600) e 200 frames ao acaso (resolução).

Uso, a partir da pasta do projeto:
    python scripts/01_inventario.py --dataset "<caminho>/Dataset_WCE"

Gera:
    outputs/01_inventario_resumo.txt   contagens e verificações  ← é este que se lê
    data/derived/manifest.csv          uma linha por ficheiro, já interpretada (base do passo 0.2)
"""
import argparse
import csv
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Números do diagnóstico / guiões, para comparação
REF = {
    "frames": 517_986,
    "ficheiros de box": 513_682,
    "box com coordenadas": 9_615,
    "casos": 320,
    "seeds (relatório)": 2_601,
    "janelas com frames": 2_593,
}
CASOS_SEM_FRAMES_REF = {"0044", "0118", "0130", "0235", "0403", "0493", "0676"}

BASE = r"^(?P<split>[^/]+)/(?P<caso>[^/]+)/GT_com_classes/"
RE_FRAME = re.compile(BASE + r"(?P<seed>\d+)/(?P<c>\d+)_(?P<frame>\d+)_(?P<s>\d+)\.(?:jpe?g|png)$", re.I)
RE_BOX = re.compile(BASE + r"(?P<seed>\d+)/Findings_(?P<sf>\d+)/(?P<c>\d+)_(?P<frame>\d+)_(?P<s>\d+)\.txt$", re.I)
RE_MATCH = re.compile(BASE + r"imagem(?P<n>\d+)_(?P<k>\d+)match_at_videoframe(?P<seed>\d+)"
                      r"(?:_Similariedade_(?P<sim>-?\d+(?:\.\d+)?))?", re.I)
RE_DESC = re.compile(BASE + r"imagem(?P<n>\d+)_(?P<resto>[^/]+)$", re.I)
RE_NUM = re.compile(r"-?\d+(?:\.\d+)?")

COLUNAS = ["caminho", "tamanho", "tipo", "split", "caso", "seed", "frame", "imagem_n",
           "box_estado", "box_valores", "descricao", "similaridade"]


def igual(a, b):
    try:
        return int(a) == int(b)
    except ValueError:
        return a == b


def ler_box(caminho):
    """Devolve (estado, valores, n_linhas) de um ficheiro de box com conteúdo."""
    try:
        texto = caminho.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"erro_leitura ({type(e).__name__})", "", 0
    linhas = [l for l in texto.splitlines() if l.strip()]
    nums = RE_NUM.findall(texto)
    if not nums:
        return ("so_espacos" if not texto.strip() else "ilegivel"), "", len(linhas)
    if len(linhas) > 1:
        return "varias_linhas", " ".join(nums), len(linhas)
    if len(nums) == 4:
        return "coords", " ".join(nums), 1
    if len(nums) == 5:
        return "5_numeros", " ".join(nums), 1
    return "outro_formato", " ".join(nums), len(linhas)


def ler_lista(caminho):
    with open(caminho, encoding="utf-8", errors="replace") as f:
        for linha in f:
            linha = linha.rstrip("\n")
            if not linha:
                continue
            cam, _, tam = linha.rpartition("\t")
            yield cam, int(tam)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="pasta Dataset_WCE (cópia de trabalho)")
    ap.add_argument("--lista", default="outputs/lista_dataset.tsv", help="lista criada na Parte 1")
    ap.add_argument("--resumo", default="outputs/01_inventario_resumo.txt")
    ap.add_argument("--manifest", default="data/derived/manifest.csv")
    ap.add_argument("--amostra-resolucao", type=int, default=200)
    args = ap.parse_args()

    ds = Path(args.dataset)
    if not Path(args.lista).is_file():
        sys.exit(f"Não encontrei {args.lista}: corre primeiro a Parte 1 (lista de ficheiros).")
    if not ds.is_dir():
        sys.exit(f"Não encontrei a pasta do dataset: {ds}")
    for p in (args.resumo, args.manifest):
        Path(p).parent.mkdir(parents=True, exist_ok=True)

    tipos = Counter()
    casos_vistos = set()              # (split, caso)
    frames_por_janela = Counter()     # (split, caso, seed) -> nº de frames
    box_por_janela = Counter()
    coords_por_janela = Counter()
    chaves_frame, chaves_box = set(), set()
    incoerentes = []                  # nome do ficheiro não bate com a pasta onde está
    box_estado = Counter()
    box_geometria = Counter()
    match = defaultdict(set)          # (split, caso) -> {(n, seed)}
    k_match = Counter()
    desc = defaultdict(lambda: defaultdict(list))   # (split, caso) -> n -> [descrições]
    outros_ext, outros_ex = Counter(), []
    rng, amostra, n_frames_vistos = random.Random(0), [], 0
    sims, sem_sim = [], 0             # valor "Similariedade" nos ficheiros match

    with open(args.manifest, "w", newline="", encoding="utf-8") as fm:
        w = csv.writer(fm)
        w.writerow(COLUNAS)
        for cam, tam in ler_lista(args.lista):
            partes = cam.split("/")
            if len(partes) >= 3:
                casos_vistos.add((partes[0], partes[1]))
            linha = dict.fromkeys(COLUNAS, "")
            linha.update(caminho=cam, tamanho=tam)

            m = RE_FRAME.match(cam)
            if m:
                s, c, sd, fr = m["split"], m["caso"], m["seed"], m["frame"]
                linha.update(tipo="frame", split=s, caso=c, seed=sd, frame=fr)
                frames_por_janela[(s, c, sd)] += 1
                chaves_frame.add(f"{s}|{c}|{sd}|{int(fr)}")
                if not (igual(m["c"], c) and igual(m["s"], sd)):
                    incoerentes.append(cam)
                n_frames_vistos += 1          # amostragem por reservatório (determinística)
                if len(amostra) < args.amostra_resolucao:
                    amostra.append(cam)
                else:
                    j = rng.randrange(n_frames_vistos)
                    if j < args.amostra_resolucao:
                        amostra[j] = cam
            elif (m := RE_BOX.match(cam)):
                s, c, sd, fr = m["split"], m["caso"], m["seed"], m["frame"]
                estado, valores = "vazio", ""
                if tam > 0:
                    estado, valores, _ = ler_box(ds / cam)
                linha.update(tipo="box", split=s, caso=c, seed=sd, frame=fr, box_estado=estado, box_valores=valores)
                box_estado[estado] += 1
                box_por_janela[(s, c, sd)] += 1
                chaves_box.add(f"{s}|{c}|{sd}|{int(fr)}")
                if not (igual(m["c"], c) and igual(m["s"], sd) and igual(m["sf"], sd)):
                    incoerentes.append(cam)
                if estado == "coords":
                    coords_por_janela[(s, c, sd)] += 1
                    x1, y1, x2, y2 = map(float, valores.split())
                    if x2 < x1 or y2 < y1:
                        box_geometria["cantos trocados"] += 1
                    if abs(x2 - x1) < 2 or abs(y2 - y1) < 2:
                        box_geometria["degenerada (largura ou altura < 2 px)"] += 1
            elif (m := RE_MATCH.match(cam)):
                s, c = m["split"], m["caso"]
                linha.update(tipo="relatorio_match", split=s, caso=c, seed=m["seed"], imagem_n=m["n"],
                             similaridade=m["sim"] or "")
                if m["sim"]:
                    sims.append(float(m["sim"]))
                else:
                    sem_sim += 1
                match[(s, c)].add((m["n"], m["seed"]))
                k_match[m["k"]] += 1
            elif (m := RE_DESC.match(cam)):
                s, c = m["split"], m["caso"]
                texto = Path(m["resto"]).stem.rstrip(". ")
                linha.update(tipo="relatorio_descricao", split=s, caso=c, imagem_n=m["n"], descricao=texto)
                desc[(s, c)][m["n"]].append(texto)
            else:
                linha.update(tipo="outro")
                outros_ext[Path(cam).suffix.lower() or "(sem extensão)"] += 1
                if len(outros_ex) < 15:
                    outros_ex.append(cam)
            tipos[linha["tipo"]] += 1
            w.writerow([linha[k] for k in COLUNAS])

    # ---------- cruzamentos ----------
    janelas = set(frames_por_janela)
    casos_com_frames = {(s, c) for (s, c, _) in janelas}
    casos_sem_frames = sorted(casos_vistos - casos_com_frames)
    seeds = {(s, c, sd) for (s, c), v in match.items() for (_, sd) in v}
    seeds_sem_janela = sorted(seeds - janelas)
    janelas_sem_seed = sorted(janelas - seeds)
    n_desc = Counter()
    exemplos_sem_desc, exemplos_multi = [], []
    for (s, c, sd) in sorted(seeds):
        ns = {n for (n, x) in match[(s, c)] if x == sd}
        ds_ = [d for n in ns for d in desc[(s, c)].get(n, [])]
        k = len(ds_)
        n_desc["0" if k == 0 else "1" if k == 1 else "2 ou mais"] += 1
        if k == 0 and len(exemplos_sem_desc) < 10:
            exemplos_sem_desc.append(f"{s}/{c} seed {sd}")
        if k > 1 and len(exemplos_multi) < 10:
            exemplos_multi.append(f"{s}/{c} seed {sd}: {' | '.join(ds_)}")
    frames_sem_box = chaves_frame - chaves_box
    box_sem_frame = chaves_box - chaves_frame
    tamanhos_janela = sorted(frames_por_janela.values())

    resol = Counter()
    try:
        from PIL import Image
        for cam in amostra:
            with Image.open(ds / cam) as im:
                resol[f"{im.size[0]}×{im.size[1]}"] += 1
    except ImportError:
        resol["(Pillow não instalado — resolução não verificada)"] += 1

    # ---------- resumo ----------
    L = []
    sec = lambda t: L.append(f"\n== {t} " + "=" * max(0, 64 - len(t)))
    L.append(f"Inventário do dataset — {ds}")
    L.append(f"Lista lida: {args.lista} ({sum(tipos.values())} ficheiros)")

    sec("Ficheiros por tipo")
    for t in ["frame", "box", "relatorio_match", "relatorio_descricao", "outro"]:
        L.append(f"  {t:<22} {tipos[t]:>9}")

    sec("Ficheiros de box por conteúdo")
    for e, n in box_estado.most_common():
        L.append(f"  {e:<22} {n:>9}")
    for g, n in box_geometria.items():
        L.append(f"  (das com coordenadas) {g}: {n}")

    sec("Por split")
    L.append(f"  {'split':<18}{'casos':>7}{'c/ frames':>11}{'seeds':>8}{'janelas':>9}{'frames':>9}{'box c/ coord':>14}")
    for sp in sorted({s for (s, _) in casos_vistos}):
        L.append(f"  {sp:<18}"
                 f"{sum(1 for (s, _) in casos_vistos if s == sp):>7}"
                 f"{sum(1 for (s, _) in casos_com_frames if s == sp):>11}"
                 f"{sum(1 for (s, _, _) in seeds if s == sp):>8}"
                 f"{sum(1 for (s, _, _) in janelas if s == sp):>9}"
                 f"{sum(n for (s, _, _), n in frames_por_janela.items() if s == sp):>9}"
                 f"{sum(n for (s, _, _), n in coords_por_janela.items() if s == sp):>14}")

    sec("Janelas")
    if tamanhos_janela:
        L.append(f"  frames por janela: mín {tamanhos_janela[0]} · mediana {statistics.median(tamanhos_janela):g}"
                 f" · máx {tamanhos_janela[-1]}")
        L.append(f"  janelas com menos de 150 frames: {sum(1 for x in tamanhos_janela if x < 150)}")
    L.append(f"  janelas com ≥ 1 box com coordenadas: {len(coords_por_janela)}")
    L.append(f"  seeds do relatório sem pasta de frames: {len(seeds_sem_janela)}"
             + (f"  (ex.: {', '.join('/'.join(x) for x in seeds_sem_janela[:5])})" if seeds_sem_janela else ""))
    L.append(f"  pastas de frames sem ficheiro 'match' do relatório: {len(janelas_sem_seed)}"
             + (f"  (ex.: {', '.join('/'.join(x) for x in janelas_sem_seed[:5])})" if janelas_sem_seed else ""))
    L.append(f"  frames sem ficheiro de box: {len(frames_sem_box)} · ficheiros de box sem frame: {len(box_sem_frame)}")
    L.append(f"  nomes que não batem com a pasta onde estão: {len(incoerentes)}"
             + (f"  (ex.: {incoerentes[0]})" if incoerentes else ""))

    sec("Ligação seed → descrição clínica")
    L.append(f"  seeds com 1 descrição: {n_desc['1']} · sem descrição: {n_desc['0']} · com 2 ou mais: {n_desc['2 ou mais']}")
    L.append(f"  ficheiros 'match' por número antes de 'match': {dict(k_match)}")
    if sims:
        q = statistics.quantiles(sims, n=20) if len(sims) >= 20 else None
        L.append(f"  'Similariedade' no nome do match: n={len(sims)} · mín {min(sims):g}"
                 + (f" · P5 {q[0]:g} · mediana {statistics.median(sims):g} · P95 {q[-1]:g}" if q else "")
                 + f" · máx {max(sims):g} · sem valor: {sem_sim}")
    for e in exemplos_sem_desc:
        L.append(f"    sem descrição: {e}")
    for e in exemplos_multi:
        L.append(f"    várias: {e}")

    sec("Casos sem frames")
    ids = sorted(c for (_, c) in casos_sem_frames)
    L.append(f"  {len(ids)}: {', '.join(ids) or '-'}")
    L.append(f"  referência (7): {', '.join(sorted(CASOS_SEM_FRAMES_REF))} → "
             + ("IGUAL" if set(ids) == CASOS_SEM_FRAMES_REF else
                f"DIFERENTE (a mais: {sorted(set(ids) - CASOS_SEM_FRAMES_REF)}, "
                f"em falta: {sorted(CASOS_SEM_FRAMES_REF - set(ids))})"))

    sec(f"Resolução ({len(amostra)} frames ao acaso, semente 0)")
    for r_, n in resol.most_common():
        L.append(f"  {r_}: {n}")

    if tipos["outro"]:
        sec("Ficheiros não reconhecidos ('outro')")
        L.append(f"  por extensão: {dict(outros_ext)}")
        for e in outros_ex:
            L.append(f"    {e}")

    obtido = {
        "frames": tipos["frame"],
        "ficheiros de box": tipos["box"],
        "box com coordenadas": box_estado["coords"],
        "casos": len(casos_vistos),
        "seeds (relatório)": len(seeds),
        "janelas com frames": len(janelas),
    }
    sec("Comparação com o diagnóstico")
    difs = []
    for k, ref in REF.items():
        ok = obtido[k] == ref
        L.append(f"  {k:<22} obtido {obtido[k]:>9} · referência {ref:>9} · {'OK' if ok else 'DIFERENÇA'}")
        if not ok:
            difs.append(k)
    L.append("\n  " + ("Tudo bate certo." if not difs else
                        "Diferenças a explicar antes de avançar: " + ", ".join(difs)))
    L.append(f"\nManifest: {args.manifest}")

    texto = "\n".join(L)
    print(texto)
    Path(args.resumo).write_text(texto + "\n", encoding="utf-8")
    print(f"Resumo gravado em {args.resumo}")


if __name__ == "__main__":
    main()
