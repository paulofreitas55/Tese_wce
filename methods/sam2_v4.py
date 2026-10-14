#!/usr/bin/env python3
"""
A5 — V4: pós-processamento temporal das previsões do SAM 2 (Guião A). Corre sobre os CSV já gravados pelo
sam2_prop.py: não volta a correr o SAM 2 e não usa a GPU nem as imagens.

Regras, por janela, sobre a presença frame a frame (pela ordem do número do frame):
  1. preencher buracos de 1 frame: um frame ausente entre dois presentes passa a presente;
  2. remover ilhas de 1 frame: um frame presente com os dois vizinhos ausentes passa a ausente se estiver a mais de
     k frames do troço do seed (k ∈ {3, 6}, configs/sam2_v4.yaml). O troço do seed são os frames presentes seguidos
     que contêm o seed; a distância conta-se até ao frame mais próximo desse troço.
O seed conta sempre como presente (é o frame anotado). Os frames sem imagem (sem linha no CSV) contam como ausentes e
não são preenchidos (continuam sem linha).

A presença depende de τ, que só se fixa na calibração. Por isso o V4 não grava presenças: reescreve o score. O novo
score de um frame é o maior τ para o qual o frame fica presente depois das regras. As duas regras são monótonas (mais
frames presentes à entrada nunca tiram frames à saída); por isso, para qualquer τ > 0, marcar "novo score ≥ τ" dá
exatamente o mesmo que marcar "score ≥ τ" e aplicar as regras. O avaliador calibra o τ como nos outros métodos.
Frames preenchidos sem máscara própria: caixa = média das caixas dos vizinhos (frame − 1 e frame + 1) que as tiverem;
area_px = a menor das áreas desses vizinhos. Os outros frames mantêm a caixa e a área.

Uso, a partir da pasta do projeto:
    python methods/sam2_v4.py --conjunto CAL
Para cada base de configs/sam2_v4.yaml (SAM2_L12 = linha base; SAM2_L29 = V3) e cada k, lê
outputs/preds/pred_<base>_<CONJUNTO>.csv e grava outputs/preds/pred_<base>_V4k<k>_<CONJUNTO>.csv, no mesmo formato,
e pred_SAM2_V4_<CONJUNTO>_resumo.txt. Depois, por variante:
    python eval/evaluate.py calibrar --metodo SAM2_L12_V4k3 --pred outputs/preds/pred_SAM2_L12_V4k3_CAL.csv
VAL_EX e TEST_BOX só no A7.
"""
import argparse
import datetime
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "eval"))
import evaluate as ev  # noqa: E402

VERSAO = "1.0"
CAIXA = ["box_x1", "box_y1", "box_x2", "box_y2"]


# ============================================================================ regras
def regras(presente, existe, i_seed, k):
    """As duas regras do V4 sobre presenças binárias.
    presente: (n,) ou (T, n) bool, frames seguidos da janela pela ordem do número do frame (uma linha por τ);
    existe: (n,) bool, frame com linha nas previsões; i_seed: posição do seed (conta sempre como presente)."""
    existe = np.asarray(existe, dtype=bool).copy()
    existe[i_seed] = True
    b = np.atleast_2d(np.asarray(presente, dtype=bool)) & existe
    b[:, i_seed] = True
    c = b.copy()
    c[:, 1:-1] |= b[:, :-2] & b[:, 2:] & existe[1:-1]                      # 1. buracos de 1 frame
    esq = np.cumprod(c[:, i_seed::-1], axis=1).sum(axis=1)                  # presentes seguidos a partir do seed
    dir_ = np.cumprod(c[:, i_seed:], axis=1).sum(axis=1)
    ini, fim = i_seed - esq + 1, i_seed + dir_ - 1                          # troço do seed
    pos = np.arange(c.shape[1])
    dist = np.maximum(ini[:, None] - pos, pos - fim[:, None]).clip(min=0)
    vizinho = np.zeros_like(c)
    vizinho[:, 1:] |= c[:, :-1]
    vizinho[:, :-1] |= c[:, 1:]
    return c & ~(~vizinho & (dist > k))                                     # 2. ilhas de 1 frame longe do troço


def novo_score(score, existe, i_seed, k):
    """score: (n,) pela ordem do número do frame (ignorado onde não existe). Devolve o novo score: o maior τ (entre os
    scores da janela) com o frame presente depois das regras, ou 0. Para qualquer τ > 0: novo ≥ τ ⇔ regras(score ≥ τ)."""
    score = np.asarray(score, dtype=float)
    existe = np.asarray(existe, dtype=bool)
    t = np.unique(np.append(score[existe], 1.0))                            # limiares candidatos (o seed vale 1)
    with np.errstate(invalid="ignore"):
        presente = score[None, :] >= t[:, None]
    p = regras(presente, existe, i_seed, k)
    return np.where(p, t[:, None], 0.0).max(axis=0)


# ============================================================================ janelas
def processar(pred, k):
    """pred: previsões de um conjunto (formato do avaliador, com caixas). Devolve (previsões do V4, contagens)."""
    chaves = [c for c in ("split", "caso", "seed") if c in pred.columns]
    tem_area = "area_px" in pred.columns
    partes = []
    n = {"subiu": 0, "desceu": 0, "caixa_dos_vizinhos": 0, "janelas_alteradas": 0, "janelas": 0}
    for _, g in pred.groupby(chaves, sort=True):
        g = g.sort_values("frame").copy()
        seed = int(g["seed"].iloc[0])
        fr = g["frame"].to_numpy(dtype=int)
        f0 = min(int(fr.min()), seed)
        pos = fr - f0
        s = np.full(max(int(fr.max()), seed) - f0 + 1, np.nan)
        s[pos] = g["score"].to_numpy(dtype=float)
        existe = np.zeros(len(s), dtype=bool)
        existe[pos] = True
        antigo = s[pos]
        novo = novo_score(s, existe, seed - f0, k)[pos]
        novo[fr == seed] = antigo[fr == seed]                               # a linha do seed (se houver) não muda
        caixa0 = g[CAIXA].to_numpy(dtype=float)
        caixa = caixa0.copy()
        area = g["area_px"].to_numpy(dtype=float) if tem_area else None
        linha = {f: i for i, f in enumerate(fr)}
        for i in np.flatnonzero((novo > antigo) & np.isnan(caixa0[:, 0])):  # preenchido sem máscara própria
            viz = [linha[f] for f in (fr[i] - 1, fr[i] + 1) if f in linha and not np.isnan(caixa0[linha[f], 0])]
            if viz:
                caixa[i] = caixa0[viz].mean(axis=0).round(6)
                if tem_area:
                    area[i] = min(area[j] for j in viz)
                n["caixa_dos_vizinhos"] += 1
        g["score"] = novo
        g[CAIXA] = caixa
        if tem_area:
            g["area_px"] = area.astype(int)
        partes.append(g)
        n["subiu"] += int((novo > antigo).sum())
        n["desceu"] += int((novo < antigo).sum())
        n["janelas_alteradas"] += int((novo != antigo).any())
        n["janelas"] += 1
    return pd.concat(partes, ignore_index=True).sort_values(chaves + ["frame"]), n


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--conjunto", required=True, choices=("CAL", "VAL_EX", "TEST_BOX", "ALL_SEEDBOX"))
    a.add_argument("--config-v4", default="configs/sam2_v4.yaml")
    args = a.parse_args(argv)
    conf = ev.ler_config(args.config_v4)
    entrada, saida = Path(conf["entrada"]), Path(conf["saida"])
    ks = sorted({int(x) for x in conf["k"]})
    git = ev._estado_git(RAIZ)
    estado = ("sem git" if git["commit"] is None else
              f"commit {git['commit']}" + (" (com alterações por guardar)" if git["alteracoes_por_guardar"] else ""))
    L = [f"V4 (pós-processamento temporal) em {args.conjunto} · {datetime.datetime.now():%Y-%m-%d %H:%M} · {estado}",
         f"sam2_v4.py v{VERSAO} · sha256 {ev._sha256(__file__)[:16]} · {Path(args.config_v4).name} sha256 "
         f"{ev._sha256(args.config_v4)[:16]} · k = {', '.join(map(str, ks))}"]
    saida.mkdir(parents=True, exist_ok=True)
    csv, contagens = {}, {}
    for base in conf["bases"]:
        f = entrada / f"pred_{base}_{args.conjunto}.csv"
        if not f.is_file():
            raise ev.ErroAvaliacao(f"Não encontrei {f}: corre primeiro o methods/sam2_prop.py em {args.conjunto}.")
        pred = pd.read_csv(f, dtype={"caso": str})
        falta = [c for c in ["caso", "seed", "frame", "score"] + CAIXA if c not in pred.columns]
        if falta:
            raise ev.ErroAvaliacao(f"{f.name}: faltam as colunas {', '.join(falta)}.")
        L += ["", f"{base} ({f.name}, sha256 {ev._sha256(f)[:16]}): {len(pred)} frames"]
        for k in ks:
            novo, n = processar(pred, k)
            destino = saida / f"pred_{base}_V4k{k}_{args.conjunto}.csv"
            novo.to_csv(destino, index=False)
            csv[(base, k)], contagens[(base, k)] = destino, n
            L.append(f"  k = {k}: score subiu em {n['subiu']} frames (buracos preenchidos; {n['caixa_dos_vizinhos']} "
                     f"sem máscara própria ficaram com a caixa dos vizinhos) · desceu em {n['desceu']} (ilhas "
                     f"removidas) · janelas com alterações: {n['janelas_alteradas']} de {n['janelas']}")
    if args.conjunto == "CAL":
        L += ["", "Calibração, uma por variante (comparam-se em CAL com a base):"]
        L += [f"  python eval/evaluate.py calibrar --metodo {b}_V4k{k} --pred {p}" for (b, k), p in csv.items()]
    texto = "\n".join(L)
    f_resumo = saida / f"pred_SAM2_V4_{args.conjunto}_resumo.txt"
    f_resumo.write_text(texto + "\n", encoding="utf-8")
    print(texto)
    return {"csv": csv, "contagens": contagens, "resumo": f_resumo}


if __name__ == "__main__":
    try:
        main()
    except ev.ErroAvaliacao as e:
        print(f"ERRO: {e}", file=sys.stderr)
        sys.exit(2)
