#!/usr/bin/env python3
"""
A5 — V6: paragem adaptativa do SAM 2 (Guião A; acrescento ao plano, decidido a 15 out). Em vez de um limite fixo de
distância ao seed (L_max), o seguimento continua enquanto o SAM 2 vê o objeto e pára de vez quando o perde.

Regra, por janela e por direção (depois e antes do seed), sobre as previsões do SAM 2 com a caixa, seguidas até 100
frames (pred_SAM2_L100_<CONJUNTO>.csv):
  andando do seed para fora (distância 1, 2, 3, …), o seguimento pára no primeiro frame de um bloco de N frames
  seguidos com score < 0,5, o ponto em que o próprio SAM 2 considera que o objeto deixou de estar visível (um frame
  sem linha conta como abaixo). Desse frame em diante, nessa direção, tudo fica com score 0, sem caixa e com área 0.
  Os frames antes da paragem não mudam (os de score baixo ficam de fora pelo τ, como sempre).
N ∈ {1, 3, 6} (configs/sam2_v6.yaml): com N = 1 pára no primeiro frame em que o SAM 2 perde o objeto; com 3 ou 6
tolera frames maus isolados (desfocagem, bolhas). Não lê rótulos nem caixas: só as previsões.

Uso, a partir da pasta do projeto:
    python methods/sam2_v6.py --conjunto CAL
Grava outputs/preds/pred_SAM2_V6n<N>_<CONJUNTO>.csv, no formato do avaliador, e pred_SAM2_V6_<CONJUNTO>_resumo.txt.
Depois, por variante:
    python eval/evaluate.py calibrar --metodo SAM2_V6n3 --pred outputs/preds/pred_SAM2_V6n3_CAL.csv
VAL_EX e TEST_BOX só no A7, e só se o V6 for adotado.
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


def ponto_de_paragem(scores, limiar, n):
    """scores: por ordem de distância ao seed (1, 2, 3, …) numa direção, NaN onde não há frame. Devolve o índice do
    primeiro frame do primeiro bloco de n scores seguidos abaixo do limiar, ou len(scores) se o seguimento nunca pára."""
    seguidos = 0
    for i, s in enumerate(scores):
        if not s >= limiar:                                                 # NaN conta como abaixo
            seguidos += 1
            if seguidos == n:
                return i - n + 1
        else:
            seguidos = 0
    return len(scores)


def processar(pred, limiar, n):
    """Aplica a paragem a todas as janelas. Devolve (previsões do V6, uma linha por direção com a distância da paragem
    ou vazio se não parou)."""
    chaves = [c for c in ("split", "caso", "seed") if c in pred.columns]
    tem_area = "area_px" in pred.columns
    partes, paragens = [], []
    for k, g in pred.groupby(chaves, sort=True):
        g = g.copy()
        seed = int(g["seed"].iloc[0])
        fr = g["frame"].to_numpy(dtype=int)
        score = dict(zip(fr, g["score"].to_numpy(dtype=float)))
        cortar = np.zeros(len(g), dtype=bool)
        for direcao, sinal in (("depois", 1), ("antes", -1)):
            lado = fr * sinal > seed * sinal
            if not lado.any():
                continue
            seq = [score.get(seed + sinal * d, np.nan) for d in range(1, int(np.abs(fr[lado] - seed).max()) + 1)]
            i = ponto_de_paragem(seq, limiar, n)
            cortar |= lado & (np.abs(fr - seed) >= i + 1)
            paragens.append({**dict(zip(chaves, k if isinstance(k, tuple) else (k,))), "direcao": direcao,
                             "distancia_paragem": i + 1 if i < len(seq) else np.nan})
        g.loc[cortar, "score"] = 0.0
        g[CAIXA] = g[CAIXA].astype(float)
        g.loc[cortar, CAIXA] = np.nan
        if tem_area:
            g.loc[cortar, "area_px"] = 0
        partes.append(g)
    return pd.concat(partes, ignore_index=True).sort_values(chaves + ["frame"]), pd.DataFrame(paragens)


def _confiantes(p, limiar):
    """Frames com score ≥ limiar, por distância ao seed: 1–12 e > 12."""
    d = (p["frame"] - p["seed"]).abs()
    c = p["score"] >= limiar
    return int((c & (d <= 12)).sum()), int((c & (d > 12)).sum())


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--conjunto", required=True, choices=("CAL", "VAL_EX", "TEST_BOX", "ALL_SEEDBOX"))
    a.add_argument("--config-v6", default="configs/sam2_v6.yaml")
    args = a.parse_args(argv)
    conf = ev.ler_config(args.config_v6)
    limiar, ns = float(conf["limiar"]), sorted({int(x) for x in conf["n"]})
    f = Path(conf["entrada"]) / f"pred_{conf['base']}_{args.conjunto}.csv"
    if not f.is_file():
        raise ev.ErroAvaliacao(f"Não encontrei {f}: corre primeiro o methods/sam2_prop.py em {args.conjunto}.")
    pred = pd.read_csv(f, dtype={"caso": str})
    falta = [c for c in ["caso", "seed", "frame", "score"] + CAIXA if c not in pred.columns]
    if falta:
        raise ev.ErroAvaliacao(f"{f.name}: faltam as colunas {', '.join(falta)}.")
    saida = Path(conf["saida"])
    saida.mkdir(parents=True, exist_ok=True)
    git = ev._estado_git(RAIZ)
    estado = ("sem git" if git["commit"] is None else
              f"commit {git['commit']}" + (" (com alterações por guardar)" if git["alteracoes_por_guardar"] else ""))
    perto, longe = _confiantes(pred, limiar)
    L = [f"V6 (paragem adaptativa) em {args.conjunto} · {datetime.datetime.now():%Y-%m-%d %H:%M} · {estado}",
         f"sam2_v6.py v{VERSAO} · sha256 {ev._sha256(__file__)[:16]} · {Path(args.config_v6).name} sha256 "
         f"{ev._sha256(args.config_v6)[:16]} · base {f.name} (sha256 {ev._sha256(f)[:16]}) · limiar {limiar:g}",
         "",
         f"Base ({conf['base']}): {len(pred)} frames · com score ≥ {limiar:g}: {perto} a 1–12 frames do seed, "
         f"{longe} a mais de 12"]
    csv = {}
    for n in ns:
        novo, par = processar(pred, limiar, n)
        destino = saida / f"pred_SAM2_V6n{n}_{args.conjunto}.csv"
        novo.to_csv(destino, index=False)
        csv[n] = destino
        dist = par["distancia_paragem"].dropna().to_numpy(dtype=float)
        p_, l_ = _confiantes(novo, limiar)
        L += [f"  N = {n}: o seguimento parou em {len(dist)} de {len(par)} direções (distância da paragem: "
              + (f"mediana {np.median(dist):g}, P90 {np.percentile(dist, 90):g}" if len(dist) else "—")
              + f") · ficam com score ≥ {limiar:g}: {p_} a 1–12 frames, {l_} a mais de 12"]
    if args.conjunto == "CAL":
        L += ["", "Calibração, uma por variante (comparam-se com o SAM2_L12 em CAL):"]
        L += [f"  python eval/evaluate.py calibrar --metodo SAM2_V6n{n} --pred {p}" for n, p in csv.items()]
    texto = "\n".join(L)
    f_resumo = saida / f"pred_SAM2_V6_{args.conjunto}_resumo.txt"
    f_resumo.write_text(texto + "\n", encoding="utf-8")
    print(texto)
    return {"csv": csv, "resumo": f_resumo}


if __name__ == "__main__":
    try:
        main()
    except ev.ErroAvaliacao as e:
        print(f"ERRO: {e}", file=sys.stderr)
        sys.exit(2)
