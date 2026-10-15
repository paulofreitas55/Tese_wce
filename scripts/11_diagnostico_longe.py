#!/usr/bin/env python3
"""
11 — Diagnóstico dos frames longe do seed, em CAL (Guião A, A5; apoio à decisão sobre o V6, 15 out).

Pergunta: depois de perder a lesão, quando o SAM 2 volta a ficar confiante mais à frente, acerta (a lesão
reapareceu) ou erra (agarrou-se a outra coisa)? E a duração do bloco confiante separa uma coisa da outra?

Usa as previsões do SAM 2 com a caixa, seguidas até 100 frames (pred_SAM2_L100_CAL.csv), e a referência PF v2, lida
com as mesmas funções do avaliador. O seed fica de fora. Por janela e por direção (depois e antes do seed):
  - troço do seed: os frames seguidos, a partir do seed, com score ≥ 0,5 (onde o V6 com N = 1 pararia);
  - blocos confiantes: depois do troço, frames seguidos com score ≥ 0,5 (o próprio SAM 2 diz que vê o objeto).
Para cada bloco: distância ao seed, comprimento, score máximo e quantos frames são visíveis (PF 1), não visíveis
(PF 0) ou em dúvida (D), também só para os frames com score ≥ 0,9. O resumo agrupa os blocos por comprimento e por
janela, e diz onde estão os frames visíveis: no troço, em blocos ou sem nada.
Só lê dados e só corre em CAL (usa os rótulos).

Uso, a partir da pasta do projeto:
    python scripts/11_diagnostico_longe.py
Gera em outputs/: 11_diagnostico_longe_resumo.txt (também no ecrã) e 11_diagnostico_longe_blocos.csv.
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

CONJUNTO = "CAL"
LIMIAR = 0.5                                                                # o SAM 2 vê o objeto
FORTE = 0.9
GRUPOS = ((1, 1, "1"), (2, 3, "2–3"), (4, 6, "4–6"), (7, 12, "7–12"), (13, 10 ** 9, "≥ 13"))


def troco_e_blocos(scores, limiar=LIMIAR):
    """scores por distância ao seed (1, 2, …), NaN onde não há frame. Devolve (t, blocos): t = nº de frames do troço
    do seed; blocos = [(i, j)] em índices (0 = distância 1) dos frames seguidos com score ≥ limiar depois do troço."""
    n, t = len(scores), 0
    while t < n and scores[t] >= limiar:
        t += 1
    blocos, i = [], t
    while i < n:
        if scores[i] >= limiar:
            j = i
            while j + 1 < n and scores[j + 1] >= limiar:
                j += 1
            blocos.append((i, j))
            i = j + 1
        else:
            i += 1
    return t, blocos


def analisar(cfg, pred):
    """Devolve (blocos, frames): uma linha por bloco confiante e uma linha por frame (sem o seed) com onde ficou."""
    janelas = ev.ler_conjunto(Path(cfg["caminhos"]["conjuntos"]) / f"{CONJUNTO}.json")
    ref, _ = ev.carregar_referencia(cfg, CONJUNTO, janelas)
    m = ref.merge(pred[ev.K3 + ["frame", "score"]], on=ev.K3 + ["frame"], how="left")
    linhas_b, linhas_f = [], []
    for (sp, caso, seed), g in m.groupby(ev.K3, sort=True):
        sc = dict(zip(g["frame"].astype(int), g["score"].astype(float)))
        pf = dict(zip(g["frame"].astype(int), g["pf"]))
        for direcao, sinal, dmax in (("depois", 1, int(g["frame"].max()) - seed),
                                     ("antes", -1, seed - int(g["frame"].min()))):
            if dmax <= 0:
                continue
            fr = [seed + sinal * d for d in range(1, dmax + 1)]
            s = [sc.get(f, np.nan) for f in fr]
            p = [pf.get(f, "") for f in fr]
            t, blocos = troco_e_blocos(s)
            onde = ["troço"] * t + ["nada"] * (len(s) - t)
            for a, b in blocos:
                onde[a:b + 1] = ["bloco"] * (b - a + 1)
                forte = [k for k in range(a, b + 1) if s[k] >= FORTE]
                linhas_b.append({"split": sp, "caso": caso, "seed": seed, "direcao": direcao, "inicio": a + 1,
                                 "fim": b + 1, "comprimento": b - a + 1, "score_max": round(max(s[a:b + 1]), 6),
                                 "pf1": p[a:b + 1].count("1"), "pf0": p[a:b + 1].count("0"),
                                 "duvida": p[a:b + 1].count("D"), "frames_forte": len(forte),
                                 "pf1_forte": sum(p[k] == "1" for k in forte),
                                 "pf0_forte": sum(p[k] == "0" for k in forte)})
            linhas_f += [{"split": sp, "caso": caso, "seed": seed, "direcao": direcao, "distancia": d + 1,
                          "score": s[d], "pf": p[d], "onde": onde[d]} for d in range(len(s))]
    colunas = ["split", "caso", "seed", "direcao", "inicio", "fim", "comprimento", "score_max", "pf1", "pf0", "duvida",
               "frames_forte", "pf1_forte", "pf0_forte"]
    return pd.DataFrame(linhas_b, columns=colunas), pd.DataFrame(linhas_f)


def _prec(a, b):
    return f"{a / (a + b):.2f}" if a + b else "—"


def resumo(blocos, frames, fontes):
    git = ev._estado_git(RAIZ)
    estado = ("sem git" if git["commit"] is None else
              f"commit {git['commit']}" + (" (com alterações por guardar)" if git["alteracoes_por_guardar"] else ""))
    tr = frames[frames["onde"] == "troço"]
    comp = tr.groupby(ev.K3 + ["direcao"]).size().reindex(
        frames[ev.K3 + ["direcao"]].drop_duplicates().set_index(ev.K3 + ["direcao"]).index, fill_value=0)
    L = [f"11 — Frames longe do seed em {CONJUNTO} (SAM 2 com a caixa, até 100 frames) · "
         f"{datetime.datetime.now():%Y-%m-%d %H:%M} · {estado}",
         "fontes: " + " · ".join(f"{Path(f).name} sha256 {ev._sha256(f)[:12]}" for f in fontes),
         "",
         f"Troço do seed (frames seguidos com score ≥ {LIMIAR:g}; é onde o V6 com N = 1 pararia): {len(tr)} frames · "
         f"PF 1 {int((tr['pf'] == '1').sum())} · PF 0 {int((tr['pf'] == '0').sum())} · D {int((tr['pf'] == 'D').sum())}"
         f" · precisão {_prec(int((tr['pf'] == '1').sum()), int((tr['pf'] == '0').sum()))}",
         f"  comprimento do troço por direção: mediana {np.median(comp):g} · P90 {np.percentile(comp, 90):g} · "
         f"máximo {comp.max()} · direções sem troço (o 1.º frame já está abaixo): {int((comp == 0).sum())} de {len(comp)}",
         "",
         f"Blocos confiantes depois do troço (frames seguidos com score ≥ {LIMIAR:g}): {len(blocos)} blocos · "
         f"{int(blocos['comprimento'].sum())} frames · PF 1 {int(blocos['pf1'].sum())} · PF 0 {int(blocos['pf0'].sum())}"
         f" · D {int(blocos['duvida'].sum())} · precisão {_prec(int(blocos['pf1'].sum()), int(blocos['pf0'].sum()))}",
         f"  {'comprimento':>11} {'blocos':>6} {'frames':>6} {'PF 1':>5} {'PF 0':>5} {'D':>4} {'precisão':>8}   "
         f"só score ≥ {FORTE:g}: {'PF 1':>5} {'PF 0':>5} {'precisão':>8}"]
    for lo, hi, nome in GRUPOS:
        b = blocos[blocos["comprimento"].between(lo, hi)]
        if len(b):
            L.append(f"  {nome:>11} {len(b):>6} {int(b['comprimento'].sum()):>6} {int(b['pf1'].sum()):>5} "
                     f"{int(b['pf0'].sum()):>5} {int(b['duvida'].sum()):>4} {_prec(b['pf1'].sum(), b['pf0'].sum()):>8}"
                     f"   {'':>17}{int(b['pf1_forte'].sum()):>5} {int(b['pf0_forte'].sum()):>5} "
                     f"{_prec(b['pf1_forte'].sum(), b['pf0_forte'].sum()):>8}")
    if len(blocos):
        por = (blocos.groupby(ev.K3).agg(blocos=("comprimento", "size"), frames=("comprimento", "sum"),
                                         pf1=("pf1", "sum"), pf0=("pf0", "sum"))
               .sort_values("frames", ascending=False))
        L += ["", f"Por janela ({len(por)} janelas com blocos; as que têm mais frames em blocos):"]
        L += [f"  {sp}/{caso}/{seed}: {r.blocos} blocos · {r.frames} frames · PF 1 {r.pf1} · PF 0 {r.pf0}"
              for (sp, caso, seed), r in por.head(8).iterrows()]
        L.append(f"  distância ao seed onde os blocos começam: mediana {blocos['inicio'].median():g} · "
                 f"P10 {blocos['inicio'].quantile(0.1):g} · P90 {blocos['inicio'].quantile(0.9):g}")
    vis = frames[frames["pf"] == "1"]
    n = max(1, len(vis))
    nada = vis[vis["onde"] == "nada"]
    L += ["",
          f"Frames visíveis (PF 1, sem o seed): {len(vis)} · no troço {int((vis['onde'] == 'troço').sum())} "
          f"({100 * (vis['onde'] == 'troço').mean() if len(vis) else 0:.0f} %) · em blocos "
          f"{int((vis['onde'] == 'bloco').sum())} ({100 * (vis['onde'] == 'bloco').sum() / n:.0f} %) · sem nada "
          f"{len(nada)} ({100 * len(nada) / n:.0f} %, dos quais {int((nada['distancia'] > 12).sum())} a mais de 12 "
          "frames do seed)"]
    if len(vis):
        top = vis.groupby(ev.K3).size().sort_values(ascending=False).head(3)
        L.append("  janelas com mais frames visíveis: " + " · ".join(
            f"{sp}/{caso}/{seed} {k} (a mais de 12: {int(((vis['caso'] == caso) & (vis['seed'] == seed) & (vis['distancia'] > 12)).sum())})"
            for (sp, caso, seed), k in top.items()))
    return "\n".join(L)


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--config", default="configs/avaliacao.yaml")
    a.add_argument("--pred", default="outputs/preds/pred_SAM2_L100_CAL.csv")
    a.add_argument("--saida", default="outputs")
    args = a.parse_args(argv)
    cfg = ev.ler_config(args.config)
    if not Path(args.pred).is_file():
        raise ev.ErroAvaliacao(f"Não encontrei {args.pred}.")
    pred = pd.read_csv(args.pred, dtype={"caso": str})
    blocos, frames = analisar(cfg, pred)
    texto = resumo(blocos, frames, [args.pred, cfg["caminhos"]["referencia_pf"]])
    saida = Path(args.saida)
    saida.mkdir(parents=True, exist_ok=True)
    (saida / "11_diagnostico_longe_resumo.txt").write_text(texto + "\n", encoding="utf-8")
    blocos.to_csv(saida / "11_diagnostico_longe_blocos.csv", index=False)
    print(texto)
    print(f"\ngravado em {saida}/11_diagnostico_longe_*")
    return {"blocos": blocos, "frames": frames, "texto": texto}


if __name__ == "__main__":
    try:
        main()
    except ev.ErroAvaliacao as e:
        print(f"ERRO: {e}", file=sys.stderr)
        sys.exit(2)
