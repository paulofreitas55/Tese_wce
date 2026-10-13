#!/usr/bin/env python3
"""
10 — Cruzamento da anotação de referência (PF v2) com as caixas da anotação original, em CAL.

Guião A, passo 0.5: "cruzar os frames visíveis do Paulo com as boxes do médico na mesma janela —
primeira estimativa da concordância entre as duas referências". As caixas foram desenhadas pelo aluno
anterior; na escrita, "caixas da anotação original".

Só lê dados: não altera a referência congelada nem as caixas. Lê as referências com as mesmas funções
do avaliador (eval/evaluate.py) e os caminhos de configs/avaliacao.yaml. O seed (offset 0) fica de fora,
como nas métricas.

Duas perguntas, em palavras simples:
  - das caixas da anotação original, quantas a PF v2 marcou como visíveis   (caixas confirmadas)
  - dos frames que a PF v2 marcou como visíveis, quantos têm caixa           (PF v2 com caixa)
Um frame sem caixa NÃO é negativo (a anotação original não marcou todos os frames em que a lesão se vê),
por isso o κ, que trata "sem caixa" como negativo, é só indicativo.
IC 95 % por bootstrap sobre janelas (parâmetros de configs/avaliacao.yaml: 1 000 reamostragens, semente 0).

Uso, a partir da pasta do projeto:
    python scripts/10_cruzamento_pf_caixas.py
Gera em outputs/:
    10_cruzamento_pf_caixas_resumo.txt          resumo (também no ecrã)
    10_cruzamento_pf_caixas_janelas.csv         contagens por janela
    10_cruzamento_pf_caixas_discordancias.csv   frames em que as duas referências discordam, para ver à mão
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
LINHAS_PF = {"1": "PF v2 = 1 (visível)", "0": "PF v2 = 0 (não visível)", "D": "PF v2 = D (dúvida)"}
COLUNAS_CX = {"caixa": "com caixa", "vazio": "sem caixa", "sem_ficheiro": "sem ficheiro", "invalida": "caixa inválida"}


def por_janela(x, W):
    """a: PF 1 com caixa · b: PF 1 sem caixa · c: PF 0 com caixa · d: PF 0 sem caixa · dc: dúvida com caixa."""
    w = x["w"].to_numpy()
    pf, cx = x["pf"].to_numpy(), x["caixa"].to_numpy()

    def n(m):
        return np.bincount(w, weights=m.astype(float), minlength=W)
    return {"a": n((pf == "1") & (cx == "caixa")), "b": n((pf == "1") & (cx == "vazio")),
            "c": n((pf == "0") & (cx == "caixa")), "d": n((pf == "0") & (cx == "vazio")),
            "dc": n((pf == "D") & (cx == "caixa"))}


def medidas(a, b, c, d):
    """Funciona com totais ou com vetores (uma posição por reamostragem do bootstrap)."""
    a, b, c, d = (np.asarray(v, dtype=float) for v in (a, b, c, d))
    n = a + b + c + d
    with np.errstate(divide="ignore", invalid="ignore"):
        po = np.where(n > 0, (a + d) / n, np.nan)
        pe = np.where(n > 0, ((a + b) * (a + c) + (c + d) * (b + d)) / n ** 2, np.nan)
        return {"caixas_confirmadas": np.where(a + c > 0, a / (a + c), np.nan),
                "pf_com_caixa": np.where(a + b > 0, a / (a + b), np.nan),
                "concordancia_positiva": np.where(2 * a + b + c > 0, 2 * a / (2 * a + b + c), np.nan),
                "concordancia": po,
                "kappa": np.where(pe < 1, (po - pe) / (1 - pe), np.nan)}


def _pct(v):
    return "—" if v is None or not np.isfinite(v) else f"{100 * v:.1f} %".replace(".", ",")


def _num(v, casas=2):
    return "—" if v is None or not np.isfinite(v) else f"{v:.{casas}f}".replace(".", ",")


def _ic(ic, pct=True):
    if not ic or None in ic:
        return ""
    f = (lambda v: f"{100 * v:.1f}".replace(".", ",")) if pct else (lambda v: _num(v))
    return f" [IC 95 % {f(ic[0])}–{f(ic[1])}{' %' if pct else ''}]"


def cruzar(cfg):
    janelas = ev.ler_conjunto(Path(cfg["caminhos"]["conjuntos"]) / f"{CONJUNTO}.json")
    ref, info = ev.carregar_referencia(cfg, CONJUNTO, janelas)
    W = len(info)
    x = ref[ref["offset"] != 0].copy()
    v = por_janela(x, W)
    tot = {k: int(v[k].sum()) for k in v}
    val = {k: float(m) for k, m in medidas(*(tot[k] for k in "abcd")).items()}
    bs = cfg["bootstrap"]
    boot = ev.Bootstrap(W, bs["reamostragens"], bs["semente"], bs["nivel"])
    ic = {}
    if boot.M is not None:
        reps = medidas(*(boot.M @ v[k] for k in "abcd"))
        ic = {k: boot.ic(r) for k, r in reps.items()}

    tabela = (pd.crosstab(x["pf"], x["caixa"]).reindex(index=list(LINHAS_PF), columns=list(COLUNAS_CX), fill_value=0)
              .fillna(0).astype(int).rename(index=LINHAS_PF, columns=COLUNAS_CX))
    tabela.index.name = tabela.columns.name = None

    dist = {}
    off = x["offset"].abs().to_numpy()
    for lo, hi in cfg["estratos"]["distancia"]:
        rot = f">{lo - 1}" if hi >= 100 else f"{lo}-{hi}"
        vd = por_janela(x[(off >= lo) & (off <= hi)], W)
        td = {k: int(vd[k].sum()) for k in "abcd"}
        md = medidas(*(td[k] for k in "abcd"))
        dist[rot] = {**td, "caixas_confirmadas": float(md["caixas_confirmadas"]), "pf_com_caixa": float(md["pf_com_caixa"])}

    k = int(np.argmax(v["a"] + v["b"]))
    resto = np.arange(W) != k
    sem = {kk: float(m) for kk, m in medidas(*(v[c][resto].sum() for c in "abcd")).items()}
    dominante = {"janela": f"{info['split'].iloc[k].replace('Dataset_', '')}/{info['caso'].iloc[k]}/{info['seed'].iloc[k]}",
                 "positivos_pf": int(v["a"][k] + v["b"][k]), "sem_ela": sem}

    jan = info[["split", "caso", "seed", "classe", "categoria"]].copy()
    for c, nome in (("a", "ambos"), ("b", "so_pf"), ("c", "so_caixa"), ("d", "nenhum"), ("dc", "caixa_com_duvida")):
        jan[nome] = v[c].astype(int)
    jan["caixas"] = jan["ambos"] + jan["so_caixa"] + jan["caixa_com_duvida"]
    jan["pf_visiveis"] = jan["ambos"] + jan["so_pf"]
    mj = medidas(v["a"], v["b"], v["c"], v["d"])
    jan["caixas_confirmadas"] = np.round(mj["caixas_confirmadas"], 3)
    jan["pf_com_caixa"] = np.round(mj["pf_com_caixa"], 3)
    jan = jan.sort_values(["so_caixa", "so_pf"], ascending=False).reset_index(drop=True)

    dis = x[((x["pf"] == "0") & (x["caixa"] == "caixa")) | ((x["pf"] == "1") & (x["caixa"] == "vazio"))].copy()
    dis["tipo"] = np.where(dis["pf"] == "0", "caixa_sem_pf", "pf_sem_caixa")
    dis = dis.merge(info[["w", "classe"]], on="w")
    dis = dis[["tipo", "split", "caso", "seed", "frame", "offset", "classe"]].sort_values(
        ["tipo", "split", "caso", "seed", "frame"]).reset_index(drop=True)

    contagens = {"janelas": W, "frames_sem_seed": int(len(x)), "duvidas": int((x["pf"] == "D").sum()),
                 "sem_ficheiro": int((x["caixa"] == "sem_ficheiro").sum()), "invalidas": int((x["caixa"] == "invalida").sum())}
    return {"totais": tot, "medidas": val, "ic95": ic, "tabela": tabela, "distancia": dist, "dominante": dominante,
            "janelas": jan, "discordancias": dis, "contagens": contagens}


def resumo(r, cfg, fontes):
    t, m, ic, c = r["totais"], r["medidas"], r["ic95"], r["contagens"]
    git = ev._estado_git(RAIZ)
    estado = ("sem git" if git["commit"] is None else
              f"commit {git['commit']}" + (" (com alterações por guardar)" if git["alteracoes_por_guardar"] else ""))
    L = [f"10 — Cruzamento PF v2 × caixas da anotação original, em {CONJUNTO}",
         f"{datetime.datetime.now():%Y-%m-%d %H:%M} · {estado}",
         *[f"sha256 {Path(f).name}: {ev._sha256(f)}" for f in fontes],
         "",
         f"{c['janelas']} janelas · {c['frames_sem_seed']} frames sem o seed",
         "",
         "Frames por estado (sem o seed):",
         r["tabela"].to_string(),
         "",
         f"Caixas confirmadas: das {t['a'] + t['c']} caixas em frames com PF v2 0 ou 1, a PF v2 marcou {t['a']} como visíveis"
         f" = {_pct(m['caixas_confirmadas'])}{_ic(ic.get('caixas_confirmadas'))}",
         f"PF v2 com caixa: dos {t['a'] + t['b']} frames que a PF v2 marcou como visíveis (com ficheiro de caixa), "
         f"{t['a']} têm caixa = {_pct(m['pf_com_caixa'])}{_ic(ic.get('pf_com_caixa'))}",
         f"Caixas em frames marcados como dúvida: {t['dc']}",
         f"Concordância positiva: {_num(m['concordancia_positiva'])}{_ic(ic.get('concordancia_positiva'), False)}"
         f" · concordância observada {_pct(m['concordancia'])}"
         f" · κ {_num(m['kappa'])}{_ic(ic.get('kappa'), False)}",
         "  (o κ trata 'sem caixa' como negativo, o que não é verdade na anotação original: só indicativo)",
         "",
         "Por distância ao seed:"]
    for rot, d in r["distancia"].items():
        L.append(f"  {rot:>5}: caixas {d['a'] + d['c']:>4} · confirmadas {_pct(d['caixas_confirmadas']):>7}"
                 f" · PF v2 visíveis {d['a'] + d['b']:>4} · com caixa {_pct(d['pf_com_caixa']):>7}")
    dm = r["dominante"]
    L += ["",
          f"Sem a janela com mais positivos ({dm['janela']}, {dm['positivos_pf']} frames visíveis na PF v2): "
          f"caixas confirmadas {_pct(dm['sem_ela']['caixas_confirmadas'])} · PF v2 com caixa {_pct(dm['sem_ela']['pf_com_caixa'])}"
          f" · κ {_num(dm['sem_ela']['kappa'])}",
          "",
          "Discordâncias, para ver à mão no CSV: "
          f"caixa_sem_pf = {int((r['discordancias']['tipo'] == 'caixa_sem_pf').sum())} (frame com caixa, PF v2 = 0) · "
          f"pf_sem_caixa = {int((r['discordancias']['tipo'] == 'pf_sem_caixa').sum())} (frame visível na PF v2, sem caixa)",
          f"Ficam de fora: {c['duvidas']} frames com dúvida (D) · {c['sem_ficheiro']} sem ficheiro de caixa · "
          f"{c['invalidas']} com caixa inválida",
          "",
          "3 linhas de exemplo (janelas com mais caixas não confirmadas):",
          r["janelas"].head(3).to_string(index=False),
          "",
          "Por verificar: ver à mão alguns frames de caixa_sem_pf (a caixa marca a mesma lesão? a PF v2 falhou-a?)."]
    return "\n".join(L)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/avaliacao.yaml")
    p.add_argument("--saida", default="outputs")
    args = p.parse_args(argv)
    cfg = ev.ler_config(args.config)
    r = cruzar(cfg)
    fontes = [cfg["caminhos"]["referencia_pf"], cfg["caminhos"]["anotacoes"],
              str(Path(cfg["caminhos"]["conjuntos"]) / f"{CONJUNTO}.json")]
    texto = resumo(r, cfg, fontes)
    saida = Path(args.saida)
    saida.mkdir(parents=True, exist_ok=True)
    (saida / "10_cruzamento_pf_caixas_resumo.txt").write_text(texto + "\n", encoding="utf-8")
    r["janelas"].to_csv(saida / "10_cruzamento_pf_caixas_janelas.csv", index=False)
    r["discordancias"].to_csv(saida / "10_cruzamento_pf_caixas_discordancias.csv", index=False)
    print(texto)
    print(f"\ngravado em {saida}/10_cruzamento_pf_caixas_*")
    return r


if __name__ == "__main__":
    try:
        main()
    except ev.ErroAvaliacao as e:
        print(f"ERRO: {e}", file=sys.stderr)
        sys.exit(2)
