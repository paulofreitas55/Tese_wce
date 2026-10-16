#!/usr/bin/env python3
"""
13 — A7: tabela mestre, estratos e figuras (Guião A). Só lê os resultados já gravados pelo avaliador: não avalia
nada, não lê previsões nem referências, e pode correr as vezes que for preciso (não regista execuções de teste).

Lê:
  configs/a6.yaml          a lista do A7 (métodos, nome de cada linha da tabela, conjuntos), o método principal e o
                           ponto de operação do critério 2 do go/no-go
  configs/a7.yaml          os métodos das figuras, os limites do go/no-go e a pasta da inspeção dos FP aparentes
  configs/avaliacao.yaml   a pasta dos resultados (outputs/eval)
  outputs/eval/CAL/<método>.json (curvas PR de CAL) e outputs/eval/VAL_EX|TEST_BOX/ (<método>.json, comparar_*.json)
  <fp_inspecao.saida>/classificacao.csv, se já tiver classificações (scripts/14_fp_inspecao.py)

Grava em outputs/13_a7/:
  a7_resumo.md          o que aparece no ecrã: tabela mestre (OP-F1 e OP-R90), comparações pareadas, critérios do
                        go/no-go, estratos do método principal e classificação dos FP aparentes (vírgula decimal)
  tabela_mestre.csv     uma linha por método × conjunto × ponto de operação × métrica (valor e IC 95 %)
  comparacoes.csv       as diferenças pareadas por janela já calculadas pelo comparar, com os nomes da tabela
  estratos.csv          todos os métodos: distância ao seed, classe, tamanho da caixa do seed e direção
  fontes.csv            os JSON lidos, com o sha256, o commit e a data de cada avaliação
  fp_classificacao.csv  as classes dos FP aparentes inspecionados (só quando a folha já tem classificações)
  fig_pr.png/.pdf       curvas PR da presença em CAL e em VAL-EX, com o OP-F1 marcado
  fig_distancia.png/.pdf  P e R em VAL-EX e R_loc@0,3 em TEST-BOX por distância ao seed, no OP-F1
As figuras precisam do matplotlib; sem ele, as tabelas são gravadas na mesma.

Uso, a partir da pasta do projeto:
    python scripts/13_a7_tabelas.py
"""
import argparse
import datetime
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "eval"))
import evaluate as ev  # noqa: E402

VERSAO = "1.0"
OPS = ("OP_F1", "OP_R90")
CLASSES_FP = ("lesao_nao_anotada", "outra_lesao", "fundo_mucosa", "nao_avaliavel")
NOME_CONJ = {"CAL": "CAL", "VAL_EX": "VAL-EX", "TEST_BOX": "TEST-BOX"}
ROTULOS = {"angiectasia": "Angiectasia", "erosao": "Erosão", "ulcera": "Úlcera", "polipo": "Pólipo",
           "eritema": "Eritema", "xantelasma": "Xantelasma", "outras": "Outras", "pequena": "Pequena",
           "media": "Média", "grande": "Grande", "sem_caixa": "Sem caixa", "antes": "Antes do seed",
           "depois": "Depois do seed"}
# figuras: paleta categórica validada para 4 séries em fundo branco (ordem fixa, nunca reciclada) e uma marca
# diferente por série, para se lerem também a preto e branco
CORES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
MARCAS = ("o", "s", "^", "D")
TINTA, TINTA_2, GRELHA, EIXO = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"


# ============================================================================ formato
def _vic(x):
    """Valor e IC 95 % de uma métrica do avaliador ({"valor", "ic95"}) ou de um número solto."""
    if not isinstance(x, dict):
        return {"valor": x, "ic95_inf": None, "ic95_sup": None}
    ic = x.get("ic95") or [None, None]
    return {"valor": x.get("valor"), "ic95_inf": ic[0], "ic95_sup": ic[1]}


def _ok(v):
    return v is not None and not (isinstance(v, float) and not math.isfinite(v))


def _pt(x, casas=3, ic=True, sinal=False):
    """0,664 [0,620–0,706], com vírgula decimal; — quando não há valor."""
    m = _vic(x)
    if not _ok(m["valor"]):
        return "—"
    f = f"{{:{'+' if sinal else ''}.{casas}f}}"
    s = f.format(m["valor"])
    if ic and _ok(m["ic95_inf"]) and _ok(m["ic95_sup"]):
        s += f" [{f.format(m['ic95_inf'])}{'; ' if sinal else '–'}{f.format(m['ic95_sup'])}]"
    return s.replace(".", ",")


def _int(x):
    return "—" if x is None else str(int(x))


def _tabela(cab, linhas):
    L = ["| " + " | ".join(cab) + " |", "|" + "|".join("---" for _ in cab) + "|"]
    return L + ["| " + " | ".join(str(c) for c in linha) + " |" for linha in linhas]


def wilson(k, n, z=1.959963984540054):
    """IC 95 % de Wilson para uma proporção k/n (frames sorteados de muitas janelas: aproximação razoável)."""
    if not n:
        return None, None
    p = k / n
    den = 1 + z * z / n
    centro = (p + z * z / (2 * n)) / den
    meia = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centro - meia), min(1.0, centro + meia)


# ============================================================================ leitura
def ler_resultado(cfg, conjunto, metodo):
    p = Path(cfg["caminhos"]["saida"]) / conjunto / f"{metodo}.json"
    if not p.is_file():
        acao = "calibrar" if conjunto == "CAL" else f"avaliar --conjunto {conjunto}"
        raise ev.ErroAvaliacao(f"Falta o resultado de {metodo} em {conjunto} ({p}): corre primeiro "
                               f"eval/evaluate.py {acao} --metodo {metodo}.")
    r = json.loads(p.read_text(encoding="utf-8"))
    if r.get("metodo") != metodo or r.get("conjunto") != conjunto or "por_op" not in r:
        raise ev.ErroAvaliacao(f"{p} não é o resultado de {metodo} em {conjunto}.")
    return p, r


def _fonte(p, r, tipo):
    reg = r.get("registo", {})
    return {"ficheiro": str(p), "sha256": ev._sha256(p), "tipo": tipo, "metodo": r.get("metodo", ""),
            "conjunto": r.get("conjunto", ""), "commit": reg.get("commit"), "data": reg.get("data")}


def ler_comparacoes(cfg, conjuntos, nomes):
    linhas, fontes = [], []
    for conj in conjuntos:
        for p in sorted((Path(cfg["caminhos"]["saida"]) / conj).glob("comparar_*.json")):
            if ".anterior_" in p.name:
                continue
            c = json.loads(p.read_text(encoding="utf-8"))
            ic = c.get("ic95") or [None, None]
            linhas.append({"conjunto": conj, "metrica": c["metrica"], "op": c["op"], "a": c["a"], "b": c["b"],
                           "linha_a": nomes.get(c["a"], c["a"]), "linha_b": nomes.get(c["b"], c["b"]),
                           "n_janelas": c["n_janelas"], "media_a": c["media_a"], "media_b": c["media_b"],
                           "diferenca": c["diferenca_media"], "ic95_inf": ic[0], "ic95_sup": ic[1],
                           "diferentes": bool(c["diferentes"]), "a_melhor": c["janelas_a_melhor"],
                           "empates": c["empates"], "b_melhor": c["janelas_b_melhor"]})
            fontes.append({"ficheiro": str(p), "sha256": ev._sha256(p), "tipo": "comparar", "metodo": f"{c['a']} vs {c['b']}",
                           "conjunto": conj, "commit": None, "data": None})
    cols = ["conjunto", "metrica", "op", "a", "b", "linha_a", "linha_b", "n_janelas", "media_a", "media_b",
            "diferenca", "ic95_inf", "ic95_sup", "diferentes", "a_melhor", "empates", "b_melhor"]
    return pd.DataFrame(linhas, columns=cols), fontes


def tabela_longa(res, linha):
    """Tabela longa de um resultado do avaliador (um método num conjunto)."""
    base = {"metodo": res["metodo"], "linha": linha, "conjunto": res["conjunto"]}
    out = []
    if isinstance(res.get("AP"), dict):
        out.append({**base, "op": "-", "tau": None, "metrica": "AP", **_vic(res["AP"]), "n_janelas": None})
    for op in OPS:
        r = res["por_op"][op]
        b = {**base, "op": op, "tau": r["tau"]}
        for k, v in r["micro"].items():
            out.append({**b, "metrica": f"{k}_micro" if k in ("P", "R", "F1") else k, **_vic(v), "n_janelas": None})
        for k, v in r["macro"].items():
            out.append({**b, "metrica": f"{k}_macro", **_vic(v), "n_janelas": v.get("n_janelas")})
        g = (r.get("ganho") or {}).get("todas")
        if g:
            for k in ("mediana", "media"):
                out.append({**b, "metrica": f"ganho_por_janela_{k}", **_vic(g[k]), "n_janelas": g["n_janelas"]})
    return out


def estratos_longos(res, linha):
    out = []
    for op in OPS:
        for tipo, blocos in res["por_op"][op].get("estratos", {}).items():
            for estrato, b in blocos.items():
                base = {"metodo": res["metodo"], "linha": linha, "conjunto": res["conjunto"], "op": op, "tipo": tipo,
                        "estrato": estrato, "n_janelas": b.get("n_janelas"), "n_frames": b.get("n_frames"),
                        "n_pos": b.get("n_pos"), "n_caixas": b.get("n_caixas")}
                out += [{**base, "metrica": k, **_vic(v)} for k, v in b.items() if isinstance(v, dict)]
    return out


def resumo_fp(caminho):
    """Classes dos FP aparentes já classificados na folha do 14_fp_inspecao.py (None se ainda não há nenhum)."""
    if not caminho.is_file():
        return None, f"ainda não há folha ({caminho}): corre o scripts/14_fp_inspecao.py"
    t = pd.read_csv(caminho, dtype=str, keep_default_na=False)
    if "classificacao" not in t.columns:
        raise ev.ErroAvaliacao(f"{caminho} não tem a coluna classificacao.")
    c = t["classificacao"].str.strip().str.lower()
    feitas = c[c != ""]
    if not len(feitas):
        return None, f"a folha ainda não tem classificações ({caminho})"
    erradas = sorted(set(feitas) - set(CLASSES_FP))
    if erradas:
        raise ev.ErroAvaliacao(f"{caminho}: valores desconhecidos na classificacao: {', '.join(erradas)} "
                               f"(usa só {', '.join(CLASSES_FP)}).")
    n, linhas = len(feitas), []
    for cl in CLASSES_FP:
        k = int((feitas == cl).sum())
        lo, hi = wilson(k, n)
        linhas.append({"classe": cl, "n": k, "fracao": k / n, "ic95_inf": lo, "ic95_sup": hi})
    return pd.DataFrame(linhas), f"{n} de {len(t)} FP classificados"


# ============================================================================ secções do resumo
def secao_mestre(res, metodos, nomes, op):
    """Tabelas de VAL-EX (presença) e TEST-BOX (caixas) num ponto de operação."""
    L, nota = [], False
    for conj in ("VAL_EX", "TEST_BOX"):
        linhas = []
        for m in metodos:
            r = res.get((m, conj))
            if r is None:
                continue
            o = r["por_op"][op]
            mi, ma = o["micro"], o["macro"]
            fora = op == "OP_R90" and "não é atingível" in r.get("pontos_operacao", {}).get("OP_R90", {}).get("regra", "")
            nota |= fora
            tau = (f"{o['tau']:.2f}".replace(".", ",") + ("*" if fora else "")
                   + (f" (A_min {o['a_min']:g} px)" if o.get("a_min") else ""))
            g = (o.get("ganho") or {}).get("todas") or {}
            ganho = "—" if g.get("mediana") is None else f"{g['mediana']:g}".replace(".", ",")
            nome = f"{nomes[m]} ({m})"
            if conj == "VAL_EX":
                linhas.append([nome, tau] + ([_pt(r.get("AP"))] if op == "OP_F1" else [])
                              + [_pt(mi.get("P")), _pt(mi.get("R")), _pt(mi.get("F1")), _pt(ma.get("F1")), ganho])
            else:
                linhas.append([nome, tau, _pt(mi.get("R_box")), _pt(mi.get("R_loc")), _pt(mi.get("mIoU")),
                               _pt(mi.get("pct_IoU>=0.5")), _pt(mi.get("taxa_fp_aparentes")), _pt(ma.get("R_loc")),
                               ganho])
        if not linhas:
            continue
        if conj == "VAL_EX":
            cab = (["Método", "τ"] + (["AP"] if op == "OP_F1" else []) +
                   ["P", "R", "F1", "F1 por janela", "Frames previstos por janela (mediana)"])
            titulo = f"### VAL-EX: a lesão está no frame? (referência PF v2) · {op.replace('_', '-')}"
        else:
            cab = ["Método", "τ", "R_box", "R_loc@0,3", "mIoU", "IoU ≥ 0,5", "FP aparentes", "R_loc por janela",
                   "Frames previstos por janela (mediana)"]
            titulo = f"### TEST-BOX: caixas originais · {op.replace('_', '-')}"
        L += [titulo, ""] + _tabela(cab, linhas) + [""]
    if nota:
        L += ["\\* recall ≥ 0,90 não atingível em CAL: o OP-R90 ficou no τ do recall máximo (τ = 0 é \"lesão em todos "
              "os frames\").", ""]
    return L


def secao_comparacoes(comps):
    if comps.empty:
        return ["Ainda não há comparações (eval/evaluate.py comparar).", ""]
    linhas = []
    for c in comps.itertuples(index=False):
        dif = _pt({"valor": c.diferenca, "ic95": [c.ic95_inf, c.ic95_sup]}, sinal=True)
        linhas.append([NOME_CONJ[c.conjunto], f"{c.metrica} ({c.op.replace('_', '-')})", c.linha_a, c.linha_b,
                       _pt(c.media_a), _pt(c.media_b), dif, "sim" if c.diferentes else "não",
                       f"{c.a_melhor} / {c.empates} / {c.b_melhor}"])
    return _tabela(["Conjunto", "Métrica por janela", "A", "B", "Média A", "Média B", "A − B [IC 95 %]",
                    "Diferentes?", "Janelas: A melhor / empates / B melhor"], linhas) + [""]


def criterios(a6, a7, res, comps, nomes):
    """Os três critérios do go/no-go, a partir dos resultados gravados (a decisão está no docs/decisoes.md)."""
    pr, g = a6["principal"], a7["go_no_go"]
    b1, op2 = g["b1"], a6["criterio2_op"]
    out = []
    c = comps[(comps["conjunto"] == "VAL_EX") & (comps["a"] == pr) & (comps["b"] == b1) & (comps["metrica"] == "F1")
              & (comps["op"] == "OP_F1")]
    if c.empty:
        out.append((f"{nomes.get(pr, pr)} melhor do que {nomes.get(b1, b1)} em VAL-EX", "falta o comparar", None))
    else:
        c = c.iloc[0]
        out.append((f"{nomes.get(pr, pr)} melhor do que {nomes.get(b1, b1)} em VAL-EX (F1 por janela; o IC da diferença "
                    "pareada exclui 0)", _pt({"valor": c["diferenca"], "ic95": [c["ic95_inf"], c["ic95_sup"]]}, sinal=True),
                    bool(c["diferentes"]) and c["diferenca"] > 0))
    t, v = res.get((pr, "TEST_BOX")), res.get((pr, "VAL_EX"))
    rl = t["por_op"][op2]["micro"].get("R_loc") if t else None
    lim2 = f"{g['r_loc_minimo']:.2f}".replace(".", ",")
    out.append((f"R_loc@0,3 micro em TEST-BOX no {op2.replace('_', '-')} ≥ {lim2}", _pt(rl),
                None if rl is None or rl.get("valor") is None else rl["valor"] >= g["r_loc_minimo"]))
    p = v["por_op"]["OP_F1"]["micro"].get("P") if v else None
    lim3 = f"{g['precisao_minima']:.2f}".replace(".", ",")
    out.append((f"Precisão micro em VAL-EX no OP-F1 ≥ {lim3}", _pt(p),
                None if p is None or p.get("valor") is None else p["valor"] >= g["precisao_minima"]))
    return out


def secao_estratos(res, principal, nome):
    v, t = res.get((principal, "VAL_EX")), res.get((principal, "TEST_BOX"))
    ev_ = v["por_op"]["OP_F1"].get("estratos", {}) if v else {}
    et = t["por_op"]["OP_F1"].get("estratos", {}) if t else {}
    L = []
    for tipo, titulo in (("distancia", "Distância ao seed (frames)"), ("classe", "Classe"),
                         ("tamanho", "Tamanho da caixa do seed"), ("direcao", "Direção")):
        chaves = list(dict.fromkeys(list(ev_.get(tipo, {})) + list(et.get(tipo, {}))))
        if not chaves:
            continue
        linhas = []
        for k in chaves:
            a, b = ev_.get(tipo, {}).get(k, {}), et.get(tipo, {}).get(k, {})
            rot = k.replace("-", "–") if tipo == "distancia" else ROTULOS.get(k, k)
            linhas.append([rot, _int(a.get("n_janelas")), _int(a.get("n_pos")), _pt(a.get("P")), _pt(a.get("R")),
                           _pt(a.get("F1")), _int(b.get("n_janelas")), _int(b.get("n_caixas")), _pt(b.get("R_box")),
                           _pt(b.get("R_loc"))])
        L += [f"#### {nome} por {titulo.lower()} (OP-F1)", ""]
        L += _tabela([titulo, "Janelas VAL-EX", "Positivos", "P", "R", "F1", "Janelas TEST-BOX", "Caixas", "R_box",
                      "R_loc@0,3"], linhas) + [""]
    return L


# ============================================================================ figuras
def _estilo(ax, fmt):
    ax.set_facecolor("white")
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(EIXO)
        ax.spines[lado].set_linewidth(0.8)
    ax.tick_params(colors=TINTA_2, labelsize=8, length=3, width=0.6)
    ax.grid(axis="y", color=GRELHA, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(fmt)


def figuras(res, metodos, nomes, saida):
    """fig_pr (curvas PR em CAL e VAL-EX) e fig_distancia (P, R e R_loc por distância ao seed). Devolve os ficheiros
    gravados e uma nota (vazia se correu bem)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.ticker import FuncFormatter
    except ImportError:
        return [], "falta o matplotlib neste ambiente (pip install matplotlib): as tabelas foram gravadas, as figuras não"
    if len(metodos) > len(CORES):
        raise ev.ErroAvaliacao(f"As figuras levam no máximo {len(CORES)} métodos (configs/a7.yaml, figuras.metodos).")
    plt.rcParams.update({"font.size": 9, "axes.labelcolor": TINTA, "axes.titlecolor": TINTA, "text.color": TINTA,
                         "font.family": "sans-serif", "pdf.fonttype": 42})
    fmt = FuncFormatter(lambda x, _: f"{x:.1f}".replace(".", ","))
    gravados = []

    def legenda(fig, nota=None):
        h = [Line2D([], [], color=CORES[i], marker=MARCAS[i], lw=1.6, ms=6, mec="white", mew=0.8, label=nomes[m])
             for i, m in enumerate(metodos)]
        if nota:
            h.append(Line2D([], [], color="none", marker="o", ms=6, mfc=TINTA_2, mec="white", label=nota))
        fig.legend(handles=h, loc="lower center", ncol=len(h), frameon=False, fontsize=8.5, handlelength=2.4)

    def gravar(fig, nome):
        for ext in ("png", "pdf"):
            f = saida / f"{nome}.{ext}"
            fig.savefig(f, dpi=200, facecolor="white")
            gravados.append(f)
        plt.close(fig)

    # curvas PR da presença (CAL e VAL-EX), com o OP-F1 de cada método
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.5), sharex=True, sharey=True)
    for ax, conj in zip(axs, ("CAL", "VAL_EX")):
        _estilo(ax, fmt)
        ax.xaxis.set_major_formatter(fmt)
        ax.grid(axis="x", color=GRELHA, linewidth=0.6)
        for i, m in enumerate(metodos):
            r = res.get((m, conj))
            if not r or "curva_pr" not in r:
                continue
            c = r["curva_pr"]
            tau = np.array(c["tau"], dtype=float)
            R = np.array([np.nan if x is None else x for x in c["R"]], dtype=float)
            P = np.array([np.nan if x is None else x for x in c["P"]], dtype=float)
            ok = np.isfinite(R) & np.isfinite(P)
            ax.plot(R[ok], P[ok], color=CORES[i], lw=1.6, zorder=2)
            k = int(np.argmin(np.abs(tau - float(r["por_op"]["OP_F1"]["tau"]))))
            if ok[k]:
                ax.plot([R[k]], [P[k]], marker=MARCAS[i], ms=7, color=CORES[i], mec="white", mew=1.0, zorder=3)
            ap = (r.get("AP") or {}).get("valor")
            if ap is not None:                                            # AP de cada método, no canto do painel
                y = 0.955 - 0.075 * i
                ax.plot([0.74], [y], transform=ax.transAxes, marker=MARCAS[i], ms=5, color=CORES[i], mec="white",
                        mew=0.6, clip_on=False, zorder=4)
                ax.annotate(f"AP {ap:.2f}".replace(".", ","), xy=(0.77, y), xycoords="axes fraction", ha="left",
                            va="center", fontsize=7.5, color=TINTA_2)
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1.02)
        ax.set_title(f"{NOME_CONJ[conj]}", fontsize=9.5, loc="left")
        ax.set_xlabel("Recall")
    axs[0].set_ylabel("Precisão")
    legenda(fig, "marca: OP-F1")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    gravar(fig, "fig_pr")

    # P e R (VAL-EX) e R_loc (TEST-BOX) por distância ao seed, no OP-F1, com IC 95 %
    paineis = [("VAL_EX", "P", "Precisão (VAL-EX)"), ("VAL_EX", "R", "Recall (VAL-EX)"),
               ("TEST_BOX", "R_loc", "R_loc@0,3 (TEST-BOX)")]
    cats = []
    for m in metodos:
        for conj in ("VAL_EX", "TEST_BOX"):
            r = res.get((m, conj))
            if r:
                cats += [k for k in r["por_op"]["OP_F1"].get("estratos", {}).get("distancia", {}) if k not in cats]
    if cats:
        fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.9), sharey=True)
        x = np.arange(len(cats))
        for ax, (conj, met, titulo) in zip(axs, paineis):
            _estilo(ax, fmt)
            for i, m in enumerate(metodos):
                r = res.get((m, conj))
                if not r:
                    continue
                bl = r["por_op"]["OP_F1"].get("estratos", {}).get("distancia", {})
                v, lo, hi = (np.array([_vic(bl.get(k, {}).get(met))[c] for k in cats], dtype=float)
                             for c in ("valor", "ic95_inf", "ic95_sup"))
                dx = (i - (len(metodos) - 1) / 2) * 0.1
                ok = np.isfinite(v)
                err = np.vstack([np.where(np.isfinite(lo), v - lo, 0), np.where(np.isfinite(hi), hi - v, 0)])
                ax.errorbar(x[ok] + dx, v[ok], yerr=err[:, ok], color=CORES[i], marker=MARCAS[i], ms=5, lw=1.4,
                            elinewidth=0.8, capsize=1.8, mec="white", mew=0.6, zorder=3)
            ax.set_xticks(x)
            ax.set_xticklabels([k.replace("-", "–") for k in cats])
            ax.set_ylim(0, 1.02)
            ax.set_title(titulo, fontsize=9, loc="left")
            ax.set_xlabel("Distância ao seed (frames)", fontsize=8.5)
        legenda(fig)
        fig.tight_layout(rect=(0, 0.1, 1, 1))
        gravar(fig, "fig_distancia")
    return gravados, ""


# ============================================================================ principal
def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--config", default="configs/avaliacao.yaml")
    a.add_argument("--config-a6", default="configs/a6.yaml")
    a.add_argument("--config-a7", default="configs/a7.yaml")
    a.add_argument("--saida", default="outputs/13_a7")
    a.add_argument("--sem-figuras", action="store_true")
    args = a.parse_args(argv)
    cfg, a6, a7 = ev.ler_config(args.config), ev.ler_config(args.config_a6), ev.ler_config(args.config_a7)
    metodos = [m["metodo"] for m in a6["metodos"]]
    nomes = {m["metodo"]: m["linha"] for m in a6["metodos"]}
    principal = a6["principal"]
    figm = list(a7["figuras"]["metodos"])
    fora = [m for m in figm + [a7["go_no_go"]["b1"]] if m not in nomes]
    if fora:
        raise ev.ErroAvaliacao(f"Métodos do configs/a7.yaml que não estão na lista do A7 (configs/a6.yaml): {fora}.")

    res, fontes, longa, estr = {}, [], [], []
    for m in a6["metodos"]:
        for conj in m.get("conjuntos", a6["conjuntos"]):
            p, r = ler_resultado(cfg, conj, m["metodo"])
            res[(m["metodo"], conj)] = r
            fontes.append(_fonte(p, r, "avaliar"))
            longa += tabela_longa(r, m["linha"])
            estr += estratos_longos(r, m["linha"])
    for m in figm:                                                        # curvas PR de CAL (só as figuras)
        p, r = ler_resultado(cfg, "CAL", m)
        res[(m, "CAL")] = r
        fontes.append(_fonte(p, r, "calibrar"))
    comps, fc = ler_comparacoes(cfg, ("VAL_EX", "TEST_BOX"), nomes)
    fontes += fc
    fp_dir = Path(a7["fp_inspecao"]["saida"])
    fp, fp_estado = resumo_fp(fp_dir / "classificacao.csv")

    saida = Path(args.saida)
    saida.mkdir(parents=True, exist_ok=True)
    gravados, nota_fig = ([], "não pedidas (--sem-figuras)") if args.sem_figuras else figuras(res, figm, nomes, saida)

    git = ev._estado_git(RAIZ)
    estado = ("sem git" if git["commit"] is None else
              f"commit {git['commit']}" + (" (com alterações por guardar)" if git["alteracoes_por_guardar"] else ""))
    agora = datetime.datetime.now()
    L = ["# A7 — tabela mestre, estratos e figuras", "",
         f"13_a7_tabelas.py v{VERSAO} · sha256 {ev._sha256(__file__)[:16]} · {agora:%Y-%m-%d %H:%M} · {estado} · "
         f"{len(fontes)} ficheiros de resultados lidos (lista e sha256 em fontes.csv)", "",
         "IC 95 % por bootstrap sobre janelas (do avaliador). P, R, F1, R_box, R_loc e mIoU são micro (frames juntos); "
         "\"por janela\" é a média das janelas.", "",
         "## Tabela mestre", ""]
    L += secao_mestre(res, metodos, nomes, "OP_F1") + secao_mestre(res, metodos, nomes, "OP_R90")
    L += ["## Comparações pareadas por janela (comparar)", ""] + secao_comparacoes(comps)
    L += ["## Critérios do go/no-go (decisão registada no docs/decisoes.md)", ""]
    cr = criterios(a6, a7, res, comps, nomes)
    for i, (txt, val, ok) in enumerate(cr, 1):
        L.append(f"{i}. {txt}: {val} → " + ("cumpre" if ok else "não cumpre" if ok is not None else "por avaliar"))
    falhas = [str(i) for i, (_, _, ok) in enumerate(cr, 1) if ok is False]
    L += ["", "Resultado pelos critérios: " + ("go" if all(ok for _, _, ok in cr) else
                                               f"no-go (falha o critério {', '.join(falhas)})" if falhas else
                                               "incompleto"), ""]
    L += ["## Estratos do método principal", ""] + secao_estratos(res, principal, nomes[principal])
    L += [f"## FP aparentes inspecionados ({nomes.get(a7['fp_inspecao']['metodo'], a7['fp_inspecao']['metodo'])}, "
          "TEST-BOX, OP-F1)", ""]
    if fp is None:
        L += [f"Ainda não: {fp_estado}.", ""]
    else:
        L += [f"{fp_estado} (IC 95 % de Wilson):", ""]
        L += _tabela(["Classe", "FP", "Fração [IC 95 %]"],
                     [[r.classe, r.n, _pt({"valor": r.fracao, "ic95": [r.ic95_inf, r.ic95_sup]}, casas=2)]
                      for r in fp.itertuples(index=False)]) + [""]
    L += ["## Ficheiros", "",
          f"Em {saida}/: a7_resumo.md, tabela_mestre.csv, comparacoes.csv, estratos.csv, fontes.csv"
          + (", fp_classificacao.csv" if fp is not None else "") + ".",
          "Figuras: " + (", ".join(sorted({f.stem for f in gravados})) + " (png e pdf)" if gravados else nota_fig) + "."]
    texto = "\n".join(L)

    mestre = pd.DataFrame(longa)
    estratos = pd.DataFrame(estr)
    mestre.to_csv(saida / "tabela_mestre.csv", index=False)
    comps.to_csv(saida / "comparacoes.csv", index=False)
    estratos.to_csv(saida / "estratos.csv", index=False)
    pd.DataFrame(fontes).to_csv(saida / "fontes.csv", index=False)
    if fp is not None:
        fp.to_csv(saida / "fp_classificacao.csv", index=False)
    (saida / "a7_resumo.md").write_text(texto + "\n", encoding="utf-8")
    print(texto)
    return {"mestre": mestre, "comparacoes": comps, "estratos": estratos, "fp": fp, "figuras": gravados,
            "criterios": cr, "texto": texto}


if __name__ == "__main__":
    try:
        main()
    except ev.ErroAvaliacao as e:
        print(f"ERRO: {e}", file=sys.stderr)
        sys.exit(2)
