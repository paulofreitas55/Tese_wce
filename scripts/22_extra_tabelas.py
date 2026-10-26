#!/usr/bin/env python3
"""
22 — Experiência A: tabela, comparações, estratos e figuras da comparação SAM 2.1 base+ / Tiny / MedSAM2. Só lê o
que o avaliador já gravou (outputs/eval/<conjunto>/<método>.json, as comparações e outputs/test_runs.csv): não corre
nenhum método nem avalia nada.

A partir de configs/extra.yaml (os métodos, a referência e as comparações declaradas a 26 out 2026):
  1. Confere que cada método (e a referência) foi avaliado em VAL_EX e TEST_BOX e quantas execuções tem em
     outputs/test_runs.csv (deve ser uma), e que as 6 comparações declaradas existem.
  2. Tabela no formato da do A7: τ do OP-F1; em VAL_EX AP, P, R, F1 e F1 por janela; em TEST_BOX R_box, R_loc@0,3,
     mIoU e taxa de FP aparentes (tudo no OP-F1, com IC 95 %).
  3. As comparações declaradas: diferença pareada por janela (a − b), IC 95 %, se o IC exclui 0, e janelas em que
     cada um é melhor.
  4. Estratos no OP-F1: distância ao seed (P e R em VAL_EX; R_loc em TEST_BOX), classe e tamanho (R_loc em TEST_BOX).
  5. Duas figuras (png e pdf): as diferenças pareadas com o IC (fig_extra_comparacoes) e P, R e R_loc por distância
     ao seed (fig_extra_distancia), com as cores validadas do A7 (base+ #2a78d6, Tiny #eb6834, MedSAM2 #1baf7a).
  6. A linha para o docs/decisoes.md com os resultados.

Uso, a partir da pasta do projeto:
    python scripts/22_extra_tabelas.py
Gera em outputs/22_extra/: resumo.md (só números agregados), tabela.csv, comparacoes.csv, estratos.csv e as figuras.

Versão 1.0 (26 out 2026).
"""
import argparse
import datetime
import json
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "eval"))
import evaluate as ev  # noqa: E402

VERSAO = "1.0"
CORES = ["#2a78d6", "#eb6834", "#1baf7a"]       # as três primeiras cores validadas do A7, pela ordem
MARCAS = ["o", "s", "^"]
TINTA, TINTA2, GRELHA, FUNDO = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
DISTANCIAS = ["1-3", "4-6", "7-12"]               # com L12 não há previsões para lá de 12
METRICA = {"VAL_EX": "F1", "TEST_BOX": "R_loc"}   # a métrica de cada comparação declarada (omissão do avaliador)


def _v(x):
    return x.get("valor") if isinstance(x, dict) else x


def _ic(x):
    ic = x.get("ic95") if isinstance(x, dict) else None
    return (ic[0], ic[1]) if ic and None not in ic else (None, None)


def _pt(x, casas=3, sinal=False):
    """Número à portuguesa (vírgula), com sinal se pedido; — se não houver."""
    if x is None:
        return "—"
    return (f"{x:+.{casas}f}" if sinal else f"{x:.{casas}f}").replace(".", ",").replace("-", "−")


def _com_ic(x, casas=3):
    lo, hi = _ic(x)
    v = _v(x)
    return _pt(v, casas) + ("" if lo is None else f" [{_pt(lo, casas)}–{_pt(hi, casas)}]")


def ler_resultado(cfg, conjunto, metodo):
    p = Path(cfg["caminhos"]["saida"]) / conjunto / f"{metodo}.json"
    if not p.is_file():
        raise ev.ErroAvaliacao(f"{metodo} não está avaliado em {conjunto} ({p}).")
    r = json.loads(p.read_text(encoding="utf-8"))
    if r.get("metodo") != metodo or r.get("conjunto") != conjunto:
        raise ev.ErroAvaliacao(f"{p} não é a avaliação de {metodo} em {conjunto}.")
    return p, r


def execucoes(cfg, metodos, conjuntos):
    reg = Path(cfg["caminhos"]["registo_testes"])
    if not reg.is_file():
        raise ev.ErroAvaliacao(f"Não encontrei {reg}: as avaliações de teste não estão registadas.")
    r = pd.read_csv(reg, dtype=str, keep_default_na=False)
    out = {}
    for m in metodos:
        for c in conjuntos:
            x = r[(r["metodo"] == m) & (r["conjunto"] == c)]
            if not len(x):
                raise ev.ErroAvaliacao(f"{m} não tem nenhuma execução registada em {c} ({reg}).")
            out[(m, c)] = {"n": len(x), "motivos": [s for s in x["motivo_repeticao"] if s]}
    return out


def tabela(res, nomes, linhas):
    t = []
    for m in nomes:
        a, b = res[(m, "VAL_EX")], res[(m, "TEST_BOX")]
        oa, ob = a["por_op"]["OP_F1"], b["por_op"]["OP_F1"]
        reg = {"metodo": m, "linha": linhas[m], "tau_OP_F1": oa["tau"]}
        for nome, x in (("AP_VAL_EX", a.get("AP")), ("P", oa["micro"]["P"]), ("R", oa["micro"]["R"]),
                        ("F1", oa["micro"]["F1"]), ("F1_por_janela", oa["macro"]["F1"]),
                        ("R_box", ob["micro"]["R_box"]), ("R_loc", ob["micro"]["R_loc"]),
                        ("mIoU", ob["micro"]["mIoU"]), ("FP_aparentes", ob["micro"]["taxa_fp_aparentes"])):
            lo, hi = _ic(x)
            reg.update({nome: _v(x), f"{nome}_ic_inf": lo, f"{nome}_ic_sup": hi, f"_{nome}": x})
        t.append(reg)
    return t


def comparacoes(cfg, pares, conjuntos):
    out = []
    for c in conjuntos:
        met = METRICA[c]
        for a, b in pares:
            p = Path(cfg["caminhos"]["saida"]) / c / f"comparar_{a}_vs_{b}_{met}_OP_F1.json"
            if not p.is_file():
                raise ev.ErroAvaliacao(f"Falta a comparação {a} − {b} em {c} ({p}): corre o comparar.")
            r = json.loads(p.read_text(encoding="utf-8"))
            out.append({"conjunto": c, "metrica": met, "a": a, "b": b, "n_janelas": r["n_janelas"],
                        "diferenca_media": r["diferenca_media"], "ic_inf": r["ic95"][0], "ic_sup": r["ic95"][1],
                        "diferentes": r["diferentes"], "a_melhor": r["janelas_a_melhor"], "empates": r["empates"],
                        "b_melhor": r["janelas_b_melhor"]})
    return out


def estratos(res, nomes):
    out = []
    for m in nomes:
        ea = res[(m, "VAL_EX")]["por_op"]["OP_F1"]["estratos"]
        eb = res[(m, "TEST_BOX")]["por_op"]["OP_F1"]["estratos"]
        for d in DISTANCIAS:
            for met in ("P", "R"):
                x = ea["distancia"].get(d, {}).get(met)
                out.append({"metodo": m, "conjunto": "VAL_EX", "estrato": "distancia", "grupo": d, "metrica": met,
                            "valor": _v(x), "ic_inf": _ic(x)[0], "ic_sup": _ic(x)[1],
                            "n": ea["distancia"].get(d, {}).get("n_pos")})
            x = eb["distancia"].get(d, {}).get("R_loc")
            out.append({"metodo": m, "conjunto": "TEST_BOX", "estrato": "distancia", "grupo": d, "metrica": "R_loc",
                        "valor": _v(x), "ic_inf": _ic(x)[0], "ic_sup": _ic(x)[1],
                        "n": eb["distancia"].get(d, {}).get("n_caixas")})
        for est in ("classe", "tamanho"):
            for g, blk in eb[est].items():
                x = blk.get("R_loc")
                out.append({"metodo": m, "conjunto": "TEST_BOX", "estrato": est, "grupo": g, "metrica": "R_loc",
                            "valor": _v(x), "ic_inf": _ic(x)[0], "ic_sup": _ic(x)[1], "n": blk.get("n_caixas")})
    return pd.DataFrame(out)


# ============================================================================ figuras
def _virgula_eixo(x, _pos=None):
    return f"{x:.1f}".replace(".", ",").replace("-", "−")


def _eixos(ax, eixo="y"):
    from matplotlib.ticker import FuncFormatter
    (ax.yaxis if eixo == "y" else ax.xaxis).set_major_formatter(FuncFormatter(_virgula_eixo))
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(GRELHA)
    ax.tick_params(colors=TINTA2, labelsize=9, length=0)
    ax.set_axisbelow(True)


def fig_comparacoes(comp, rotulos, destino):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    conj = [c for c in METRICA if any(r["conjunto"] == c for r in comp)]
    lo = min(min(r["ic_inf"] for r in comp), 0.0)
    hi = max(max(r["ic_sup"] for r in comp), 0.0)
    folga = 0.08 * (hi - lo or 1)
    fig, axs = plt.subplots(1, len(conj), figsize=(4.6 * len(conj), 2.6), sharex=True, squeeze=False)
    for ax, c in zip(axs[0], conj):
        rs = [r for r in comp if r["conjunto"] == c][::-1]
        for i, r in enumerate(rs):
            cheio = r["diferentes"]
            ax.plot([r["ic_inf"], r["ic_sup"]], [i, i], color=TINTA, lw=2, solid_capstyle="round", zorder=2)
            ax.plot(r["diferenca_media"], i, "o", ms=8, mfc=TINTA if cheio else FUNDO, mec=TINTA, mew=2, zorder=3)
            ax.annotate(_pt(r["diferenca_media"], 3, sinal=True), (r["ic_sup"], i), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=8.5, color=TINTA2)
        ax.axvline(0, color=TINTA2, lw=1, zorder=1)
        ax.set_yticks(range(len(rs)))
        ax.set_yticklabels([f"{rotulos[r['a']]} − {rotulos[r['b']]}" for r in rs], fontsize=9, color=TINTA)
        ax.set_ylim(-0.6, len(rs) - 0.4)
        ax.set_xlim(lo - folga, hi + 3 * folga)
        ax.grid(axis="x", color=GRELHA, lw=1)
        ax.set_title(f"{c.replace('_', '-')} · {'F1' if METRICA[c] == 'F1' else 'R_loc@0,3'} por janela (OP-F1)",
                     fontsize=10, color=TINTA, loc="left")
        ax.set_xlabel("diferença média e IC 95 % (a − b)", fontsize=9, color=TINTA2)
        _eixos(ax, "x")
    fig.text(0.01, -0.04, "Ponto cheio: o IC exclui 0 (diferentes). Ponto vazio: sem diferença demonstrada.",
             fontsize=8, color=TINTA2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(destino.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight", facecolor=FUNDO)
    plt.close(fig)


def fig_distancia(est, nomes, rotulos, destino):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paineis = [("VAL_EX", "P", "VAL-EX · precisão"), ("VAL_EX", "R", "VAL-EX · recall"),
               ("TEST_BOX", "R_loc", "TEST-BOX · R_loc@0,3")]
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
    desvio = {m: (i - (len(nomes) - 1) / 2) * 0.08 for i, m in enumerate(nomes)}
    for ax, (c, met, titulo) in zip(axs, paineis):
        fins = []
        for i, m in enumerate(nomes):
            s = est[(est["metodo"] == m) & (est["conjunto"] == c) & (est["estrato"] == "distancia")
                    & (est["metrica"] == met)].set_index("grupo").reindex(DISTANCIAS)
            x = [k + desvio[m] for k in range(len(DISTANCIAS))]
            y = s["valor"].astype(float).to_numpy()
            ax.vlines(x, s["ic_inf"].astype(float), s["ic_sup"].astype(float), color=CORES[i], lw=1, alpha=0.6,
                      zorder=2)
            ax.plot(x, y, "-", color=CORES[i], lw=2, solid_joinstyle="round", solid_capstyle="round", zorder=3)
            ax.plot(x, y, MARCAS[i], ms=8, color=CORES[i], mec=FUNDO, mew=1.5, zorder=4)
            if y[-1] == y[-1]:                      # sem NaN
                fins.append([float(y[-1]), rotulos[m]])
        if c == "TEST_BOX":                       # rótulos diretos só no último painel, sem se sobreporem
            fins.sort()
            for j in range(1, len(fins)):
                fins[j][0] = max(fins[j][0], fins[j - 1][0] + 0.06)
            x_fim = len(DISTANCIAS) - 1 + max(desvio.values())
            for yy, nome in fins:
                ax.annotate(nome, (x_fim, yy), xytext=(10, 0), textcoords="offset points", va="center",
                            fontsize=8.5, color=TINTA2)
        ax.set_xticks(range(len(DISTANCIAS)))
        ax.set_xticklabels([d.replace("-", "–") for d in DISTANCIAS], fontsize=9)
        ax.set_xlim(-0.4, len(DISTANCIAS) - 0.4 + (0.7 if c == "TEST_BOX" else 0))
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y", color=GRELHA, lw=1)
        ax.set_title(titulo, fontsize=10, color=TINTA, loc="left")
        ax.set_xlabel("distância ao seed (frames)", fontsize=9, color=TINTA2)
        _eixos(ax)
    from matplotlib.lines import Line2D
    pegas = [Line2D([0], [0], color=CORES[i], lw=2, marker=MARCAS[i], ms=7, mec=FUNDO, mew=1.5)
             for i in range(len(nomes))]
    fig.legend(pegas, [rotulos[m] for m in nomes], loc="upper center", ncol=len(nomes), bbox_to_anchor=(0.5, 1.08),
               fontsize=9, frameon=False, labelcolor=TINTA)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(destino.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight", facecolor=FUNDO)
    plt.close(fig)


# ============================================================================ principal
def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--config", default="configs/avaliacao.yaml")
    a.add_argument("--config-extra", default="configs/extra.yaml")
    a.add_argument("--saida", default="outputs/22_extra")
    a.add_argument("--sem-figuras", action="store_true")
    args = a.parse_args(argv)
    cfg = ev.ler_config(args.config)
    e = ev.ler_config(args.config_extra)
    ref = e["referencia"]
    nomes = [ref] + [m["metodo"] for m in e["metodos"]]
    rotulos = {ref: "base+", **{m["metodo"]: m["linha"].replace("SAM2 ", "") for m in e["metodos"]}}
    linhas = {ref: "SAM2 base+ (principal)", **{m["metodo"]: m["linha"] for m in e["metodos"]}}
    conjuntos = list(e.get("conjuntos", ["VAL_EX", "TEST_BOX"]))
    pares = [tuple(p) for p in e.get("comparacoes", [])]

    res, fontes = {}, {}
    for m in nomes:
        for c in conjuntos:
            p, r = ler_resultado(cfg, c, m)
            res[(m, c)], fontes[(m, c)] = r, p
    n_exec = execucoes(cfg, nomes, conjuntos)
    tab = tabela(res, nomes, linhas)
    comp = comparacoes(cfg, pares, conjuntos)
    est = estratos(res, nomes)

    saida = Path(args.saida)
    saida.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{k: v for k, v in t.items() if not k.startswith("_")} for t in tab]).to_csv(
        saida / "tabela.csv", index=False)
    pd.DataFrame(comp).to_csv(saida / "comparacoes.csv", index=False)
    est.to_csv(saida / "estratos.csv", index=False)
    if not args.sem_figuras:
        fig_comparacoes(comp, rotulos, saida / "fig_extra_comparacoes")
        fig_distancia(est, nomes, rotulos, saida / "fig_extra_distancia")

    agora = datetime.datetime.now()
    git = ev._estado_git(RAIZ)
    estado = ("sem git" if git["commit"] is None else
              f"commit {git['commit']}" + (" (com alterações por guardar)" if git["alteracoes_por_guardar"] else ""))
    L = ["# Experiência A: SAM 2.1 base+ / Tiny / MedSAM2 — resultados do teste",
         "",
         f"22_extra_tabelas.py v{VERSAO} · {agora:%Y-%m-%d %H:%M} · {estado} · só números agregados.",
         "Cada método foi avaliado uma vez em VAL-EX e TEST-BOX, no OP-F1 fixado em CAL; o base+ (SAM2_L12) é o do A7.",
         "",
         "## Tabela (OP-F1)",
         "",
         "| Método | τ | AP VAL-EX | P | R | F1 | F1 por janela | R_box | R_loc@0,3 | mIoU | FP aparentes |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for t in tab:
        L.append(f"| {t['linha']} | {_pt(t['tau_OP_F1'], 2)} | {_pt(t['AP_VAL_EX'])} | {_pt(t['P'])} | {_pt(t['R'])} | "
                 f"{_pt(t['F1'])} | {_com_ic(t['_F1_por_janela'])} | {_pt(t['R_box'])} | {_com_ic(t['_R_loc'])} | "
                 f"{_pt(t['mIoU'])} | {_pt(t['FP_aparentes'])} |")
    L += ["", "P, R, F1 e F1 por janela são de VAL-EX; R_box, R_loc, mIoU e FP aparentes são de TEST-BOX.",
          "", "## Comparações declaradas (diferença pareada por janela, a − b, no OP-F1)", "",
          "| Conjunto | Métrica | a − b | Diferença [IC 95 %] | Resultado | Janelas a melhor / empate / b melhor |",
          "| --- | --- | --- | --- | --- | --- |"]
    for r in comp:
        L.append(f"| {r['conjunto'].replace('_', '-')} | {r['metrica']} por janela | {rotulos[r['a']]} − "
                 f"{rotulos[r['b']]} | {_pt(r['diferenca_media'], 3, True)} [{_pt(r['ic_inf'], 3, True)}; "
                 f"{_pt(r['ic_sup'], 3, True)}] | {'diferentes' if r['diferentes'] else 'sem diferença demonstrada'} "
                 f"| {r['a_melhor']} / {r['empates']} / {r['b_melhor']} |")
    L += ["", f"{len(comp)} comparações, sem correção para comparações múltiplas (declarado a 26 out).", "",
          "## Por distância ao seed (OP-F1)", "",
          "| Método | VAL-EX P (1–3 / 4–6 / 7–12) | VAL-EX R (1–3 / 4–6 / 7–12) | TEST-BOX R_loc (1–3 / 4–6 / 7–12) |",
          "| --- | --- | --- | --- |"]
    for m in nomes:
        def serie(c, met):
            s = est[(est["metodo"] == m) & (est["conjunto"] == c) & (est["estrato"] == "distancia")
                    & (est["metrica"] == met)].set_index("grupo").reindex(DISTANCIAS)["valor"]
            return " / ".join(_pt(None if pd.isna(x) else float(x)) for x in s)
        L.append(f"| {linhas[m]} | {serie('VAL_EX', 'P')} | {serie('VAL_EX', 'R')} | {serie('TEST_BOX', 'R_loc')} |")
    L += ["", "## Proveniência", ""]
    for (m, c), p in fontes.items():
        n = n_exec[(m, c)]
        L.append(f"- {m} em {c}: {p.name} (sha256 {ev._sha256(p)[:16]}) · {n['n']} execução(ões) registada(s)"
                 + (f" · repetida: {'; '.join(n['motivos'])}" if n["motivos"] else ""))
    if not args.sem_figuras:
        L += ["", "Figuras: fig_extra_comparacoes (png, pdf) e fig_extra_distancia (png, pdf)."]

    def c_txt(a_, b_, c):
        r = next(x for x in comp if x["a"] == a_ and x["b"] == b_ and x["conjunto"] == c)
        return (f"{_pt(r['diferenca_media'], 3, True)} [{_pt(r['ic_inf'], 3, True)}; {_pt(r['ic_sup'], 3, True)}]"
                f" ({'diferentes' if r['diferentes'] else 'sem diferença demonstrada'})")
    por = {t["metodo"]: t for t in tab}
    partes = []
    for a_, b_ in pares:
        partes.append(f"{rotulos[a_]} − {rotulos[b_]}: F1 por janela {c_txt(a_, b_, 'VAL_EX')}, R_loc "
                      f"{c_txt(a_, b_, 'TEST_BOX')}")
    linha = (f"- {agora:%Y-%m-%d}: experiência A, avaliação única (tag extra-congelado; scripts/22_extra_tabelas.py "
             f"v{VERSAO}, outputs/22_extra/). F1 por janela em VAL-EX (OP-F1): "
             + ", ".join(f"{rotulos[m]} {_pt(por[m]['F1_por_janela'])}" for m in nomes)
             + "; R_loc@0,3 em TEST-BOX: "
             + ", ".join(f"{rotulos[m]} {_pt(por[m]['R_loc'])}" for m in nomes)
             + ". Comparações declaradas (diferença pareada por janela, IC 95 %, sem correção para comparações "
               "múltiplas): " + "; ".join(partes)
             + ". O método principal continua o SAM2_L12; nada muda no go/no-go, na anotação expandida nem no "
               "Dataset WCE v2.")
    L += ["", "## Linha para o docs/decisoes.md", "", linha]
    texto = "\n".join(L)
    (saida / "resumo.md").write_text(texto + "\n", encoding="utf-8")
    print(texto)
    return {"resumo": saida / "resumo.md", "linha": linha, "tabela": tab, "comparacoes": comp, "estratos": est}


if __name__ == "__main__":
    try:
        main()
    except ev.ErroAvaliacao as e:
        print(f"ERRO: {e}", file=sys.stderr)
        sys.exit(2)
