#!/usr/bin/env python3
"""
26 — Experiência B (detetor YOLO): tabela, comparação declarada, estratos, mAP e figuras do YOLO treinado com a anotação
original (YOLO_ORIG) e com o Dataset WCE v2 (YOLO_V2). Só lê o que já foi gravado: as avaliações do avaliador
(outputs/eval/<conjunto>/<modelo>.json, as comparações e outputs/test_runs.csv), os configs/final_<modelo>.yaml e os logs
do val.py (outputs/logs/yolo_map_<modelo>_<referência>.log). Não corre nenhum modelo nem avalia nada.

  1. Confere que cada modelo foi avaliado em VAL_EX e TEST_BOX (nas janelas das 6 classes) e quantas execuções tem em
     outputs/test_runs.csv (deve ser uma), e que as duas comparações declaradas existem.
  2. Tabela no formato do A7: τ do OP-F1; em VAL_EX AP, P, R, F1 e F1 por janela; em TEST_BOX R_box, R_loc@0,3, mIoU e
     taxa de FP aparentes (tudo no OP-F1, com IC 95 %).
  3. A comparação declarada a 26 out: YOLO_V2 − YOLO_ORIG, diferença pareada por janela (F1 em VAL_EX, R_loc em
     TEST_BOX), IC 95 % e se exclui 0.
  4. Estratos no OP-F1: distância ao seed (P e R em VAL_EX, R_loc em TEST_BOX, incluindo > 12, porque o YOLO vê todos
     os frames da janela), classe e tamanho (R_loc em TEST_BOX).
  5. Secundário: mAP50 e mAP50-95 por classe (val.py), nos frames com caixa das janelas de TEST_BOX, contra as caixas
     originais e contra as da v2. A precisão e o recall que o val.py escreve não entram: são os do melhor corte no
     próprio teste.
  6. Duas figuras (png e pdf): P, R e R_loc por distância ao seed (fig_yolo_distancia) e R_loc por classe em TEST_BOX
     (fig_yolo_classes), com as duas primeiras cores validadas da tese (original #2a78d6, v2 #eb6834).
  7. A linha para o docs/decisoes.md com os resultados.

Uso, no sam_tese, a partir da pasta do projeto:
    python scripts/26_yolo_tabelas.py
Gera em outputs/26_yolo/: resumo.md (só números agregados), tabela.csv, comparacoes.csv, estratos.csv, map.csv e as
figuras.

Versão 1.0 (27 out 2026).
"""
import argparse
import datetime
import json
import re
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "eval"))
import evaluate as ev  # noqa: E402

VERSAO = "1.0"
NOMES = ["YOLO_ORIG", "YOLO_V2"]
ROTULOS = {"YOLO_ORIG": "original", "YOLO_V2": "v2"}
LINHAS = {"YOLO_ORIG": "YOLO, anotação original", "YOLO_V2": "YOLO, Dataset WCE v2"}
PAR = ("YOLO_V2", "YOLO_ORIG")
CONJUNTOS = ("VAL_EX", "TEST_BOX")
METRICA = {"VAL_EX": "F1", "TEST_BOX": "R_loc"}
DISTANCIAS = ["1-3", "4-6", "7-12", ">12"]
CLASSES = ["angiectasia", "erosao", "ulcera", "polipo", "eritema", "xantelasma"]
NOMES_PT = {"angiectasia": "angiectasia", "erosao": "erosão", "ulcera": "úlcera", "polipo": "pólipo",
            "eritema": "eritema", "xantelasma": "xantelasma", "all": "todas"}
REFS = ("original", "v2")
CORES = {"YOLO_ORIG": "#2a78d6", "YOLO_V2": "#eb6834"}      # validadas (validate_palette.js, modo claro)
MARCAS = {"YOLO_ORIG": "o", "YOLO_V2": "s"}
TINTA, TINTA2, GRELHA, FUNDO = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


def _v(x):
    return x.get("valor") if isinstance(x, dict) else x


def _ic(x):
    ic = x.get("ic95") if isinstance(x, dict) else None
    return (ic[0], ic[1]) if ic and None not in ic else (None, None)


def _pt(x, casas=3, sinal=False):
    """Número à portuguesa (vírgula), com sinal se pedido; — se não houver."""
    if x is None or x != x:
        return "—"
    return (f"{x:+.{casas}f}" if sinal else f"{x:.{casas}f}").replace(".", ",").replace("-", "−")


def _com_ic(x, casas=3):
    lo, hi = _ic(x)
    return _pt(_v(x), casas) + ("" if lo is None else f" [{_pt(lo, casas)}–{_pt(hi, casas)}]")


# ============================================================================ leitura
def ler_resultado(cfg, conjunto, metodo):
    p = Path(cfg["caminhos"]["saida"]) / conjunto / f"{metodo}.json"
    if not p.is_file():
        raise ev.ErroAvaliacao(f"{metodo} não está avaliado em {conjunto} ({p}).")
    r = json.loads(p.read_text(encoding="utf-8"))
    if r.get("metodo") != metodo or r.get("conjunto") != conjunto:
        raise ev.ErroAvaliacao(f"{p} não é a avaliação de {metodo} em {conjunto}.")
    return p, r


def execucoes(cfg):
    reg = Path(cfg["caminhos"]["registo_testes"])
    if not reg.is_file():
        raise ev.ErroAvaliacao(f"Não encontrei {reg}: as avaliações de teste não estão registadas.")
    r = pd.read_csv(reg, dtype=str, keep_default_na=False)
    out = {}
    for m in NOMES:
        for c in CONJUNTOS:
            x = r[(r["metodo"] == m) & (r["conjunto"] == c)]
            if not len(x):
                raise ev.ErroAvaliacao(f"{m} não tem nenhuma execução registada em {c} ({reg}).")
            out[(m, c)] = {"n": len(x), "motivos": [s for s in x["motivo_repeticao"] if s]}
    return out


def ler_map(log):
    """mAP50 e mAP50-95 por classe (e «all») de um log do val.py; fica a última tabela do ficheiro."""
    p = Path(log)
    if not p.is_file():
        raise ev.ErroAvaliacao(f"Falta {p}: corre o val.py desse modelo e dessa referência (passo 4 do resumo do 25).")
    texto = p.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
    padrao = re.compile(rf"^\s*({'|'.join(['all'] + CLASSES)})\s+(\d+)\s+(\d+)\s+({NUM})\s+({NUM})\s+({NUM})\s+({NUM})\s*$")
    out = {}
    for linha in texto.splitlines():
        m = padrao.match(linha)
        if m:
            out[m.group(1)] = {"imagens": int(m.group(2)), "caixas": int(m.group(3)), "mAP50": float(m.group(6)),
                               "mAP50_95": float(m.group(7))}
    if "all" not in out:
        raise ev.ErroAvaliacao(f"{p} não tem a tabela final do val.py (a linha «all»). O val.py acabou bem?")
    return out


# ============================================================================ tabelas
def tabela(res):
    t = []
    for m in NOMES:
        a, b = res[(m, "VAL_EX")], res[(m, "TEST_BOX")]
        oa, ob = a["por_op"]["OP_F1"], b["por_op"]["OP_F1"]
        reg = {"metodo": m, "linha": LINHAS[m], "tau_OP_F1": oa["tau"]}
        for nome, x in (("AP_VAL_EX", a.get("AP")), ("P", oa["micro"]["P"]), ("R", oa["micro"]["R"]),
                        ("F1", oa["micro"]["F1"]), ("F1_por_janela", oa["macro"]["F1"]),
                        ("R_box", ob["micro"]["R_box"]), ("R_loc", ob["micro"]["R_loc"]),
                        ("mIoU", ob["micro"]["mIoU"]), ("FP_aparentes", ob["micro"]["taxa_fp_aparentes"])):
            lo, hi = _ic(x)
            reg.update({nome: _v(x), f"{nome}_ic_inf": lo, f"{nome}_ic_sup": hi, f"_{nome}": x})
        reg["janelas_VAL_EX"] = a["contagens"]["janelas"]
        reg["janelas_TEST_BOX"] = b["contagens"]["janelas"]
        t.append(reg)
    return t


def comparacoes(cfg):
    out = []
    for c in CONJUNTOS:
        met = METRICA[c]
        a, b = PAR
        p = Path(cfg["caminhos"]["saida"]) / c / f"comparar_{a}_vs_{b}_{met}_OP_F1.json"
        if not p.is_file():
            raise ev.ErroAvaliacao(f"Falta a comparação {a} − {b} em {c} ({p}): corre o comparar.")
        r = json.loads(p.read_text(encoding="utf-8"))
        out.append({"conjunto": c, "metrica": met, "a": a, "b": b, "n_janelas": r["n_janelas"],
                    "diferenca_media": r["diferenca_media"], "ic_inf": r["ic95"][0], "ic_sup": r["ic95"][1],
                    "diferentes": r["diferentes"], "a_melhor": r["janelas_a_melhor"], "empates": r["empates"],
                    "b_melhor": r["janelas_b_melhor"]})
    return out


def estratos(res):
    out = []
    for m in NOMES:
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
                if _v(x) is None:
                    continue
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


def _legenda(fig, plt):
    from matplotlib.lines import Line2D
    pegas = [Line2D([0], [0], color=CORES[m], lw=2, marker=MARCAS[m], ms=7, mec=FUNDO, mew=1.5) for m in NOMES]
    fig.legend(pegas, [LINHAS[m] for m in NOMES], loc="upper center", ncol=len(NOMES), bbox_to_anchor=(0.5, 1.08),
               fontsize=9, frameon=False, labelcolor=TINTA)


def fig_distancia(est, destino):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paineis = [("VAL_EX", "P", "VAL-EX · precisão"), ("VAL_EX", "R", "VAL-EX · recall"),
               ("TEST_BOX", "R_loc", "TEST-BOX · R_loc@0,3")]
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
    desvio = {m: (i - (len(NOMES) - 1) / 2) * 0.1 for i, m in enumerate(NOMES)}
    for ax, (c, met, titulo) in zip(axs, paineis):
        fins = []
        for m in NOMES:
            s = est[(est["metodo"] == m) & (est["conjunto"] == c) & (est["estrato"] == "distancia")
                    & (est["metrica"] == met)].set_index("grupo").reindex(DISTANCIAS)
            x = [k + desvio[m] for k in range(len(DISTANCIAS))]
            y = s["valor"].astype(float).to_numpy()
            ax.vlines(x, s["ic_inf"].astype(float), s["ic_sup"].astype(float), color=CORES[m], lw=1, alpha=0.6,
                      zorder=2)
            ax.plot(x, y, "-", color=CORES[m], lw=2, solid_joinstyle="round", solid_capstyle="round", zorder=3)
            ax.plot(x, y, MARCAS[m], ms=8, color=CORES[m], mec=FUNDO, mew=1.5, zorder=4)
            if y[-1] == y[-1]:
                fins.append([float(y[-1]), ROTULOS[m]])
        if c == "TEST_BOX":                        # rótulos diretos só no último painel, sem se sobreporem
            fins.sort()
            for j in range(1, len(fins)):
                fins[j][0] = max(fins[j][0], fins[j - 1][0] + 0.06)
            x_fim = len(DISTANCIAS) - 1 + max(desvio.values())
            for yy, nome in fins:
                ax.annotate(nome, (x_fim, yy), xytext=(10, 0), textcoords="offset points", va="center",
                            fontsize=8.5, color=TINTA2)
        ax.set_xticks(range(len(DISTANCIAS)))
        ax.set_xticklabels([d.replace("-", "–").replace(">", "> ") for d in DISTANCIAS], fontsize=9)
        ax.set_xlim(-0.4, len(DISTANCIAS) - 0.4 + (0.6 if c == "TEST_BOX" else 0))
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y", color=GRELHA, lw=1)
        ax.set_title(titulo, fontsize=10, color=TINTA, loc="left")
        ax.set_xlabel("distância ao seed (frames)", fontsize=9, color=TINTA2)
        _eixos(ax)
    _legenda(fig, plt)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(destino.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight", facecolor=FUNDO)
    plt.close(fig)


def fig_classes(est, destino):
    """R_loc@0,3 em TEST-BOX por classe: um ponto por modelo, com o IC, e o nº de caixas no rótulo."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = est[(est["conjunto"] == "TEST_BOX") & (est["estrato"] == "classe")]
    classes = [c for c in CLASSES if c in set(s["grupo"])]
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(classes) + 1.1))
    desvio = {m: (i - (len(NOMES) - 1) / 2) * 0.22 for i, m in enumerate(NOMES)}
    rotulos = []
    for k, cl in enumerate(classes[::-1]):
        n = None
        for m in NOMES:
            r = s[(s["metodo"] == m) & (s["grupo"] == cl)]
            if not len(r):
                continue
            r = r.iloc[0]
            y = k + desvio[m]
            ax.hlines(y, r["ic_inf"], r["ic_sup"], color=CORES[m], lw=1, alpha=0.6, zorder=2)
            ax.plot(r["valor"], y, MARCAS[m], ms=8, color=CORES[m], mec=FUNDO, mew=1.5, zorder=3)
            n = r["n"] if n is None else n
        rotulos.append(f"{NOMES_PT[cl]} ({'—' if n is None else int(n)})")
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(rotulos, fontsize=9, color=TINTA)
    ax.set_ylim(-0.6, len(classes) - 0.4)
    ax.set_xlim(0, 1.02)
    ax.grid(axis="x", color=GRELHA, lw=1)
    ax.set_title("TEST-BOX · R_loc@0,3 por classe (OP-F1; entre parênteses, o nº de caixas)", fontsize=10,
                 color=TINTA, loc="left")
    _eixos(ax, "x")
    _legenda(fig, plt)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(destino.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight", facecolor=FUNDO)
    plt.close(fig)


# ============================================================================ principal
def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--config", default="configs/avaliacao.yaml")
    a.add_argument("--configs", default="configs", help="onde estão os final_<modelo>.yaml")
    a.add_argument("--logs", default="outputs/logs")
    a.add_argument("--saida", default="outputs/26_yolo")
    a.add_argument("--sem-figuras", action="store_true")
    args = a.parse_args(argv)
    cfg = ev.ler_config(args.config)

    res, fontes = {}, {}
    for m in NOMES:
        for c in CONJUNTOS:
            p, r = ler_resultado(cfg, c, m)
            res[(m, c)], fontes[(m, c)] = r, p
    n_exec = execucoes(cfg)
    finais = {}
    for m in NOMES:
        f = Path(args.configs) / f"final_{m}.yaml"
        if not f.is_file():
            raise ev.ErroAvaliacao(f"Não encontrei {f} (o congelamento, scripts/25_yolo_congelar.py).")
        finais[m] = ev.ler_config(f)
    tab = tabela(res)
    comp = comparacoes(cfg)
    est = estratos(res)
    mp = {(m, ref): ler_map(Path(args.logs) / f"yolo_map_{m}_{ref}.log") for m in NOMES for ref in REFS}
    linhas_map = [{"metodo": m, "referencia": ref, "classe": cl, **v} for (m, ref), d in mp.items()
                  for cl, v in d.items()]

    saida = Path(args.saida)
    saida.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{k: v for k, v in t.items() if not k.startswith("_")} for t in tab]).to_csv(
        saida / "tabela.csv", index=False)
    pd.DataFrame(comp).to_csv(saida / "comparacoes.csv", index=False)
    est.to_csv(saida / "estratos.csv", index=False)
    pd.DataFrame(linhas_map).to_csv(saida / "map.csv", index=False)
    if not args.sem_figuras:
        fig_distancia(est, saida / "fig_yolo_distancia")
        fig_classes(est, saida / "fig_yolo_classes")

    agora = datetime.datetime.now()
    git = ev._estado_git(RAIZ)
    estado = ("sem git" if git["commit"] is None else
              f"commit {git['commit']}" + (" (com alterações por guardar)" if git["alteracoes_por_guardar"] else ""))
    por = {t["metodo"]: t for t in tab}
    L = ["# Experiência B: YOLO com a anotação original e com o Dataset WCE v2 — resultados do teste",
         "",
         f"26_yolo_tabelas.py v{VERSAO} · {agora:%Y-%m-%d %H:%M} · {estado} · só números agregados.",
         f"Cada modelo foi avaliado uma vez, nas janelas das 6 classes (VAL-EX {por['YOLO_ORIG']['janelas_VAL_EX']}, "
         f"TEST-BOX {por['YOLO_ORIG']['janelas_TEST_BOX']}), no OP-F1 fixado em CAL.",
         "",
         "## Treino", "",
         "| Modelo | Épocas feitas | Melhor época | Validação (casos de CAL, anotação do próprio modelo): mAP50 / mAP50-95 |",
         "| --- | --- | --- | --- |"]
    for m in NOMES:
        t = finais[m]["modelo"]["treino"]
        vm = t["validacao_na_melhor"]
        L.append(f"| {LINHAS[m]} | {t['epocas_feitas']} | {t['melhor_epoca']} | {_pt(vm['mAP50'])} / {_pt(vm['mAP50_95'])} |")
    L += ["", "A validação de cada modelo usa a sua anotação: os números não se comparam entre si.", "",
          "## Tabela (OP-F1)", "",
          "| Modelo | τ | AP VAL-EX | P | R | F1 | F1 por janela | R_box | R_loc@0,3 | mIoU | FP aparentes |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for t in tab:
        L.append(f"| {t['linha']} | {_pt(t['tau_OP_F1'], 2)} | {_pt(t['AP_VAL_EX'])} | {_pt(t['P'])} | {_pt(t['R'])} | "
                 f"{_pt(t['F1'])} | {_com_ic(t['_F1_por_janela'])} | {_pt(t['R_box'])} | {_com_ic(t['_R_loc'])} | "
                 f"{_pt(t['mIoU'])} | {_pt(t['FP_aparentes'])} |")
    L += ["", "P, R, F1 e F1 por janela são de VAL-EX (contra a anotação PF v2); R_box, R_loc, mIoU e FP aparentes são de "
          "TEST-BOX (contra as caixas originais da janela).",
          "", "## Comparação declarada (v2 − original, diferença pareada por janela, no OP-F1)", "",
          "| Conjunto | Métrica | Diferença [IC 95 %] | Resultado | Janelas v2 melhor / empate / original melhor |",
          "| --- | --- | --- | --- | --- |"]
    for r in comp:
        L.append(f"| {r['conjunto'].replace('_', '-')} ({r['n_janelas']} janelas) | {r['metrica']} por janela | "
                 f"{_pt(r['diferenca_media'], 3, True)} [{_pt(r['ic_inf'], 3, True)}; {_pt(r['ic_sup'], 3, True)}] | "
                 f"{'diferentes' if r['diferentes'] else 'sem diferença demonstrada'} | {r['a_melhor']} / {r['empates']} "
                 f"/ {r['b_melhor']} |")
    L += ["", "Duas comparações, declaradas a 26 out, sem correção para comparações múltiplas.", "",
          "## Por distância ao seed (OP-F1)", "",
          "| Modelo | VAL-EX P (1–3 / 4–6 / 7–12 / > 12) | VAL-EX R (1–3 / 4–6 / 7–12 / > 12) | "
          "TEST-BOX R_loc (1–3 / 4–6 / 7–12 / > 12) |",
          "| --- | --- | --- | --- |"]
    for m in NOMES:
        def serie(c, met):
            s = est[(est["metodo"] == m) & (est["conjunto"] == c) & (est["estrato"] == "distancia")
                    & (est["metrica"] == met)].set_index("grupo").reindex(DISTANCIAS)["valor"]
            return " / ".join(_pt(None if pd.isna(x) else float(x)) for x in s)
        L.append(f"| {LINHAS[m]} | {serie('VAL_EX', 'P')} | {serie('VAL_EX', 'R')} | {serie('TEST_BOX', 'R_loc')} |")
    L += ["", "## Por classe (TEST-BOX, R_loc@0,3, OP-F1)", "",
          "| Classe | Caixas | " + " | ".join(LINHAS[m] for m in NOMES) + " |", "| --- | --- | --- | --- |"]
    for cl in CLASSES:
        s = est[(est["conjunto"] == "TEST_BOX") & (est["estrato"] == "classe") & (est["grupo"] == cl)]
        if not len(s):
            continue
        cel = []
        for m in NOMES:
            r = s[s["metodo"] == m]
            cel.append("—" if not len(r) else f"{_pt(r.iloc[0]['valor'])} [{_pt(r.iloc[0]['ic_inf'])}–"
                       f"{_pt(r.iloc[0]['ic_sup'])}]")
        n = s.iloc[0]["n"]
        L.append(f"| {NOMES_PT[cl]} | {'—' if pd.isna(n) else int(n)} | " + " | ".join(cel) + " |")
    L += ["", "## Secundário: mAP por classe (val.py, frames com caixa das janelas de TEST-BOX das 6 classes)", "",
          "| Classe | " + " | ".join(f"{LINHAS[m]} · caixas {ref}" for ref in REFS for m in NOMES) + " |",
          "| --- | " + " | ".join("---" for _ in REFS for _ in NOMES) + " |"]
    for cl in ["all"] + CLASSES:
        cel = []
        for ref in REFS:
            for m in NOMES:
                x = mp[(m, ref)].get(cl)
                cel.append("—" if x is None else f"{_pt(x['mAP50'])} / {_pt(x['mAP50_95'])}")
        L.append(f"| {NOMES_PT[cl]} | " + " | ".join(cel) + " |")
    L += ["", "Em cada célula: mAP50 / mAP50-95. «caixas original»: as caixas originais (imagens com caixa original); "
          "«caixas v2»: as da v2 (mais as imagens que só têm caixas novas). Um frame sem caixa não entra (não é um "
          "negativo), por isso estes números não se comparam com os do trabalho anterior (outra divisão e frames de "
          "fundo). A precisão e o recall que o val.py escreve não entram: são os do melhor corte no próprio teste.",
          "", "## Proveniência", ""]
    for (m, c), p in fontes.items():
        n = n_exec[(m, c)]
        L.append(f"- {m} em {c}: {p.name} (sha256 {ev._sha256(p)[:16]}) · {n['n']} execução(ões) registada(s)"
                 + (f" · repetida: {'; '.join(n['motivos'])}" if n["motivos"] else ""))
    for m in NOMES:
        L.append(f"- {m}: best.pt sha256 {finais[m]['modelo']['pesos_sha256'][:16]} · τ do OP-F1 "
                 f"{_pt(finais[m]['pontos_operacao']['OP_F1']['tau'], 2)} (congelado: {finais[m]['congelado']['data']})")
    if not args.sem_figuras:
        L += ["", "Figuras: fig_yolo_distancia (png, pdf) e fig_yolo_classes (png, pdf)."]

    def c_txt(c):
        r = next(x for x in comp if x["conjunto"] == c)
        return (f"{_pt(r['diferenca_media'], 3, True)} [{_pt(r['ic_inf'], 3, True)}; {_pt(r['ic_sup'], 3, True)}]"
                f" ({'diferentes' if r['diferentes'] else 'sem diferença demonstrada'})")
    linha = (f"- {agora:%Y-%m-%d}: experiência B, avaliação única (tag yolo-congelado; scripts/26_yolo_tabelas.py "
             f"v{VERSAO}, outputs/26_yolo/), nas janelas das 6 classes, no OP-F1 fixado em CAL. F1 por janela em VAL-EX: "
             f"original {_pt(por['YOLO_ORIG']['F1_por_janela'])}, v2 {_pt(por['YOLO_V2']['F1_por_janela'])}; R_loc@0,3 "
             f"em TEST-BOX: original {_pt(por['YOLO_ORIG']['R_loc'])}, v2 {_pt(por['YOLO_V2']['R_loc'])}. Comparação "
             f"declarada (v2 − original, diferença pareada por janela, IC 95 %): VAL-EX {c_txt('VAL_EX')}; TEST-BOX "
             f"{c_txt('TEST_BOX')}. Secundário, mAP50 nos frames de teste com caixa: contra as caixas originais "
             f"original {_pt(mp[('YOLO_ORIG', 'original')]['all']['mAP50'])}, v2 "
             f"{_pt(mp[('YOLO_V2', 'original')]['all']['mAP50'])}; contra as da v2 original "
             f"{_pt(mp[('YOLO_ORIG', 'v2')]['all']['mAP50'])}, v2 {_pt(mp[('YOLO_V2', 'v2')]['all']['mAP50'])}. Nada "
             "muda no SAM2_L12, no go/no-go, na anotação expandida nem no Dataset WCE v2.")
    L += ["", "## Linha para o docs/decisoes.md", "", linha]
    texto = "\n".join(L)
    (saida / "resumo.md").write_text(texto + "\n", encoding="utf-8")
    print(texto)
    return {"resumo": saida / "resumo.md", "linha": linha, "tabela": tab, "comparacoes": comp, "estratos": est,
            "map": mp}


if __name__ == "__main__":
    try:
        main()
    except ev.ErroAvaliacao as e:
        print(f"ERRO: {e}", file=sys.stderr)
        sys.exit(2)
