#!/usr/bin/env python3
"""
12 — A6: congelamento antes da avaliação final (Guião A). Só lê as calibrações de CAL e os ficheiros de previsões já
gerados: não corre nenhum método e não calcula nada em VAL_EX nem em TEST_BOX.

A partir de configs/a6.yaml (a lista do A7 e as decisões do A6):
  1. Para cada método do A7 grava configs/final_<método>.yaml com: o script e a configuração que o geraram (sha256),
     os parâmetros, os pontos de operação fixados em CAL (OP-F1 e OP-R90: τ, A_min, regra), o ficheiro da
     calibração (sha256) e os ficheiros de previsões de VAL_EX e TEST_BOX já gerados (sha256), para o A7 poder
     confirmar que avalia exatamente esses. O do método principal leva também as decisões do A6 (ponto de operação
     do critério 2 do go/no-go e nível "média" da anotação expandida).
  2. Estabilidade em CAL, como o guião pede no A6: para cada método calibrado, a distribuição do F1 por janela no
     OP-F1 e a sensibilidade à janela dominante (o F1 micro sem ela, somando TP, FP e FN por janela).
  3. Nível "média" da anotação expandida (A8), para o método principal: em CAL, quantos frames, quantos certos e quanto
     recall acrescenta a faixa τ_inferior ≤ score < τ(OP-F1), para vários τ_inferior (a partir da grelha da
     calibração; os frames em dúvida não contam, como nas métricas).
Depois: git commit e git tag a6-congelado. A partir daí não se mexe em nada antes do A7.

Uso, a partir da pasta do projeto:
    python scripts/12_a6_congelar.py
Gera configs/final_<método>.yaml (um por método do A7), outputs/12_a6_resumo.txt, outputs/12_a6_estabilidade.csv e
outputs/12_a6_nivel_media.csv.
"""
import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "eval"))
import evaluate as ev  # noqa: E402

VERSAO = "1.0"
OPS = ("OP_F1", "OP_R90")


def _caminho(p):
    """Caminhos relativos a partir da pasta do projeto."""
    p = Path(p)
    return p if p.is_absolute() else RAIZ / p


def _valor(x):
    return x.get("valor") if isinstance(x, dict) else x


def _f1(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else None


def ler_calibracao(cfg, metodo):
    p = Path(cfg["caminhos"]["saida"]) / "CAL" / f"{metodo}.json"
    if not p.is_file():
        raise ev.ErroAvaliacao(f"Falta a calibração de {metodo} em CAL ({p}): corre primeiro "
                               f"eval/evaluate.py calibrar --metodo {metodo}.")
    cal = json.loads(p.read_text(encoding="utf-8"))
    if cal.get("metodo") != metodo or cal.get("conjunto") != "CAL" or "pontos_operacao" not in cal:
        raise ev.ErroAvaliacao(f"{p} não é uma calibração de {metodo} em CAL.")
    return p, cal


def sensibilidade(cal, janela):
    """F1 micro no OP-F1 com todas as janelas e sem a janela dada (somando TP, FP e FN por janela)."""
    tot, jan = np.zeros(3), np.zeros(3)
    achou = False
    for j in cal["por_janela"]:
        c = np.array([j["OP_F1"].get(k, 0) for k in ("TP", "FP", "FN")], dtype=float)
        tot += c
        if (j["split"], str(j["caso"]), int(j["seed"])) == janela:
            jan, achou = c, True
    return {"F1_micro": _f1(*tot), "F1_micro_sem_dominante": _f1(*(tot - jan)) if achou else None,
            "dominante_TP": int(jan[0]), "dominante_FP": int(jan[1]), "dominante_FN": int(jan[2]),
            "positivos": int(tot[0] + tot[2]), "positivos_dominante": int(jan[0] + jan[2])}


def estabilidade(cfg, janela):
    """Uma linha por método calibrado em CAL (todos os outputs/eval/CAL/<método>.json)."""
    linhas = []
    for p in sorted((Path(cfg["caminhos"]["saida"]) / "CAL").glob("*.json")):
        if p.name.startswith("comparar_") or ".anterior_" in p.name:
            continue
        cal = json.loads(p.read_text(encoding="utf-8"))
        if "calibracao" not in cal:
            continue
        f1j = cal["calibracao"]["F1_por_janela_em_OP_F1"]
        op = cal["por_op"]["OP_F1"]
        linhas.append({"metodo": cal["metodo"], "AP": _valor(cal.get("AP")), "tau_OP_F1": op["tau"],
                       "P_micro": _valor(op["micro"].get("P")), "R_micro": _valor(op["micro"].get("R")),
                       "F1_micro": _valor(op["micro"].get("F1")), "F1_macro": _valor(op["macro"].get("F1")),
                       "F1_janela_media": f1j["media"], "F1_janela_dp": f1j["desvio_padrao"],
                       "F1_janela_mediana": f1j["mediana"], "F1_janela_iqr_inf": f1j["iqr"][0],
                       "F1_janela_iqr_sup": f1j["iqr"][1], "F1_janela_min": f1j["min"], **sensibilidade(cal, janela)})
    return pd.DataFrame(linhas)


def faixa_media(cal, limites):
    """Faixa τ_inferior ≤ score < τ(OP-F1) em CAL, a partir da grelha da calibração (mesmo A_min do OP-F1)."""
    op = cal["pontos_operacao"]["OP_F1"]
    tau_f1, a_min = float(op["tau"]), float(op.get("a_min", 0))
    grelha = {round(float(r["tau"]), 6): r for r in cal["calibracao"]["grelha"] if float(r["a_min"]) == a_min}
    alto = grelha[round(tau_f1, 6)]
    pos = alto["TP"] + alto["FN"]
    linhas = []
    for lo in sorted(limites):
        r = grelha.get(round(float(lo), 6))
        if r is None or not float(lo) < tau_f1:
            continue
        tp, fp = r["TP"] - alto["TP"], r["FP"] - alto["FP"]
        linhas.append({"tau_inferior": float(lo), "tau_superior": tau_f1, "frames": int(tp + fp), "certos": int(tp),
                       "errados": int(fp), "precisao": tp / (tp + fp) if tp + fp else None,
                       "recall_so_alta": alto["TP"] / pos if pos else None,
                       "recall_alta_e_media": r["TP"] / pos if pos else None})
    return tau_f1, pd.DataFrame(linhas)


def _num(x, casas=3):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{casas}f}"


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--config", default="configs/avaliacao.yaml")
    a.add_argument("--config-a6", default="configs/a6.yaml")
    a.add_argument("--saida-configs", default="configs", help="onde gravar os final_<método>.yaml")
    a.add_argument("--saida", default="outputs")
    args = a.parse_args(argv)
    import yaml
    cfg = ev.ler_config(args.config)
    a6 = ev.ler_config(args.config_a6)
    metodos = a6["metodos"]
    nomes = [m["metodo"] for m in metodos]
    principal = a6["principal"]
    if principal not in nomes:
        raise ev.ErroAvaliacao(f"O método principal ({principal}) não está na lista do A7 (configs/a6.yaml).")
    if a6["criterio2_op"] not in OPS:
        raise ev.ErroAvaliacao(f"criterio2_op tem de ser OP_F1 ou OP_R90 (está {a6['criterio2_op']}).")
    janela = (a6["janela_dominante"][0], str(a6["janela_dominante"][1]), int(a6["janela_dominante"][2]))
    git = ev._estado_git(RAIZ)
    agora = datetime.datetime.now()
    previsoes = Path(a6["previsoes"])

    # 1. YAML final por método
    cals, gravados = {}, []
    for m in metodos:
        p_cal, cal = ler_calibracao(cfg, m["metodo"])
        cals[m["metodo"]] = cal
        conjuntos = m.get("conjuntos", a6["conjuntos"])
        prev = {}
        for c in conjuntos:
            f = previsoes / f"pred_{m['metodo']}_{c}.csv"
            if not f.is_file():
                raise ev.ErroAvaliacao(f"Faltam as previsões de {m['metodo']} em {c} ({f}).")
            prev[c] = {"ficheiro": str(f), "sha256": ev._sha256(f)}
        fonte = {}
        for chave in ("script", "config"):
            f = _caminho(m[chave])
            if not f.is_file():
                raise ev.ErroAvaliacao(f"{m['metodo']}: não encontrei {m[chave]}.")
            fonte.update({chave: m[chave], f"{chave}_sha256": ev._sha256(f)})
        final = {"metodo": m["metodo"], "linha_tabela_mestre": m["linha"], "principal": m["metodo"] == principal,
                 "gerado_por": fonte, "parametros": m.get("parametros", {}),
                 "pontos_operacao": {k: {"tau": float(v["tau"]), "a_min": float(v.get("a_min", 0)),
                                         "regra": v.get("regra", "")} for k, v in cal["pontos_operacao"].items()},
                 "calibracao": {"ficheiro": str(p_cal), "sha256": ev._sha256(p_cal),
                                "commit": cal.get("registo", {}).get("commit"),
                                "previsoes_sha256": cal.get("registo", {}).get("previsoes_sha256")},
                 "conjuntos_A7": list(conjuntos), "previsoes": prev}
        if m["metodo"] == principal:
            tau_f1 = float(cal["pontos_operacao"]["OP_F1"]["tau"])
            nm = a6.get("nivel_media")
            if nm is not None and not 0 < float(nm) < tau_f1:
                raise ev.ErroAvaliacao(f"nivel_media tem de estar entre 0 e o τ do OP-F1 ({tau_f1:g}) ou ser null.")
            final["decisoes_A6"] = {
                "criterio2_op": a6["criterio2_op"],
                "niveis_A8": {"alta": {"score_minimo": tau_f1},
                              "media": None if nm is None else {"score_minimo": float(nm), "score_abaixo_de": tau_f1}}}
        final["congelado"] = {"data": agora.isoformat(timespec="seconds"), "script": f"12_a6_congelar.py v{VERSAO}",
                              "commit": git["commit"], "alteracoes_por_guardar": git["alteracoes_por_guardar"]}
        destino = Path(args.saida_configs) / f"final_{m['metodo']}.yaml"
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text("# Congelado no A6 (scripts/12_a6_congelar.py): não alterar depois da tag a6-congelado.\n"
                           + yaml.safe_dump(final, allow_unicode=True, sort_keys=False), encoding="utf-8")
        gravados.append(destino)

    # 2. estabilidade e 3. nível "média"
    est = estabilidade(cfg, janela)
    tau_f1, media = faixa_media(cals[principal], a6.get("limites_media", [0.5, 0.6, 0.7, 0.8]))
    _, media_toda = faixa_media(cals[principal], np.round(np.arange(0.0, tau_f1, 0.02), 2))

    estado = ("sem git" if git["commit"] is None else
              f"commit {git['commit']}" + (" (com alterações por guardar)" if git["alteracoes_por_guardar"] else ""))
    L = [f"12 — A6: congelamento antes do A7 · {agora:%Y-%m-%d %H:%M} · {estado}",
         f"12_a6_congelar.py v{VERSAO} · sha256 {ev._sha256(__file__)[:16]} · {Path(args.config_a6).name} sha256 "
         f"{ev._sha256(args.config_a6)[:16]}",
         "",
         f"1. Métodos do A7 ({len(metodos)}; principal: {principal}) — YAML final em {args.saida_configs}/:"]
    for m in metodos:
        op = cals[m["metodo"]]["pontos_operacao"]
        L.append(f"  {m['metodo']:<20} {m['linha']:<20} OP-F1 τ {float(op['OP_F1']['tau']):.2f} · OP-R90 τ "
                 f"{float(op['OP_R90']['tau']):.2f} · conjuntos {', '.join(m.get('conjuntos', a6['conjuntos']))}")
    L += ["", f"   Decisões no YAML do principal: critério 2 do go/no-go no {a6['criterio2_op']} · nível \"média\": "
              + ("sem" if a6.get("nivel_media") is None else f"{float(a6['nivel_media']):g} ≤ score < {tau_f1:g}"),
          "",
          f"2. Estabilidade em CAL (F1 por janela no OP-F1; sensibilidade sem {janela[0]}/{janela[1]}/{janela[2]}):",
          f"  {'método':<20} {'AP':>6} {'F1 micro':>8} {'sem ela':>8} {'F1 janela: média':>17} {'dp':>6} "
          f"{'mediana':>7} {'IQR':>13} {'mín':>6}"]
    for r in est.itertuples(index=False):
        L.append(f"  {r.metodo:<20} {_num(r.AP):>6} {_num(r.F1_micro):>8} {_num(r.F1_micro_sem_dominante):>8} "
                 f"{_num(r.F1_janela_media):>17} {_num(r.F1_janela_dp):>6} {_num(r.F1_janela_mediana):>7} "
                 f"{_num(r.F1_janela_iqr_inf, 2) + '–' + _num(r.F1_janela_iqr_sup, 2):>13} {_num(r.F1_janela_min):>6}")
    if len(est):
        d = est.iloc[0]
        L.append(f"  (a janela dominante tem {d['positivos_dominante']} dos {d['positivos']} frames visíveis de CAL)")
    L += ["", f"3. Nível \"média\" do {principal} em CAL: faixa τ_inferior ≤ score < {tau_f1:g} (o nível \"alta\" é "
              f"score ≥ {tau_f1:g})",
          f"  {'τ inferior':>10} {'frames':>7} {'certos':>7} {'errados':>8} {'precisão':>9} {'recall só alta':>15} "
          f"{'recall alta + média':>20}"]
    for r in media.itertuples(index=False):
        L.append(f"  {r.tau_inferior:>10.2f} {r.frames:>7} {r.certos:>7} {r.errados:>8} {_num(r.precisao, 2):>9} "
                 f"{_num(r.recall_so_alta, 2):>15} {_num(r.recall_alta_e_media, 2):>20}")
    L += ["  (todos os τ da grelha em outputs/12_a6_nivel_media.csv)",
          "",
          "Depois de escolher o nível \"média\" (configs/a6.yaml, nivel_media) e voltar a correr:",
          "  git add -A && git commit -m \"A6: congelamento\" && git tag a6-congelado"]
    texto = "\n".join(L)
    saida = Path(args.saida)
    saida.mkdir(parents=True, exist_ok=True)
    (saida / "12_a6_resumo.txt").write_text(texto + "\n", encoding="utf-8")
    est.to_csv(saida / "12_a6_estabilidade.csv", index=False)
    media_toda.to_csv(saida / "12_a6_nivel_media.csv", index=False)
    print(texto)
    return {"yaml": gravados, "estabilidade": est, "media": media, "tau_f1": tau_f1}


if __name__ == "__main__":
    try:
        main()
    except ev.ErroAvaliacao as e:
        print(f"ERRO: {e}", file=sys.stderr)
        sys.exit(2)
