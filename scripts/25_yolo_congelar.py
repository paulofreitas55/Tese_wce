#!/usr/bin/env python3
"""
25 — Experiência B (detetor YOLO): congelamento antes da avaliação única dos dois modelos (YOLO_ORIG e YOLO_V2). Só lê
o que já está no disco (treinos, previsões, calibrações de CAL): não corre nenhum modelo e não avalia nada em VAL_EX
nem em TEST_BOX.

Confere, para cada modelo, antes de gravar o que quer que seja:
  - o treino (outputs/yolo/runs/<modelo>/opt.yaml) foi feito com o desenho do docs/decisoes.md (26 out): YOLOv5x COCO,
    100 épocas, batch 8, 416 px, paragem ao fim de 20 épocas sem melhoria, semente 0, 2 workers, sem cache, SGD, e com
    os dados certos (YOLO_ORIG → data/derived/yolo/original/treino.yaml; YOLO_V2 → …/v2/treino.yaml);
  - as previsões de CAL, VAL_EX e TEST_BOX são as da última corrida do scripts/24_yolo_prever.py (a marca e o resumo em
    outputs/yolo/prever/<modelo>/), com os pesos best.pt tal como estão agora e com os subconjuntos das 6 classes
    tal como estão agora;
  - a calibração em CAL (outputs/eval/CAL/<modelo>.json) foi feita com essas previsões de CAL e com o subconjunto das
    6 classes de CAL;
  - o modelo ainda não foi avaliado em nenhum conjunto de teste (outputs/test_runs.csv).
Grava configs/final_<modelo>.yaml (no formato do A6 e da experiência A: o treino, os pesos e os dados com sha256, os
pontos de operação de CAL, a calibração e as previsões de teste) e outputs/25_yolo_congelar_resumo.txt, com os comandos
da avaliação única, das duas comparações declaradas, do mAP (secundário) e a linha para o docs/decisoes.md. Não reescreve
um final_<modelo>.yaml que já exista (só com --forcar, com linha no decisoes.md). Se alguma coisa não bate, não grava
nada. Depois: git commit, git tag yolo-congelado e git push.

Uso, no sam_tese, a partir da pasta do projeto:
    python scripts/25_yolo_congelar.py
Versão 1.0 (27 out 2026).
"""
import argparse
import datetime
import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "eval"))
import evaluate as ev  # noqa: E402

VERSAO = "1.0"
MODELOS = {"YOLO_ORIG": ("original", "YOLO, anotação original"), "YOLO_V2": ("v2", "YOLO, Dataset WCE v2")}
COMPARACAO = ("YOLO_V2", "YOLO_ORIG")                   # a − b, declarada a 26 out
CONJUNTOS = ("CAL", "VAL_EX", "TEST_BOX")
TESTE = ("VAL_EX", "TEST_BOX")
DESENHO = {"epochs": 100, "batch_size": 8, "imgsz": 416, "patience": 20, "seed": 0, "workers": 2, "optimizer": "SGD",
           "single_cls": False}
TAG = "yolo-congelado"


def _num(x, casas=3):
    return "—" if x is None else f"{x:.{casas}f}"


def _virgula(x, casas=3):
    return "—" if x is None else f"{x:.{casas}f}".replace(".", ",")


def _valor(x):
    return x.get("valor") if isinstance(x, dict) else x


def _mesmo(a, b):
    return Path(a).resolve() == Path(b).resolve()


# ============================================================================ conferências
def ler_treino(runs, metodo, variante, dados):
    """O opt.yaml e o results.csv do treino; recusa um treino que não seja o do desenho."""
    pasta = Path(runs) / metodo
    f_opt, f_res, pesos = pasta / "opt.yaml", pasta / "results.csv", pasta / "weights" / "best.pt"
    for f in (f_opt, f_res, pesos):
        if not f.is_file():
            raise ev.ErroAvaliacao(f"{metodo}: não encontrei {f} (o treino acabou?).")
    opt = ev.ler_config(f_opt)
    mal = [f"{k} = {opt.get(k)!r} (desenho: {v!r})" for k, v in DESENHO.items() if opt.get(k) != v]
    if not str(opt.get("weights", "")).endswith("yolov5x.pt"):
        mal.append(f"weights = {opt.get('weights')!r} (desenho: yolov5x.pt)")
    if opt.get("cache"):
        mal.append(f"cache = {opt.get('cache')!r} (desenho: sem cache)")
    certo = Path(dados) / variante / "treino.yaml"
    if not _mesmo(opt.get("data", ""), certo):
        mal.append(f"data = {opt.get('data')} (devia ser {certo})")
    if mal:
        raise ev.ErroAvaliacao(f"{metodo}: o treino não é o do desenho: " + "; ".join(mal) + ". Não gravei nada.")
    r = pd.read_csv(f_res)
    r.columns = [c.strip() for c in r.columns]
    fit = 0.1 * r["metrics/mAP_0.5"] + 0.9 * r["metrics/mAP_0.5:0.95"]      # a do YOLOv5, que escolhe o best.pt
    b = int(fit.idxmax())
    hyp = pasta / "hyp.yaml"
    treino = {"pasta": str(pasta), "opt_yaml_sha256": ev._sha256(f_opt),
              "hyp_yaml_sha256": ev._sha256(hyp) if hyp.is_file() else None,
              **{k: opt.get(k) for k in ("weights", "data", "epochs", "batch_size", "imgsz", "patience", "seed",
                                          "workers", "optimizer")},
              "dados_sha256": ev._sha256(certo),
              "epocas_feitas": int(len(r)), "melhor_epoca": int(r.loc[b, "epoch"]),
              "validacao_na_melhor": {"mAP50": float(r.loc[b, "metrics/mAP_0.5"]),
                                      "mAP50_95": float(r.loc[b, "metrics/mAP_0.5:0.95"]),
                                      "P": float(r.loc[b, "metrics/precision"]), "R": float(r.loc[b, "metrics/recall"])}}
    return treino, pesos


def sha_conjuntos(pasta):
    """O mesmo resumo dos três subconjuntos que o script 24 escreve na marca."""
    return hashlib.sha256("".join(ev._sha256(Path(pasta) / f"{n}_6classes.csv") for n in CONJUNTOS)
                          .encode()).hexdigest()


def ler_geracao(prever, previsoes, metodo, pesos, conjuntos, script24):
    """A marca e o resumo da última corrida do 24: confere os pesos, os subconjuntos e os ficheiros de previsões."""
    pasta = Path(prever) / metodo
    f_marca, f_resumo = pasta / "configuracao.txt", pasta / "resumo.txt"
    for f in (f_marca, f_resumo):
        if not f.is_file():
            raise ev.ErroAvaliacao(f"{metodo}: não encontrei {f}: as previsões foram geradas pelo scripts/24_yolo_prever.py?")
    marca = f_marca.read_text(encoding="utf-8").strip()
    v24 = re.search(r'^VERSAO = "([^"]+)"', Path(script24).read_text(encoding="utf-8"), re.M)
    if not v24 or not marca.startswith(f"24_yolo_prever.py v{v24.group(1)} "):
        raise ev.ErroAvaliacao(f"{metodo}: as previsões foram geradas por outra versão do 24 ({marca[:40]}…).")
    m = re.search(r"pesos (\S+) sha256 ([0-9a-f]{64})", marca)
    if not m or not _mesmo(m.group(1), pesos):
        raise ev.ErroAvaliacao(f"{metodo}: as previsões não foram geradas com {pesos} ({marca}).")
    if m.group(2) != ev._sha256(pesos):
        raise ev.ErroAvaliacao(f"{metodo}: os pesos {pesos} mudaram depois das previsões. Não gravei nada.")
    sc = re.search(r"conjuntos sha256 ([0-9a-f]{16})", marca)
    if not sc or sc.group(1) != sha_conjuntos(conjuntos)[:16]:
        raise ev.ErroAvaliacao(f"{metodo}: os subconjuntos das 6 classes ({conjuntos}) mudaram depois das previsões.")
    no_resumo = dict(re.findall(r"(pred_\S+\.csv) sha256 ([0-9a-f]{16})", f_resumo.read_text(encoding="utf-8")))
    prev = {}
    for c in CONJUNTOS:
        f = Path(previsoes) / f"pred_{metodo}_{c}.csv"
        if not f.is_file():
            raise ev.ErroAvaliacao(f"Faltam as previsões de {metodo} em {c} ({f}).")
        sha = ev._sha256(f)
        if no_resumo.get(f.name) != sha[:16]:
            raise ev.ErroAvaliacao(f"{metodo}: {f.name} não é o da última corrida do 24 (sha256 {sha[:16]}…, no resumo "
                                   f"{no_resumo.get(f.name)}). Não gravei nada.")
        prev[c] = {"ficheiro": str(f), "sha256": sha}
    yolov5 = re.search(r"yolov5 (\S+)", marca)
    return marca, prev, (yolov5.group(1) if yolov5 else "?")


def ler_calibracao(cfg, metodo, f_cal_prev, f_cal_conj):
    p = Path(cfg["caminhos"]["saida"]) / "CAL" / f"{metodo}.json"
    if not p.is_file():
        raise ev.ErroAvaliacao(f"Falta a calibração de {metodo} em CAL ({p}): corre primeiro eval/evaluate.py calibrar "
                               f"--metodo {metodo} … --conjunto-ficheiro {f_cal_conj}.")
    cal = json.loads(p.read_text(encoding="utf-8"))
    if cal.get("metodo") != metodo or cal.get("conjunto") != "CAL" or "pontos_operacao" not in cal:
        raise ev.ErroAvaliacao(f"{p} não é uma calibração de {metodo} em CAL.")
    reg = cal.get("registo", {})
    if reg.get("previsoes_sha256") != ev._sha256(f_cal_prev):
        raise ev.ErroAvaliacao(f"{f_cal_prev} mudou depois da calibração de {metodo}: volta a calibrar em CAL.")
    if reg.get("conjunto_sha256") != ev._sha256(f_cal_conj):
        raise ev.ErroAvaliacao(f"A calibração de {metodo} não foi feita com {f_cal_conj} (as janelas das 6 classes de "
                               f"CAL): foi com {reg.get('conjunto')}. Volta a calibrar com --conjunto-ficheiro.")
    return p, cal


def ja_avaliados(cfg, metodos):
    reg = Path(cfg["caminhos"]["registo_testes"])
    if not reg.is_file():
        return []
    r = pd.read_csv(reg, dtype=str, keep_default_na=False)
    r = r[r["metodo"].isin(metodos) & r["conjunto"].isin(TESTE)]
    return [f"{m} em {c} ({d})" for m, c, d in zip(r["metodo"], r["conjunto"], r["data"])]


# ============================================================================ principal
def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--config", default="configs/avaliacao.yaml")
    a.add_argument("--dados", default="data/derived/yolo", help="pasta dos datasets YOLO (script 23)")
    a.add_argument("--conjuntos", default="data/derived/yolo/conjuntos")
    a.add_argument("--runs", default="outputs/yolo/runs")
    a.add_argument("--prever", default="outputs/yolo/prever")
    a.add_argument("--previsoes", default="outputs/preds")
    a.add_argument("--script24", default="scripts/24_yolo_prever.py")
    a.add_argument("--saida-configs", default="configs")
    a.add_argument("--saida", default="outputs")
    a.add_argument("--forcar", action="store_true", help="reescreve os final_<modelo>.yaml (registar no decisoes.md)")
    args = a.parse_args(argv)
    import yaml
    cfg = ev.ler_config(args.config)
    nomes = list(MODELOS)
    destinos = {n: Path(args.saida_configs) / f"final_{n}.yaml" for n in nomes}
    existem = [str(d) for d in destinos.values() if d.exists()]
    if existem and not args.forcar:
        raise ev.ErroAvaliacao(f"Já existem {', '.join(existem)}: o congelamento faz-se uma vez. Para o refazer, "
                               "--forcar, com linha no decisoes.md.")
    avaliados = ja_avaliados(cfg, nomes)
    if avaliados:
        raise ev.ErroAvaliacao(f"Já há avaliações de teste destes modelos ({'; '.join(avaliados)}): congelar agora já "
                               "não garante nada.")
    conj = Path(args.conjuntos)
    f_conj = {c: conj / f"{c}_6classes.csv" for c in CONJUNTOS}
    for f in f_conj.values():
        if not f.is_file():
            raise ev.ErroAvaliacao(f"Não encontrei {f} (é gerado pelo scripts/23_yolo_dados.py).")
    if not Path(args.script24).is_file():
        raise ev.ErroAvaliacao(f"Não encontrei {args.script24}.")

    # 1. conferir tudo antes de gravar
    prep = {}
    for n, (variante, linha) in MODELOS.items():
        treino, pesos = ler_treino(args.runs, n, variante, args.dados)
        marca, prev, yolov5 = ler_geracao(args.prever, args.previsoes, n, pesos, conj, args.script24)
        p_cal, cal = ler_calibracao(cfg, n, prev["CAL"]["ficheiro"], f_conj["CAL"])
        op = cal["por_op"]["OP_F1"]["micro"]
        numeros = {"AP": _valor(cal.get("AP")), "P": _valor(op.get("P")), "R": _valor(op.get("R")),
                   "F1": _valor(op.get("F1")), "R_loc": _valor(op.get("R_loc"))}
        final = {"metodo": n, "linha_tabela_mestre": linha, "etapa": "experiência B",
                 "modelo": {"pesos": str(pesos), "pesos_sha256": ev._sha256(pesos), "yolov5": yolov5,
                            "pesos_iniciais": "yolov5x.pt (COCO, release v7.0 do YOLOv5)", "treino": treino},
                 "gerado_por": {"script": args.script24, "script_sha256": ev._sha256(args.script24)},
                 "geracao_previsoes": marca,
                 "pontos_operacao": {k: {"tau": float(v["tau"]), "a_min": float(v.get("a_min", 0)),
                                         "regra": v.get("regra", "")} for k, v in cal["pontos_operacao"].items()},
                 "calibracao": {"ficheiro": str(p_cal), "sha256": ev._sha256(p_cal),
                                "commit": cal.get("registo", {}).get("commit"),
                                "previsoes_sha256": cal.get("registo", {}).get("previsoes_sha256"),
                                "conjunto": str(f_conj["CAL"]), "conjunto_sha256": ev._sha256(f_conj["CAL"])},
                 "conjuntos_teste": list(TESTE),
                 "conjuntos_ficheiros": {c: {"ficheiro": str(f_conj[c]), "sha256": ev._sha256(f_conj[c])} for c in TESTE},
                 "previsoes": {c: prev[c] for c in TESTE}}
        prep[n] = (final, numeros)

    # 2. gravar
    git = ev._estado_git(RAIZ)
    agora = datetime.datetime.now()
    for n in nomes:
        final, _ = prep[n]
        final["congelado"] = {"data": agora.isoformat(timespec="seconds"), "script": f"25_yolo_congelar.py v{VERSAO}",
                              "commit": git["commit"], "alteracoes_por_guardar": git["alteracoes_por_guardar"]}
        d = destinos[n]
        d.parent.mkdir(parents=True, exist_ok=True)
        d.write_text(f"# Congelado na experiência B (scripts/25_yolo_congelar.py): não alterar depois da tag {TAG}.\n"
                     + yaml.safe_dump(final, allow_unicode=True, sort_keys=False), encoding="utf-8")

    # 3. resumo, comandos e linha para o decisoes.md
    estado = ("sem git" if git["commit"] is None else
              f"commit {git['commit']}" + (" (com alterações por guardar)" if git["alteracoes_por_guardar"] else ""))
    L = [f"25 — Experiência B: congelamento antes da avaliação única · {agora:%Y-%m-%d %H:%M} · {estado}",
         f"25_yolo_congelar.py v{VERSAO} · sha256 {ev._sha256(__file__)[:16]}",
         "",
         "Modelos congelados (pontos de operação e números de CAL, nas janelas das 6 classes):"]
    partes_linha = []
    for n in nomes:
        final, x = prep[n]
        po, t = final["pontos_operacao"], final["modelo"]["treino"]
        L.append(f"  {n:<10} OP-F1 τ {po['OP_F1']['tau']:.2f} · OP-R90 τ {po['OP_R90']['tau']:.2f} · CAL: AP "
                 f"{_num(x['AP'])} · P {_num(x['P'])} · R {_num(x['R'])} · F1 {_num(x['F1'])} · R_loc {_num(x['R_loc'])}")
        L.append(f"  {'':<10} treino: {t['epocas_feitas']} épocas, melhor {t['melhor_epoca']} (validação mAP50 "
                 f"{_num(t['validacao_na_melhor']['mAP50'])}) · best.pt sha256 {final['modelo']['pesos_sha256'][:16]}")
        partes_linha.append(f"{n} OP-F1 τ {_virgula(po['OP_F1']['tau'], 2)} (CAL: AP {_virgula(x['AP'])}, F1 "
                            f"{_virgula(x['F1'])}, R_loc {_virgula(x['R_loc'])}; treino {t['epocas_feitas']} épocas, "
                            f"melhor {t['melhor_epoca']}; best.pt sha256 {final['modelo']['pesos_sha256'][:16]})")
    gravados = [str(destinos[n]) for n in nomes]
    raiz = RAIZ.as_posix()
    L += ["", f"Gravados: {', '.join(gravados)}", "",
          "A seguir, por esta ordem:",
          f"  1. git add {' '.join(gravados)} scripts/25_yolo_congelar.py tests/test_yolo_congelar.py docs/decisoes.md",
          f"     git commit -m \"Experiência B: congelamento\" && git tag {TAG} && git push origin main {TAG}",
          "  2. avaliação única (no sam_tese; uma vez por modelo e conjunto):"]
    for n in nomes:
        for c in TESTE:
            L.append(f"     python eval/evaluate.py avaliar --metodo {n} --conjunto {c} --pred "
                     f"{prep[n][0]['previsoes'][c]['ficheiro']} --conjunto-ficheiro {f_conj[c]}")
    L.append(f"  3. comparação declarada ({COMPARACAO[0]} − {COMPARACAO[1]}, por janela no OP-F1: F1 em VAL_EX, R_loc em "
             "TEST_BOX):")
    for c in TESTE:
        L.append(f"     python eval/evaluate.py comparar --conjunto {c} --a {COMPARACAO[0]} --b {COMPARACAO[1]}")
    L.append("  4. mAP50 e mAP50-95 por classe (secundário; no ambiente yolo, a partir de ml_models/yolov5):")
    for n in nomes:
        for ref in ("original", "v2"):
            L.append(f"     python val.py --data {raiz}/{Path(args.dados).as_posix()}/{ref}/teste.yaml --weights "
                     f"{raiz}/{Path(args.runs).as_posix()}/{n}/weights/best.pt --task test --imgsz 416 --batch-size 16 "
                     f"--workers 2 --verbose --project {raiz}/outputs/yolo/val --name {n}_{ref} 2>&1 | tee "
                     f"{raiz}/outputs/logs/yolo_map_{n}_{ref}.log")
    linha = (f"- {agora:%Y-%m-%d}: experiência B, congelamento antes do teste (scripts/25_yolo_congelar.py v{VERSAO}, tag "
             f"{TAG}): " + "; ".join(partes_linha) + ". Treinos conferidos com o desenho de 26 out; previsões de CAL, "
             "VAL-EX e TEST-BOX geradas antes de congelar (scripts/24_yolo_prever.py), nas janelas das 6 classes; cada "
             "modelo é avaliado uma só vez, com a comparação declarada (YOLO_V2 − YOLO_ORIG: F1 por janela em VAL-EX e "
             "R_loc@0,3 por janela em TEST-BOX) e o mAP por classe como secundário.")
    L += ["", "Linha para o docs/decisoes.md:", linha]
    texto = "\n".join(L)
    saida = Path(args.saida)
    saida.mkdir(parents=True, exist_ok=True)
    f_resumo = saida / "25_yolo_congelar_resumo.txt"
    f_resumo.write_text(texto + "\n", encoding="utf-8")
    print(texto)
    return {"yaml": [destinos[n] for n in nomes], "resumo": f_resumo, "linha": linha, "texto": texto}


if __name__ == "__main__":
    try:
        main()
    except ev.ErroAvaliacao as e:
        print(f"ERRO: {e}", file=sys.stderr)
        sys.exit(2)
