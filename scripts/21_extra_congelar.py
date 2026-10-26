#!/usr/bin/env python3
"""
21 — Experiência A: congelamento antes da avaliação única (SAM 2.1 Tiny e MedSAM2). Só lê as calibrações de CAL, as
marcas das partes e os ficheiros de previsões já gerados: não corre nenhum método e não avalia nada em VAL_EX nem em
TEST_BOX.

A partir de configs/extra.yaml (os métodos e as comparações declarados em docs/decisoes.md, 24 e 26 out 2026):
  1. Confere, para cada método:
     - a calibração em CAL (outputs/eval/CAL/<método>.json), feita com as previsões de CAL que estão no disco;
     - as previsões de VAL_EX e TEST_BOX;
     - que as previsões de CAL, VAL_EX e TEST_BOX foram geradas pela mesma versão do sam2_prop.py, com o YAML do
       modelo tal como está agora e com o mesmo prompt (a marca configuracao.txt das partes);
     - que o método ainda não foi avaliado em nenhum conjunto de teste (outputs/test_runs.csv).
  2. Grava configs/final_<método>.yaml, no formato do A6: o script e o YAML (sha256), o checkpoint e o config do
     modelo (sha256), os parâmetros, os pontos de operação fixados em CAL (OP-F1 e OP-R90), a calibração (sha256 e
     commit) e as previsões de teste (sha256). Não reescreve um final_<método>.yaml que já exista (só com --forcar,
     com linha no decisoes.md).
  3. Confere que a referência (o SAM2_L12, congelado no A6 e avaliado no A7) tem o final_<referência>.yaml e as
     avaliações de VAL_EX e TEST_BOX, e escreve no resumo os comandos da avaliação única, os das comparações declaradas
     e a linha para o decisoes.md.
Se alguma coisa não bate, não grava nada. Depois: git commit, git tag extra-congelado e git push.

Uso, a partir da pasta do projeto:
    python scripts/21_extra_congelar.py
Gera configs/final_<método>.yaml (um por método) e outputs/21_extra_congelar_resumo.txt.

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
OPS = ("OP_F1", "OP_R90")


def _caminho(p):
    """Caminhos relativos a partir da pasta do projeto."""
    p = Path(p)
    return p if p.is_absolute() else RAIZ / p


def _valor(x):
    return x.get("valor") if isinstance(x, dict) else x


def _num(x, casas=3):
    return "—" if x is None else f"{x:.{casas}f}"


def _virgula(x, casas=3):
    return "—" if x is None else f"{x:.{casas}f}".replace(".", ",")


def ler_calibracao(cfg, metodo):
    p = Path(cfg["caminhos"]["saida"]) / "CAL" / f"{metodo}.json"
    if not p.is_file():
        raise ev.ErroAvaliacao(f"Falta a calibração de {metodo} em CAL ({p}): corre primeiro "
                               f"eval/evaluate.py calibrar --metodo {metodo}.")
    cal = json.loads(p.read_text(encoding="utf-8"))
    if cal.get("metodo") != metodo or cal.get("conjunto") != "CAL" or "pontos_operacao" not in cal:
        raise ev.ErroAvaliacao(f"{p} não é uma calibração de {metodo} em CAL.")
    return p, cal


def marca_das_partes(conf_modelo, prompt, conjunto):
    """A marca configuracao.txt que o sam2_prop.py deixa na pasta das partes de cada conjunto."""
    sufixo = "" if prompt == "caixa" else f"_{prompt}"
    p = Path(f"{conf_modelo['partes']}{sufixo}") / conjunto / "configuracao.txt"
    if not p.is_file():
        raise ev.ErroAvaliacao(f"Não encontrei {p}: as previsões de {conjunto} foram geradas pelo sam2_prop.py?")
    return p.read_text(encoding="utf-8").strip()


def ja_avaliados(cfg, metodos, conjuntos):
    reg = Path(cfg["caminhos"]["registo_testes"])
    if not reg.is_file():
        return []
    r = pd.read_csv(reg, dtype=str, keep_default_na=False)
    r = r[r["metodo"].isin(metodos) & r["conjunto"].isin(conjuntos)]
    return [f"{m} em {c} ({d})" for m, c, d in zip(r["metodo"], r["conjunto"], r["data"])]


def preparar_metodo(cfg, e, m, conjuntos):
    """Confere tudo o que diz respeito a um método e devolve o conteúdo do final_<método>.yaml (sem o bloco
    'congelado') e os números de CAL para o resumo. Não grava nada."""
    metodo = m["metodo"]
    p_cal, cal = ler_calibracao(cfg, metodo)
    prev_dir = Path(e["previsoes"])
    f_cal = prev_dir / f"pred_{metodo}_CAL.csv"
    if not f_cal.is_file():
        raise ev.ErroAvaliacao(f"Faltam as previsões de {metodo} em CAL ({f_cal}).")
    usado = cal.get("registo", {}).get("previsoes_sha256")
    if usado and usado != ev._sha256(f_cal):
        raise ev.ErroAvaliacao(f"{f_cal} mudou depois da calibração de {metodo}: volta a calibrar em CAL.")
    prev = {}
    for c in conjuntos:
        f = prev_dir / f"pred_{metodo}_{c}.csv"
        if not f.is_file():
            raise ev.ErroAvaliacao(f"Faltam as previsões de {metodo} em {c} ({f}).")
        prev[c] = {"ficheiro": str(f), "sha256": ev._sha256(f)}
    fonte = {}
    for chave in ("script", "config"):
        f = _caminho(m[chave])
        if not f.is_file():
            raise ev.ErroAvaliacao(f"{metodo}: não encontrei {m[chave]}.")
        fonte.update({chave: m[chave], f"{chave}_sha256": ev._sha256(f)})
    conf_modelo = ev.ler_config(_caminho(m["config"]))
    prompt = m.get("parametros", {}).get("prompt", "caixa")
    marcas = {c: marca_das_partes(conf_modelo, prompt, c) for c in ["CAL"] + list(conjuntos)}
    if len(set(marcas.values())) != 1:
        txt = "; ".join(f"{c}: {v}" for c, v in marcas.items())
        raise ev.ErroAvaliacao(f"{metodo}: as previsões dos conjuntos não foram geradas da mesma maneira "
                               f"(configuracao.txt das partes): {txt}")
    marca = marcas["CAL"]
    esperado = f"{Path(m['config']).name} sha256 {fonte['config_sha256']}"
    if esperado not in marca or not marca.endswith(f"prompt {prompt}"):
        raise ev.ErroAvaliacao(f"{metodo}: as previsões foram geradas com outro YAML ou outro prompt ({marca}); "
                               f"o YAML atual é {m['config']} (sha256 {fonte['config_sha256'][:16]}).")
    modelo = {}
    ck = Path(conf_modelo["checkpoint"])
    if not ck.is_file():
        raise ev.ErroAvaliacao(f"{metodo}: não encontrei o checkpoint {ck}.")
    modelo.update({"checkpoint": str(conf_modelo["checkpoint"]), "checkpoint_sha256": ev._sha256(ck),
                   "modelo_cfg": conf_modelo["modelo_cfg"]})
    if conf_modelo.get("modelo_cfg_fora_do_pacote"):
        modelo["modelo_cfg_sha256"] = ev._sha256(conf_modelo["modelo_cfg"])
    op = cal["por_op"]["OP_F1"]["micro"]
    numeros = {"AP": _valor(cal.get("AP")), "P": _valor(op.get("P")), "R": _valor(op.get("R")),
               "F1": _valor(op.get("F1")), "R_loc": _valor(op.get("R_loc"))}
    final = {"metodo": metodo, "linha_tabela_mestre": m["linha"], "etapa": e.get("etapa", "experiência A"),
             "gerado_por": fonte, "geracao_previsoes": marca, "modelo": modelo,
             "parametros": m.get("parametros", {}),
             "pontos_operacao": {k: {"tau": float(v["tau"]), "a_min": float(v.get("a_min", 0)),
                                     "regra": v.get("regra", "")} for k, v in cal["pontos_operacao"].items()},
             "calibracao": {"ficheiro": str(p_cal), "sha256": ev._sha256(p_cal),
                            "commit": cal.get("registo", {}).get("commit"),
                            "previsoes_sha256": cal.get("registo", {}).get("previsoes_sha256")},
             "conjuntos_teste": list(conjuntos), "previsoes": prev}
    return final, numeros


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--config", default="configs/avaliacao.yaml")
    a.add_argument("--config-extra", default="configs/extra.yaml")
    a.add_argument("--saida-configs", default="configs", help="onde gravar os final_<método>.yaml")
    a.add_argument("--saida", default="outputs")
    a.add_argument("--forcar", action="store_true", help="reescreve os final_<método>.yaml (registar no decisoes.md)")
    args = a.parse_args(argv)
    import yaml
    cfg = ev.ler_config(args.config)
    e = ev.ler_config(args.config_extra)
    metodos = e["metodos"]
    nomes = [m["metodo"] for m in metodos]
    conjuntos = list(e.get("conjuntos", ["VAL_EX", "TEST_BOX"]))
    ref = e["referencia"]
    if ref in nomes:
        raise ev.ErroAvaliacao(f"A referência ({ref}) já está congelada: não pode estar na lista de métodos.")
    destinos = {n: Path(args.saida_configs) / f"final_{n}.yaml" for n in nomes}
    existem = [str(d) for d in destinos.values() if d.exists()]
    if existem and not args.forcar:
        raise ev.ErroAvaliacao(f"Já existem {', '.join(existem)}: o congelamento faz-se uma vez. Para o refazer, "
                               "--forcar, com linha no decisoes.md.")
    avaliados = ja_avaliados(cfg, nomes, conjuntos)
    if avaliados:
        raise ev.ErroAvaliacao(f"Já há avaliações de teste destes métodos ({'; '.join(avaliados)}): congelar agora "
                               "já não garante nada.")
    # referência: congelada no A6 e avaliada no A7
    f_ref = Path(args.saida_configs) / f"final_{ref}.yaml"
    if not f_ref.is_file():
        raise ev.ErroAvaliacao(f"Não encontrei {f_ref} (a referência congelada no A6).")
    for c in conjuntos:
        if not (Path(cfg["caminhos"]["saida"]) / c / f"{ref}.json").is_file():
            raise ev.ErroAvaliacao(f"A referência {ref} não tem a avaliação de {c}: as comparações não correm.")
    for par in e.get("comparacoes", []):
        for x in par:
            if x not in nomes and x != ref:
                raise ev.ErroAvaliacao(f"A comparação {par} usa {x}, que não é um dos métodos nem a referência.")

    # 1. conferir tudo antes de gravar
    preparados = {m["metodo"]: preparar_metodo(cfg, e, m, conjuntos) for m in metodos}

    # 2. gravar
    git = ev._estado_git(RAIZ)
    agora = datetime.datetime.now()
    tag = e.get("tag", "extra-congelado")
    for m in metodos:
        final, _ = preparados[m["metodo"]]
        final["referencia"] = {"metodo": ref, "final": str(f_ref), "sha256": ev._sha256(f_ref)}
        final["congelado"] = {"data": agora.isoformat(timespec="seconds"), "script": f"21_extra_congelar.py v{VERSAO}",
                              "commit": git["commit"], "alteracoes_por_guardar": git["alteracoes_por_guardar"]}
        d = destinos[m["metodo"]]
        d.parent.mkdir(parents=True, exist_ok=True)
        d.write_text(f"# Congelado na {final['etapa']} (scripts/21_extra_congelar.py): não alterar depois da tag "
                     f"{tag}.\n" + yaml.safe_dump(final, allow_unicode=True, sort_keys=False), encoding="utf-8")

    # 3. resumo, comandos e linha para o decisoes.md
    estado = ("sem git" if git["commit"] is None else
              f"commit {git['commit']}" + (" (com alterações por guardar)" if git["alteracoes_por_guardar"] else ""))
    L = [f"21 — Experiência A: congelamento antes da avaliação única · {agora:%Y-%m-%d %H:%M} · {estado}",
         f"21_extra_congelar.py v{VERSAO} · sha256 {ev._sha256(__file__)[:16]} · {Path(args.config_extra).name} "
         f"sha256 {ev._sha256(args.config_extra)[:16]}",
         f"referência: {ref} (congelado no A6: {f_ref.name} sha256 {ev._sha256(f_ref)[:16]}; avaliado no A7)",
         "",
         "Métodos congelados (pontos de operação e números de CAL):"]
    partes_linha = []
    for m in metodos:
        final, n = preparados[m["metodo"]]
        po = final["pontos_operacao"]
        L.append(f"  {m['metodo']:<12} {m['linha']:<10} OP-F1 τ {po['OP_F1']['tau']:.2f} · OP-R90 τ "
                 f"{po['OP_R90']['tau']:.2f} · CAL: AP {_num(n['AP'])} · P {_num(n['P'])} · R {_num(n['R'])} · "
                 f"F1 {_num(n['F1'])} · R_loc {_num(n['R_loc'])}")
        L.append(f"  {'':<12} previsões: {final['geracao_previsoes']}")
        partes_linha.append(f"{m['metodo']} OP-F1 τ {_virgula(po['OP_F1']['tau'], 2)} (CAL: AP {_virgula(n['AP'])}, "
                            f"F1 {_virgula(n['F1'])}, R_loc {_virgula(n['R_loc'])})")
    gravados = ", ".join(str(destinos[n]) for n in nomes)
    L += ["", f"Gravados: {gravados}",
          "",
          "A seguir, por esta ordem:",
          f"  1. git add {args.config_extra} {gravados.replace(', ', ' ')} scripts/21_extra_congelar.py "
          "tests/test_extra_congelar.py docs/decisoes.md",
          f"     git commit -m \"Experiência A: congelamento\" && git tag {tag} && git push origin main {tag}",
          "  2. avaliação única (uma vez por método e conjunto):"]
    prev_dir = Path(e["previsoes"])
    for n in nomes:
        for c in conjuntos:
            L.append(f"     python eval/evaluate.py avaliar --metodo {n} --conjunto {c} "
                     f"--pred {prev_dir / f'pred_{n}_{c}.csv'}")
    L.append("  3. comparações declaradas (diferença pareada por janela no OP-F1: F1 em VAL_EX, R_loc em TEST_BOX):")
    for a_, b_ in e.get("comparacoes", []):
        for c in conjuntos:
            L.append(f"     python eval/evaluate.py comparar --conjunto {c} --a {a_} --b {b_}")
    linha = (f"- {agora:%Y-%m-%d}: experiência A, congelamento antes do teste (scripts/21_extra_congelar.py v{VERSAO}, "
             f"tag {tag}): " + "; ".join(partes_linha) + ". As previsões de VAL-EX e TEST-BOX foram geradas antes de "
             "congelar, com o mesmo sam2_prop.py e os mesmos YAML de CAL; cada método é avaliado uma só vez, com as "
             "comparações declaradas a 26 out.")
    L += ["", "Linha para o docs/decisoes.md:", linha]
    texto = "\n".join(L)
    saida = Path(args.saida)
    saida.mkdir(parents=True, exist_ok=True)
    f_resumo = saida / "21_extra_congelar_resumo.txt"
    f_resumo.write_text(texto + "\n", encoding="utf-8")
    print(texto)
    return {"yaml": list(destinos.values()), "resumo": f_resumo, "linha": linha}


if __name__ == "__main__":
    try:
        main()
    except ev.ErroAvaliacao as e:
        print(f"ERRO: {e}", file=sys.stderr)
        sys.exit(2)
