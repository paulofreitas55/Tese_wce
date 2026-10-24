#!/usr/bin/env python3
"""
17b — Guião C: contexto da ferramenta de revisão (tools/rever.py v1.3). Para cada janela da lista da revisão, as caixas
dos OUTROS frames seguidos (1 ≤ |offset| ≤ 12), só com as coordenadas:
  original  a caixa da anotação original (desenhada pelo Rodrigo), com as regras do avaliador (eval/evaluate.py)
  modelo    a caixa do SAM 2 na anotação expandida v1, em todos os frames seguidos, qualquer que seja a confiança
            (se só entrassem as de confiança alta, a falta da caixa denunciava a confiança)
Nunca entra nada dos frames que estão na lista: nem a caixa original (nos controlos com caixa original ficaria à vista e
deixavam de servir) nem a do modelo (é a própria caixa a rever, que a ferramenta já lê da lista). Não tem score, nível
nem tipo de item: a revisão continua às cegas e a ferramenta continua sem abrir a chave nem o expanded_v1.csv.

Lê a lista (itens_v1.csv), data/derived/expanded_v1.csv (confirma o sha256 do A8) e data/derived/annotations_v1.csv.
Grava, só de leitura (e sem reescrever um ficheiro diferente, salvo --forcar):
  data/derived/guiao_c/contexto_v1.csv    split, caso, seed, frame, fonte (original ou modelo), box_x1..box_y2 (0–1)
e o resumo em outputs/17b_c_contexto_resumo.txt, com os sha256 e a linha para o docs/decisoes.md.

Uso, a partir da pasta do projeto, depois do source …/inicio.sh:
    python scripts/17b_c_contexto.py
    python scripts/17b_c_contexto.py --itens data/derived/guiao_c/piloto_v1.csv --saida data/derived/guiao_c/contexto_piloto_v1.csv
Versão 1.0 (24 out 2026).
"""
import argparse
import datetime
import importlib.util
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("c_itens_17", RAIZ / "scripts" / "17_c_itens.py")
ci = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ci)                       # reutiliza a leitura do script 17 (não grava nada ao importar)
ev = ci.ev

VERSAO = "1.0"
K3 = ["split", "caso", "seed"]
K4 = K3 + ["frame"]
CAIXA = ["box_x1", "box_y1", "box_x2", "box_y2"]
COL_CONTEXTO = K4 + ["fonte"] + CAIXA


def ler_lista(caminho):
    p = Path(caminho)
    if not p.is_file():
        raise ev.ErroAvaliacao(f"Não encontrei a lista {p} (é gravada pelo scripts/17_c_itens.py).")
    t = pd.read_csv(p, dtype={"caso": str, "split": str})
    falta = [c for c in ci.COL_ITENS if c not in t.columns]
    if falta:
        raise ev.ErroAvaliacao(f"{p} não tem as colunas {falta}: tem de ser uma lista do script 17.")
    t["caso"] = t["caso"].str.strip().str.zfill(4)
    for c in ("seed", "frame"):
        t[c] = t[c].astype(int)
    return t


def contexto(lista, expandida, ref, alcance):
    """Linhas do contexto: caixas originais e do modelo dos frames seguidos que não estão na lista."""
    janelas = set(zip(lista["split"], lista["caso"], lista["seed"]))
    na_lista = set(zip(lista["split"], lista["caso"], lista["seed"], lista["frame"]))

    def fora_da_lista(t):
        k3 = [k in janelas for k in zip(t["split"], t["caso"], t["seed"])]
        k4 = [k not in na_lista for k in zip(t["split"], t["caso"], t["seed"], t["frame"])]
        off = (t["frame"] - t["seed"]).abs()
        return t[pd.Series(k3, index=t.index) & pd.Series(k4, index=t.index) & (off >= 1) & (off <= alcance)]

    m = fora_da_lista(expandida[expandida["tem_caixa"]])
    m = m[K4 + CAIXA].assign(fonte="modelo")
    o = fora_da_lista(ref[ref["caixa"] == "caixa"])
    o = o[K4 + ["rx1", "ry1", "rx2", "ry2"]].rename(columns=dict(zip(["rx1", "ry1", "rx2", "ry2"], CAIXA))).assign(fonte="original")
    t = pd.concat([o, m], ignore_index=True)[COL_CONTEXTO]
    for c in CAIXA:
        t[c] = t[c].astype(float).clip(0, 1).round(6)
    t["seed"], t["frame"] = t["seed"].astype(int), t["frame"].astype(int)
    t = t.sort_values(K4 + ["fonte"], kind="mergesort").reset_index(drop=True)
    escondidas = ref[(ref["caixa"] == "caixa") & pd.Series([k in na_lista for k in zip(ref["split"], ref["caso"], ref["seed"], ref["frame"])],
                                                          index=ref.index)]
    return t, len(janelas), len(escondidas)


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--config", default="configs/avaliacao.yaml")
    a.add_argument("--itens", default="data/derived/guiao_c/itens_v1.csv")
    a.add_argument("--expandida", default="data/derived/expanded_v1.csv")
    a.add_argument("--saida", default="data/derived/guiao_c/contexto_v1.csv")
    a.add_argument("--resumo", default="outputs/17b_c_contexto_resumo.txt")
    a.add_argument("--alcance", type=int, default=12, help="frames seguidos para cada lado do seed (SAM2_L12: 12)")
    a.add_argument("--sha-expandida", default=ci.SHA_A8, help="início do sha256 esperado do expanded_v1.csv ('' não confirma)")
    a.add_argument("--forcar", action="store_true", help="substitui um contexto já gravado com outro conteúdo")
    args = a.parse_args(argv)
    cfg = ev.ler_config(args.config)
    lista = ler_lista(args.itens)
    sha_e = ev._sha256(args.expandida) if Path(args.expandida).is_file() else ""
    e = ci.ler_expandida(args.expandida)
    if args.sha_expandida and not sha_e.startswith(args.sha_expandida):
        raise ev.ErroAvaliacao(f"{args.expandida} não é o do A8: sha256 {sha_e[:16]}…, esperado {args.sha_expandida}….")
    janelas = sorted(set(zip(lista["split"], lista["caso"], lista["seed"])))
    sem = [k for k in janelas if k not in set(zip(e["split"], e["caso"], e["seed"]))]
    if sem:
        raise ev.ErroAvaliacao(f"{len(sem)} janelas da lista não estão em {args.expandida} (ex.: {ev._txt(sem[0])}).")
    ref, _ = ci.referencia(cfg, [ev.chave(*k) for k in janelas])
    t, n_jan, n_esc = contexto(lista, e, ref, args.alcance)
    destino = Path(args.saida)
    estado = ci.gravar_congelados([(destino, t.to_csv(index=False, float_format="%.6f"))], args.forcar)[0]
    sha_c, sha_l = ev._sha256(destino), ev._sha256(args.itens)

    git = ev._estado_git(RAIZ)
    n_o, n_m = int((t["fonte"] == "original").sum()), int((t["fonte"] == "modelo").sum())
    L = [f"17b — Guião C: contexto da ferramenta de revisão · {datetime.datetime.now():%Y-%m-%d %H:%M} · "
         + (f"commit {git['commit']}" if git["commit"] else "sem git")
         + (" (com alterações por guardar!)" if git["alteracoes_por_guardar"] else ""),
         f"17b_c_contexto.py v{VERSAO} · lista {args.itens} (sha256 {sha_l[:16]}) · {args.expandida} (sha256 {sha_e[:16]})",
         "",
         f"{n_jan} janelas da lista · frames seguidos até {args.alcance} para cada lado do seed",
         f"  caixas originais (Rodrigo) nos outros frames: {ci._n(n_o)} · nos frames da lista ficam escondidas: {ci._n(n_esc)}",
         f"  caixas do modelo nos outros frames (todas as confianças, sem score): {ci._n(n_m)}",
         f"{destino}: {estado} · sha256 {sha_c}",
         "  Sem nenhum frame da lista, sem score, nível nem tipo de item: não é preciso abri-lo.",
         ""]
    linha = (f"- {datetime.date.today().isoformat()}: Guião C, contexto da ferramenta v1.3 (scripts/17b_c_contexto.py "
             f"v{VERSAO}, commit {git['commit'] or 'sem git'}): {destino.name} sha256 {sha_c[:16]}, com as caixas originais "
             f"({ci._n(n_o)}) e do modelo ({ci._n(n_m)}, todas as confianças) dos outros frames seguidos (|offset| ≤ "
             f"{args.alcance}) das {n_jan} janelas da lista ({Path(args.itens).name} sha256 {sha_l[:16]}); nenhum frame da "
             f"lista entra (as {ci._n(n_esc)} caixas originais desses frames ficam escondidas) e não há score, nível nem "
             "tipo de item.")
    L += ["Linha para o docs/decisoes.md:", linha]
    texto = "\n".join(L)
    rs = Path(args.resumo)
    rs.parent.mkdir(parents=True, exist_ok=True)
    rs.write_text(texto + "\n", encoding="utf-8")
    print(texto)
    return {"contexto": t, "estado": estado, "sha256": sha_c, "escondidas": n_esc, "linha": linha, "texto": texto}


if __name__ == "__main__":
    try:
        main()
    except ev.ErroAvaliacao as e:
        print(f"ERRO: {e}", file=sys.stderr)
        sys.exit(2)
