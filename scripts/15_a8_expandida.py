#!/usr/bin/env python3
"""
15 — A8: anotação expandida v1 (Guião A). Junta num só ficheiro as previsões do método principal congelado no A6
(SAM2_L12) nas janelas de ALL_SEEDBOX, com o nível de confiança de cada frame, e resume o ganho (por janela, distância,
classe e split). Não corre o SAM 2 nem avalia nada.

Lê:
  configs/final_SAM2_L12.yaml                  o método congelado no A6: τ e A_min do OP-F1, L_max, níveis do A8
  outputs/preds/pred_SAM2_L12_ALL_SEEDBOX.csv  as previsões do A8 (methods/sam2_prop.py --conjunto ALL_SEEDBOX)
  sets/ALL_SEEDBOX.json                        as janelas (confirma que estão todas e que não sobra nenhuma)
  data/derived/janelas_v1.csv                  a classe de cada janela
  data/derived/annotations_v1.csv              as caixas originais (separa os frames novos dos que já tinham caixa)
  outputs/eval/VAL_EX/SAM2_L12.json            se existir: a precisão do OP-F1 em VAL-EX, para o resumo

Grava:
  data/derived/expanded_v1.csv   um frame seguido por linha (1 ≤ |offset| ≤ L_max; o seed fica de fora), com as colunas
                                 do guião (caso, split, seed, frame, score, box_x1..box_y2, nivel, metodo, versao_config)
                                 e ainda offset, area_px e caixa_original (1 se o frame já tinha caixa original)
      nivel: alta  = score ≥ score mínimo do nível alta (o τ do OP-F1) e área ≥ A_min do OP-F1
             media = só se o A6 a tiver definido (decidido a 15 out: sem nível média)
             baixa = o resto
  outputs/15_a8_resumo.txt       o que aparece no ecrã
  outputs/15_a8_por_classe.csv   o resumo por classe
As máscaras ficam onde o A8 as gravou (outputs/masks/ALL_SEEDBOX/, uma por janela); aqui só se contam.
Se o expanded_v1.csv já existir com outro conteúdo, não é reescrito: usa --forcar ou grava uma versão nova (--saida).

Uso, a partir da pasta do projeto:
    python scripts/15_a8_expandida.py
"""
import argparse
import datetime
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "eval"))
import evaluate as ev  # noqa: E402

VERSAO = "1.0"
CAIXA = ["box_x1", "box_y1", "box_x2", "box_y2"]
FAIXAS = ((1, 3), (4, 6), (7, 12), (13, 10 ** 6))


def ler_final(caminho):
    """Método principal congelado: τ e A_min do OP-F1, L_max e níveis do A8 (decisoes_A6)."""
    f = ev.ler_config(caminho)
    niveis = (f.get("decisoes_A6") or {}).get("niveis_A8")
    if not niveis or "alta" not in niveis:
        raise ev.ErroAvaliacao(f"{caminho} não tem decisoes_A6.niveis_A8: tem de ser o YAML final do método principal (A6).")
    op = f["pontos_operacao"]["OP_F1"]
    media = niveis.get("media")
    return {"metodo": f["metodo"], "tau": float(op["tau"]), "a_min": float(op.get("a_min", 0) or 0),
            "l_max": int(f["parametros"]["l_max"]), "alta": float(niveis["alta"]["score_minimo"]),
            "media": None if media is None else float(media["score_minimo"])}


def caixas_originais(cfg, janelas):
    """{(split, caso, seed): frames com caixa original válida (box_src medico_seed ou medico_vizinho, 4 coordenadas e
    x2 > x1, y2 > y1, a regra do avaliador), sem o seed}."""
    alvo, casos, out = set(janelas), {k[1] for k in janelas}, {k: set() for k in janelas}
    for b in pd.read_csv(cfg["caminhos"]["anotacoes"], dtype=str, keep_default_na=False,
                         usecols=["split", "caso", "seed", "frame", "offset", "box_src"] + CAIXA, chunksize=200_000):
        b = b[b["caso"].isin(casos) & b["box_src"].isin(["medico_seed", "medico_vizinho"])]
        if not len(b):
            continue
        v = b[CAIXA].apply(pd.to_numeric, errors="coerce")
        ok = (v.notna().all(axis=1) & (v["box_x2"] > v["box_x1"]) & (v["box_y2"] > v["box_y1"])
              & (pd.to_numeric(b["offset"], errors="coerce") != 0))
        for sp, caso, seed, fr in zip(b.loc[ok, "split"], b.loc[ok, "caso"], b.loc[ok, "seed"], b.loc[ok, "frame"]):
            k = (sp, caso, int(seed))
            if k in alvo:
                out[k].add(int(fr))
    return out


def classes(cfg, janelas):
    jt = pd.read_csv(cfg["caminhos"]["janelas"], dtype=str, keep_default_na=False)
    principais = set(cfg["estratos"]["classes_principais"])
    m = {ev.chave(s, c, e): cat for s, c, e, cat in zip(jt["split"], jt["caso"], jt["seed"], jt["categoria"])}
    falta = [k for k in janelas if k not in m]
    if falta:
        raise ev.ErroAvaliacao(f"{len(falta)} janelas não estão em {cfg['caminhos']['janelas']} (ex.: {ev._txt(falta[0])}).")
    return {k: m[k] if m[k] in principais else "outras" for k in janelas}


def ler_previsoes(caminho, janelas, l_max):
    p = pd.read_csv(caminho, dtype={"caso": str, "split": str}, keep_default_na=False)
    falta = [c for c in ["split", "caso", "seed", "frame", "score"] + CAIXA if c not in p.columns]
    if falta:
        raise ev.ErroAvaliacao(f"{caminho}: faltam as colunas {falta}.")
    p["caso"] = p["caso"].str.zfill(4)
    p["seed"], p["frame"] = p["seed"].astype(int), p["frame"].astype(int)
    k = set(zip(p["split"], p["caso"], p["seed"]))
    faltam, sobram = [j for j in janelas if j not in k], k - set(janelas)
    if faltam:
        raise ev.ErroAvaliacao(f"{len(faltam)} janelas de ALL_SEEDBOX sem previsões em {caminho} (ex.: {ev._txt(faltam[0])}).")
    if sobram:
        raise ev.ErroAvaliacao(f"{caminho} tem {len(sobram)} janelas que não são de ALL_SEEDBOX (ex.: {ev._txt(sorted(sobram)[0])}).")
    p["offset"] = p["frame"] - p["seed"]
    p = p[(p["offset"] != 0) & (p["offset"].abs() <= l_max)].copy()
    p["score"] = pd.to_numeric(p["score"], errors="coerce")
    if p["score"].isna().any() or ((p["score"] < 0) | (p["score"] > 1)).any():
        raise ev.ErroAvaliacao(f"{caminho}: há scores em falta ou fora de [0, 1].")
    p["area_px"] = pd.to_numeric(p["area_px"], errors="coerce").fillna(0) if "area_px" in p.columns else 0.0
    p["tem_caixa"] = (p[CAIXA].astype(str) != "").all(axis=1).to_numpy()
    return p


def _md(v):
    v = np.asarray(v, dtype=float)
    if not len(v):
        return "—"
    return (f"mediana {np.median(v):g} (IQR {np.percentile(v, 25):g}–{np.percentile(v, 75):g}) · média "
            f"{v.mean():.2f}").replace(".", ",")


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--config", default="configs/avaliacao.yaml")
    a.add_argument("--final", default="configs/final_SAM2_L12.yaml")
    a.add_argument("--pred", default="outputs/preds/pred_SAM2_L12_ALL_SEEDBOX.csv")
    a.add_argument("--conjunto", default="ALL_SEEDBOX")
    a.add_argument("--mascaras", default="outputs/masks/ALL_SEEDBOX")
    a.add_argument("--saida", default="data/derived/expanded_v1.csv")
    a.add_argument("--saida-resumo", default="outputs")
    a.add_argument("--forcar", action="store_true", help="substitui um expanded_v1.csv com outro conteúdo")
    args = a.parse_args(argv)
    cfg, fin = ev.ler_config(args.config), ler_final(args.final)
    if abs(fin["alta"] - fin["tau"]) > 1e-9:
        raise ev.ErroAvaliacao(f"O nível alta ({fin['alta']:g}) não é o τ do OP-F1 ({fin['tau']:g}) em {args.final}.")
    janelas = ev.ler_conjunto(Path(cfg["caminhos"]["conjuntos"]) / f"{args.conjunto}.json")
    if not Path(args.pred).is_file():
        raise ev.ErroAvaliacao(f"Não encontrei {args.pred} (corre primeiro o A8: methods/sam2_prop.py --conjunto "
                               f"{args.conjunto}).")
    p = ler_previsoes(args.pred, janelas, fin["l_max"])
    orig = caixas_originais(cfg, janelas)
    classe = classes(cfg, janelas)

    alta = (p["score"] >= fin["alta"]).to_numpy()
    if fin["a_min"]:
        alta &= (p["area_px"] >= fin["a_min"]).to_numpy()
    media = (~alta & (p["score"] >= fin["media"]).to_numpy()) if fin["media"] is not None else np.zeros(len(p), bool)
    chaves = list(zip(p["split"], p["caso"], p["seed"]))
    p["caixa_original"] = [int(f in orig[k]) for k, f in zip(chaves, p["frame"])]
    versao = f"final_{fin['metodo']}.yaml sha256 {ev._sha256(args.final)[:12]}"
    out = pd.DataFrame({"caso": p["caso"], "split": p["split"], "seed": p["seed"], "frame": p["frame"],
                        "score": p["score"].round(6), **{c: p[c] for c in CAIXA},
                        "nivel": np.where(alta, "alta", np.where(media, "media", "baixa")), "metodo": fin["metodo"],
                        "versao_config": versao, "offset": p["offset"], "area_px": p["area_px"],
                        "caixa_original": p["caixa_original"]}).sort_values(["split", "caso", "seed", "frame"])

    # gravar sem apagar uma versão diferente
    destino = Path(args.saida)
    destino.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".csv", dir=destino.parent, delete=False, encoding="utf-8") as t:
        out.to_csv(t, index=False)
    tmp = Path(t.name)
    novo = ev._sha256(tmp)
    if destino.is_file() and ev._sha256(destino) != novo and not args.forcar:
        tmp.unlink()
        raise ev.ErroAvaliacao(f"{destino} já existe com outro conteúdo. Se é para o substituir, usa --forcar; se é uma "
                               "versão nova, grava-a com --saida data/derived/expanded_v2.csv.")
    estado_fich = "igual ao que já existia" if destino.is_file() and ev._sha256(destino) == novo else "gravado"
    tmp.replace(destino)

    # resumo
    e = out.assign(alta=out["nivel"] == "alta", novo=(out["nivel"] == "alta") & (out["caixa_original"] == 0),
                   sem_caixa=(out["nivel"] == "alta") & ~p.loc[out.index, "tem_caixa"].to_numpy())
    pj = e.groupby(["split", "caso", "seed"]).agg(frames=("frame", "size"), alta=("alta", "sum"), novos=("novo", "sum"))
    pj["originais"] = [len(orig[k]) for k in pj.index]
    pj["antes"] = 1 + pj["originais"]                                       # o seed mais as caixas originais
    pj["depois"] = pj["antes"] + pj["novos"]
    pj["classe"] = [classe[k] for k in pj.index]
    n_alta, n_novos = int(e["alta"].sum()), int(e["novo"].sum())
    git = ev._estado_git(RAIZ)
    estado = ("sem git" if git["commit"] is None else
              f"commit {git['commit']}" + (" (com alterações por guardar)" if git["alteracoes_por_guardar"] else ""))
    L = [f"15 — A8: anotação expandida v1 · {datetime.datetime.now():%Y-%m-%d %H:%M} · {estado}",
         f"15_a8_expandida.py v{VERSAO} · sha256 {ev._sha256(__file__)[:16]} · {versao} · previsões {args.pred} sha256 "
         f"{ev._sha256(args.pred)[:16]}"]
    a8 = Path(args.pred).with_name(f"pred_SAM2_{args.conjunto}_resumo.txt")
    if a8.is_file():
        L += ["A8: " + linha for linha in a8.read_text(encoding="utf-8").splitlines()[:2]]
    L += ["",
          (f"{fin['metodo']}: nível alta = score ≥ {fin['alta']:g}" + (f" e área ≥ {fin['a_min']:g} px" if fin["a_min"] else "")
           + (f" · média = {fin['media']:g} ≤ score < {fin['alta']:g}" if fin["media"] is not None else
              " · sem nível média (A6)") + f" · frames seguidos até {fin['l_max']} para cada lado").replace(".", ","),
          f"{destino}: {len(out)} frames de {len(pj)} janelas, {estado_fich} (sha256 {novo[:16]})",
          "",
          (f"Nível alta: {n_alta} frames ({100 * n_alta / max(1, len(out)):.1f} % dos seguidos)"
           + (f" · média: {int((out['nivel'] == 'media').sum())}" if fin["media"] is not None else "")
           + f" · baixa: {int((out['nivel'] == 'baixa').sum())}").replace(".", ","),
          f"  dos frames alta: {n_alta - n_novos} já tinham caixa original · {n_novos} são novos · "
          f"{int(e['sem_caixa'].sum())} sem caixa (presença sem localização)",
          f"  alta por janela: {_md(pj['alta'])} · janelas sem nenhum frame alta: {int((pj['alta'] == 0).sum())}",
          f"  novos por janela: {_md(pj['novos'])}",
          f"Caixas por lesão (seed + originais → + novos alta): antes {int(pj['antes'].sum())}, depois "
          f"{int(pj['depois'].sum())} (×{pj['depois'].sum() / max(1, pj['antes'].sum()):.2f})".replace(".", ","),
          f"  por janela: antes {_md(pj['antes'])} · depois {_md(pj['depois'])}"]
    try:
        v = json.loads((Path(cfg["caminhos"]["saida"]) / "VAL_EX" / f"{fin['metodo']}.json").read_text(encoding="utf-8"))
        pr = v["por_op"]["OP_F1"]["micro"]["P"]
        L.append(f"  (em VAL-EX, a precisão do OP-F1 foi {pr['valor']:.3f} [{pr['ic95'][0]:.3f}–{pr['ic95'][1]:.3f}]: "
                 "espera-se que cerca de 9 em cada 10 frames alta tenham a lesão; confirma-se na validação, Guião C)"
                 .replace(".", ","))
    except (OSError, KeyError, TypeError, ValueError):
        pass
    L += ["", "Por distância ao seed:"]
    d = e["offset"].abs()
    for lo, hi in FAIXAS:
        m = (d >= lo) & (d <= hi)
        if m.any():
            rot = f"{lo}–{hi}" if hi < 10 ** 6 else f">{lo - 1}"
            L.append(f"  {rot:>5}: {int(m.sum()):>6} frames · alta {int(e.loc[m, 'alta'].sum()):>6} "
                     f"({100 * e.loc[m, 'alta'].mean():.0f} %) · novos {int(e.loc[m, 'novo'].sum()):>6}")
    pc = (pj.groupby("classe").agg(janelas=("alta", "size"), originais=("originais", "sum"), alta=("alta", "sum"),
                                   novos=("novos", "sum"), antes=("antes", "sum"), depois=("depois", "sum"),
                                   alta_mediana=("alta", "median"), novos_mediana=("novos", "median"),
                                   sem_alta=("alta", lambda s: int((s == 0).sum())))
          .sort_values("janelas", ascending=False))
    L += ["", "Por classe:",
          f"  {'classe':<12} {'janelas':>7} {'alta':>6} {'novos':>6} {'novos/janela (mediana)':>23} "
          f"{'sem alta':>8} {'caixas antes → depois':>22}"]
    for cl, r in pc.iterrows():
        L.append(f"  {cl:<12} {int(r.janelas):>7} {int(r.alta):>6} {int(r.novos):>6} {r.novos_mediana:>23g} "
                 f"{int(r.sem_alta):>8} {f'{int(r.antes)} → {int(r.depois)}':>22}")
    L += ["", "Por split:"]
    for sp, g in pj.groupby(level="split"):
        L.append(f"  {sp}: {len(g)} janelas · alta {int(g['alta'].sum())} · novos {int(g['novos'].sum())} · caixas "
                 f"{int(g['antes'].sum())} → {int(g['depois'].sum())}")
    m_dir = Path(args.mascaras)
    n_m = len(list(m_dir.glob("*.json.gz"))) if m_dir.is_dir() else 0
    L += ["", f"Máscaras: {n_m} ficheiros em {m_dir}/ (um por janela; esperados {len(pj)})"
          + ("" if n_m == len(pj) else " — atenção: não batem com o número de janelas"),
          "",
          "A seguir: a validação humana (Guião C) parte dos frames alta, sobretudo os novos (caixa_original = 0)."]
    texto = "\n".join(L)
    rs = Path(args.saida_resumo)
    rs.mkdir(parents=True, exist_ok=True)
    (rs / "15_a8_resumo.txt").write_text(texto + "\n", encoding="utf-8")
    pc.to_csv(rs / "15_a8_por_classe.csv")
    print(texto)
    return {"expandida": out, "por_janela": pj, "por_classe": pc, "sha256": novo, "texto": texto}


if __name__ == "__main__":
    try:
        main()
    except ev.ErroAvaliacao as e:
        print(f"ERRO: {e}", file=sys.stderr)
        sys.exit(2)
