#!/usr/bin/env python3
"""
Passo 0.8 — Avaliador único (Guião A, secção 5). A mesma função, avaliar(), serve todos os métodos
dos Guiões A, B e C.

Três comandos:
  calibrar  Só em CAL. Varre τ (0 a 1, passo 0,02), fixa OP-F1 (máximo F1 micro) e OP-R90 (maior τ com
            recall micro ≥ 0,90; se nenhum chegar lá, o τ com o recall máximo) e avalia CAL nesses dois
            pontos. Nota: o guião diz "menor τ com recall ≥ 0,90", mas o menor τ daria sempre τ = 0
            (tudo presente); o que tem sentido é o maior τ que ainda garante o recall.
  avaliar   VAL_EX ou TEST_BOX, com os pontos de operação calibrados em CAL (outputs/eval/CAL/<método>.json).
            Cada método corre UMA vez em cada conjunto de teste: a execução fica em outputs/test_runs.csv e
            uma segunda é recusada, a não ser com --repetir "<motivo>" (o motivo fica registado).
  comparar  Diferença pareada por janela entre dois métodos já avaliados no mesmo conjunto: média e IC 95 %
            por bootstrap; "diferentes" quando o IC exclui 0.

Previsões (pred_<método>_<conjunto>.csv), uma linha por frame:
  caso, seed, frame, score  [, split] [, box_x1, box_y1, box_x2, box_y2] [, area_px]
  - score ∈ [0, 1]; presença prevista = score ≥ τ (e area_px ≥ A_min, se houver area_px)
  - caixas normalizadas 0–1 (como em annotations_v1.csv), x1 ≤ x2, y1 ≤ y2; vazias quando não há caixa
  - frame sem linha = ausente (com aviso); a linha do seed, se existir, é ignorada

Referência:
  presença (CAL, VAL_EX)     anotação do Paulo, data/derived/paulo_frame_labels.csv: 1 positivo, 0 negativo;
                             duvidoso fica fora das métricas e é contado à parte
  caixas (CAL, TEST_BOX)     caixas do médico da PRÓPRIA janela (box_* de annotations_v1.csv; as consolidadas
                             boxc_* não entram): frame com caixa = positivo. Um frame sem caixa NÃO é negativo,
                             por isso em TEST_BOX não há precisão, só a taxa de FP aparentes
  O seed (offset 0) é excluído de todas as métricas.

Métricas: P, R, F1 micro (frames juntos) e macro (média por janela); AP (área sob a curva PR, varrendo τ);
R_box (fração dos frames com caixa em que há presença prevista); R_loc (presença e IoU ≥ 0,3; métrica
principal de TEST_BOX); mIoU e % IoU ≥ 0,3 / 0,5 entre os frames com as duas caixas; ganho bruto (frames
previstos por janela, sem o seed); estratos por classe, distância ao seed, tamanho da caixa do seed e direção.
IC 95 % por bootstrap sobre janelas (1 000 reamostragens, semente 0).
Convenções por janela: P indefinida sem previsões; R indefinida sem positivos; F1 = 1 quando não há
positivos nem previsões (nada a encontrar, nada previsto). As médias macro usam as janelas definidas.

Uso (a partir da pasta do projeto):
  python eval/evaluate.py calibrar --metodo B1 --pred outputs/preds/pred_B1_CAL.csv
  python eval/evaluate.py avaliar  --metodo B1 --conjunto VAL_EX --pred outputs/preds/pred_B1_VAL_EX.csv
  python eval/evaluate.py comparar --conjunto VAL_EX --a SAM2 --b B1
Parâmetros em configs/avaliacao.yaml. Resultados em outputs/eval/<conjunto>/<método>.json e numa tabela
longa, outputs/eval/resultados.csv (a base da tabela mestre).
"""
import argparse
import datetime
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

VERSAO = "1.0"
K3 = ["split", "caso", "seed"]
CAIXA = ["box_x1", "box_y1", "box_x2", "box_y2"]
REF = ["rx1", "ry1", "rx2", "ry2"]
PRV = ["px1", "py1", "px2", "py2"]
TAMANHOS = ["pequena", "media", "grande", "sem_caixa"]


class ErroAvaliacao(Exception):
    """Erro nos dados ou no uso; a mensagem diz o que corrigir."""


# ============================================================================ utilitários
def _f(x, casas=6):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, casas) if np.isfinite(x) else None


def _div(a, b, vazio=np.nan):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b > 0, a / np.where(b > 0, b, 1.0), vazio)


def _limpo(o):
    if isinstance(o, dict):
        return {str(k): _limpo(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_limpo(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return _f(o)
    return o


def _sha256(caminho):
    h = hashlib.sha256()
    with open(caminho, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def chave(split, caso, seed):
    return (str(split), str(caso).strip().zfill(4), int(float(seed)))


def _txt(k):
    return "/".join(str(x) for x in k)


def iou(a, b):
    """IoU entre pares de caixas xyxy (arrays n×4). Caixas sem área dão 0."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    ix = np.clip(np.minimum(a[:, 2], b[:, 2]) - np.maximum(a[:, 0], b[:, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, 3], b[:, 3]) - np.maximum(a[:, 1], b[:, 1]), 0, None)
    inter = ix * iy
    uniao = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter
    return _div(inter, uniao, 0.0)


# ============================================================================ leitura
def ler_config(caminho):
    p = Path(caminho)
    if not p.is_file():
        raise ErroAvaliacao(f"Não encontrei a configuração {p}.")
    try:
        import yaml
    except ImportError:
        raise ErroAvaliacao("Falta o PyYAML neste ambiente: pip install pyyaml")
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def grelha_tau(cfg):
    g = cfg["limiar"]["grelha"]
    n = int(round((g["fim"] - g["inicio"]) / g["passo"]))
    return np.round(g["inicio"] + g["passo"] * np.arange(n + 1), 10)


def ler_conjunto(caminho):
    """Lista de janelas (split, caso, seed) de um conjunto: JSON {"janelas": [...]} ou CSV."""
    p = Path(caminho)
    if not p.is_file():
        raise ErroAvaliacao(f"Não encontrei o conjunto {p} (os conjuntos são criados no passo 0.6).")
    if p.suffix.lower() == ".json":
        d = json.loads(p.read_text(encoding="utf-8"))
        linhas = d["janelas"] if isinstance(d, dict) else d
    else:
        linhas = pd.read_csv(p, dtype=str).to_dict("records")
    ks = [chave(r["split"], r["caso"], r["seed"]) for r in linhas]
    if len(set(ks)) != len(ks):
        raise ErroAvaliacao(f"O conjunto {p} tem janelas repetidas.")
    if not ks:
        raise ErroAvaliacao(f"O conjunto {p} está vazio.")
    return ks


def carregar_referencia(cfg, nome, janelas):
    """Frames das janelas do conjunto, com a referência que o protocolo manda usar nesse conjunto.

    Devolve (ref, info). ref: uma linha por frame — w, split, caso, seed, frame, offset, pf ∈ {1, 0, D, ""},
    caixa ∈ {caixa, vazio, sem_ficheiro, invalida, nao_usada}, rx1..ry2 (caixa do médico na própria janela).
    info: uma linha por janela — w, split, caso, seed, categoria, classe, area_seed, tercil."""
    if nome not in cfg["conjuntos"]:
        raise ErroAvaliacao(f"Conjunto desconhecido: {nome} (conheço {', '.join(cfg['conjuntos'])}).")
    conf, cam = cfg["conjuntos"][nome], cfg["caminhos"]
    usa_pf, usa_cx = "pf" in conf["referencia"], "caixas" in conf["referencia"]
    if conf.get("split"):
        errados = [k for k in janelas if k[0] != conf["split"]]
        if errados:
            raise ErroAvaliacao(f"{nome} só pode ter janelas de {conf['split']}; {len(errados)} não são "
                                f"(ex.: {_txt(errados[0])}).")
    alvo, casos = set(janelas), {k[1] for k in janelas}
    if not Path(cam["anotacoes"]).is_file():
        raise ErroAvaliacao(f"Não encontrei {cam['anotacoes']}.")

    partes = []
    colunas = K3 + ["frame", "offset", "box_src"] + CAIXA + ["excluida"]
    for b in pd.read_csv(cam["anotacoes"], dtype=str, keep_default_na=False, usecols=colunas, chunksize=200_000):
        b = b[b["caso"].isin(casos)]
        if len(b):
            b = b[[k in alvo for k in zip(b["split"], b["caso"], b["seed"].astype(int))]]
            if len(b):
                partes.append(b)
    if not partes:
        raise ErroAvaliacao(f"Nenhuma janela de {nome} está em {cam['anotacoes']}.")
    a = pd.concat(partes, ignore_index=True)
    for c in ("seed", "frame", "offset"):
        a[c] = a[c].astype(int)
    presentes = set(zip(a["split"], a["caso"], a["seed"]))
    faltam = [k for k in janelas if k not in presentes]
    if faltam:
        raise ErroAvaliacao(f"{len(faltam)} janelas de {nome} não estão em {cam['anotacoes']} (ex.: {_txt(faltam[0])}).")
    exc = a.loc[a["excluida"] == "1", K3].drop_duplicates()
    if len(exc):
        raise ErroAvaliacao(f"{nome} tem janelas excluídas (ex.: {_txt(tuple(exc.iloc[0]))}).")
    if a.duplicated(K3 + ["frame"]).any():
        raise ErroAvaliacao("annotations_v1.csv tem frames repetidos dentro da mesma janela.")

    v = a[CAIXA].apply(pd.to_numeric, errors="coerce")
    src = a["box_src"].to_numpy()
    medico = np.isin(src, ["medico_seed", "medico_vizinho"])
    valida = (medico & v.notna().all(axis=1).to_numpy()
              & (v["box_x2"] > v["box_x1"]).to_numpy() & (v["box_y2"] > v["box_y1"]).to_numpy())
    ref = a[K3 + ["frame", "offset"]].copy()
    ref["caixa"] = np.select([valida, medico, src == "vazio", src == "sem_ficheiro"],
                             ["caixa", "invalida", "vazio", "sem_ficheiro"], "invalida")
    for r, c in zip(REF, CAIXA):
        ref[r] = np.where(valida, v[c].to_numpy(), np.nan)

    # janelas: categoria, classe e tercil da área da caixa do seed
    if not Path(cam["janelas"]).is_file():
        raise ErroAvaliacao(f"Não encontrei {cam['janelas']}.")
    jt = pd.read_csv(cam["janelas"], dtype=str, keep_default_na=False)
    if "categoria" not in jt.columns:
        raise ErroAvaliacao(f"{cam['janelas']} não tem a coluna 'categoria' (falta correr o 05 --aplicar).")
    jmap = {chave(s, c, e): (cat, les) for s, c, e, cat, les in
            zip(jt["split"], jt["caso"], jt["seed"], jt["categoria"], jt.get("lesao", pd.Series("1", index=jt.index)))}
    sd = ref[(ref["offset"] == 0) & (ref["caixa"] == "caixa")]
    area = {(s, c, e): (x2 - x1) * (y2 - y1)
            for s, c, e, x1, y1, x2, y2 in zip(sd["split"], sd["caso"], sd["seed"], sd["rx1"], sd["ry1"], sd["rx2"], sd["ry2"])}
    principais = cfg["estratos"]["classes_principais"]
    t1, t2 = cfg["estratos"]["tercis_area_seed"]
    linhas = []
    for w, k in enumerate(janelas):
        if k not in jmap:
            raise ErroAvaliacao(f"{_txt(k)} não está em {cam['janelas']}.")
        cat, les = jmap[k]
        if les != "1":
            raise ErroAvaliacao(f"{_txt(k)} não é uma janela de lesão ({cat}); os conjuntos só têm janelas de lesão.")
        ar = area.get(k, np.nan)
        terc = "sem_caixa" if not np.isfinite(ar) else ("pequena" if ar < t1 else "media" if ar < t2 else "grande")
        linhas.append({"w": w, "split": k[0], "caso": k[1], "seed": k[2], "categoria": cat,
                       "classe": cat if cat in principais else "outras", "area_seed": ar, "tercil": terc})
    info = pd.DataFrame(linhas)
    wmap = {k: w for w, k in enumerate(janelas)}
    ref["w"] = [wmap[k] for k in zip(ref["split"], ref["caso"], ref["seed"])]

    if usa_pf:
        if not Path(cam["referencia_pf"]).is_file():
            raise ErroAvaliacao(f"Não encontrei {cam['referencia_pf']} (consolida a anotação: tools/anotar.py --consolidar).")
        pf = pd.read_csv(cam["referencia_pf"], dtype=str, keep_default_na=False)
        if not {"split", "caso", "seed", "frame", "visivel"} <= set(pf.columns):
            raise ErroAvaliacao(f"{cam['referencia_pf']} tem de ter as colunas split, caso, seed, frame, visivel.")
        pf["caso"] = pf["caso"].str.zfill(4)
        pf["seed"] = pf["seed"].astype(int)
        pf["frame"] = pf["frame"].astype(int)
        pf = pf[[k in alvo for k in zip(pf["split"], pf["caso"], pf["seed"])]].copy()
        pf["pf"] = pf["visivel"].map({"1": "1", "0": "0", "duvidoso": "D", "D": "D"})
        if pf["pf"].isna().any():
            raise ErroAvaliacao(f"Valores de 'visivel' desconhecidos: {sorted(set(pf.loc[pf['pf'].isna(), 'visivel']))[:5]}.")
        if pf.duplicated(K3 + ["frame"]).any():
            raise ErroAvaliacao("A anotação do Paulo tem frames repetidos.")
        m = ref.merge(pf[K3 + ["frame", "pf"]], on=K3 + ["frame"], how="outer", indicator=True)
        sobra = m[m["_merge"] == "right_only"]
        if len(sobra):
            raise ErroAvaliacao(f"A anotação do Paulo tem {len(sobra)} frames que não existem nas janelas "
                                f"(ex.: {_txt(tuple(sobra[K3 + ['frame']].iloc[0]))}).")
        sem = m[m["_merge"] == "left_only"]
        if len(sem):
            por = sem.groupby(K3).size()
            raise ErroAvaliacao(f"A anotação do Paulo não cobre {len(sem)} frames de {len(por)} janelas "
                                f"(ex.: {_txt(por.index[0])}: {int(por.iloc[0])} frames). Completa-as e volta a consolidar.")
        ref = m.drop(columns="_merge")
    else:
        ref["pf"] = ""
    if not usa_cx:
        ref["caixa"] = "nao_usada"
        ref[REF] = np.nan
    ref["seed"] = ref["seed"].astype(int)
    return ref.sort_values(["w", "frame"]).reset_index(drop=True), info


def ler_previsoes(fonte):
    """Lê e valida as previsões (caminho de um CSV ou DataFrame). Erros explicam o que corrigir."""
    if isinstance(fonte, pd.DataFrame):
        d, nome = fonte.copy(), "as previsões"
    else:
        p = Path(fonte)
        if not p.is_file():
            raise ErroAvaliacao(f"Não encontrei {p}.")
        d, nome = pd.read_csv(p, dtype={"caso": str, "split": str}), p.name
    falta = [c for c in ("caso", "seed", "frame", "score") if c not in d.columns]
    if falta:
        raise ErroAvaliacao(f"{nome}: faltam as colunas {falta}.")
    cx = [c for c in CAIXA if c in d.columns]
    if cx and len(cx) != 4:
        raise ErroAvaliacao(f"{nome}: para caixas são precisas as quatro colunas {CAIXA}.")
    d["caso"] = d["caso"].astype(str).str.strip().str.zfill(4)
    for c in ("seed", "frame"):
        v = pd.to_numeric(d[c], errors="coerce")
        if v.isna().any() or (v != v.round()).any():
            raise ErroAvaliacao(f"{nome}: a coluna {c} tem valores que não são números inteiros.")
        d[c] = v.astype(int)
    s = pd.to_numeric(d["score"], errors="coerce")
    if s.isna().any():
        raise ErroAvaliacao(f"{nome}: {int(s.isna().sum())} linhas sem score numérico (escreve 0 quando o método não deteta nada).")
    if ((s < 0) | (s > 1)).any():
        raise ErroAvaliacao(f"{nome}: há scores fora de [0, 1] (mín. {s.min():.3f}, máx. {s.max():.3f}); normaliza-os.")
    d["score"] = s.astype(float)
    if cx:
        v = d[CAIXA].apply(pd.to_numeric, errors="coerce")
        n_na = v.isna().sum(axis=1)
        if ((n_na > 0) & (n_na < 4)).any():
            raise ErroAvaliacao(f"{nome}: há caixas incompletas (só algumas das quatro coordenadas).")
        ok = n_na == 0
        e = 1e-6
        mal = ok & ((v.min(axis=1) < -e) | (v.max(axis=1) > 1 + e) | (v["box_x2"] < v["box_x1"]) | (v["box_y2"] < v["box_y1"]))
        if mal.any():
            raise ErroAvaliacao(f"{nome}: {int(mal.sum())} caixas fora de 0–1 ou com x2 < x1 / y2 < y1 "
                                "(as caixas são normalizadas pelo tamanho da imagem, como em annotations_v1).")
        d[CAIXA] = v.clip(0, 1)
    else:
        for c in CAIXA:
            d[c] = np.nan
    if "area_px" in d.columns:
        a = pd.to_numeric(d["area_px"], errors="coerce")
        if (a < 0).any():
            raise ErroAvaliacao(f"{nome}: area_px negativa.")
        d["area_px"] = a
    dup = d[d.duplicated(["caso", "seed", "frame"], keep=False)]
    if len(dup):
        r = dup.iloc[0]
        raise ErroAvaliacao(f"{nome}: {len(dup)} linhas repetidas (ex.: caso {r['caso']}, seed {r['seed']}, frame {r['frame']}).")
    return d.rename(columns=dict(zip(CAIXA, PRV)))


def juntar(ref, info, pred):
    """Junta as previsões à referência. Recusa janelas fora do conjunto e frames que não existem."""
    avisos = []
    split_de = {(c, s): sp for sp, c, s in zip(info["split"], info["caso"], info["seed"])}
    kp = list(zip(pred["caso"], pred["seed"]))
    fora = sorted({k for k in kp if k not in split_de})
    if fora:
        raise ErroAvaliacao(f"{len(fora)} janelas das previsões não pertencem ao conjunto "
                            f"(ex.: {', '.join(f'{c}/{s}' for c, s in fora[:3])}).")
    if "split" in pred.columns:
        mal = [k for sp, k in zip(pred["split"], kp) if str(sp) != split_de[k]]
        if mal:
            raise ErroAvaliacao(f"O split das previsões não bate com o do conjunto (ex.: {mal[0][0]}/{mal[0][1]}).")
    cols = ["caso", "seed", "frame", "score"] + PRV + (["area_px"] if "area_px" in pred.columns else [])
    chk = pred[["caso", "seed", "frame"]].merge(ref[["caso", "seed", "frame"]], how="left", indicator=True)
    sobra = chk[chk["_merge"] == "left_only"]
    if len(sobra):
        r = sobra.iloc[0]
        raise ErroAvaliacao(f"{len(sobra)} frames das previsões não existem nas janelas "
                            f"(ex.: caso {r['caso']}, seed {r['seed']}, frame {r['frame']}).")
    d = ref.merge(pred[cols], on=["caso", "seed", "frame"], how="left", indicator=True)
    d["tem_prev"] = (d["_merge"] == "both").to_numpy()
    d = d.drop(columns="_merge")
    nseed = d["offset"].to_numpy() != 0
    falta = nseed & ~d["tem_prev"].to_numpy()
    if falta.any():
        sem = d[nseed].groupby("w")["tem_prev"].any()
        vazias = info.set_index("w").loc[sem[~sem].index]
        txt = f"{int(falta.sum())} frames sem previsão contam como ausentes"
        if len(vazias):
            txt += (f"; {len(vazias)} janelas sem nenhuma previsão (ex.: "
                    + ", ".join(f"{r.caso}/{r.seed}" for r in vazias.head(3).itertuples()) + ")")
        avisos.append(txt)
    d["tem_cx_prev"] = d[PRV].notna().all(axis=1).to_numpy()
    ambas = (d["caixa"] == "caixa").to_numpy() & d["tem_cx_prev"].to_numpy()
    d["iou"] = np.nan
    if ambas.any():
        d.loc[ambas, "iou"] = iou(d.loc[ambas, PRV].to_numpy(), d.loc[ambas, REF].to_numpy())
    return d.sort_values(["w", "frame"]).reset_index(drop=True), avisos


# ============================================================================ núcleo
def presenca(d, tau, a_min=0):
    s = d["score"].to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        p = np.isfinite(s) & (s >= tau)
    if a_min and "area_px" in d.columns:
        p &= d["area_px"].fillna(0).to_numpy(dtype=float) >= a_min
    return p


def _contar(d, W, pres, iou_loc, limiares, mascara=None):
    """Contagens por janela (vetores de comprimento W). O seed nunca entra."""
    w = d["w"].to_numpy()
    base = d["offset"].to_numpy() != 0
    if mascara is not None:
        base = base & mascara
    pf, cx = d["pf"].to_numpy(), d["caixa"].to_numpy()
    iu, tem = d["iou"].to_numpy(dtype=float), d["tem_cx_prev"].to_numpy()

    def n(m):
        return np.bincount(w, weights=m.astype(float), minlength=W)

    y1, y0 = base & (pf == "1"), base & (pf == "0")
    pos_cx, vaz = base & (cx == "caixa"), base & (cx == "vazio")
    ambas = pos_cx & pres & tem
    tol = 1e-9                                  # IoU igual ao limiar (a menos do arredondamento) conta como atingido
    with np.errstate(invalid="ignore"):
        c = {"frames": n(base), "tp": n(pres & y1), "fp": n(pres & y0), "fn": n(~pres & y1), "tn": n(~pres & y0),
             "duv": n(base & (pf == "D")), "ganho": n(pres & base),
             "n_cx": n(pos_cx), "hit": n(pos_cx & pres), "ambas": n(ambas), "loc": n(ambas & (iu >= iou_loc - tol)),
             "iou_soma": np.bincount(w, weights=np.where(ambas, iu, 0.0), minlength=W),
             "fp_ap": n(vaz & pres), "pres_aval": n((pos_cx | vaz) & pres)}
        for t in limiares:
            c[f"iou_{t}"] = n(ambas & (iu >= t - tol))
    return c


def _por_janela_met(c):
    return {"P": _div(c["tp"], c["tp"] + c["fp"]),
            "R": _div(c["tp"], c["tp"] + c["fn"]),
            "F1": _div(2 * c["tp"], 2 * c["tp"] + c["fp"] + c["fn"], vazio=1.0),
            "R_box": _div(c["hit"], c["n_cx"]),
            "R_loc": _div(c["loc"], c["n_cx"]),
            "mIoU": _div(c["iou_soma"], c["ambas"]),
            "taxa_fp_aparentes": _div(c["fp_ap"], c["pres_aval"])}


class Bootstrap:
    """Reamostragem de janelas com reposição; a mesma matriz serve todas as métricas de uma avaliação."""

    def __init__(self, W, reamostragens, semente, nivel):
        self.nivel = nivel
        self.M = None
        if reamostragens:
            idx = np.random.default_rng(semente).integers(0, W, size=(reamostragens, W))
            self.M = np.stack([np.bincount(r, minlength=W) for r in idx]).astype(float)

    def ic(self, reps):
        r = np.asarray(reps, dtype=float)
        r = r[np.isfinite(r)]
        if not len(r):
            return None
        a = (1 - self.nivel) / 2 * 100
        lo, hi = np.percentile(r, [a, 100 - a])
        return [_f(lo), _f(hi)]

    def razao(self, num, den, sel, vazio=np.nan):
        val = _div(num[sel].sum(), den[sel].sum(), vazio)
        if self.M is None:
            return {"valor": _f(val), "ic95": None}
        Ms = self.M[:, sel]
        return {"valor": _f(val), "ic95": self.ic(_div(Ms @ num[sel], Ms @ den[sel], vazio))}

    def media(self, v, sel):
        ok = sel & np.isfinite(v)
        out = {"valor": _f(v[ok].mean()) if ok.any() else None, "ic95": None, "n_janelas": int(ok.sum())}
        if self.M is not None and ok.any():
            Ms = self.M[:, ok]
            out["ic95"] = self.ic(_div(Ms @ v[ok], Ms.sum(axis=1)))
        return out


def _bloco(c, boot, sel, usa_pf, usa_cx, limiares):
    out = {"n_janelas": int(((c["frames"] > 0) & sel).sum()), "n_frames": int(c["frames"][sel].sum()),
           "ganho_frames": int(c["ganho"][sel].sum())}
    if usa_pf:
        out.update({"n_pos": int((c["tp"] + c["fn"])[sel].sum()), "n_neg": int((c["fp"] + c["tn"])[sel].sum()),
                    "n_duvidosos": int(c["duv"][sel].sum()),
                    "P": boot.razao(c["tp"], c["tp"] + c["fp"], sel),
                    "R": boot.razao(c["tp"], c["tp"] + c["fn"], sel),
                    "F1": boot.razao(2 * c["tp"], 2 * c["tp"] + c["fp"] + c["fn"], sel)})
    if usa_cx:
        out.update({"n_caixas": int(c["n_cx"][sel].sum()), "n_com_as_duas_caixas": int(c["ambas"][sel].sum()),
                    "R_box": boot.razao(c["hit"], c["n_cx"], sel),
                    "R_loc": boot.razao(c["loc"], c["n_cx"], sel),
                    "mIoU": boot.razao(c["iou_soma"], c["ambas"], sel)})
        for t in limiares:
            out[f"pct_IoU>={t}"] = boot.razao(c[f"iou_{t}"], c["ambas"], sel)
        out["n_fp_aparentes"] = int(c["fp_ap"][sel].sum())
        out["taxa_fp_aparentes"] = boot.razao(c["fp_ap"], c["pres_aval"], sel)
    return out


def ap(d, boot, W, a_min=0):
    """AP = Σ (R_k − R_{k−1}) · P_k, varrendo τ por todos os scores. Frames sem previsão nunca entram como
    presentes, mas contam como positivos por encontrar (o recall pode não chegar a 1)."""
    base = (d["offset"].to_numpy() != 0) & np.isin(d["pf"].to_numpy(), ["1", "0"])
    y = (d["pf"].to_numpy() == "1")[base].astype(float)
    s = d["score"].to_numpy(dtype=float)[base].copy()
    if a_min and "area_px" in d.columns:
        s[d["area_px"].fillna(0).to_numpy(dtype=float)[base] < a_min] = np.nan
    w = d["w"].to_numpy()[base]
    if y.sum() == 0:
        return {"valor": None, "ic95": None}
    fin = np.isfinite(s)
    if not fin.any():
        return {"valor": 0.0, "ic95": [0.0, 0.0] if boot.M is not None else None}
    o = np.argsort(-s[fin], kind="mergesort")
    ys, ss, ws = y[fin][o], s[fin][o], w[fin][o]
    fim = np.r_[np.nonzero(np.diff(ss))[0], len(ss) - 1]       # último frame de cada valor de score
    pos_j = np.bincount(w, weights=y, minlength=W)

    def calc(peso, npos):
        tp = np.cumsum(peso * ys, axis=1)[:, fim]
        fp = np.cumsum(peso * (1 - ys), axis=1)[:, fim]
        prec = _div(tp, tp + fp, 0.0)
        rec = _div(tp, npos[:, None], 0.0)
        dr = np.diff(np.concatenate([np.zeros((len(npos), 1)), rec], axis=1), axis=1)
        return np.where(npos > 0, (dr * prec).sum(axis=1), np.nan)

    out = {"valor": _f(calc(np.ones((1, len(ys))), np.array([y.sum()]))[0]), "ic95": None}
    if boot.M is not None:
        out["ic95"] = boot.ic(calc(boot.M[:, ws], boot.M @ pos_j))
    return out


def curva_pr(d, grelha, a_min=0):
    """P, R e F1 micro em cada τ da grelha (para a calibração e para as figuras)."""
    base = (d["offset"].to_numpy() != 0) & np.isin(d["pf"].to_numpy(), ["1", "0"])
    y = (d["pf"].to_numpy() == "1")[base]
    s = d["score"].to_numpy(dtype=float)[base]
    ok = np.isfinite(s)
    if a_min and "area_px" in d.columns:
        ok &= d["area_px"].fillna(0).to_numpy(dtype=float)[base] >= a_min
    with np.errstate(invalid="ignore"):
        pres = (s[None, :] >= grelha[:, None]) & ok[None, :]
    tp = (pres & y).sum(axis=1)
    fp = (pres & ~y).sum(axis=1)
    fn = int(y.sum()) - tp
    return pd.DataFrame({"tau": grelha, "TP": tp, "FP": fp, "FN": fn, "P": _div(tp, tp + fp),
                         "R": _div(tp, tp + fn), "F1": _div(2 * tp, 2 * tp + fp + fn)})


def calibrar_ops(d, cfg):
    """OP-F1 = máximo F1 micro (empates: menor A_min, depois maior τ). OP-R90 = maior τ com recall micro ≥ 0,90,
    com o mesmo A_min."""
    grelha = grelha_tau(cfg)
    a_mins = [0]
    if "area_px" in d.columns and d["area_px"].notna().any():
        a_mins = [float(a) for a in cfg["limiar"].get("a_min", [0])]
    T = pd.concat([curva_pr(d, grelha, a).assign(a_min=a) for a in a_mins], ignore_index=True)
    if not ((T["TP"] + T["FN"]) > 0).any():
        raise ErroAvaliacao("CAL não tem positivos além do seed: não é possível calibrar.")
    f1 = T["F1"].fillna(-1)
    cand = T[f1 == f1.max()]
    cand = cand[cand["a_min"] == cand["a_min"].min()]
    best = cand.sort_values("tau").iloc[-1]
    Ta = T[T["a_min"] == best["a_min"]]
    alvo = cfg["limiar"]["recall_op_r90"]
    ok = Ta[Ta["R"] >= alvo - 1e-12]
    if len(ok):
        r90 = ok.sort_values("tau").iloc[-1]
        regra = f"maior τ com recall micro ≥ {alvo:.2f} em CAL"
    else:
        rmax = Ta["R"].max()
        r90 = Ta[Ta["R"] == rmax].sort_values("tau").iloc[-1]
        regra = f"recall ≥ {alvo:.2f} não é atingível em CAL (máximo {rmax:.3f}): τ com o recall máximo"
    ops = {"OP_F1": {"tau": float(best["tau"]), "a_min": float(best["a_min"]),
                     "regra": "máximo F1 micro em CAL (empates: menor A_min, depois maior τ)"},
           "OP_R90": {"tau": float(r90["tau"]), "a_min": float(best["a_min"]), "regra": regra}}
    return ops, T


def _ganho(g, info, cfg):
    def resumo(v):
        v = np.asarray(v, dtype=float)
        if not len(v):
            return None
        return {"mediana": _f(np.median(v)), "iqr": [_f(np.percentile(v, 25)), _f(np.percentile(v, 75))],
                "media": _f(v.mean()), "n_janelas": int(len(v))}
    classes = list(cfg["estratos"]["classes_principais"]) + ["outras"]
    return {"todas": resumo(g), "por_classe": {cl: resumo(g[(info["classe"] == cl).to_numpy()]) for cl in classes}}


def _estratos(d, info, pres, c, boot, cfg, usa_pf, usa_cx):
    W, lim, iou_loc = len(info), cfg["iou"]["limiares"], cfg["iou"]["loc"]
    todas = np.ones(W, dtype=bool)
    out = {"classe": {}, "tamanho": {}, "distancia": {}, "direcao": {}}
    for cl in list(cfg["estratos"]["classes_principais"]) + ["outras"]:
        out["classe"][cl] = _bloco(c, boot, (info["classe"] == cl).to_numpy(), usa_pf, usa_cx, lim)
    for t in TAMANHOS:
        sel = (info["tercil"] == t).to_numpy()
        if t != "sem_caixa" or sel.any():
            out["tamanho"][t] = _bloco(c, boot, sel, usa_pf, usa_cx, lim)
    off = d["offset"].to_numpy()
    for lo, hi in cfg["estratos"]["distancia"]:
        rot = f">{lo - 1}" if hi >= 100 else f"{lo}-{hi}"
        m = (np.abs(off) >= lo) & (np.abs(off) <= hi)
        out["distancia"][rot] = _bloco(_contar(d, W, pres, iou_loc, lim, m), boot, todas, usa_pf, usa_cx, lim)
    for rot, m in (("antes", off < 0), ("depois", off > 0)):
        out["direcao"][rot] = _bloco(_contar(d, W, pres, iou_loc, lim, m), boot, todas, usa_pf, usa_cx, lim)
    return out


def _contagens(d, info, usa_pf, usa_cx):
    ns = d["offset"].to_numpy() != 0
    x = d[ns]
    out = {"janelas": int(len(info)), "frames": int(len(d)), "seeds_excluidos": int((~ns).sum()),
           "frames_sem_previsao": int((~x["tem_prev"]).sum()),
           "janelas_sem_previsao": int((~x.groupby("w")["tem_prev"].any()).sum()),
           "janelas_por_classe": info["classe"].value_counts().to_dict(),
           "janelas_por_tamanho": info["tercil"].value_counts().to_dict()}
    if usa_pf:
        out.update({"positivos": int((x["pf"] == "1").sum()), "negativos": int((x["pf"] == "0").sum()),
                    "duvidosos": int((x["pf"] == "D").sum()),
                    "janelas_sem_positivos_alem_do_seed": int((x.groupby("w")["pf"].apply(lambda s: (s == "1").sum()) == 0).sum())})
    if usa_cx:
        cx = x["caixa"]
        out.update({"frames_com_caixa": int((cx == "caixa").sum()), "frames_vazios": int((cx == "vazio").sum()),
                    "frames_sem_ficheiro_de_caixa": int((cx == "sem_ficheiro").sum()),
                    "caixas_invalidas": int((cx == "invalida").sum()),
                    "janelas_sem_caixa_alem_do_seed": int((x.groupby("w")["caixa"].apply(lambda s: (s == "caixa").sum()) == 0).sum())})
    return out


def avaliar(d, info, cfg, ops, usa_pf, usa_cx, avisos=()):
    """A função única de avaliação.

    d: frames do conjunto já juntos com as previsões (juntar); info: uma linha por janela;
    ops: {"OP_F1": {"tau", "a_min"}, "OP_R90": {...}}; usa_pf / usa_cx: que referências existem."""
    W = len(info)
    bs = cfg["bootstrap"]
    boot = Bootstrap(W, bs["reamostragens"], bs["semente"], bs["nivel"])
    lim, iou_loc = cfg["iou"]["limiares"], cfg["iou"]["loc"]
    todas = np.ones(W, dtype=bool)
    res = {"contagens": _contagens(d, info, usa_pf, usa_cx), "avisos": list(avisos),
           "pontos_operacao": ops, "referencia": {"presenca_pf": usa_pf, "caixas_do_medico": usa_cx}}
    if usa_pf:
        a0 = ops["OP_F1"].get("a_min", 0)
        res["AP"] = ap(d, boot, W, a0)
        cur = curva_pr(d, grelha_tau(cfg), a0)
        res["curva_pr"] = {k: [_f(v) for v in cur[k]] for k in ("tau", "P", "R", "F1")}
    res["por_op"], pj = {}, {}
    for nome, op in ops.items():
        pres = presenca(d, op["tau"], op.get("a_min", 0))
        c = _contar(d, W, pres, iou_loc, lim)
        mj = _por_janela_met(c)
        macro = {}
        if usa_pf:
            macro.update({"P": boot.media(mj["P"], todas), "R": boot.media(mj["R"], todas),
                          "F1": boot.media(mj["F1"], todas),
                          "F1_janelas_com_positivos": boot.media(np.where(c["tp"] + c["fn"] > 0, mj["F1"], np.nan), todas)})
        if usa_cx:
            macro.update({"R_box": boot.media(mj["R_box"], todas), "R_loc": boot.media(mj["R_loc"], todas),
                          "mIoU": boot.media(mj["mIoU"], todas)})
        res["por_op"][nome] = {"tau": op["tau"], "a_min": op.get("a_min", 0),
                               "micro": _bloco(c, boot, todas, usa_pf, usa_cx, lim), "macro": macro,
                               "ganho": _ganho(c["ganho"], info, cfg),
                               "estratos": _estratos(d, info, pres, c, boot, cfg, usa_pf, usa_cx)}
        pj[nome] = (c, mj)
    linhas = []
    for i, r in enumerate(info.itertuples(index=False)):
        l = {"split": r.split, "caso": r.caso, "seed": int(r.seed), "categoria": r.categoria, "classe": r.classe,
             "tercil": r.tercil, "area_seed": _f(r.area_seed)}
        for nome, (c, mj) in pj.items():
            e = {"ganho": int(c["ganho"][i])}
            if usa_pf:
                e.update({"TP": int(c["tp"][i]), "FP": int(c["fp"][i]), "FN": int(c["fn"][i]), "TN": int(c["tn"][i]),
                          "duvidosos": int(c["duv"][i]), "P": _f(mj["P"][i]), "R": _f(mj["R"][i]), "F1": _f(mj["F1"][i])})
            if usa_cx:
                e.update({"caixas": int(c["n_cx"][i]), "R_box": _f(mj["R_box"][i]), "R_loc": _f(mj["R_loc"][i]),
                          "mIoU": _f(mj["mIoU"][i]), "fp_aparentes": int(c["fp_ap"][i])})
            l[nome] = e
        linhas.append(l)
    res["por_janela"] = linhas
    return res


def diferenca_pareada(a, b, metrica, op, cfg):
    """Diferença por janela (a − b) de uma métrica, num ponto de operação; IC por bootstrap sobre janelas."""
    def valores(r):
        return {(j["split"], j["caso"], j["seed"]): (np.nan if j[op].get(metrica) is None else j[op][metrica])
                for j in r["por_janela"]}
    if a.get("conjunto") != b.get("conjunto"):
        raise ErroAvaliacao("Os dois resultados são de conjuntos diferentes.")
    va, vb = valores(a), valores(b)
    if set(va) != set(vb):
        raise ErroAvaliacao("Os dois métodos não foram avaliados nas mesmas janelas.")
    ks = sorted(va)
    x, y = np.array([va[k] for k in ks], dtype=float), np.array([vb[k] for k in ks], dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if not ok.any():
        raise ErroAvaliacao(f"A métrica {metrica} não está definida em nenhuma janela.")
    dl = (x - y)[ok]
    bs = cfg["bootstrap"]
    boot = Bootstrap(len(dl), bs["reamostragens"], bs["semente"], bs["nivel"])
    ic = boot.ic(_div(boot.M @ dl, boot.M.sum(axis=1)))
    return {"metrica": metrica, "op": op, "a": a.get("metodo"), "b": b.get("metodo"), "conjunto": a.get("conjunto"),
            "n_janelas": int(ok.sum()), "janelas_sem_valor": int((~ok).sum()),
            "media_a": _f(x[ok].mean()), "media_b": _f(y[ok].mean()), "diferenca_media": _f(dl.mean()), "ic95": ic,
            "diferentes": bool(ic is not None and (ic[0] > 0 or ic[1] < 0)),
            "janelas_a_melhor": int((dl > 0).sum()), "empates": int((dl == 0).sum()), "janelas_b_melhor": int((dl < 0).sum())}


def fp_aparentes(d, op, cfg):
    """Frames com presença prevista e sem caixa do médico (TEST_BOX), com a amostra para a inspeção manual."""
    m = presenca(d, op["tau"], op.get("a_min", 0)) & (d["offset"].to_numpy() != 0) & (d["caixa"].to_numpy() == "vazio")
    t = d.loc[m, K3 + ["frame", "offset", "score"]].reset_index(drop=True)
    t["amostra"] = 0
    n = min(int(cfg["fp_aparentes"]["amostra"]), len(t))
    if n:
        esc = np.random.default_rng(cfg["fp_aparentes"]["semente"]).choice(len(t), size=n, replace=False)
        t.loc[esc, "amostra"] = 1
    t["classificacao"] = ""     # à mão: lesao_nao_anotada · outra_lesao · fundo_mucosa · nao_avaliavel
    return t


# ============================================================================ linha de comandos
def _proveniencia(cfg, args, fich_conjunto, t0, usa_pf):
    cam = cfg["caminhos"]
    refs = [cam["anotacoes"]] + ([cam["referencia_pf"]] if usa_pf else [])
    vd = Path(cam.get("versao_dataset", ""))
    versao = "desconhecida (falta docs/DATASET_VERSION.txt: corre o 01c)"
    if vd.is_file():
        for linha in vd.read_text(encoding="utf-8").splitlines():
            if "sha256 do conteúdo" in linha:
                versao = linha.split()[-1]
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                timeout=5).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        commit = None
    return {"data": datetime.datetime.now().isoformat(timespec="seconds"), "duracao_s": round(time.time() - t0, 1),
            "versao_avaliador": VERSAO, "avaliador_sha256": _sha256(__file__),
            "config": str(args.config), "config_sha256": _sha256(args.config),
            "previsoes": str(args.pred), "previsoes_sha256": _sha256(args.pred),
            "conjunto": str(fich_conjunto), "conjunto_sha256": _sha256(fich_conjunto),
            "referencias_sha256": {str(f): _sha256(f) for f in refs},
            "versao_dataset": versao, "commit": commit, "python": platform.python_version(),
            "numpy": np.__version__, "pandas": pd.__version__,
            "bootstrap": cfg["bootstrap"]}


def _atualizar_resultados(cfg, res):
    linhas = []

    def add(op, metrica, x, n=None):
        ic = x.get("ic95") or [None, None]
        linhas.append({"metodo": res["metodo"], "conjunto": res["conjunto"], "op": op, "metrica": metrica,
                       "valor": x.get("valor"), "ic95_inf": ic[0], "ic95_sup": ic[1], "n": n, "data": res["registo"]["data"]})
    if "AP" in res:
        add("-", "AP", res["AP"])
    for op, r in res["por_op"].items():
        add(op, "tau", {"valor": r["tau"]})
        for k, v in r["micro"].items():
            if isinstance(v, dict):
                add(op, f"{k}_micro" if k in ("P", "R", "F1") else k, v)
        for k, v in r["macro"].items():
            add(op, f"{k}_macro", v, v.get("n_janelas"))
    p = Path(cfg["caminhos"]["saida"]) / "resultados.csv"
    novo = pd.DataFrame(linhas)
    if p.exists():
        velho = pd.read_csv(p, dtype=str, keep_default_na=False)
        velho = velho[~((velho["metodo"] == res["metodo"]) & (velho["conjunto"] == res["conjunto"]))]
        novo = pd.concat([velho, novo.astype(str).replace({"None": "", "nan": ""})], ignore_index=True)
    novo.to_csv(p, index=False)


def _gravar_json(caminho, obj):
    caminho.parent.mkdir(parents=True, exist_ok=True)
    tmp = caminho.with_name(caminho.name + ".tmp")
    tmp.write_text(json.dumps(_limpo(obj), ensure_ascii=False, indent=1, allow_nan=False), encoding="utf-8")
    tmp.replace(caminho)


def _fmt(x):
    if not isinstance(x, dict) or x.get("valor") is None:
        return "—"
    ic = x.get("ic95")
    return f"{x['valor']:.3f}" + (f" [{ic[0]:.3f}–{ic[1]:.3f}]" if ic and None not in ic else "")


def _ecra(res):
    L = [f"\n{res['metodo']} em {res['conjunto']}: {res['contagens']['janelas']} janelas · {res['contagens']['frames']} frames"]
    for a in res["avisos"]:
        L.append(f"  aviso: {a}")
    if "AP" in res:
        L.append(f"  AP {_fmt(res['AP'])}")
    for op, r in res["por_op"].items():
        m = r["micro"]
        partes = [f"τ = {r['tau']:.2f}" + (f", A_min = {r['a_min']:g}" if r.get("a_min") else "")]
        if "F1" in m:
            partes += [f"P {_fmt(m['P'])}", f"R {_fmt(m['R'])}", f"F1 {_fmt(m['F1'])}", f"F1 macro {_fmt(r['macro']['F1'])}"]
        if "R_loc" in m:
            partes += [f"R_box {_fmt(m['R_box'])}", f"R_loc {_fmt(m['R_loc'])}", f"mIoU {_fmt(m['mIoU'])}",
                       f"FP aparentes {_fmt(m['taxa_fp_aparentes'])}"]
        L.append(f"  {op}: " + " · ".join(partes))
    print("\n".join(L))


def _preparar(args, cfg, conjunto):
    conf = cfg["conjuntos"][conjunto]
    fich = Path(args.conjunto_ficheiro or Path(cfg["caminhos"]["conjuntos"]) / f"{conjunto}.json")
    janelas = ler_conjunto(fich)
    ref, info = carregar_referencia(cfg, conjunto, janelas)
    d, avisos = juntar(ref, info, ler_previsoes(args.pred))
    return conf, fich, d, info, avisos


def cmd_calibrar(args):
    t0 = time.time()
    cfg = ler_config(args.config)
    conf, fich, d, info, avisos = _preparar(args, cfg, "CAL")
    if not conf.get("calibracao"):
        raise ErroAvaliacao("A configuração não permite calibrar em CAL.")
    ops, T = calibrar_ops(d, cfg)
    res = {"metodo": args.metodo, "conjunto": "CAL"}
    res.update(avaliar(d, info, cfg, ops, True, "caixas" in conf["referencia"], avisos))
    f1 = np.array([j["OP_F1"]["F1"] for j in res["por_janela"] if j["OP_F1"]["F1"] is not None], dtype=float)
    res["calibracao"] = {"grelha": T[["a_min", "tau", "TP", "FP", "FN", "P", "R", "F1"]].to_dict("records"),
                         "F1_por_janela_em_OP_F1": {"media": _f(f1.mean()), "desvio_padrao": _f(f1.std(ddof=1)) if len(f1) > 1 else None,
                                                     "mediana": _f(np.median(f1)), "iqr": [_f(np.percentile(f1, 25)), _f(np.percentile(f1, 75))],
                                                     "min": _f(f1.min()), "max": _f(f1.max())}}
    res["registo"] = _proveniencia(cfg, args, fich, t0, True)
    saida = Path(cfg["caminhos"]["saida"]) / "CAL" / f"{args.metodo}.json"
    _gravar_json(saida, res)
    _atualizar_resultados(cfg, _limpo(res))
    _ecra(res)
    print(f"\n  pontos de operação fixados em CAL: OP-F1 τ = {ops['OP_F1']['tau']:.2f} · OP-R90 τ = {ops['OP_R90']['tau']:.2f}"
          f" ({ops['OP_R90']['regra']})\n  gravado em {saida}")


def cmd_avaliar(args):
    t0 = time.time()
    cfg = ler_config(args.config)
    conj = args.conjunto
    if conj not in cfg["conjuntos"]:
        raise ErroAvaliacao(f"Conjunto desconhecido: {conj}.")
    conf = cfg["conjuntos"][conj]
    reg = Path(cfg["caminhos"]["registo_testes"])
    if conf.get("execucao_unica") and reg.exists():
        r = pd.read_csv(reg, dtype=str, keep_default_na=False)
        ant = r[(r["metodo"] == args.metodo) & (r["conjunto"] == conj)]
        if len(ant) and not args.repetir:
            raise ErroAvaliacao(f"{args.metodo} já foi avaliado em {conj} ({ant['data'].iloc[-1]}). Cada método corre "
                                "uma só vez nos conjuntos de teste. Se for mesmo preciso (ex.: erro no avaliador), "
                                "repete com --repetir \"<motivo>\": fica registado.")
    op_path = Path(args.op or Path(cfg["caminhos"]["saida"]) / "CAL" / f"{args.metodo}.json")
    if not op_path.is_file():
        raise ErroAvaliacao(f"Não encontrei {op_path}: primeiro calibra o método em CAL (comando calibrar).")
    cal = json.loads(op_path.read_text(encoding="utf-8"))
    if cal.get("conjunto") != "CAL" or "pontos_operacao" not in cal:
        raise ErroAvaliacao(f"{op_path} não é uma calibração em CAL.")
    if cal.get("metodo") != args.metodo:
        raise ErroAvaliacao(f"{op_path} é a calibração de {cal.get('metodo')}, não de {args.metodo}.")
    origem = f"{op_path} (sha256 {_sha256(op_path)[:12]})"
    ops = {k: {"tau": v["tau"], "a_min": v.get("a_min", 0), "regra": v.get("regra", ""), "origem": origem}
           for k, v in cal["pontos_operacao"].items()}
    conf, fich, d, info, avisos = _preparar(args, cfg, conj)
    if cal.get("registo", {}).get("config_sha256") not in (None, _sha256(args.config)):
        avisos.append("a configuração (configs/avaliacao.yaml) mudou desde a calibração em CAL")
    res = {"metodo": args.metodo, "conjunto": conj}
    res.update(avaliar(d, info, cfg, ops, "pf" in conf["referencia"], "caixas" in conf["referencia"], avisos))
    res["registo"] = _proveniencia(cfg, args, fich, t0, "pf" in conf["referencia"])
    if args.repetir:
        res["registo"]["motivo_repeticao"] = args.repetir
    saida = Path(cfg["caminhos"]["saida"]) / conj / f"{args.metodo}.json"
    if saida.exists() and conf.get("execucao_unica"):
        saida.replace(saida.with_name(f"{args.metodo}.anterior_{datetime.datetime.now():%Y%m%d_%H%M%S}.json"))
    _gravar_json(saida, res)
    _atualizar_resultados(cfg, _limpo(res))
    if "caixas" in conf["referencia"]:
        fp = fp_aparentes(d, ops["OP_F1"], cfg)
        fp.to_csv(saida.with_name(f"{args.metodo}_fp_aparentes.csv"), index=False)
    if conf.get("execucao_unica"):
        linha = pd.DataFrame([{"data": res["registo"]["data"], "metodo": args.metodo, "conjunto": conj,
                               "previsoes": args.pred, "previsoes_sha256": res["registo"]["previsoes_sha256"],
                               "calibracao": origem, "avaliador_sha256": res["registo"]["avaliador_sha256"],
                               "config_sha256": res["registo"]["config_sha256"], "versao_dataset": res["registo"]["versao_dataset"],
                               "commit": res["registo"]["commit"] or "", "motivo_repeticao": args.repetir or "", "resultado": str(saida)}])
        reg.parent.mkdir(parents=True, exist_ok=True)
        linha.to_csv(reg, mode="a", header=not reg.exists(), index=False)
    _ecra(res)
    print(f"\n  gravado em {saida}" + (f" · execução registada em {reg}" if conf.get("execucao_unica") else ""))


def cmd_comparar(args):
    cfg = ler_config(args.config)
    base = Path(cfg["caminhos"]["saida"]) / args.conjunto
    rs = []
    for m in (args.a, args.b):
        p = base / f"{m}.json"
        if not p.is_file():
            raise ErroAvaliacao(f"Não encontrei {p}: avalia primeiro {m} em {args.conjunto}.")
        rs.append(json.loads(p.read_text(encoding="utf-8")))
    metrica = args.metrica or ("R_loc" if args.conjunto == "TEST_BOX" else "F1")
    r = diferenca_pareada(rs[0], rs[1], metrica, args.op, cfg)
    saida = base / f"comparar_{args.a}_vs_{args.b}_{metrica}_{args.op}.json"
    _gravar_json(saida, r)
    print(f"\n{metrica} por janela em {args.conjunto} ({args.op}), {r['n_janelas']} janelas: "
          f"{args.a} {r['media_a']:.3f} · {args.b} {r['media_b']:.3f}\n"
          f"  diferença média {r['diferenca_media']:+.3f} [{r['ic95'][0]:+.3f}, {r['ic95'][1]:+.3f}] → "
          + ("diferentes (o IC exclui 0)" if r["diferentes"] else "sem diferença demonstrada (o IC inclui 0)")
          + f"\n  janelas em que {args.a} é melhor: {r['janelas_a_melhor']} · empates {r['empates']} · {args.b} melhor: {r['janelas_b_melhor']}"
          + f"\n  gravado em {saida}")


def main(argv=None):
    ap_ = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap_.add_subparsers(dest="comando", required=True)
    for nome in ("calibrar", "avaliar"):
        s = sub.add_parser(nome)
        s.add_argument("--metodo", required=True, help="nome do método (ex.: B1, SAM2_box)")
        s.add_argument("--pred", required=True, help="CSV de previsões")
        s.add_argument("--config", default="configs/avaliacao.yaml")
        s.add_argument("--conjunto-ficheiro", help="ficheiro do conjunto em vez de sets/<conjunto>.json")
        if nome == "avaliar":
            s.add_argument("--conjunto", required=True, choices=["VAL_EX", "TEST_BOX", "CAL"])
            s.add_argument("--op", help="calibração em CAL (por omissão outputs/eval/CAL/<método>.json)")
            s.add_argument("--repetir", help="motivo para repetir uma avaliação de teste (fica registado)")
    s = sub.add_parser("comparar")
    s.add_argument("--conjunto", required=True, choices=["CAL", "VAL_EX", "TEST_BOX"])
    s.add_argument("--a", required=True)
    s.add_argument("--b", required=True)
    s.add_argument("--metrica", help="F1 (omissão em CAL/VAL_EX), R_loc (omissão em TEST_BOX), R_box, P, R, mIoU")
    s.add_argument("--op", default="OP_F1", choices=["OP_F1", "OP_R90"])
    s.add_argument("--config", default="configs/avaliacao.yaml")
    args = ap_.parse_args(argv)
    try:
        {"calibrar": cmd_calibrar, "avaliar": cmd_avaliar, "comparar": cmd_comparar}[args.comando](args)
    except ErroAvaliacao as e:
        print(f"ERRO: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
