#!/usr/bin/env python3
"""
23 — Experiência B (detetor YOLO): os dois datasets YOLO, um com a anotação original e outro com o Dataset WCE v2,
pelo desenho fixado no docs/decisoes.md a 26 out 2026, antes de treinar.

Regras (as do decisoes.md):
  - só janelas de ALL-SEEDBOX (±100 frames à volta do seed):
      treino     = casos de Dataset_Treino que não são de CAL;
      validação  = os casos de CAL (escolha da época; o corte escolhe-se depois em CAL, com o avaliador);
      teste      = as janelas de TEST_BOX, só para o mAP do val.py (secundário; não entra no treino);
  - só as caixas das 6 classes, pela classe_v2 da janela: 0 angiectasia, 1 erosao, 2 ulcera, 3 polipo, 4 eritema,
    5 xantelasma (as de «outras» não entram);
  - anotação original = estado original; v2 = original + nova_aceite;
  - só frames com pelo menos uma caixa: um frame sem caixa não é um negativo;
  - cada frame físico (split, caso, frame) entra uma vez, com as caixas das janelas que o têm. Caixas que se sobrepõem
    (IoU ≥ 0,3) são a mesma lesão e fica uma: a original antes da nova, depois a da janela com o seed mais perto do
    frame (como no passo 0.4), depois a do seed mais pequeno. As que não se sobrepõem ficam todas;
  - as imagens não se copiam: são ligações simbólicas para as do $DS.

Lê:
  data/derived/Dataset_WCE_v2/caixas_v2.csv   as caixas da v2 (confirma o sha256 do C10)
  sets/CAL.json, sets/VAL_EX.json, sets/TEST_BOX.json   (a pasta vem do configs/avaliacao.yaml)
  $DS/<split>/<caso>/GT_com_classes/<seed>/<caso>_<frame>_<seed>.jpg
Grava em data/derived/yolo/ (fora do git):
  original/, v2/   images/{train,val,test}/ (ligações), labels/{train,val,test}/ (uma linha por caixa: classe xc yc w h,
                   normalizados 0–1), treino.yaml (train e val, para o train.py), teste.yaml (com test, para o val.py)
                   e manifesto.csv (uma linha por imagem: de onde vem e quantas caixas tem)
  conjuntos/       CAL_6classes.csv, VAL_EX_6classes.csv e TEST_BOX_6classes.csv: as janelas das 6 classes de cada
                   conjunto (split, caso, seed, classe), para o --conjunto-ficheiro do avaliador e para a inferência
e o resumo em outputs/23_yolo_dados_resumo.txt (só números agregados: pode colar-se no chat), com a linha para o
docs/decisoes.md. Se data/derived/yolo/ já existir, não a reescreve: --forcar apaga-a e volta a criá-la.

Uso, a partir da pasta do projeto, depois do source …/inicio.sh e do export DS=…:
    python scripts/23_yolo_dados.py
Versão 1.0 (26 out 2026).
"""
import argparse
import datetime
import hashlib
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "eval"))
import evaluate as ev  # noqa: E402

VERSAO = "1.0"
SHA_CAIXAS = "cb77148173f0cae3"          # caixas_v2.csv do C10 (25 out 2026)
TREINO, TESTE = "Dataset_Treino", "Dataset_Teste"
ABREV = {TREINO: "tr", TESTE: "te"}
VARIANTES = {"original": ("original",), "v2": ("original", "nova_aceite")}
PARTES = ("train", "val", "test")
NOME_PARTE = {"train": "treino", "val": "validação", "test": "teste"}
CONJUNTOS = ("CAL", "VAL_EX", "TEST_BOX")
K3 = ["split", "caso", "seed"]
CAIXA = ["box_x1", "box_y1", "box_x2", "box_y2"]
PRECISAS = K3 + ["frame", "offset", "estado", "classe_v2", "all_seedbox"] + CAIXA
NOMES_PT = {"angiectasia": "angiectasia", "erosao": "erosão", "ulcera": "úlcera", "polipo": "pólipo",
            "eritema": "eritema", "xantelasma": "xantelasma"}


def _n(x):
    """Inteiro com espaço nos milhares, como na tese: 10 240."""
    return f"{int(x):,}".replace(",", " ")


def _pl(n, um, varios):
    return f"{_n(n)} {um if n == 1 else varios}"


# ============================================================================ leitura
def _inteiros(s, nome, fich):
    v = pd.to_numeric(s, errors="coerce")
    if v.isna().any() or (v != v.round()).any():
        raise ev.ErroAvaliacao(f"{fich}: a coluna {nome} tem valores que não são números inteiros.")
    return v.astype(int)


def ler_caixas(caminho, sha_esperado):
    """As caixas da v2 (uma linha por caixa e janela), com tipos certos. Recusa um ficheiro que não seja o do C10."""
    p = Path(caminho)
    if not p.is_file():
        raise ev.ErroAvaliacao(f"Não encontrei {p} (é gerado pelo scripts/19_c_dataset_v2.py).")
    sha = ev._sha256(p)
    if sha_esperado and not sha.startswith(sha_esperado):
        raise ev.ErroAvaliacao(f"{p} não é o do Dataset WCE v2 (C10): sha256 {sha[:16]}…, esperado {sha_esperado}….")
    c = pd.read_csv(p, dtype=str, keep_default_na=False)
    falta = [x for x in PRECISAS if x not in c.columns]
    if falta:
        raise ev.ErroAvaliacao(f"{p.name}: faltam as colunas {falta}.")
    c["split"] = c["split"].str.strip()
    c["caso"] = c["caso"].str.strip().str.zfill(4)
    for x in ("seed", "frame", "offset"):
        c[x] = _inteiros(c[x], x, p.name)
    c["all_seedbox"] = pd.to_numeric(c["all_seedbox"], errors="coerce").fillna(0).astype(int)
    c["estado"] = c["estado"].str.strip()
    c["classe_v2"] = c["classe_v2"].str.strip()
    estranhos = sorted(set(c["estado"]) - {"original", "nova_aceite"})
    if estranhos:
        raise ev.ErroAvaliacao(f"{p.name}: estados desconhecidos {estranhos} (só há original e nova_aceite).")
    if "excluida" in c.columns:
        c = c[c["excluida"].str.strip() != "1"].copy()
    dup = c[c.duplicated(K3 + ["frame"], keep=False)]
    if len(dup):
        r = dup.iloc[0]
        raise ev.ErroAvaliacao(f"{p.name}: há frames com mais de uma caixa na mesma janela (ex.: {r['split']}/{r['caso']}/"
                               f"{r['seed']}, frame {r['frame']}).")
    v = c[CAIXA].apply(pd.to_numeric, errors="coerce")
    e = 1e-6
    mal = v.isna().any(axis=1) | (v.min(axis=1) < -e) | (v.max(axis=1) > 1 + e) \
        | (v["box_x2"] <= v["box_x1"]) | (v["box_y2"] <= v["box_y1"])
    if mal.any():
        r = c[mal].iloc[0]
        raise ev.ErroAvaliacao(f"{p.name}: {_pl(int(mal.sum()), 'caixa inválida', 'caixas inválidas')} (ex.: "
                               f"{r['split']}/{r['caso']}/{r['seed']}, frame {r['frame']}).")
    c[CAIXA] = v.clip(0, 1)
    return c.reset_index(drop=True), sha


def classes_das_janelas(c, conjuntos, cfg, principais):
    """{janela: classe} para as janelas dos conjuntos: a classe_v2 da v2 (uma por janela); para uma janela sem caixas na
    v2, a da janelas_v1.csv (categoria principal ou «outras»). Devolve também quantas vieram da janelas_v1."""
    g = c.groupby(K3)["classe_v2"].agg(lambda s: sorted(set(s)))
    varias = [k for k, v in g.items() if len(v) > 1]
    if varias:
        raise ev.ErroAvaliacao(f"A janela {ev._txt(varias[0])} tem mais de uma classe_v2 no caixas_v2.csv.")
    cls = {(s, ca, int(se)): v[0] for (s, ca, se), v in g.items()}
    faltam = sorted({k for js in conjuntos.values() for k in js if k not in cls})
    da_v1 = 0
    if faltam:
        p = Path(cfg["caminhos"]["janelas"])
        if not p.is_file():
            raise ev.ErroAvaliacao(f"{len(faltam)} janelas dos conjuntos não têm caixas na v2 e não encontrei {p}.")
        jt = pd.read_csv(p, dtype=str, keep_default_na=False)
        cat = {ev.chave(r["split"], r["caso"], r["seed"]): r["categoria"].strip() for r in jt.to_dict("records")}
        for k in faltam:
            if k not in cat:
                raise ev.ErroAvaliacao(f"A janela {ev._txt(k)} não está no caixas_v2.csv nem em {p}.")
            cls[k] = cat[k] if cat[k] in principais else "outras"
            da_v1 += 1
    return cls, da_v1


# ============================================================================ juntar por frame físico
def juntar_frames(t, estados, limiar):
    """Uma entrada por frame físico (split, caso, frame), com as caixas que ficam depois de juntar as das várias janelas.
    t: as caixas de uma parte (treino, validação ou teste), já só das 6 classes. Devolve (frames, contagens)."""
    x = t[t["estado"].isin(estados)].copy()
    x["prio"] = (x["estado"] != "original").astype(int)
    x["dist"] = x["offset"].abs()
    x = x.sort_values(["split", "caso", "frame", "prio", "dist", "seed"], kind="mergesort")
    frames, cont = [], {"juntadas": 0, "classes_diferentes": 0, "nova_em_original": 0, "em_varias_janelas": 0}
    cols = ["seed", "offset", "estado", "classe_v2"] + CAIXA
    for (sp, caso, fr), g in x.groupby(["split", "caso", "frame"], sort=True):
        linhas = g[cols].to_dict("records")
        if g["seed"].nunique() > 1:
            cont["em_varias_janelas"] += 1
        ficam = []
        for r in linhas:
            b = [r[k] for k in CAIXA]
            sobre = [k for k in ficam if ev.iou(b, [k[c] for c in CAIXA])[0] >= limiar]
            if sobre:
                cont["juntadas"] += 1
                cont["classes_diferentes"] += int(r["classe_v2"] != sobre[0]["classe_v2"])
                cont["nova_em_original"] += int(r["estado"] == "nova_aceite" and sobre[0]["estado"] == "original")
                sobre[0]["juntou"] = sobre[0].get("juntou", 0) + 1
            else:
                ficam.append(dict(r))
        frames.append({"split": sp, "caso": caso, "frame": int(fr), "caixas": ficam,
                       "seeds": list(dict.fromkeys(int(s) for s in g["seed"]))})
    return frames, cont


def caminho_imagem(ds, sp, caso, frame, seed):
    return ds / sp / caso / "GT_com_classes" / str(seed) / f"{caso}_{frame}_{seed}.jpg"


def linha_yolo(cx, idx):
    x1, y1, x2, y2 = (float(cx[c]) for c in CAIXA)
    return f"{idx[cx['classe_v2']]} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}"


# ============================================================================ gravar
def gravar_variante(pasta, final, nome, partes, ds, idx, principais, seeds_do_caso):
    """Ligações, rótulos, YAML e manifesto de uma variante. Devolve contagens por parte e a impressão digital.
    A imagem de cada frame vem da cópia da janela da caixa que ficou; se faltar, das outras janelas do caso que têm o
    frame, da mais perto para a mais longe (assim a mesma imagem serve as duas variantes)."""
    base = pasta / nome
    h = hashlib.sha256()
    man, cont, sem_imagem = [], {}, []
    for parte in PARTES:
        (base / "images" / parte).mkdir(parents=True)
        (base / "labels" / parte).mkdir(parents=True)
        n_img, n_cx = 0, {cl: 0 for cl in principais}
        for f in partes[parte]:
            outras = sorted((s for s in seeds_do_caso.get((f["split"], f["caso"]), ()) if s not in f["seeds"]),
                            key=lambda s: (abs(s - f["frame"]), s))
            orig = next((p for p in (caminho_imagem(ds, f["split"], f["caso"], f["frame"], s)
                                     for s in f["seeds"] + outras) if p.is_file()), None)
            if orig is None:
                sem_imagem.append(f"{f['split']}/{f['caso']}/frame {f['frame']}")
                continue
            stem = f"{ABREV[f['split']]}_{f['caso']}_{f['frame']}"
            linhas = [linha_yolo(cx, idx) for cx in f["caixas"]]
            texto = "\n".join(linhas) + "\n"
            os.symlink(orig, base / "images" / parte / f"{stem}.jpg")
            (base / "labels" / parte / f"{stem}.txt").write_text(texto, encoding="utf-8")
            h.update(f"{parte}/{stem}\n{texto}".encode())
            n_img += 1
            for cx in f["caixas"]:
                n_cx[cx["classe_v2"]] += 1
            man.append({"imagem": f"{stem}.jpg", "parte": parte, "split": f["split"], "caso": f["caso"],
                        "frame": f["frame"], "seed_imagem": int(orig.parent.name), "n_caixas": len(f["caixas"]),
                        "classes": ";".join(str(idx[cx["classe_v2"]]) for cx in f["caixas"]),
                        "estados": ";".join(cx["estado"] for cx in f["caixas"]),
                        "juntadas": sum(cx.get("juntou", 0) for cx in f["caixas"]), "origem": str(orig)})
        cont[parte] = {"imagens": n_img, "caixas": sum(n_cx.values()), "por_classe": n_cx}
    nomes = "names:\n" + "".join(f"  {i}: {cl}\n" for i, cl in enumerate(principais))
    cab = (f"# Experiência B — dataset YOLO «{nome}» (scripts/23_yolo_dados.py v{VERSAO}). As imagens são ligações para o "
           "$DS; os rótulos são das 6 classes.\n")
    (base / "treino.yaml").write_text(
        cab + "# Só treino e validação (os casos de CAL): o teste está no teste.yaml e o train.py nunca o vê.\n"
        f"path: {final / nome}\ntrain: images/train\nval: images/val\n" + nomes, encoding="utf-8")
    (base / "teste.yaml").write_text(
        cab + "# Para o val.py --task test (mAP nos frames com caixa das janelas de TEST_BOX das 6 classes).\n"
        f"path: {final / nome}\ntrain: images/train\nval: images/val\ntest: images/test\n" + nomes, encoding="utf-8")
    pd.DataFrame(man).to_csv(base / "manifesto.csv", index=False)
    return cont, h.hexdigest(), sem_imagem


# ============================================================================ principal
def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--config", default="configs/avaliacao.yaml")
    a.add_argument("--caixas", default="data/derived/Dataset_WCE_v2/caixas_v2.csv")
    a.add_argument("--ds", default=os.environ.get("DS", ""), help="pasta do Dataset WCE (por omissão, a variável DS)")
    a.add_argument("--saida", default="data/derived/yolo")
    a.add_argument("--resumo", default="outputs/23_yolo_dados_resumo.txt")
    a.add_argument("--sha-caixas", default=SHA_CAIXAS, help="início do sha256 esperado do caixas_v2.csv ('' não confirma)")
    a.add_argument("--iou-mesma", type=float, default=0.3, help="IoU a partir do qual duas caixas são a mesma lesão")
    a.add_argument("--forcar", action="store_true", help="apaga a pasta de saída, se existir, e volta a criá-la")
    args = a.parse_args(argv)
    cfg = ev.ler_config(args.config)
    principais = list(cfg["estratos"]["classes_principais"])
    idx = {cl: i for i, cl in enumerate(principais)}
    if not args.ds:
        raise ev.ErroAvaliacao("Falta a pasta do dataset: faz export DS=/mnt/paulo/tese_2026/Update_Final/raw/Dataset_WCE "
                               "(ou passa --ds).")
    ds = Path(args.ds).resolve()
    if not ds.is_dir():
        raise ev.ErroAvaliacao(f"Não encontrei a pasta do dataset {ds}.")
    saida = Path(args.saida).resolve()
    if saida.exists() and not args.forcar:
        raise ev.ErroAvaliacao(f"{saida} já existe: não a reescrevo. Para a refazer, corre com --forcar (apaga-a e volta a "
                               "criá-la).")

    # 1. entradas
    c, sha_c = ler_caixas(args.caixas, args.sha_caixas)
    pasta_conj = Path(cfg["caminhos"]["conjuntos"])
    conj = {n: ev.ler_conjunto(pasta_conj / f"{n}.json") for n in CONJUNTOS}
    cls, da_v1 = classes_das_janelas(c, conj, cfg, principais)
    casos_cal = {k[1] for k in conj["CAL"]}
    if any(k[0] != TREINO for k in conj["CAL"]) or any(k[0] != TESTE for n in ("VAL_EX", "TEST_BOX") for k in conj[n]):
        raise ev.ErroAvaliacao("CAL tem de ser de Dataset_Treino e VAL_EX e TEST_BOX de Dataset_Teste.")

    # 2. partes: só ALL-SEEDBOX e só as 6 classes
    t = c[c["all_seedbox"] == 1].copy()
    jan = list(zip(t["split"], t["caso"], t["seed"]))
    tb = set(conj["TEST_BOX"])
    t["parte"] = np.where(t["split"] == TREINO, np.where(t["caso"].isin(casos_cal), "val", "train"),
                          np.where([k in tb for k in jan], "test", ""))
    janelas_teste_fora = len({k for k, p in zip(jan, t["parte"]) if p == "" and k[0] == TESTE})
    tb_sem_v2 = len(tb - set(jan))
    t = t[t["parte"] != ""]
    n_outras = int((~t["classe_v2"].isin(principais)).sum())
    t = t[t["classe_v2"].isin(principais)]
    casos = {p: set(t.loc[t["parte"] == p, "caso"].map(str) + "@" + t.loc[t["parte"] == p, "split"]) for p in PARTES}
    if casos["train"] & casos["val"] or casos["test"] & (casos["train"] | casos["val"]):
        raise ev.ErroAvaliacao("Há casos em mais de uma parte (treino, validação, teste). Não gravei nada.")
    n_jan = {p: t.loc[t["parte"] == p, K3].drop_duplicates().shape[0] for p in PARTES}

    # 3. juntar por frame físico, nas duas variantes
    juntos, cont_j = {}, {}
    for v, estados in VARIANTES.items():
        juntos[v], cont_j[v] = {}, {}
        for p in PARTES:
            juntos[v][p], cont_j[v][p] = juntar_frames(t[t["parte"] == p], estados, args.iou_mesma)

    # 4. gravar numa pasta temporária ao lado e trocar no fim (tudo ou nada)
    saida.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{saida.name}.", dir=saida.parent))
    try:
        res = {}
        seeds_do_caso = c.groupby(["split", "caso"])["seed"].agg(lambda s: sorted(set(int(x) for x in s))).to_dict()
        for v in VARIANTES:
            res[v] = gravar_variante(tmp, saida, v, juntos[v], ds, idx, principais, seeds_do_caso)
        (tmp / "conjuntos").mkdir()
        sub = {}
        for n in CONJUNTOS:
            linhas = [{"split": k[0], "caso": k[1], "seed": k[2], "classe": cls[k]} for k in conj[n]
                      if cls[k] in principais]
            pd.DataFrame(linhas, columns=K3 + ["classe"]).to_csv(tmp / "conjuntos" / f"{n}_6classes.csv", index=False)
            sub[n] = (len(linhas), len(conj[n]))
        if saida.exists():
            shutil.rmtree(saida)
        os.replace(tmp, saida)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)

    # 5. resumo (só números agregados)
    git = ev._estado_git(RAIZ)
    commit = git["commit"] or "sem git"
    agora = datetime.datetime.now()
    L = [f"23 — Experiência B: dados do YOLO · {agora:%Y-%m-%d %H:%M} · "
         + (f"commit {git['commit']}" if git["commit"] else "sem git")
         + (" (com alterações por guardar!)" if git["alteracoes_por_guardar"] else ""),
         f"23_yolo_dados.py v{VERSAO} · {Path(args.caixas).name} sha256 {sha_c[:16]} · caixas da mesma lesão: IoU ≥ "
         f"{str(args.iou_mesma).replace('.', ',')}",
         "Classes (índice YOLO): " + " · ".join(f"{i} {NOMES_PT.get(cl, cl)}" for i, cl in enumerate(principais)),
         "",
         f"Casos: treino {_n(len(casos['train']))} ({_n(n_jan['train'])} janelas) · validação = casos de CAL "
         f"{_n(len(casos['val']))} ({_n(n_jan['val'])} janelas) · teste {_n(len(casos['test']))} ({_n(n_jan['test'])} "
         "janelas de TEST_BOX)",
         f"Caixas de «outras» que não entram: {_n(n_outras)}",
         "",
         "                     imagens  caixas  " + "  ".join(f"{NOMES_PT.get(cl, cl)[:7]:>7}" for cl in principais)]
    for v in VARIANTES:
        for p in PARTES:
            r = res[v][0][p]
            L.append(f"  {v:<8} {NOME_PARTE[p]:<10} {_n(r['imagens']):>7} {_n(r['caixas']):>7}  "
                     + "  ".join(f"{_n(r['por_classe'][cl]):>7}" for cl in principais))
    L.append("")
    for v in VARIANTES:
        j = {k: sum(cont_j[v][p][k] for p in PARTES) for k in cont_j[v]["train"]}
        L.append(f"Juntadas ({v}): {_pl(j['juntadas'], 'caixa', 'caixas')} (a mesma lesão noutra janela"
                 + (f"; {_pl(j['nova_em_original'], 'nova', 'novas')} sobre uma original" if v == "v2" else "")
                 + f") · {_n(j['classes_diferentes'])} com classe diferente · "
                 f"{_pl(j['em_varias_janelas'], 'frame', 'frames')} com caixas de mais de uma janela")
    for v in VARIANTES:
        s = res[v][2]
        if s:
            L.append(f"atenção: {_pl(len(s), 'frame', 'frames')} com caixa sem imagem no $DS ({v}; ex.: {s[0]}): "
                     "ficaram de fora")
    if janelas_teste_fora or tb_sem_v2:
        L.append(f"atenção: {janelas_teste_fora} janelas de teste de ALL-SEEDBOX fora de TEST_BOX · {tb_sem_v2} janelas "
                 "de TEST_BOX sem caixas na v2")
    if da_v1:
        L.append(f"atenção: {_pl(da_v1, 'janela', 'janelas')} dos conjuntos sem caixas na v2 (classe da janelas_v1.csv)")
    L += ["",
          "Janelas das 6 classes para o avaliador (conjuntos/): "
          + " · ".join(f"{n} {_n(sub[n][0])} de {_n(sub[n][1])}" for n in CONJUNTOS),
          f"Pasta: {saida}/ (original/, v2/, conjuntos/)",
          "Impressões digitais (rótulos): " + " · ".join(f"{v} {res[v][1][:16]}" for v in VARIANTES),
          ""]

    def _parte(v):
        r = res[v][0]
        return ", ".join(f"{NOME_PARTE[p]} {_n(r[p]['imagens'])} imagens / {_n(r[p]['caixas'])} caixas" for p in PARTES)

    linha = (f"- {datetime.date.today().isoformat()}: experiência B, dados do YOLO (scripts/23_yolo_dados.py v{VERSAO}, "
             f"commit {commit}; caixas_v2.csv sha256 {sha_c[:16]}): {Path(args.saida).as_posix()}/ com original/ e v2/ (ligações para o "
             f"$DS, rótulos das 6 classes, treino.yaml e teste.yaml) e conjuntos/ (janelas das 6 classes: "
             + ", ".join(f"{n.replace('_', '-')} {_n(sub[n][0])}" for n in CONJUNTOS)
             + f"). Original: {_parte('original')}. v2: {_parte('v2')}. Validação = os {_n(len(casos['val']))} casos de "
             f"CAL. Caixas da mesma lesão juntadas por frame físico (IoU ≥ 0,3): original "
             f"{_n(sum(cont_j['original'][p]['juntadas'] for p in PARTES))}, v2 "
             f"{_n(sum(cont_j['v2'][p]['juntadas'] for p in PARTES))}. Impressões digitais: original "
             f"{res['original'][1][:16]}, v2 {res['v2'][1][:16]}.")
    L += ["Linha para o docs/decisoes.md:", linha]
    texto = "\n".join(L)
    rs = Path(args.resumo)
    rs.parent.mkdir(parents=True, exist_ok=True)
    rs.write_text(texto + "\n", encoding="utf-8")
    print(texto)
    return {"saida": saida, "contagens": {v: res[v][0] for v in VARIANTES}, "juntadas": cont_j,
            "impressoes": {v: res[v][1] for v in VARIANTES}, "sem_imagem": {v: res[v][2] for v in VARIANTES},
            "subconjuntos": sub, "casos": casos, "linha": linha, "texto": texto, "resumo": rs}


if __name__ == "__main__":
    try:
        main()
    except ev.ErroAvaliacao as e:
        print(f"ERRO: {e}", file=sys.stderr)
        sys.exit(2)
