#!/usr/bin/env python3
"""
18 — Guião C, C9: análise da revisão humana das caixas novas da anotação expandida v1, com as regras fixadas antes da
revisão (docs/decisoes.md, 24 out 2026). Tudo por script, a partir das respostas consolidadas (tools/rever.py
--consolidar) e da chave escondida da lista (scripts/17_c_itens.py); nada se conta à mão.

Dois passos:

1) Classe das janelas com mais de uma lesão no texto (regra de 24 out; aplica-se às tabelas por classe dos Guiões A e C):
       python scripts/18_c_analise.py --preparar-classes
   Grava data/derived/guiao_c/classes_mistas.csv (uma linha por janela) e outputs/18_c_analise/classes_mistas.png (o
   seed com a caixa azul e a imagem do relatório de cada uma) e pergunta, janela a janela, que lesão está dentro da caixa
   do seed. Enter = não é claro: fica a primeira mencionada no texto. Grava depois de cada resposta; se for interrompido,
   volta a correr-se e só pergunta as que faltam. Não abre a chave da lista nem a anotação expandida.

2) Análise, depois de consolidar a revisão:
       python scripts/18_c_analise.py
   - Precisão das caixas novas = aceitar / (aceitar + rejeitar). «Não dá para decidir» (N) fica fora da conta e é
     contado à parte; na análise de sensibilidade, as N contam como rejeitadas. Número principal: as caixas novas do
     teste; o treino à parte, sem as janelas de CAL (onde se escolheu o τ); e o total. IC 95 % por bootstrap por janela
     (configs/avaliacao.yaml). Estratos: distância ao seed, classe, tamanho da caixa do seed; as caixas em frames sem
     ficheiro de caixa à parte. Uma «mal posta» (só existe com --ajustar) conta como rejeitada.
   - Controlos às cegas (se a lista os tiver): taxa de aceitação nos com caixa original e nos abaixo do limiar; nos
     primeiros, «aceitar» contra IoU ≥ 0,5 entre a caixa do SAM 2 e a caixa original.
   - Concordâncias sem trabalho extra: com a PF v2 (janelas de CAL e VAL-EX) e com a classificação dos 50 FP (20–22 out).
   - Ganho: caixas por lesão antes (seed + caixas originais, a regra do script 15) e depois (+ novas aceites), por classe
     e por split. Tempo por caixa, sem as pausas (> 120 s). Janelas com a nota «seed duvidoso».
   - Guião A: o R_loc por classe em TEST-BOX (OP-F1) refeito com a classe nova das janelas com mais de uma lesão no
     texto; a tabela antiga é recalculada e conferida com a do avaliador.
   Grava em outputs/18_c_analise/: resumo.md (o que aparece no ecrã, com a linha para o docs/decisoes.md), precisao.csv,
   controlos.csv, concordancias.csv, ganho.csv, tempo.csv, guiao_a_rloc_classe.csv e as figuras fig_c_precisao e
   fig_c_ganho (png e pdf). O resumo só tem números agregados: pode colar-se no chat.

Versão 1.0 (24 out 2026).
"""
import argparse
import datetime
import json
import os
import re
import sys
import tempfile
import textwrap
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "eval"))
import evaluate as ev  # noqa: E402

VERSAO = "1.0"
K3 = ["split", "caso", "seed"]
K4 = K3 + ["frame"]
CAIXA = ["box_x1", "box_y1", "box_x2", "box_y2"]
REF = ["rx1", "ry1", "rx2", "ry2"]
RESPOSTAS = ("aceitar", "rejeitar", "nao_avaliavel", "mal_posta")
PAUSA_MS = 120_000
TESTE, TREINO = "Dataset_Teste", "Dataset_Treino"
NOTA_SEED = "seed duvidoso"
LESOES = {"angiectasia": r"(?<!linf)angiect|angiodispl", "erosao": r"eros", "ulcera": r"ulcer", "polipo": r"polip",
          "eritema": r"eritem", "xantelasma": r"xantel", "linfangiectasia": r"linfang", "sangue": r"sangu|hemorr|coagul"}
FP_PARA_RESPOSTA = {"lesao_nao_anotada": "aceitar", "fundo_mucosa": "rejeitar", "outra_lesao": "rejeitar",
                    "nao_avaliavel": "nao_avaliavel"}
COL_CLASSES = ["n", "split", "caso", "seed", "opcoes", "primeira", "categoria_v1", "classe_caixa", "descricao"]
NOMES = {"angiectasia": "angiectasia", "erosao": "erosão", "ulcera": "úlcera", "polipo": "pólipo", "eritema": "eritema",
         "xantelasma": "xantelasma", "outras": "outras", "linfangiectasia": "linfangiectasia", "sangue": "sangue",
         "pequena": "pequena", "media": "média", "grande": "grande", TESTE: "teste", TREINO: "treino",
         "teste": "teste (principal)", "treino_sem_cal": "treino sem CAL", "total": "total"}
AZUL, CINZA, TINTA, TINTA2, GRELHA = "#2a78d6", "#80868b", "#1f2328", "#59636e", "#e6e8eb"
CIANO = "#00e5ff"                                   # a cor da caixa do seed na ferramenta de revisão
RE_MATCH = re.compile(r"match_at_videoframe(\d+)", re.I)
RE_IMAGEM_N = re.compile(r"^imagem(\d+)_", re.I)


# ============================================================================ utilitários
def _n(x):
    """2269 → «2 269», como na tese."""
    return f"{int(x):,}".replace(",", " ")


def _pl(n, um, varios):
    """1 → «1 janela»; 2 → «2 janelas»."""
    return f"{_n(n)} {um if int(n) == 1 else varios}"


def _p(x, casas=3):
    if x is None or (isinstance(x, (float, np.floating)) and not np.isfinite(x)):
        return "—"
    return f"{float(x):.{casas}f}".replace(".", ",")


def _ic(r, casas=3):
    """{"valor", "ic95"} → «0,714 [0,600–0,820]»."""
    if not r or r.get("valor") is None:
        return "—"
    ic = r.get("ic95")
    return _p(r["valor"], casas) + (f" [{_p(ic[0], casas)}–{_p(ic[1], casas)}]" if ic and None not in ic else "")


def _tira(t):
    """Sem acentos e em minúsculas (para procurar palavras no texto livre)."""
    return unicodedata.normalize("NFKD", str(t)).encode("ascii", "ignore").decode().lower()


def lesoes_no_texto(texto):
    """Lesões nomeadas no texto, pela ordem da primeira menção (linfangiectasia não conta como angiectasia)."""
    t, pos = _tira(texto), {}
    for c, rx in LESOES.items():
        m = re.search(rx, t)
        if m:
            pos[c] = m.start()
    return sorted(pos, key=lambda c: (pos[c], c))


def kappa(a, b):
    """κ de Cohen entre duas classificações binárias."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    if not len(a):
        return None
    po = float((a == b).mean())
    pa, pb = a.mean(), b.mean()
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe >= 1 - 1e-12 else float((po - pe) / (1 - pe))


def concordancia(nome, a, b, rot_a, rot_b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    return {"comparacao": nome, "a": rot_a, "b": rot_b, "n": int(len(a)),
            "concordancia": float((a == b).mean()) if len(a) else None, "kappa": kappa(a, b),
            "a_sim_b_sim": int((a & b).sum()), "a_sim_b_nao": int((a & ~b).sum()),
            "a_nao_b_sim": int((~a & b).sum()), "a_nao_b_nao": int((~a & ~b).sum())}


def _gravar_csv(t, destino):
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=destino.parent, delete=False,
                                     prefix=f".{destino.name}.", suffix=".tmp") as f:
        t.to_csv(f, index=False)
    os.replace(f.name, destino)


def referencia(cfg, janelas):
    """Anotação original de cada frame e classe/tercil de cada janela, com as regras do avaliador (como no script 17)."""
    cfg2 = dict(cfg)
    cfg2["conjuntos"] = dict(cfg["conjuntos"])
    cfg2["conjuntos"]["_GUIAO_C"] = {"referencia": ["caixas"]}
    return ev.carregar_referencia(cfg2, "_GUIAO_C", janelas)


def _principal(c, cfg):
    return c if c in cfg["estratos"]["classes_principais"] else "outras"


# ============================================================================ 1) classes das janelas com duas lesões
def janelas_mistas(cfg):
    """Janelas de ALL_SEEDBOX cuja descrição clínica nomeia mais de uma lesão."""
    alvo = set(ev.ler_conjunto(Path(cfg["caminhos"]["conjuntos"]) / "ALL_SEEDBOX.json"))
    jt = pd.read_csv(cfg["caminhos"]["janelas"], dtype=str, keep_default_na=False)
    if "descricao" not in jt.columns:
        raise ev.ErroAvaliacao(f"{cfg['caminhos']['janelas']} não tem a coluna 'descricao'.")
    linhas = []
    for r in jt.to_dict("records"):
        k = ev.chave(r["split"], r["caso"], r["seed"])
        les = lesoes_no_texto(r["descricao"]) if k in alvo else []
        if len(les) > 1:
            linhas.append({"split": k[0], "caso": k[1], "seed": k[2], "opcoes": "|".join(les), "primeira": les[0],
                           "categoria_v1": r.get("categoria", ""), "classe_caixa": "", "descricao": r["descricao"]})
    t = pd.DataFrame(linhas, columns=COL_CLASSES[1:]).sort_values(K3, kind="mergesort").reset_index(drop=True)
    t.insert(0, "n", np.arange(1, len(t) + 1))
    return t


def ler_classes(caminho, cfg):
    """{(split, caso, seed): classe} das janelas com mais de uma lesão no texto: a lesão dentro da caixa do seed
    (classe_caixa) ou, se não for clara («primeira» ou vazia), a primeira mencionada. Devolve também as contagens."""
    p = Path(caminho)
    if not p.is_file():
        raise ev.ErroAvaliacao(f"Falta {p}: a classe das janelas com mais de uma lesão no texto escolhe-se antes da "
                               "análise. Corre primeiro: python scripts/18_c_analise.py --preparar-classes")
    t = pd.read_csv(p, dtype=str, keep_default_na=False)
    falta = [c for c in ("split", "caso", "seed", "opcoes", "primeira", "classe_caixa") if c not in t.columns]
    if falta:
        raise ev.ErroAvaliacao(f"{p}: faltam as colunas {falta}.")
    out, cont = {}, {"janelas": len(t), "escolhidas": 0, "nao_claras": 0, "por_responder": 0, "mudam": 0}
    for r in t.to_dict("records"):
        op = r["opcoes"].split("|")
        c = _tira(r["classe_caixa"]).strip()
        if c in ("", "primeira"):
            cont["por_responder" if c == "" else "nao_claras"] += 1
            c = r["primeira"]
        elif c not in op:
            raise ev.ErroAvaliacao(f"{p}, linha {r.get('n', '?')}: classe_caixa = {r['classe_caixa']!r} não é uma das lesões "
                                   f"do texto ({', '.join(op)}). Corrige-a ou deixa «primeira».")
        else:
            cont["escolhidas"] += 1
        k = ev.chave(r["split"], r["caso"], r["seed"])
        out[k] = _principal(c, cfg)
        cont["mudam"] += int(out[k] != _principal(r.get("categoria_v1", ""), cfg))
    return out, cont


def resumo_classes(cont):
    return (f"{cont['escolhidas']} com a lesão da caixa do seed, {_pl(cont['nao_claras'], 'não clara', 'não claras')} "
            f"(fica a primeira mencionada), {cont['por_responder']} por responder; "
            f"{_pl(cont['mudam'], 'muda', 'mudam')} de classe")


def _imagens_janela(ds, split, caso, seed):
    gt = Path(ds) / split / caso / "GT_com_classes"
    seed_img = gt / str(seed) / f"{caso}_{seed}_{seed}.jpg"
    rel = []
    if gt.is_dir():
        for f in gt.iterdir():
            m = RE_MATCH.search(f.name)
            if f.is_file() and m and int(m.group(1)) == int(seed) and f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                n = RE_IMAGEM_N.match(f.name)
                rel.append((int(n.group(1)) if n else 0, f.name, f))
    return seed_img, [f for _, _, f in sorted(rel)]


def figura_classes(t, cfg, ds, destino):
    """Uma figura com as janelas: o seed com a caixa azul e a imagem do relatório, numeradas como na folha."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from PIL import Image
    except ImportError:
        return None
    ks = [ev.chave(s, c, e) for s, c, e in zip(t["split"], t["caso"], t["seed"])]
    ref, _ = referencia(cfg, ks)
    sd = ref[(ref["offset"] == 0) & (ref["caixa"] == "caixa")].set_index(K3)
    por_linha = 2                                                   # duas janelas por linha: seed + relatório de cada
    linhas = max(1, -(-len(t) // por_linha))
    fig, axs = plt.subplots(linhas, 2 * por_linha, figsize=(14, 4.1 * linhas), squeeze=False)
    for ax in axs.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    def vazio(ax, txt):
        ax.imshow(np.full((64, 64, 3), 255, dtype=np.uint8))
        ax.text(0.5, 0.5, txt, ha="center", va="center", color=TINTA2, transform=ax.transAxes)

    def mostrar(ax, f):
        try:
            im = Image.open(f).convert("RGB")
        except (OSError, ValueError):
            vazio(ax, "(sem imagem)")
            return None
        ax.imshow(im)
        return im.size

    for i, (r, k) in enumerate(zip(t.to_dict("records"), ks)):
        a_seed, a_rel = axs[i // por_linha][2 * (i % por_linha)], axs[i // por_linha][2 * (i % por_linha) + 1]
        seed_img, rels = _imagens_janela(ds, *k)
        tam = mostrar(a_seed, seed_img)
        if tam and k in sd.index:
            b = sd.loc[k, REF].to_numpy(dtype=float)
            a_seed.add_patch(Rectangle((b[0] * tam[0], b[1] * tam[1]), (b[2] - b[0]) * tam[0], (b[3] - b[1]) * tam[1],
                                       fill=False, ec=CIANO, lw=2.2))
        if rels:
            mostrar(a_rel, rels[0])
        else:
            vazio(a_rel, "(sem imagem do relatório)")
        a_seed.set_title(f"{r['n']}. caso {k[1]} · seed {k[2]}: o seed (caixa azul)", fontsize=9, color=TINTA, loc="left")
        a_rel.set_title("imagem do relatório" + (f" (+{len(rels) - 1})" if len(rels) > 1 else ""), fontsize=9,
                        color=TINTA, loc="left")
        texto = [textwrap.fill(x.replace("_", " ").strip(), 62) for x in r["descricao"].split(" | ")]
        opcoes = " ou ".join(NOMES.get(o, o) for o in r["opcoes"].split("|"))
        a_seed.set_xlabel("\n".join(texto) + f"\nopções: {opcoes}", fontsize=8, color=TINTA2, loc="left")
    fig.tight_layout()
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino, dpi=110, facecolor="white")
    plt.close(fig)
    return destino


def perguntar(t, destino, entrada, figura=None):
    """Pergunta, janela a janela, que lesão está dentro da caixa do seed. Grava depois de cada resposta; só pergunta as
    que ainda não têm resposta."""
    por_fazer = [i for i in t.index if t.at[i, "classe_caixa"].strip() == ""]
    if not por_fazer:
        return 0
    print(f"\nAbre a imagem {figura or '(sem imagem)'} (clica nela no explorador do VS Code) e responde, para cada "
          "janela, que lesão está dentro da caixa azul do seed. Ctrl+C pára: o que já respondeste fica gravado.")
    for i in por_fazer:
        op = t.at[i, "opcoes"].split("|")
        menu = " · ".join(f"{j} = {NOMES.get(o, o)}" for j, o in enumerate(op, 1))
        print(f"\nJanela {t.at[i, 'n']} de {len(t)} (caso {t.at[i, 'caso']}, seed {t.at[i, 'seed']}): {menu}")
        while True:
            x = str(entrada(f"  Qual está na caixa azul? (Enter = não é claro: fica {NOMES.get(op[0], op[0])}) ")).strip()
            if x == "":
                t.at[i, "classe_caixa"] = "primeira"
                break
            if x.isdigit() and 1 <= int(x) <= len(op):
                t.at[i, "classe_caixa"] = op[int(x) - 1]
                break
            print(f"  Responde com um número de 1 a {len(op)}, ou só Enter.")
        _gravar_csv(t, destino)
    return len(por_fazer)


def preparar_classes(args, cfg, entrada=None):
    if not args.ds or not Path(args.ds).is_dir():
        raise ev.ErroAvaliacao('Falta a pasta do dataset, para a imagem: faz o export DS=…/raw/Dataset_WCE (ou usa --ds "$DS").')
    destino = Path(args.classes)
    novo = janelas_mistas(cfg)
    if destino.is_file():
        t = pd.read_csv(destino, dtype=str, keep_default_na=False)
        if sorted(zip(t["split"], t["caso"].str.zfill(4), pd.to_numeric(t["seed"]).astype(int))) != \
                sorted(zip(novo["split"], novo["caso"], novo["seed"])):
            raise ev.ErroAvaliacao(f"{destino} não tem as janelas que encontro agora nos textos (mudou o janelas_v1.csv?). "
                                   "Confirma e, se for para recomeçar, apaga-o.")
        estado = "já existia (mantido)"
    else:
        t = novo.astype(str)
        _gravar_csv(t, destino)
        estado = "gravado"
    print(f"{destino}: {len(t)} janelas de ALL_SEEDBOX com mais de uma lesão no texto · {estado}")
    combinacoes = [" + ".join(NOMES.get(o, o) for o in sorted(x.split("|"))) for x in t["opcoes"]]
    for comb, n in pd.Series(combinacoes, dtype=str).value_counts().items():
        print(f"  {comb}: {n}")
    fig = figura_classes(t, cfg, args.ds, Path(args.saida) / "classes_mistas.png") if len(t) else None
    if fig:
        print(f"Imagem: {fig}")
    elif len(t):
        print("Sem matplotlib ou Pillow: não fiz a imagem.")
    if entrada is None and sys.stdin.isatty():
        entrada = input
    if entrada is not None:
        perguntar(t, destino, entrada, fig)
    _, cont = ler_classes(destino, cfg)
    txt = f"\nClasse das {cont['janelas']} janelas com mais de uma lesão no texto: {resumo_classes(cont)}"
    if cont["por_responder"]:
        txt += ("\n  Faltam respostas: volta a correr este comando num terminal (ou escreve a lesão, ou «primeira», na "
                f"coluna classe_caixa de {destino}).")
    print(txt)
    return {"tabela": t, "figura": fig, "contagens": cont}


# ============================================================================ 2) análise: leitura
def ler_itens(caminho):
    p = Path(caminho)
    if not p.is_file():
        raise ev.ErroAvaliacao(f"Não encontrei {p} (gravado pelo scripts/17_c_itens.py).")
    t = pd.read_csv(p, dtype={"caso": str, "split": str})
    t["caso"] = t["caso"].str.zfill(4)
    for c in ("item", "seed", "frame"):
        t[c] = t[c].astype(int)
    return t


def ler_respostas(caminho, itens):
    p = Path(caminho)
    if not p.is_file():
        raise ev.ErroAvaliacao(f"Não encontrei {p}: consolida primeiro a revisão "
                               "(python tools/rever.py --itens data/derived/guiao_c/itens_v1.csv --revisor paulo --consolidar).")
    r = pd.read_csv(p, dtype={"caso": str, "split": str, "decisao": str, "nota": str}, keep_default_na=False)
    falta = [c for c in ["item"] + K4 + ["decisao", "nota", "dur_ms"] if c not in r.columns]
    if falta:
        raise ev.ErroAvaliacao(f"{p}: faltam as colunas {falta}.")
    r["caso"] = r["caso"].str.zfill(4)
    for c in ("item", "seed", "frame"):
        r[c] = pd.to_numeric(r[c]).astype(int)
    mal = sorted(set(r["decisao"]) - set(RESPOSTAS))
    if mal:
        raise ev.ErroAvaliacao(f"{p}: respostas desconhecidas {mal}.")
    if r["item"].duplicated().any() or set(r["item"]) != set(itens["item"]):
        raise ev.ErroAvaliacao(f"{p} não tem exatamente uma resposta por caixa da lista ({_n(len(itens))} caixas; tem "
                               f"{_n(len(r))} linhas). Consolida outra vez.")
    r["dur_ms"] = pd.to_numeric(r["dur_ms"], errors="coerce")
    return r


def ler_chave(caminho):
    p = Path(caminho)
    if not p.is_file():
        raise ev.ErroAvaliacao(f"Não encontrei {p} (gravado pelo scripts/17_c_itens.py).")
    k = pd.read_csv(p, dtype={"caso": str, "split": str}, keep_default_na=False)
    k["caso"] = k["caso"].str.zfill(4)
    for c in ("item", "seed", "frame", "offset", "caixa_original", "cal", "val_ex", "test_box"):
        k[c] = pd.to_numeric(k[c]).astype(int)
    return k


def juntar(resp, chave, itens):
    d = resp.merge(chave, on="item", how="left", suffixes=("", "_chave"))
    d = d.merge(itens[["item"] + K4 + CAIXA], on="item", how="left", suffixes=("", "_itens"))
    for sufixo, nome in (("_chave", "a chave"), ("_itens", "a lista")):
        if d[[c + sufixo for c in K4]].isna().any().any() or \
                not all((d[c] == d[c + sufixo]).all() for c in K4):
            raise ev.ErroAvaliacao(f"As respostas não batem com {nome} (janela ou frame diferentes no mesmo item): são de "
                                   "outra lista?")
    return d.drop(columns=[c + s for c in K4 for s in ("_chave", "_itens")])


# ============================================================================ 2) análise: medidas
class Contas:
    """Precisão, com IC por bootstrap sobre as janelas da lista, de qualquer subconjunto das caixas revistas."""

    def __init__(self, d, cfg):
        ks = list(zip(d["split"], d["caso"], d["seed"]))
        self.janelas = sorted(set(ks))
        idx = {k: i for i, k in enumerate(self.janelas)}
        self.w = np.array([idx[k] for k in ks], dtype=int)
        self.dec = d["decisao"].to_numpy()
        bs = cfg["bootstrap"]
        self.boot = ev.Bootstrap(len(self.janelas), bs["reamostragens"], bs["semente"], bs["nivel"])

    def medir(self, m):
        m = np.asarray(m, bool)
        W = len(self.janelas)

        def conta(x):
            return np.bincount(self.w[m], weights=x[m].astype(float), minlength=W)
        a = conta(self.dec == "aceitar")
        r = conta(np.isin(self.dec, ["rejeitar", "mal_posta"]))
        n = conta(self.dec == "nao_avaliavel")
        sel = (a + r + n) > 0
        return {"precisao": self.boot.razao(a, a + r, sel), "precisao_n_rejeitadas": self.boot.razao(a, a + r + n, sel),
                "aceitar": int(a.sum()), "rejeitar": int(r.sum()), "nao_decidir": int(n.sum()), "janelas": int(sel.sum())}


def _linha(sub, estrato, x):
    p, s = x["precisao"], x["precisao_n_rejeitadas"]
    ic, ics = p.get("ic95") or [None, None], s.get("ic95") or [None, None]
    return {"subconjunto": sub, "estrato": estrato, "caixas": x["aceitar"] + x["rejeitar"] + x["nao_decidir"],
            "aceitar": x["aceitar"], "rejeitar": x["rejeitar"], "nao_decidir": x["nao_decidir"], "janelas": x["janelas"],
            "precisao": p.get("valor"), "ic95_inf": ic[0], "ic95_sup": ic[1],
            "precisao_n_rejeitadas": s.get("valor"), "nr_ic95_inf": ics[0], "nr_ic95_sup": ics[1]}


def tempo(d):
    t = d["dur_ms"]
    ok = t.notna() & (t >= 0) & (t <= PAUSA_MS)
    linhas = []
    grupos = [("todas", ok)] + [(r, ok & (d["decisao"] == r)) for r in RESPOSTAS if (d["decisao"] == r).any()]
    for nome, m in grupos:
        s = t[m] / 1000
        linhas.append({"respostas": nome, "caixas": int(m.sum()),
                       "mediana_s": float(s.median()) if len(s) else None,
                       "p90_s": float(s.quantile(0.9)) if len(s) else None,
                       "media_s": float(s.mean()) if len(s) else None, "total_min": float(s.sum() / 60)})
    return pd.DataFrame(linhas), int((t.notna() & (t > PAUSA_MS)).sum()), int(t.isna().sum())


def guiao_a(caminho, over, cfg):
    """R_loc por classe em TEST-BOX (OP-F1): a tabela do Guião A recalculada a partir das contagens por janela (e
    conferida com a do avaliador) e refeita com a classe nova das janelas com mais de uma lesão no texto."""
    p = Path(caminho)
    if not p.is_file():
        return None, f"não calculado: falta {p}"
    res = json.loads(p.read_text(encoding="utf-8"))
    pj = res["por_janela"]
    caixas = np.array([float(j["OP_F1"].get("caixas") or 0) for j in pj])
    loc = np.round(np.array([float(j["OP_F1"].get("R_loc") or 0) for j in pj]) * caixas)
    c1 = np.array([j["classe"] for j in pj])
    c2 = np.array([over.get(ev.chave(j["split"], j["caso"], j["seed"]), c) for j, c in zip(pj, c1)])
    bs = cfg["bootstrap"]
    boot = ev.Bootstrap(len(pj), bs["reamostragens"], bs["semente"], bs["nivel"])
    est = (res.get("por_op", {}).get("OP_F1", {}).get("estratos", {}) or {}).get("classe", {})
    linhas, confere = [], True
    for c in list(cfg["estratos"]["classes_principais"]) + ["outras"]:
        r1, r2 = boot.razao(loc, caixas, c1 == c), boot.razao(loc, caixas, c2 == c)
        if c in est:
            v = est[c].get("R_loc", {})
            confere &= (v.get("valor") == r1["valor"]) and (v.get("ic95") == r1["ic95"])
        linhas.append({"classe": c, "janelas_antes": int((c1 == c).sum()), "caixas_antes": int(caixas[c1 == c].sum()),
                       "rloc_antes": r1["valor"], "ic95_inf_antes": (r1["ic95"] or [None, None])[0],
                       "ic95_sup_antes": (r1["ic95"] or [None, None])[1],
                       "janelas_agora": int((c2 == c).sum()), "caixas_agora": int(caixas[c2 == c].sum()),
                       "rloc_agora": r2["valor"], "ic95_inf_agora": (r2["ic95"] or [None, None])[0],
                       "ic95_sup_agora": (r2["ic95"] or [None, None])[1]})
    mud = int((c1 != c2).sum())
    txt = (f"{mud} das {len(pj)} janelas de TEST-BOX {'muda' if mud == 1 else 'mudam'} de classe; a tabela antiga, recalculada, "
           + ("confere com a do avaliador" if est and confere else
              "NÃO confere com a do avaliador: confirma o ficheiro" if est else "(sem estratos no ficheiro para conferir)"))
    return pd.DataFrame(linhas), txt


# ============================================================================ 2) análise: figuras
def figuras(saida, prec, ganho_cl, classes, faixas, ref_p):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.ticker import FuncFormatter
    except ImportError:
        return "sem matplotlib: figuras não feitas"
    virgula = FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ","))
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c9cdd2", "axes.labelcolor": TINTA, "xtick.color": TINTA2,
                         "ytick.color": TINTA2, "axes.spines.top": False, "axes.spines.right": False})
    # precisão por distância e por classe: teste (cheio) e total (vazio), com o IC 95 %
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 4.1), gridspec_kw={"width_ratios": [len(faixas) + 0.6, len(classes)]})
    for ax, (tipo, cats) in zip(axs, (("distancia", faixas), ("classe", classes))):
        for sub, cheio, dx in (("teste", True, -0.12), ("total", False, 0.12)):
            xs, ys, lo, hi = [], [], [], []
            for i, c in enumerate(cats):
                r = prec[(prec["subconjunto"] == sub) & (prec["estrato"] == f"{tipo} {c}")].iloc[0]
                if pd.isna(r["precisao"]):
                    continue
                v = float(r["precisao"])
                xs.append(i + dx)
                ys.append(v)
                lo.append(v - float(r["ic95_inf"]) if pd.notna(r["ic95_inf"]) else 0.0)
                hi.append(float(r["ic95_sup"]) - v if pd.notna(r["ic95_sup"]) else 0.0)
            if xs:
                ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o", ms=7, color=AZUL, mfc=AZUL if cheio else "white", mew=1.8,
                            elinewidth=1.5, capsize=0, label=NOMES[sub] if tipo == "distancia" else None, zorder=3)
        if ref_p is not None:
            ax.axhline(ref_p, color=CINZA, lw=1.2, ls="--", zorder=1)
        n_ = {s: [int(prec[(prec["subconjunto"] == s) & (prec["estrato"] == f"{tipo} {c}")][["aceitar", "rejeitar"]]
                      .to_numpy().sum()) for c in cats] for s in ("teste", "total")}
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels([f"{NOMES.get(c, c.replace('-', '–'))}\n{_n(a)} · {_n(b)}"
                            for c, a, b in zip(cats, n_["teste"], n_["total"])], fontsize=8)
        ax.set_ylim(0, 1.04)
        ax.yaxis.set_major_formatter(virgula)
        ax.set_xlim(-0.6, len(cats) - 0.4)
        ax.grid(axis="y", color=GRELHA, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_title("por distância ao seed (frames)" if tipo == "distancia" else "por classe", fontsize=9, color=TINTA,
                     loc="left")
    axs[0].set_ylabel("precisão das caixas novas")
    h, rotulos = axs[0].get_legend_handles_labels()
    if ref_p is not None:
        h.append(Line2D([], [], color=CINZA, lw=1.2, ls="--"))
        rotulos.append(f"precisão de presença em VAL-EX (Guião A): {_p(ref_p)}")
    fig.legend(h, rotulos, loc="lower center", ncol=len(h), frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.0),
               title="por baixo de cada estrato: caixas com A ou R (teste · total)", title_fontsize=7.5)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(saida / f"fig_c_precisao.{ext}", dpi=150, facecolor="white")
    plt.close(fig)
    # ganho por classe: caixas originais (com o seed) + novas aceites
    g = ganho_cl.sort_values("depois").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.4, 0.45 * len(g) + 1.3))
    y = np.arange(len(g))
    ax.barh(y, g["antes"], color=CINZA, height=0.62, edgecolor="white", lw=1.5, label="caixas originais (com o seed)")
    ax.barh(y, g["aceites"], left=g["antes"], color=AZUL, height=0.62, edgecolor="white", lw=1.5, label="novas aceites")
    xmax = float(g["depois"].max())
    for yi, r in zip(y, g.itertuples()):
        ax.text(r.depois + xmax * 0.015, yi, f"{_n(r.antes)} → {_n(r.depois)} (+{_p(r.pct, 0)} %)", va="center",
                fontsize=8, color=TINTA)
    ax.set_yticks(y)
    ax.set_yticklabels([NOMES.get(c, c) for c in g["classe"]])
    ax.set_xlim(0, xmax * 1.34)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: _n(v)))
    ax.set_xlabel("caixas por lesão (soma das janelas de ALL-SEEDBOX)")
    ax.grid(axis="x", color=GRELHA, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(saida / f"fig_c_ganho.{ext}", dpi=150, facecolor="white")
    plt.close(fig)
    return "fig_c_precisao e fig_c_ganho (png e pdf)"


# ============================================================================ 2) análise
def analisar(args, cfg):
    est = cfg["estratos"]
    classes = list(est["classes_principais"]) + ["outras"]
    faixas = [f"{lo}-{hi}" for lo, hi in est["distancia"] if hi < 100]
    tamanhos = ["pequena", "media", "grande"]
    over, cont_cl = ler_classes(args.classes, cfg)
    itens = ler_itens(args.itens)
    d = juntar(ler_respostas(args.respostas, itens), ler_chave(args.chave), itens)
    d["classe_v1"] = d["classe"]
    d["classe"] = [over.get(k, c) for k, c in zip(zip(d["split"], d["caso"], d["seed"]), d["classe"])]
    d["seed_duvidoso"] = d["nota"].map(lambda t: NOTA_SEED in _tira(t))
    nova = (d["tipo"] == "nova").to_numpy()
    if not nova.any():
        raise ev.ErroAvaliacao("A lista não tem caixas novas (tipo nova).")
    contas = Contas(d, cfg)

    # 1. precisão das caixas novas
    subs = {"teste": nova & (d["split"] == TESTE).to_numpy(),
            "treino_sem_cal": nova & (d["split"] == TREINO).to_numpy() & (d["cal"] == 0).to_numpy(),
            "total": nova}
    estratos = ([(f"distancia {f}", (d["faixa"] == f).to_numpy()) for f in faixas]
                + [(f"classe {c}", (d["classe"] == c).to_numpy()) for c in classes]
                + [(f"tamanho {t}", (d["tercil"] == t).to_numpy()) for t in tamanhos]
                + [("sem ficheiro de caixa", (d["caixa_ref"] == "sem_ficheiro").to_numpy())])
    R, linhas = {}, []
    for sub, m in subs.items():
        for nome, e in [("todas", np.ones(len(d), bool))] + estratos:
            R[(sub, nome)] = contas.medir(m & e)
            linhas.append(_linha(sub, nome, R[(sub, nome)]))
    for s in (TESTE, TREINO):
        x = contas.medir(nova & (d["split"] == s).to_numpy())
        linhas.append(_linha("total", f"split {NOMES[s]}", x))
    prec = pd.DataFrame(linhas)

    # 2. controlos às cegas
    todas = ev.ler_conjunto(Path(cfg["caminhos"]["conjuntos"]) / "ALL_SEEDBOX.json")
    ref, info = referencia(cfg, todas)
    ctrl, conc = [], []
    for tipo in ("controlo_original", "controlo_baixa"):
        m = (d["tipo"] == tipo).to_numpy()
        if m.any():
            x = contas.medir(m)
            ctrl.append({"tipo": tipo, "caixas": int(m.sum()), "aceitar": x["aceitar"], "rejeitar": x["rejeitar"],
                         "nao_decidir": x["nao_decidir"], "janelas": x["janelas"],
                         "taxa_aceitacao": x["precisao"]["valor"], "ic95_inf": (x["precisao"]["ic95"] or [None, None])[0],
                         "ic95_sup": (x["precisao"]["ic95"] or [None, None])[1]})
    iou_txt = None
    co = d[(d["tipo"] == "controlo_original") & (d["decisao"] != "nao_avaliavel")]
    co = co.merge(ref[K4 + ["caixa"] + REF], on=K4, how="left")
    co = co[co["caixa"] == "caixa"]
    if len(co):
        iou = ev.iou(co[CAIXA].to_numpy(dtype=float), co[REF].to_numpy(dtype=float))
        acc = (co["decisao"] == "aceitar").to_numpy()
        conc.append(concordancia("controlos com caixa original: aceitar × IoU ≥ 0,5", acc, iou >= 0.5 - 1e-9,
                                 "aceitar (revisão)", "IoU ≥ 0,5 com a caixa original"))
        med = [_p(float(np.median(iou[s])) if s.any() else None, 2) for s in (acc, ~acc)]
        iou_txt = (f"aceitar × IoU ≥ 0,5 com a caixa original: concordância {_p(conc[-1]['concordancia'])} · κ "
                   f"{_p(conc[-1]['kappa'], 2)} ({_n(len(co))} caixas com A ou R) · IoU mediano {med[0]} nas aceites e "
                   f"{med[1]} nas rejeitadas")

    # 3. concordâncias sem trabalho extra
    pf_txt = fp_txt = "não calculada (ficheiro em falta)"
    pfp = Path(cfg["caminhos"]["referencia_pf"])
    if pfp.is_file():
        pf = pd.read_csv(pfp, dtype=str, keep_default_na=False)
        pf["caso"] = pf["caso"].str.zfill(4)
        pf["seed"], pf["frame"] = pf["seed"].astype(int), pf["frame"].astype(int)
        m = d[nova].merge(pf[K4 + ["visivel"]], on=K4, how="inner")
        m = m[(m["decisao"] != "nao_avaliavel") & m["visivel"].isin(["1", "0"])]
        a, b = (m["decisao"] == "aceitar").to_numpy(), (m["visivel"] == "1").to_numpy()
        conc.append(concordancia("caixas novas: revisão × PF v2", a, b, "aceitar (revisão)", "visível (PF v2)"))
        x = conc[-1]
        pf_txt = (f"{_n(x['n'])} caixas novas com A ou R e frame 1 ou 0 na PF v2 · concordância {_p(x['concordancia'])}"
                  f" · κ {_p(x['kappa'], 2)} · aceitou {x['a_sim_b_sim']} dos {x['a_sim_b_sim'] + x['a_nao_b_sim']} frames "
                  f"visíveis e rejeitou {x['a_nao_b_nao']} dos {x['a_sim_b_nao'] + x['a_nao_b_nao']} não visíveis")
    fpp = Path(args.fp)
    if fpp.is_file():
        fp = pd.read_csv(fpp, dtype=str, keep_default_na=False)
        fp["caso"] = fp["caso"].str.zfill(4)
        fp["seed"], fp["frame"] = fp["seed"].astype(int), fp["frame"].astype(int)
        m = d[nova].merge(fp[K4 + ["classificacao"]], on=K4, how="inner")
        m["antes"] = m["classificacao"].map(FP_PARA_RESPOSTA)
        mm = m[(m["antes"].isin(["aceitar", "rejeitar"])) & (m["decisao"] != "nao_avaliavel")]
        conc.append(concordancia("caixas novas: revisão × 50 FP (20–22 out)", (mm["decisao"] == "aceitar").to_numpy(),
                                 (mm["antes"] == "aceitar").to_numpy(), "aceitar (revisão)", "lesão não anotada (50 FP)"))
        fp_txt = (f"{len(m)} dos {len(fp)} FP estão na lista; {len(mm)} com resposta A ou R nas duas vezes · concordância "
                  f"{_p(conc[-1]['concordancia'])} · κ {_p(conc[-1]['kappa'], 2)}")

    # 4. ganho: caixas por lesão antes (seed + originais, regra do script 15) e depois (+ novas aceites)
    orig = ref[(ref["offset"] != 0) & (ref["caixa"] == "caixa")].groupby(K3).size()
    aceites = d[nova & (d["decisao"] == "aceitar").to_numpy()].groupby(K3).size()
    propostas = d[nova].groupby(K3).size()
    jcl = dict(zip(zip(info["split"], info["caso"], info["seed"]), info["classe"]))
    g = pd.DataFrame({"split": [k[0] for k in todas], "classe": [over.get(k, jcl[k]) for k in todas],
                      "antes": [1 + int(orig.get(k, 0)) for k in todas],
                      "propostas": [int(propostas.get(k, 0)) for k in todas],
                      "aceites": [int(aceites.get(k, 0)) for k in todas]})
    g["depois"] = g["antes"] + g["aceites"]

    def soma(por):
        t = g.groupby(por)[["antes", "propostas", "aceites", "depois"]].sum()
        t["janelas"] = g.groupby(por).size()
        return t.reset_index().rename(columns={por: "grupo_valor"})
    ganho_cl, ganho_sp = soma("classe"), soma("split")
    ordem = {c: i for i, c in enumerate(classes)}
    ganho_cl = ganho_cl.sort_values("grupo_valor", key=lambda s: s.map(ordem)).reset_index(drop=True)
    tot = pd.DataFrame([{"grupo_valor": "total", "antes": g["antes"].sum(), "propostas": g["propostas"].sum(),
                         "aceites": g["aceites"].sum(), "depois": g["depois"].sum(), "janelas": len(g)}])
    ganho = pd.concat([ganho_cl.assign(grupo="classe"), ganho_sp.assign(grupo="split"), tot.assign(grupo="total")],
                      ignore_index=True)
    ganho["pct"] = 100 * ganho["aceites"] / ganho["antes"]
    ganho = ganho[["grupo", "grupo_valor", "janelas", "antes", "propostas", "aceites", "depois", "pct"]]
    ganho_cl = ganho[ganho["grupo"] == "classe"].rename(columns={"grupo_valor": "classe"})

    # 5. tempo e seed duvidoso
    tem, n_pausas, n_sem = tempo(d)
    jan = d[nova].groupby(K3).agg(duv=("seed_duvidoso", "any"))
    duv = set(jan.index[jan["duv"]])
    em_duv = np.array([k in duv for k in zip(d["split"], d["caso"], d["seed"])])
    n_incoer = int((nova & em_duv & (d["decisao"] != "nao_avaliavel").to_numpy()).sum())

    # 6. Guião A
    ga, ga_txt = guiao_a(args.eval_testbox, over, cfg)

    # 7. gravar
    saida = Path(args.saida)
    saida.mkdir(parents=True, exist_ok=True)
    prec.to_csv(saida / "precisao.csv", index=False)
    pd.DataFrame(ctrl).to_csv(saida / "controlos.csv", index=False)
    pd.DataFrame(conc).to_csv(saida / "concordancias.csv", index=False)
    ganho.to_csv(saida / "ganho.csv", index=False)
    tem.to_csv(saida / "tempo.csv", index=False)
    if ga is not None:
        ga.to_csv(saida / "guiao_a_rloc_classe.csv", index=False)
    ref_p = ref_loc = None
    try:
        v = json.loads((Path(cfg["caminhos"]["saida"]) / "VAL_EX" / "SAM2_L12.json").read_text(encoding="utf-8"))
        ref_p = float(v["por_op"]["OP_F1"]["micro"]["P"]["valor"])
    except (OSError, KeyError, TypeError, ValueError):
        pass
    try:
        tb = json.loads(Path(args.eval_testbox).read_text(encoding="utf-8"))["por_op"]["OP_F1"]["micro"]
        ref_loc = (float(tb["R_loc"]["valor"]), float(tb["R_box"]["valor"]))
    except (OSError, KeyError, TypeError, ValueError):
        pass
    figs = figuras(saida, prec, ganho_cl, classes, faixas, ref_p)

    # 8. resumo (só números agregados)
    sha = ev._sha256(args.respostas)
    git = ev._estado_git(RAIZ)
    commit = git["commit"] or "sem git"
    cont = d["decisao"].value_counts()
    xt, xr, xo = R[("teste", "todas")], R[("treino_sem_cal", "todas")], R[("total", "todas")]
    tg = ganho[ganho["grupo"] == "total"].iloc[0]
    t0 = tem.iloc[0]
    n_jan = len(jan)
    L = ["# Guião C — análise da revisão (C9)", "",
         f"scripts/18_c_analise.py v{VERSAO} · {datetime.datetime.now():%Y-%m-%d %H:%M} · "
         + (f"commit {git['commit']}" if git["commit"] else "sem git")
         + (" (com alterações por guardar)" if git["alteracoes_por_guardar"] else "")
         + f" · {Path(args.respostas).name} sha256 {sha[:16]}", "",
         f"**Respostas:** {_n(len(d))} caixas ({_n(nova.sum())} novas + {_n(len(d) - nova.sum())} controlos) · aceitar "
         f"{_n(cont.get('aceitar', 0))} · rejeitar {_n(cont.get('rejeitar', 0))} · não dá para decidir "
         f"{_n(cont.get('nao_avaliavel', 0))}"
         + (f" · mal posta {_n(cont.get('mal_posta', 0))} (contam como rejeitadas)" if cont.get("mal_posta", 0) else ""),
         "", "## Precisão das caixas novas (aceitar / (aceitar + rejeitar))", "",
         f"- **Teste (número principal): {_ic(xt['precisao'])}** · {_n(xt['aceitar'])} aceites, {_n(xt['rejeitar'])} "
         f"rejeitadas, {_n(xt['nao_decidir'])} sem decisão, em {_n(xt['janelas'])} janelas",
         f"- Análise de sensibilidade (N contadas como rejeitadas), teste: {_ic(xt['precisao_n_rejeitadas'])}",
         f"- Treino sem CAL: {_ic(xr['precisao'])} ({_n(xr['aceitar'] + xr['rejeitar'])} caixas com A ou R) · total: "
         f"{_ic(xo['precisao'])} ({_n(xo['aceitar'] + xo['rejeitar'])})",
         "- Guião A, para comparar: precisão de presença em VAL-EX " + _p(ref_p)
         + (f"; em TEST-BOX, quando diz presente, acerta o sítio {_p(ref_loc[0] / ref_loc[1], 2)} das vezes (R_loc "
            f"{_p(ref_loc[0])} / R_box {_p(ref_loc[1])})" if ref_loc else ""),
         "", "| Estrato | Teste | Total |", "| --- | --- | --- |"]
    rot = {"distancia": "distância", "classe": "classe", "tamanho": "tamanho"}
    for nome, _ in estratos:
        tipo, _, val = nome.partition(" ")
        txt_rot = (f"{rot[tipo]} {NOMES.get(val, val.replace('-', '–'))}" if tipo in rot else nome)
        cel = [f"{_ic(R[(s, nome)]['precisao'])} (n = {R[(s, nome)]['aceitar'] + R[(s, nome)]['rejeitar']})"
               if R[(s, nome)]["precisao"]["valor"] is not None else "—" for s in ("teste", "total")]
        L.append(f"| {txt_rot} | {cel[0]} | {cel[1]} |")
    L += ["", f"n = caixas com A ou R. Classe das {cont_cl['janelas']} janelas com mais de uma lesão no texto: "
          f"{resumo_classes(cont_cl)}.",
          "", "## Controlos às cegas", ""]
    nomes_ctrl = {"controlo_original": "com caixa original", "controlo_baixa": "abaixo do limiar (nível baixa)"}
    for c in ctrl:
        L.append(f"- {nomes_ctrl[c['tipo']]}: {_n(c['caixas'])} caixas · aceitou {_n(c['aceitar'])}, rejeitou "
                 f"{_n(c['rejeitar'])}, sem decisão {_n(c['nao_decidir'])} · taxa de aceitação "
                 f"{_ic({'valor': c['taxa_aceitacao'], 'ic95': [c['ic95_inf'], c['ic95_sup']]})}")
        if c["tipo"] == "controlo_original" and iou_txt:
            L.append(f"  - {iou_txt}")
    if not ctrl:
        L.append("- a lista não tem controlos (--sem-controlos)")
    L += ["", "## Concordâncias sem trabalho extra", "", f"- PF v2 (janelas de CAL e VAL-EX): {pf_txt}",
          f"- 50 FP: {fp_txt}", "", "## Ganho de anotação (todas as janelas de ALL-SEEDBOX)", "",
          f"- Caixas por lesão: **{_n(tg['antes'])} → {_n(tg['depois'])} (+{_p(tg['pct'], 1)} %)**; com todas as "
          f"{_n(tg['propostas'])} novas propostas seriam {_n(tg['antes'] + tg['propostas'])} "
          f"(+{_p(100 * tg['propostas'] / tg['antes'], 1)} %)",
          f"- Por janela (mediana): {_p(float(g['antes'].median()), 1)} → {_p(float(g['depois'].median()), 1)} caixas",
          "", "| Grupo | Janelas | Antes | Novas propostas | Aceites | Depois | Ganho |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
    for x in ganho.itertuples():
        nome = NOMES.get(x.grupo_valor, x.grupo_valor)
        nome = nome if x.grupo == "classe" else f"**{nome}**"
        L.append(f"| {nome} | {_n(x.janelas)} | {_n(x.antes)} | {_n(x.propostas)} | {_n(x.aceites)} | {_n(x.depois)} | "
                 f"+{_p(x.pct, 1)} % |")
    L += ["", "## Tempo", "",
          f"- {_n(t0['caixas'])} caixas com tempo: mediana {_p(t0['mediana_s'], 1)} s · P90 {_p(t0['p90_s'], 1)} s · "
          f"total {_p(t0['total_min'], 0)} min (ficam fora {_pl(n_pausas, 'pausa', 'pausas')} de mais de 120 s e "
          f"{_pl(n_sem, 'caixa sem tempo', 'caixas sem tempo')})",
          "", "## Seed duvidoso", "",
          f"- {len(duv)} das {_n(n_jan)} janelas com caixas novas ({_p(100 * len(duv) / max(1, n_jan), 1)} %) "
          f"{'tem' if len(duv) == 1 else 'têm'} a nota «{NOTA_SEED}»"
          + (f"; atenção: {_pl(n_incoer, 'caixa nova dessas janelas não está', 'caixas novas dessas janelas não estão')} "
             "em N (a regra é N em todas)" if n_incoer else ""),
          "", "## Guião A: R_loc por classe em TEST-BOX (OP-F1), com a classe nova", "", f"- {ga_txt}"]
    if ga is not None:
        L += ["", "| Classe | Antes | Agora |", "| --- | --- | --- |"]
        for x in ga.itertuples():
            antes = {"valor": x.rloc_antes, "ic95": None if pd.isna(x.ic95_inf_antes) else [x.ic95_inf_antes, x.ic95_sup_antes]}
            agora = {"valor": x.rloc_agora, "ic95": None if pd.isna(x.ic95_inf_agora) else [x.ic95_inf_agora, x.ic95_sup_agora]}
            L.append(f"| {NOMES.get(x.classe, x.classe)} | {_ic(antes)} ({_n(x.caixas_antes)} caixas) | {_ic(agora)} "
                     f"({_n(x.caixas_agora)} caixas) |")
    co_txt = "; ".join(f"{nomes_ctrl[c['tipo']]}: aceitou {_n(c['aceitar'])} de {_n(c['aceitar'] + c['rejeitar'])}"
                       for c in ctrl)
    linha = (f"- {datetime.date.today().isoformat()}: Guião C, análise C9 (scripts/18_c_analise.py v{VERSAO}, commit "
             f"{commit}; {Path(args.respostas).name} sha256 {sha[:16]}): {_n(nova.sum())} caixas novas revistas. Precisão "
             f"(aceitar / (aceitar + rejeitar)) no teste {_ic(xt['precisao'])}; com as N como rejeitadas "
             f"{_ic(xt['precisao_n_rejeitadas'])}; treino sem CAL {_ic(xr['precisao'])}; total {_ic(xo['precisao'])}. "
             + (f"Controlos às cegas: {co_txt}. " if ctrl else "")
             + f"Classe das {cont_cl['janelas']} janelas com mais de uma lesão no texto ({Path(args.classes).name} sha256 "
             f"{ev._sha256(args.classes)[:16]}): {cont_cl['escolhidas']} pela lesão da caixa do seed, "
             f"{cont_cl['nao_claras'] + cont_cl['por_responder']} pela primeira mencionada. "
             + f"Caixas por lesão {_n(tg['antes'])} → {_n(tg['depois'])} (+{_p(tg['pct'], 1)} %). Tempo: mediana "
             f"{_p(t0['mediana_s'], 1)} s por caixa, {_p(t0['total_min'], 0)} min no total. Seed duvidoso em "
             f"{_pl(len(duv), 'janela', 'janelas')}. Resultados em outputs/18_c_analise/.")
    L += ["", f"Figuras: {figs}.", "", "## Linha para o docs/decisoes.md", "", linha]
    texto = "\n".join(L)
    (saida / "resumo.md").write_text(texto + "\n", encoding="utf-8")
    print(texto)
    return {"R": R, "precisao": prec, "controlos": pd.DataFrame(ctrl), "concordancias": pd.DataFrame(conc),
            "ganho": ganho, "tempo": tem, "pausas": n_pausas, "sem_tempo": n_sem, "seed_duvidoso": len(duv),
            "incoerentes": n_incoer, "guiao_a": ga, "guiao_a_txt": ga_txt, "classes": cont_cl, "texto": texto,
            "linha": linha}


def main(argv=None, entrada=None):
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--config", default="configs/avaliacao.yaml")
    a.add_argument("--itens", default="data/derived/guiao_c/itens_v1.csv")
    a.add_argument("--chave", default="data/derived/guiao_c/itens_chave_v1.csv")
    a.add_argument("--respostas", default="data/derived/guiao_c/decisoes_paulo.csv")
    a.add_argument("--classes", default="data/derived/guiao_c/classes_mistas.csv")
    a.add_argument("--fp", default="outputs/14_fp_inspecao/classificacao.csv")
    a.add_argument("--eval-testbox", default="outputs/eval/TEST_BOX/SAM2_L12.json")
    a.add_argument("--saida", default="outputs/18_c_analise")
    a.add_argument("--ds", default=os.environ.get("DS"), help="pasta do dataset (só para a imagem do --preparar-classes)")
    a.add_argument("--preparar-classes", action="store_true",
                   help="classe das janelas com mais de uma lesão no texto: folha, imagem e perguntas")
    args = a.parse_args(argv)
    cfg = ev.ler_config(args.config)
    if args.preparar_classes:
        return preparar_classes(args, cfg, entrada)
    return analisar(args, cfg)


if __name__ == "__main__":
    try:
        main()
    except ev.ErroAvaliacao as e:
        print(f"ERRO: {e}", file=sys.stderr)
        sys.exit(2)
    except (KeyboardInterrupt, EOFError):
        print("\nParado. O que já respondeste ficou gravado.")
        sys.exit(1)
