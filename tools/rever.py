#!/usr/bin/env python3
"""
Guião C, passo C3 — Ferramenta de revisão das caixas da anotação expandida (validação humana).

Abre um servidor local (só nesta máquina) com uma página onde se revê, uma a uma, cada caixa da lista congelada pelo
scripts/17_c_itens.py. A regra está escrita na página e em docs/instrucoes_revisao.md.

  - À esquerda, sempre visível: o frame do seed com a caixa da anotação original (azul), a imagem do relatório e a
    descrição clínica.
  - À direita: o frame à vista, ampliado. Roda do rato = zoom; arrastar = mover; duplo clique = tamanho normal.
    Cores: azul = caixa da anotação original (desenhada pelo Rodrigo) · roxo cheio = a caixa a rever · roxo tracejado =
    caixa do modelo noutro frame. H esconde e mostra as caixas; S mostra o seed em grande.
  - Em baixo, a tira com todos os frames seguidos (seed ± 12), cada um com as suas caixas e, nos frames a rever, a
    resposta já dada (A, R, N) ou «?». ← → andam frame a frame dentro da janela; um clique abre o frame em grande.
  - As caixas dos outros frames vêm do ficheiro de contexto (scripts/17b_c_contexto.py), que só tem coordenadas e nunca
    tem nada dos frames da lista: a caixa original nunca aparece no próprio frame a rever (nos controlos com caixa
    original fica escondida) e não há score, nível nem tipo de item. A ferramenta recusa um contexto com outras colunas
    ou com caixas em frames da lista.
  - Três respostas, só nos frames a rever (gravam e passam à próxima caixa por responder):
        A aceitar (a caixa marca a lesão do seed) · R rejeitar (não marca) · N não dá para decidir
    Z desfaz a última resposta desta sessão · U vai à próxima sem resposta · Esc volta à caixa · PgUp/PgDn mudam de
    janela.
  - Modo ajustar (só com --ajustar, não usado na revisão): aparece também o M (mal posta), com que se desenha a caixa
    certa com o rato (Esc: mal posta sem caixa).
  - Às cegas: a lista só pode ter as colunas item, split, caso, seed, frame e box_x1..box_y2. A ferramenta recusa listas
    com mais colunas (score, nível, tipo, caixa original…) e nunca abre o ficheiro da chave nem o expanded_v1.csv.
  - Grava cada resposta logo em <saida>/<revisor>/decisoes.csv e num registo de todas as alterações (_registo.csv),
    com a hora e o tempo que a caixa esteve no ecrã (o cronómetro não aparece). Retoma na primeira caixa sem resposta.
  - Cada revisor tem a sua pasta, ligada a uma só lista (pelo sha256): a ferramenta recusa misturar listas e nunca lê
    as decisões de outro revisor.

Uso, a partir da pasta do projeto, no tmux (fica a correr), depois do source …/inicio.sh e do export DS=…:
    python scripts/17b_c_contexto.py                                  (uma vez: grava o contexto)
    python tools/rever.py --itens data/derived/guiao_c/itens_v1.csv --revisor paulo
e abrir http://localhost:8766 no browser (no portátil: separador PORTS do VS Code). Ctrl+C para parar: o que foi
respondido já está gravado.

No fim da revisão:
    python tools/rever.py --itens data/derived/guiao_c/itens_v1.csv --revisor paulo --consolidar
grava data/derived/guiao_c/decisoes_<revisor>.csv (só de leitura), depois de verificar que todas as caixas têm resposta.
Versões: 1.0 (24 out 2026) · 1.1 (24 out 2026): enquanto se revê uma janela, a seguinte (dados e imagens) carrega em
segundo plano, e mudar de janela já não espera pelas gravações pendentes (menos espera, sobretudo à distância) ·
1.2 (24 out 2026), depois do piloto: só três respostas (aceitar, rejeitar, não dá para decidir); saem o M (fica só no
modo ajustar), o O, o D e o botão do seed; o seed vê-se em grande com a caixa (tecla S ou clique na tira) ·
1.3 (24 out 2026): a tira passa a ter todos os frames seguidos, com as caixas originais e do modelo dos outros frames
(contexto do script 17b) e a resposta de cada caixa; ← → andam frame a frame; depois de responder, segue para a próxima
caixa por responder; PgUp/PgDn mudam de janela.
"""
import argparse
import csv
import datetime
import hashlib
import io
import json
import os
import re
import sys
import tempfile
import threading
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

try:
    from PIL import Image
except ImportError:                                            # sem Pillow: miniaturas = imagem inteira
    Image = None

VERSAO = "1.3"
DECISOES = ("aceitar", "mal_posta", "rejeitar", "nao_avaliavel")
COL_LISTA = ["item", "split", "caso", "seed", "frame", "box_x1", "box_y1", "box_x2", "box_y2"]
COL_DEC = ["item", "split", "caso", "seed", "frame", "decisao", "nota", "box_aj_x1", "box_aj_y1", "box_aj_x2",
           "box_aj_y2", "modo", "dur_ms", "t_decisao", "sessao", "revisor"]
COL_REG = ["timestamp", "sessao", "item", "antes", "depois", "nota", "box_aj", "dur_ms"]
COL_CONTEXTO = ["split", "caso", "seed", "frame", "fonte", "box_x1", "box_y1", "box_x2", "box_y2"]
FONTES = {"original": "o", "modelo": "m"}
NOTA_NAO_ID = "não identifico a lesão no seed"
MARCA = "_rever_ferramenta.txt"
REGISTO = "_registo.csv"
FICH_DEC = "decisoes.csv"
RE_MATCH = re.compile(r"match_at_videoframe(\d+)", re.I)
RE_IMAGEM_N = re.compile(r"^imagem(\d+)_", re.I)
RE_NUM = re.compile(r"-?\d+(?:\.\d+)?")
RE_REVISOR = re.compile(r"^[a-z0-9][a-z0-9_-]{0,30}$")


class ErroRever(Exception):
    """Erro nos dados ou no uso; a mensagem diz o que corrigir."""


def agora():
    return datetime.datetime.now().isoformat(timespec="seconds")


def sha256(caminho):
    h = hashlib.sha256()
    with open(caminho, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def gravar_csv(caminho, cabecalho, linhas):
    """Escrita atómica: nunca fica um ficheiro a meio se a máquina ou a ligação caírem."""
    tmp = caminho.with_name(caminho.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cabecalho)
        w.writerows(linhas)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, caminho)


def caixa_valida(b):
    return (isinstance(b, (list, tuple)) and len(b) == 4 and all(isinstance(x, (int, float)) for x in b)
            and all(-1e-6 <= x <= 1 + 1e-6 for x in b) and b[2] > b[0] and b[3] > b[1])


def nota_com(antiga, extra=None):
    """As notas rápidas (O, D e o botão do seed) saem quando a resposta muda; o texto escrito à mão fica."""
    rapidas = ("outra lesão", "dúvida", NOTA_NAO_ID)
    partes = [s.strip() for s in str(antiga or "").split(" · ") if s.strip() and s.strip() not in rapidas]
    if extra:
        partes.insert(0, extra)
    return " · ".join(partes)


# ============================================================================ leitura
def ler_lista(caminho):
    """Lista congelada (scripts/17_c_itens.py). Só pode ter as colunas de COL_LISTA: a ferramenta tem de ser cega."""
    p = Path(caminho)
    if not p.is_file():
        raise ErroRever(f"Não encontrei a lista {p} (é gravada pelo scripts/17_c_itens.py).")
    with open(p, encoding="utf-8", newline="") as fh:
        r = csv.DictReader(fh)
        cols = list(r.fieldnames or [])
        linhas = list(r)
    extra = [c for c in cols if c not in COL_LISTA]
    if extra:
        raise ErroRever(f"A lista {p} tem colunas a mais ({', '.join(extra)}). A revisão é às cegas: a ferramenta só "
                        f"aceita {', '.join(COL_LISTA)}. Usa o itens_v1.csv (ou o piloto_v1.csv), nunca o ficheiro da chave.")
    falta = [c for c in COL_LISTA if c not in cols]
    if falta:
        raise ErroRever(f"A lista {p} não tem as colunas {', '.join(falta)}.")
    itens, vistos, frames = [], set(), set()
    for n, r in enumerate(linhas, 2):
        try:
            it = {"item": int(r["item"]), "split": r["split"].strip(), "caso": r["caso"].strip().zfill(4),
                  "seed": int(float(r["seed"])), "frame": int(float(r["frame"])),
                  "box": [round(float(r[c]), 6) for c in COL_LISTA[5:]]}
        except (TypeError, ValueError):
            raise ErroRever(f"A lista {p} tem valores inválidos na linha {n}.")
        if not caixa_valida(it["box"]):
            raise ErroRever(f"A lista {p} tem uma caixa inválida na linha {n} (item {it['item']}).")
        k = (it["split"], it["caso"], it["seed"], it["frame"])
        if it["item"] in vistos or k in frames:
            raise ErroRever(f"A lista {p} tem itens repetidos (linha {n}).")
        vistos.add(it["item"])
        frames.add(k)
        itens.append(it)
    if not itens:
        raise ErroRever(f"A lista {p} está vazia.")
    itens.sort(key=lambda d: d["item"])
    vistas, anterior = set(), None
    for it in itens:
        k = (it["split"], it["caso"], it["seed"])
        if k != anterior:
            if k in vistas:
                raise ErroRever(f"Na lista {p}, as caixas da janela {'/'.join(map(str, k))} não estão seguidas.")
            vistas.add(k)
            anterior = k
    return itens


def ler_tabela(caminho):
    if not Path(caminho).is_file():
        print(f"Aviso: não encontrei {caminho}; a descrição clínica não vai aparecer.")
        return {}
    with open(caminho, encoding="utf-8", newline="") as fh:
        return {(r["split"], r["caso"].strip().zfill(4), int(float(r["seed"]))): r for r in csv.DictReader(fh)}


def ler_contexto(caminho, itens):
    """Caixas dos outros frames seguidos (scripts/17b_c_contexto.py): só coordenadas. Recusa colunas a mais e caixas em
    frames da lista, para a revisão continuar às cegas. Devolve {(split, caso, seed): {frame: {"o": caixa, "m": caixa}}}
    só das janelas da lista, e quantas janelas da lista ficaram sem nenhuma caixa de contexto."""
    p = Path(caminho)
    if not p.is_file():
        raise ErroRever(f"Não encontrei o contexto {p}: grava-o primeiro com python scripts/17b_c_contexto.py "
                        "(sem ele não aparecem as caixas dos outros frames; --sem-contexto abre sem elas).")
    with open(p, encoding="utf-8", newline="") as fh:
        r = csv.DictReader(fh)
        cols = list(r.fieldnames or [])
        linhas = list(r)
    extra = [c for c in cols if c not in COL_CONTEXTO]
    if extra:
        raise ErroRever(f"O contexto {p} tem colunas a mais ({', '.join(extra)}). A revisão é às cegas: só aceito "
                        f"{', '.join(COL_CONTEXTO)}. Grava-o com o scripts/17b_c_contexto.py.")
    falta = [c for c in COL_CONTEXTO if c not in cols]
    if falta:
        raise ErroRever(f"O contexto {p} não tem as colunas {', '.join(falta)}.")
    janelas = {(it["split"], it["caso"], it["seed"]) for it in itens}
    na_lista = {(it["split"], it["caso"], it["seed"], it["frame"]) for it in itens}
    ctx = {}
    for n, r in enumerate(linhas, 2):
        try:
            k = (r["split"].strip(), r["caso"].strip().zfill(4), int(float(r["seed"])))
            f = int(float(r["frame"]))
            b = [round(min(max(float(r[c]), 0.0), 1.0), 6) for c in COL_CONTEXTO[5:]]   # cortada à imagem
        except (TypeError, ValueError):
            raise ErroRever(f"O contexto {p} tem valores inválidos na linha {n}.")
        fonte = FONTES.get(r["fonte"].strip())
        if fonte is None:
            raise ErroRever(f"O contexto {p} tem uma fonte desconhecida na linha {n} ({r['fonte']!r}).")
        if not caixa_valida(b):
            raise ErroRever(f"O contexto {p} tem uma caixa inválida na linha {n}.")
        if k + (f,) in na_lista:
            raise ErroRever(f"O contexto {p} tem uma caixa no frame {f} da janela {'/'.join(map(str, k))}, que está na lista "
                            "de revisão: não é o contexto desta lista (ou foi mexido). Grava-o outra vez com o 17b.")
        if k not in janelas:
            continue
        caixas = ctx.setdefault(k, {}).setdefault(f, {})
        if fonte in caixas:
            raise ErroRever(f"O contexto {p} tem duas caixas {r['fonte']} no frame {f} da janela {'/'.join(map(str, k))}.")
        caixas[fonte] = b
    return ctx, len(janelas - set(ctx))


def preparar_saida(base, revisor, lista, lista_sha, criar=True):
    """Uma pasta por revisor, ligada a uma só lista. Recusa pastas com CSV de outra origem ou de outra lista."""
    if not RE_REVISOR.match(revisor or ""):
        raise ErroRever(f"Nome de revisor inválido: {revisor!r}. Usa letras minúsculas, números, _ ou - "
                        "(ex.: paulo, piloto, medica, paulo_reteste).")
    pasta = Path(base) / revisor
    marca = pasta / MARCA
    if pasta.exists():
        if not marca.exists():
            alheios = [f.name for f in pasta.glob("*.csv")]
            if alheios:
                raise ErroRever(f"A pasta {pasta} já tem CSV que não foram criados por esta ferramenta (ex.: {alheios[0]}). "
                                "Escolhe outro --revisor ou outra --saida.")
        else:
            m = re.search(r"lista_sha256: ([0-9a-f]{64})", marca.read_text(encoding="utf-8"))
            if m and m.group(1) != lista_sha:
                raise ErroRever(f"A pasta {pasta} é de outra lista de itens (sha256 {m.group(1)[:12]}…; esta é "
                                f"{lista_sha[:12]}…). Cada revisor fica ligado a uma só lista: usa outro --revisor.")
    elif not criar:
        raise ErroRever(f"Não existe a pasta {pasta}: este revisor ainda não respondeu a nada.")
    pasta.mkdir(parents=True, exist_ok=True)
    if not marca.exists():
        marca.write_text(f"Pasta criada por tools/rever.py v{VERSAO} a {agora()}.\nrevisor: {revisor}\nlista: {lista}\n"
                         f"lista_sha256: {lista_sha}\nNão pôr aqui outros CSV.\n", encoding="utf-8")
    return pasta


def ler_decisoes(pasta, por_item):
    f = Path(pasta) / FICH_DEC
    dec = {}
    if not f.exists():
        return dec
    with open(f, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            item = int(r["item"])
            if item not in por_item:
                raise ErroRever(f"{f} tem respostas a itens que não estão na lista (ex.: item {item}).")
            b = [r.get(c, "") for c in COL_DEC[7:11]]
            dec[item] = {"decisao": r.get("decisao", ""), "nota": r.get("nota", ""),
                         "box_aj": [float(x) for x in b] if all(x != "" for x in b) else None,
                         "modo": r.get("modo", ""), "dur_ms": r.get("dur_ms", ""), "t_decisao": r.get("t_decisao", ""),
                         "sessao": r.get("sessao", "")}
    return dec


@lru_cache(maxsize=4096)
def miniatura(caminho, lado=160):
    with Image.open(caminho) as im:
        im = im.convert("RGB")
        im.thumbnail((lado, lado))
        b = io.BytesIO()
        im.save(b, "JPEG", quality=82)
        return b.getvalue()


class Janela:
    """Frames, imagens do relatório e caixa do seed de uma janela (como no tools/anotar.py)."""

    def __init__(self, j, ds, split, caso, seed, info):
        self.j, self.split, self.caso, self.seed = j, split, caso, int(seed)
        self.pasta = Path(ds) / split / caso / "GT_com_classes" / str(self.seed)
        self.problemas, self.avisos = [], []
        rx = re.compile(rf"^{re.escape(caso)}_(\d+)_{self.seed}\.jpe?g$", re.I)
        self.ficheiros = {}
        if self.pasta.is_dir():
            with os.scandir(self.pasta) as it:
                for f in it:
                    m = rx.match(f.name)
                    if m:
                        self.ficheiros[int(m.group(1))] = Path(f.path)
        self.frames = sorted(self.ficheiros)
        if not self.frames:
            self.problemas.append("não encontrei frames nesta janela")
        elif self.seed not in self.ficheiros:
            self.problemas.append("não existe o frame do seed")
        gt = self.pasta.parent
        rel = []
        if gt.is_dir():
            for f in gt.iterdir():
                m = RE_MATCH.search(f.name)
                if f.is_file() and m and int(m.group(1)) == self.seed and f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    n = RE_IMAGEM_N.match(f.name)
                    rel.append((int(n.group(1)) if n else 0, f.name, f))
        self.relatorios = [f for _, _, f in sorted(rel)]
        if not self.relatorios:
            self.avisos.append("sem imagem do relatório")
        self.caixa = self._caixa_seed()
        self.descricao = info.get("descricao", "")
        self.categoria = info.get("categoria", "")

    def _caixa_seed(self):
        """Lê SÓ a caixa do seed. Cantos por ordem de desenho → ordenados; coordenadas 0–1."""
        f = self.pasta / f"Findings_{self.seed}" / f"{self.caso}_{self.seed}_{self.seed}.txt"
        if not f.is_file():
            self.avisos.append("o seed não tem ficheiro de caixa")
            return None
        v = [float(x) for x in RE_NUM.findall(f.read_text(encoding="utf-8", errors="replace"))]
        if len(v) != 4:
            self.avisos.append("o seed não tem caixa" if not v else "caixa do seed ilegível")
            return None
        w = h = None
        if Image is not None and self.seed in self.ficheiros:
            try:
                with Image.open(self.ficheiros[self.seed]) as im:
                    w, h = im.size
            except OSError:
                pass
        if not w:
            self.avisos.append("não consegui ler o tamanho da imagem do seed")
            return None
        x1, x2 = sorted((v[0], v[2]))
        y1, y2 = sorted((v[1], v[3]))
        return [x1 / w, y1 / h, x2 / w, y2 / h]


class Estado:
    def __init__(self, ds, itens, pasta, tabela, revisor, ajustar=False, contexto=None, alcance=12):
        self.lock = threading.Lock()
        self.ds, self.itens, self.pasta, self.tabela = ds, itens, Path(pasta), tabela
        self.contexto, self.alcance = contexto or {}, int(alcance)
        self.revisor, self.ajustar, self.sessao = revisor, bool(ajustar), agora()
        self.por_item = {it["item"]: i for i, it in enumerate(itens)}
        self.janelas, jmap = [], {}
        for i, it in enumerate(itens):
            k = (it["split"], it["caso"], it["seed"])
            if k not in jmap:
                jmap[k] = len(self.janelas)
                self.janelas.append({"k": k, "idx": []})
            self.janelas[jmap[k]]["idx"].append(i)
            it["j"] = jmap[k]
        self._cache = {}
        self.dec = ler_decisoes(self.pasta, self.por_item)

    # ---------------------------------------------------------------- utilitários
    def janela_obj(self, j):
        if j not in self._cache:
            k = self.janelas[j]["k"]
            self._cache[j] = Janela(j, self.ds, *k, self.tabela.get(k, {}))
        return self._cache[j]

    def _item(self, item):
        try:
            item = int(item)
        except (TypeError, ValueError):
            raise ValueError(f"item inválido: {item!r}")
        if item not in self.por_item:
            raise ValueError(f"o item {item} não está na lista")
        return item, self.itens[self.por_item[item]]

    def respondidos(self):
        return sum(1 for d in self.dec.values() if d.get("decisao") in DECISOES)

    def _gravar(self):
        linhas = []
        for it in self.itens:
            d = self.dec.get(it["item"])
            if not d or not (d.get("decisao") or d.get("nota")):
                continue
            b = d.get("box_aj") or ["", "", "", ""]
            linhas.append([it["item"], it["split"], it["caso"], it["seed"], it["frame"], d.get("decisao", ""),
                           d.get("nota", ""), *b, d.get("modo", ""), d.get("dur_ms", ""), d.get("t_decisao", ""),
                           d.get("sessao", ""), self.revisor])
        gravar_csv(self.pasta / FICH_DEC, COL_DEC, linhas)

    def _registar(self, item, antes, depois, nota="", box_aj=None, dur_ms=""):
        f = self.pasta / REGISTO
        novo = not f.exists()
        with open(f, "a", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            if novo:
                w.writerow(COL_REG)
            w.writerow([agora(), self.sessao, item, antes, depois, nota,
                        "" if not box_aj else " ".join(f"{x:.6f}" for x in box_aj), dur_ms])

    # ---------------------------------------------------------------- API
    def estado(self):
        lista = [[it["item"], it["j"], int(self.dec.get(it["item"], {}).get("decisao") in DECISOES)] for it in self.itens]
        atual = next((i for i, x in enumerate(lista) if not x[2]), 0)
        return {"revisor": self.revisor, "ajustar": self.ajustar, "versao": VERSAO, "total": len(lista), "alcance": self.alcance,
                "respondidos": self.respondidos(), "atual": atual, "itens": lista,
                "janelas": [{"j": j, "split": w["k"][0], "caso": w["k"][1], "seed": w["k"][2], "n": len(w["idx"])}
                            for j, w in enumerate(self.janelas)]}

    def janela(self, j):
        j = int(j)
        if not 0 <= j < len(self.janelas):
            raise ValueError("janela inexistente")
        jo = self.janela_obj(j)
        avisos = list(jo.problemas) + list(jo.avisos)
        itens = []
        for i in self.janelas[j]["idx"]:
            it = self.itens[i]
            d = self.dec.get(it["item"], {})
            if it["frame"] not in jo.ficheiros:
                avisos.append(f"não existe a imagem do frame {it['frame']}")
            itens.append({"item": it["item"], "frame": it["frame"], "offset": it["frame"] - it["seed"], "box": it["box"],
                          "decisao": d.get("decisao", ""), "nota": d.get("nota", ""), "box_aj": d.get("box_aj")})
        # frames seguidos (seed ± alcance) com as caixas do contexto: o = original (no seed, a do seed), m = modelo.
        # Nos frames da lista não há nenhuma (o contexto não as tem; a caixa a rever vem da lista).
        ctx = self.contexto.get(self.janelas[j]["k"], {})
        seguidos = [{"f": f, "o": jo.caixa if f == jo.seed else ctx.get(f, {}).get("o"), "m": ctx.get(f, {}).get("m")}
                    for f in jo.frames if abs(f - jo.seed) <= self.alcance]
        return {"j": j, "split": jo.split, "caso": jo.caso, "seed": jo.seed, "frames": jo.frames, "caixa": jo.caixa,
                "relatorios": list(range(len(jo.relatorios))), "descricao": jo.descricao, "categoria": jo.categoria,
                "avisos": avisos, "itens": itens, "seguidos": seguidos}

    def decidir(self, item, decisao, nota=None, box_aj=None, dur_ms=None):
        item, it = self._item(item)
        if decisao not in DECISOES:
            raise ValueError(f"resposta inválida: {decisao!r}")
        if box_aj is not None:
            if not self.ajustar:
                raise ValueError("caixa ajustada recebida, mas a ferramenta não está no modo ajustar (--ajustar)")
            if decisao != "mal_posta":
                raise ValueError("só uma caixa mal posta pode ter caixa ajustada")
            if not caixa_valida(box_aj):
                raise ValueError("caixa ajustada inválida")
            box_aj = [round(min(max(float(x), 0.0), 1.0), 6) for x in box_aj]
        try:
            dur = int(dur_ms) if dur_ms is not None and 0 <= int(dur_ms) <= 86_400_000 else ""
        except (TypeError, ValueError):
            dur = ""
        with self.lock:
            antes = self.dec.get(item, {})
            nota = " ".join(str(antes.get("nota", "") if nota is None else nota).split())
            self.dec[item] = {"decisao": decisao, "nota": nota, "box_aj": box_aj, "modo": "ajustar" if self.ajustar else "marcar",
                              "dur_ms": dur, "t_decisao": agora(), "sessao": self.sessao}
            self._registar(item, antes.get("decisao", ""), decisao, nota, box_aj, dur)
            self._gravar()
        return {"ok": True, "respondidos": self.respondidos()}

    def apagar(self, item):
        item, _ = self._item(item)
        with self.lock:
            antes = self.dec.pop(item, None)
            if antes:
                resto = nota_com(antes.get("nota", ""))
                if resto:                                      # a nota escrita à mão fica, sem resposta
                    self.dec[item] = {"decisao": "", "nota": resto, "box_aj": None, "modo": "", "dur_ms": "",
                                      "t_decisao": "", "sessao": ""}
                self._registar(item, antes.get("decisao", ""), "", resto)
                self._gravar()
        return {"ok": True, "respondidos": self.respondidos()}

    def nota(self, item, texto):
        item, _ = self._item(item)
        with self.lock:
            d = self.dec.setdefault(item, {"decisao": "", "nota": "", "box_aj": None, "modo": "", "dur_ms": "",
                                           "t_decisao": "", "sessao": ""})
            d["nota"] = " ".join(str(texto).split())
            self._registar(item, d.get("decisao", ""), d.get("decisao", ""), d["nota"])
            self._gravar()
        return {"ok": True}

    def nao_identifico(self, j):
        j = int(j)
        if not 0 <= j < len(self.janelas):
            raise ValueError("janela inexistente")
        with self.lock:
            for i in self.janelas[j]["idx"]:
                item = self.itens[i]["item"]
                antes = self.dec.get(item, {})
                nota = nota_com(antes.get("nota", ""), NOTA_NAO_ID)
                self.dec[item] = {"decisao": "nao_avaliavel", "nota": nota, "box_aj": None,
                                  "modo": "ajustar" if self.ajustar else "marcar", "dur_ms": "", "t_decisao": agora(),
                                  "sessao": self.sessao}
                self._registar(item, antes.get("decisao", ""), "nao_avaliavel", nota)
            self._gravar()
        return {"ok": True, "respondidos": self.respondidos()}


class Pedidos(BaseHTTPRequestHandler):
    est = None

    def log_message(self, *a):                                 # silencioso: só os erros aparecem
        pass

    def _enviar(self, corpo, tipo, codigo=200, cache=False):
        self.send_response(codigo)
        self.send_header("Content-Type", tipo)
        self.send_header("Content-Length", str(len(corpo)))
        self.send_header("Cache-Control", "max-age=86400" if cache else "no-store")
        self.end_headers()
        self.wfile.write(corpo)

    def _json(self, obj, codigo=200):
        self._enviar(json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", codigo)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path == "/":
                return self._enviar(PAGINA.encode("utf-8"), "text/html; charset=utf-8")
            if u.path == "/api/estado":
                return self._json(self.est.estado())
            if u.path == "/api/janela":
                return self._json(self.est.janela(int(q["j"])))
            if u.path in ("/img", "/rel"):
                j = int(q["j"])
                if not 0 <= j < len(self.est.janelas):
                    raise KeyError("janela inexistente")
                jo = self.est.janela_obj(j)
                if q.get("w") != f"{jo.caso}_{jo.seed}":
                    raise KeyError("a janela não corresponde")       # nunca servir a imagem de outra janela
                if u.path == "/img":
                    f = jo.ficheiros[int(q["f"])]
                    if q.get("t") and Image is not None:
                        return self._enviar(miniatura(str(f)), "image/jpeg", cache=True)
                    return self._enviar(f.read_bytes(), "image/jpeg", cache=True)
                f = jo.relatorios[int(q["k"])]
                return self._enviar(f.read_bytes(), "image/png" if f.suffix.lower() == ".png" else "image/jpeg", cache=True)
            self._json({"erro": "não existe"}, 404)
        except (KeyError, ValueError, IndexError) as e:
            self._json({"erro": f"pedido inválido ({type(e).__name__}: {e})"}, 400)
        except OSError as e:
            self._json({"erro": f"não consegui ler o ficheiro ({e})"}, 500)

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            d = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/api/decidir":
                return self._json(self.est.decidir(d["item"], d["decisao"], d.get("nota"), d.get("box_aj"), d.get("dur_ms")))
            if self.path == "/api/apagar":
                return self._json(self.est.apagar(d["item"]))
            if self.path == "/api/nota":
                return self._json(self.est.nota(d["item"], d.get("nota", "")))
            if self.path == "/api/nao_identifico":
                return self._json(self.est.nao_identifico(d["j"]))
            self._json({"erro": "não existe"}, 404)
        except (KeyError, ValueError, IndexError, TypeError, json.JSONDecodeError) as e:
            self._json({"erro": f"pedido inválido ({type(e).__name__}: {e})"}, 400)
        except OSError as e:
            self._json({"erro": f"não consegui gravar ({e})"}, 500)


# ============================================================================ consolidar
def consolidar(itens, pasta, destino, revisor, lista_sha):
    """Junta as respostas num só ficheiro, só se todas as caixas tiverem resposta. O ficheiro fica só de leitura e não
    é reescrito com outro conteúdo."""
    por_item = {it["item"]: i for i, it in enumerate(itens)}
    dec = ler_decisoes(pasta, por_item)
    faltam = [it["item"] for it in itens if dec.get(it["item"], {}).get("decisao") not in DECISOES]
    if faltam:
        raise ErroRever(f"Não gravei nada: {len(faltam)} caixas ainda não têm resposta (ex.: itens {faltam[:5]}). "
                        "Abre a ferramenta e usa o U para ires às que faltam.")
    linhas, cont = [], {d: 0 for d in DECISOES}
    for it in itens:
        d = dec[it["item"]]
        cont[d["decisao"]] += 1
        b = d.get("box_aj") or ["", "", "", ""]
        linhas.append([it["item"], it["split"], it["caso"], it["seed"], it["frame"], d["decisao"], d.get("nota", ""), *b,
                       d.get("modo", ""), d.get("dur_ms", ""), d.get("t_decisao", ""), d.get("sessao", ""), revisor])
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(COL_DEC)
    w.writerows(linhas)
    txt = buf.getvalue()
    destino = Path(destino)
    if destino.exists():
        if destino.read_text(encoding="utf-8") != txt:
            raise ErroRever(f"{destino} já existe com outro conteúdo: as decisões consolidadas não se reescrevem. Se "
                            "alguma resposta mudou depois de consolidar, grava uma versão nova (--consolidado …_v2.csv) e "
                            "regista o motivo no docs/decisoes.md.")
        estado = "igual ao que já existia"
    else:
        destino.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=destino.parent, delete=False,
                                         prefix=f".{destino.name}.", suffix=".tmp") as t:
            t.write(txt)
            t.flush()
            os.fsync(t.fileno())
        os.chmod(t.name, 0o444)
        os.replace(t.name, destino)
        estado = "gravado"
    n = len(itens)
    nomes = {"aceitar": "aceitar", "mal_posta": "mal posta", "rejeitar": "rejeitar", "nao_avaliavel": "não avaliável"}
    print(f"Consolidado ({estado}): {n} caixas · " + " · ".join(f"{nomes[k]} {v}" for k, v in cont.items()))
    print(f"  {destino} · sha256 {sha256(destino)}")
    print(f"  lista: sha256 {lista_sha} · revisor: {revisor}")
    print(f"Linha para o docs/decisoes.md:\n- {datetime.date.today().isoformat()}: Guião C, respostas do revisor {revisor} "
          f"consolidadas (tools/rever.py v{VERSAO}): {n} caixas; {destino} sha256 {sha256(destino)[:16]} (lista sha256 "
          f"{lista_sha[:16]}). A partir daqui o ficheiro não muda.")
    return {"destino": destino, "contagens": cont, "estado": estado}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=os.environ.get("DS"), help="pasta Dataset_WCE, só leitura (por omissão, $DS)")
    ap.add_argument("--itens", required=True, help="lista congelada: data/derived/guiao_c/itens_v1.csv ou piloto_v1.csv")
    ap.add_argument("--revisor", required=True, help="nome da pasta das respostas (ex.: paulo, piloto)")
    ap.add_argument("--saida", default="data/derived/guiao_c/revisao")
    ap.add_argument("--tabela", default="data/derived/janelas_v1.csv", help="para a descrição clínica")
    ap.add_argument("--contexto", default="data/derived/guiao_c/contexto_v1.csv",
                    help="caixas dos outros frames (scripts/17b_c_contexto.py)")
    ap.add_argument("--sem-contexto", action="store_true", help="abre sem as caixas dos outros frames")
    ap.add_argument("--alcance", type=int, default=12, help="frames seguidos para cada lado do seed, na tira")
    ap.add_argument("--porta", type=int, default=8766)
    ap.add_argument("--ajustar", action="store_true", help="modo ajustar: depois do M, desenha-se a caixa certa")
    ap.add_argument("--consolidar", action="store_true", help="juntar as respostas num só ficheiro (fim da revisão)")
    ap.add_argument("--consolidado", help="por omissão data/derived/guiao_c/decisoes_<revisor>.csv")
    args = ap.parse_args(argv)

    itens = ler_lista(args.itens)
    lista_sha = sha256(args.itens)
    if args.consolidar:
        pasta = preparar_saida(args.saida, args.revisor, args.itens, lista_sha, criar=False)
        destino = args.consolidado or str(Path(args.saida).parent / f"decisoes_{args.revisor}.csv")
        return consolidar(itens, pasta, destino, args.revisor, lista_sha)

    if not args.dataset or not Path(args.dataset).is_dir():
        raise ErroRever('Não encontrei a pasta do dataset: faz o export DS=…/raw/Dataset_WCE (ou usa --dataset "$DS").')
    if args.sem_contexto:
        contexto, sem_ctx = {}, None
    else:
        contexto, sem_ctx = ler_contexto(args.contexto, itens)
    pasta = preparar_saida(args.saida, args.revisor, args.itens, lista_sha)
    est = Estado(args.dataset, itens, pasta, ler_tabela(args.tabela), args.revisor, args.ajustar, contexto, args.alcance)
    if sem_ctx is None:
        print("Aviso: sem contexto (--sem-contexto): na tira não aparecem as caixas dos outros frames.")
    elif sem_ctx:
        print(f"Aviso: {sem_ctx} de {len(est.janelas)} janelas sem nenhuma caixa no contexto {args.contexto}. "
              "Se forem muitas, o contexto é de outra lista: grava-o outra vez com o scripts/17b_c_contexto.py.")
    if contexto:
        n_o = sum(1 for w in contexto.values() for c in w.values() if "o" in c)
        n_m = sum(1 for w in contexto.values() for c in w.values() if "m" in c)
        print(f"Contexto: {n_o} caixas originais e {n_m} do modelo nos outros frames (seed ± {args.alcance}).")
    sem_pasta = [w["k"] for w in est.janelas
                 if not (Path(args.dataset) / w["k"][0] / w["k"][1] / "GT_com_classes" / str(w["k"][2])).is_dir()]
    for k in sem_pasta[:5]:
        print(f"Aviso: não encontrei a pasta da janela {'/'.join(map(str, k))}")
    if sem_pasta:
        print(f"Aviso: {len(sem_pasta)} janelas sem pasta no dataset. Confirma o $DS.")
    if Image is None:
        print("Aviso: sem Pillow, as miniaturas são as imagens inteiras (mais lentas).")
    print(f"{len(itens)} caixas em {len(est.janelas)} janelas · {est.respondidos()} já com resposta · revisor {args.revisor} · "
          f"modo {'ajustar' if args.ajustar else 'só marcar'} · a gravar em {pasta}/")
    Pedidos.est = est
    try:
        srv = ThreadingHTTPServer(("127.0.0.1", args.porta), Pedidos)
    except OSError:
        raise ErroRever(f"A porta {args.porta} está ocupada (a ferramenta já está a correr noutra janela?). "
                        f"Usa --porta {args.porta + 1}.")
    print(f"Pronto: abre http://localhost:{args.porta} no browser. Ctrl+C para parar.", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nParado. Tudo o que foi respondido está gravado.")
    finally:
        srv.server_close()


PAGINA = r"""<!doctype html>
<html lang="pt-PT">
<head>
<meta charset="utf-8">
<title>Revisão das caixas · Guião C</title>
<style>
:root{--bg:#14171c;--card:#1d2229;--card2:#262c35;--fg:#e7eaee;--mut:#9aa4b1;--line:#353d48;
  --ac:#3fb950;--mp:#e3b341;--rj:#f85149;--na:#8b949e;--prop:#ff00ff;--seed:#00e5ff;--aj:#3fb950;--erro:#ff6b6b}
*{box-sizing:border-box}
html,body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,Ubuntu,sans-serif}
header{position:sticky;top:0;z-index:10;display:flex;flex-wrap:wrap;gap:8px 12px;align-items:center;padding:8px 14px;background:var(--card);border-bottom:1px solid var(--line)}
button,select,input{font:inherit;color:var(--fg);background:var(--card2);border:1px solid var(--line);border-radius:6px;padding:5px 10px}
button{cursor:pointer}
button:hover:not(:disabled){border-color:var(--mut)}
button:disabled{opacity:.4;cursor:default}
input:disabled{opacity:.45}
#jsel{max-width:min(460px,55vw)}
.prog{flex:1;min-width:120px;height:8px;background:var(--card2);border-radius:4px;overflow:hidden}
.prog>div{height:100%;background:var(--ac);width:0;transition:width .2s}
#gravado{color:var(--mut);font-size:13px;min-width:130px;text-align:right}
#gravado.erro{color:var(--erro);font-weight:700}
#modo{font-size:12px;color:var(--mut)}
main{display:grid;grid-template-columns:340px minmax(0,1fr);gap:12px;padding:10px 12px}
.ref,.ver{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px}
.ref{display:flex;flex-direction:column;gap:6px;align-self:start}
.ref h3{margin:4px 0 0;font-size:11.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--mut);font-weight:600}
.img-caixa{position:relative;line-height:0;overflow:hidden;border-radius:6px}
.img-caixa img{display:block;width:100%;border-radius:6px;background:#000}
.bbox{position:absolute;pointer-events:none;box-shadow:0 0 0 1px #000c}
.bbox.seed{border:3px solid var(--seed)}
.bbox.orig{border:2px solid var(--seed)}
.bbox.prop{border:2px solid var(--prop)}
.bbox.mod{border:2px dashed var(--prop);box-shadow:none}
.bbox.aj{border:2px dashed var(--aj)}
.bbox.desenho{border:2px dashed #fff}
#rel img{width:100%;max-height:150px;object-fit:contain;background:#000;border-radius:6px;margin-bottom:4px}
.desc{font-size:15px;font-weight:600}
.mut{color:var(--mut)}
.cor{font-weight:700}
.cor.orig{color:var(--seed)}
.cor.prop{color:var(--prop)}
.cor.mod{color:var(--prop);text-decoration:underline dashed;text-underline-offset:3px}
.ver{display:flex;flex-direction:column;align-items:center;gap:7px;min-width:0}
.info{display:flex;gap:14px;align-items:baseline;flex-wrap:wrap;justify-content:center;min-height:30px}
.info .fr{font-size:18px;font-weight:700}
.info .off{font-size:18px;color:var(--mut)}
#ctxtxt{color:var(--mp);font-weight:700}
.badge{padding:3px 12px;border-radius:999px;font-weight:700;border:1px solid var(--line)}
.badge.aceitar{background:var(--ac);color:#04130a;border-color:var(--ac)}
.badge.mal_posta{background:var(--mp);color:#1a1200;border-color:var(--mp)}
.badge.rejeitar{background:var(--rj);color:#fff;border-color:var(--rj)}
.badge.nao_avaliavel{background:var(--na);color:#fff;border-color:var(--na)}
.palco{position:relative;overflow:hidden;border-radius:8px;background:#000;width:min(54vh,46vw);aspect-ratio:1/1;cursor:grab;user-select:none}
.palco.desenhar{cursor:crosshair}
.zoom{position:absolute;left:0;top:0;width:100%;height:100%;transform-origin:0 0}
.zoom img{display:block;width:100%;height:100%;object-fit:fill;pointer-events:none}
.botoes{display:flex;gap:8px;flex-wrap:wrap;justify-content:center}
.botoes button{min-width:118px;font-weight:600}
.ba{border-color:var(--ac)!important}.bm{border-color:var(--mp)!important}.br{border-color:var(--rj)!important}.bn{border-color:var(--na)!important}
.aviso{min-height:20px;color:var(--mp);font-weight:600;text-align:center}
.ajuda{color:var(--mut);font-size:12.5px;text-align:center;max-width:900px}
.regra{font-size:12.5px;line-height:1.4}
.regra p{margin:0 0 5px}
.regra b{font-weight:700}
.tira{width:100%;display:flex;flex-direction:column;align-items:center;gap:3px}
#tally{font-size:12.5px}
#tally .t-a{color:var(--ac)} #tally .t-r{color:var(--rj)} #tally .t-n{color:var(--na)}
#thumbs{position:relative;display:flex;gap:4px;overflow-x:auto;max-width:100%;padding:3px 3px 5px}
.th{position:relative;flex:0 0 auto;width:66px;border:2px solid var(--line);border-radius:5px;cursor:pointer;line-height:0;overflow:hidden;background:#000;user-select:none}
.th img{width:100%;aspect-ratio:1/1;object-fit:cover;display:block}
.th .mb{position:absolute;pointer-events:none;border:1.5px solid}
.th .mb.orig{border-color:var(--seed)}
.th .mb.prop{border-color:var(--prop)}
.th .mb.mod{border:1.5px dashed var(--prop)}
.th .lab{position:absolute;left:2px;top:2px;line-height:1;font-size:10px;font-weight:700;padding:2px 3px;border-radius:3px;background:#000b;color:#fff}
.th.seed .lab{background:var(--seed);color:#000}
.th .res{position:absolute;right:2px;bottom:2px;min-width:15px;line-height:15px;font-size:10.5px;font-weight:800;text-align:center;border-radius:3px;padding:0 2px;color:#fff;background:#000c;border:1px solid #fffa}
.th .res.aceitar{background:var(--ac);color:#04130a;border-color:var(--ac)}
.th .res.rejeitar{background:var(--rj);border-color:var(--rj)}
.th .res.nao_avaliavel{background:var(--na);border-color:var(--na)}
.th .res.mal_posta{background:var(--mp);color:#1a1200;border-color:var(--mp)}
.th.item{border-color:#6b4f7a}
.th.vista{outline:2px solid #fff;outline-offset:1px;border-color:#fff}
@media (max-width:900px){main{grid-template-columns:1fr}.palco{width:min(92vw,70vh)}}
</style>
</head>
<body>
<header>
  <button id="jprev" title="janela anterior (PgUp)">◀</button>
  <select id="jsel" title="janela"></select>
  <button id="jnext" title="janela seguinte (PgDn)">▶</button>
  <span id="cont"></span>
  <div class="prog"><div id="progbar"></div></div>
  <span id="modo"></span>
  <span id="gravado">—</span>
</header>
<main>
  <aside class="ref">
    <h3>Seed com a caixa da anotação original</h3>
    <div class="img-caixa"><img id="seedimg" alt=""><div class="bbox seed" id="seedbox" hidden></div></div>
    <div class="mut" id="seedtxt"></div>
    <h3>Imagem do relatório</h3>
    <div id="rel"></div>
    <h3>Descrição clínica</h3>
    <div class="desc" id="desc"></div>
    <div class="mut" id="cat"></div>
    <h3>Regra</h3>
    <div class="regra">
      <p><b>A · aceitar</b>: a caixa <span class="cor prop">roxa cheia</span> marca a lesão do seed (a da caixa azul do seed), mais ou menos como a desenharias.</p>
      <p><b>R · rejeitar</b>: não marca a lesão do seed: está noutra coisa, ou a lesão não está no frame.</p>
      <p><b>N · não dá para decidir</b>: a imagem não deixa ver, não reconheces a lesão do seed, ou não tens a certeza de que é a mesma.</p>
      <p class="mut"><span class="cor orig">azul</span>: caixa original (do Rodrigo; pode ter erros) · <span class="cor prop">roxo cheio</span>: caixa a rever · <span class="cor mod">roxo tracejado</span>: caixa do modelo noutro frame.</p>
      <p class="mut">Se a caixa azul do seed não marca nada claro: nota «seed duvidoso» e N em todas as caixas da janela.</p>
    </div>
  </aside>
  <section class="ver">
    <div class="info"><span class="fr" id="fr"></span><span class="off" id="off"></span>
      <span class="badge" id="badge">sem resposta</span><span id="ctxtxt"></span></div>
    <div class="palco" id="palco"><div class="zoom" id="zoom"><img id="atual" alt="">
      <div class="bbox orig" id="origbox" hidden></div><div class="bbox mod" id="modbox" hidden></div>
      <div class="bbox prop" id="propbox" hidden></div><div class="bbox aj" id="ajbox" hidden></div>
      <div class="bbox desenho" id="desenho" hidden></div></div></div>
    <div class="tira"><div id="thumbs"></div><span class="mut" id="tally"></span></div>
    <div class="aviso" id="aviso"></div>
    <div class="botoes">
      <button class="ba" data-d="aceitar">A · aceitar</button>
      <button class="br" data-d="rejeitar">R · rejeitar</button>
      <button class="bn" data-d="nao_avaliavel">N · não dá para decidir</button>
      <button class="bm" data-d="mal_posta" id="bmal" hidden>M · mal posta (desenhar)</button>
    </div>
    <div class="ajuda">← → frame a frame (ou clica na tira) · A R N respondem nos frames com ? · U próxima por responder ·
      Esc volta à caixa · S seed · Z desfaz · H esconde as caixas · PgUp PgDn mudam de janela · roda do rato: zoom ·
      duplo clique: tamanho normal</div>
    <input id="nota" placeholder="nota desta caixa (opcional; Enter grava)" style="width:min(560px,100%)">
  </section>
</main>
<script>
const $ = id => document.getElementById(id);
const NOME = {aceitar: "aceitar", mal_posta: "mal posta", rejeitar: "rejeitar", nao_avaliavel: "não dá para decidir"};
const LETRA = {aceitar: "A", mal_posta: "M", rejeitar: "R", nao_avaliavel: "N"};
const RAPIDAS = ["outra lesão", "dúvida", "não identifico a lesão no seed"];
let E = null, J = null, pos = 0, vista = null, mostrado = 0, pilha = [], seq = 0, carregando = false;
let desenhar = false, ini = null, mostrarCaixa = true;
let zoom = {s: 1, x: 0, y: 0}, arrasto = null;
let fila = Promise.resolve(), pendentes = 0;
const cacheJ = {}, aCarregar = {};                // janelas já carregadas (com as respostas desta sessão) e a carregar

function notaPara(antiga, extra) {
  // as notas rápidas antigas saem quando a resposta muda; o texto escrito à mão fica
  const partes = (antiga || "").split(" · ").map(s => s.trim()).filter(s => s && !RAPIDAS.includes(s));
  if (extra) partes.unshift(extra);
  return partes.join(" · ");
}

const esc = s => String(s).replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
const offTxt = o => o === 0 ? "seed" : (o > 0 ? "+" + o : "−" + (-o));
const Wd = w => `j=${w.j}&w=${w.caso}_${w.seed}`;
const W = () => Wd(J);
const imgDe = (w, f, t) => `/img?${Wd(w)}&f=${f}` + (t ? "&t=1" : "");
const img = (f, t) => imgDe(J, f, t);
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const itemAtual = () => J.itens.find(x => x.item === E.itens[pos][0]);
const itemDoFrame = f => J.itens.find(x => x.frame === f) || null;
const seguido = f => J.seguidos.find(s => s.f === f) || {};

async function get(url) {
  const r = await fetch(url, {cache: "no-store"});
  const d = await r.json();
  if (!r.ok || d.erro) throw new Error(d.erro || r.statusText);
  return d;
}
function post(url, corpo) {
  pendentes++; mostrarGravacao();
  const p = fila.then(async () => {
    const r = await fetch(url, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(corpo)});
    let d = {}; try { d = await r.json(); } catch (e) {}
    if (!r.ok || d.erro) throw new Error(d.erro || ("HTTP " + r.status));
    return d;
  });
  fila = p.catch(() => {});
  return p.then(d => { pendentes--; mostrarGravacao(); return d; }, e => { pendentes--; falhou(e); });
}
function mostrarGravacao() {
  const g = $("gravado");
  g.classList.remove("erro");
  g.textContent = pendentes ? "a gravar…" : "gravado ✓ " + new Date().toLocaleTimeString("pt-PT");
}
function falhou(e) {
  const g = $("gravado");
  g.classList.add("erro"); g.textContent = "ERRO ao gravar";
  alert("Não consegui gravar: " + e.message + "\n\nA página vai ser recarregada a partir do que está gravado no disco.");
  location.reload();
}
function aviso(t) {
  const a = $("aviso"); a.textContent = t;
  clearTimeout(aviso.tm); aviso.tm = setTimeout(() => a.textContent = "", 3200);
}
function caixa(el, c) {
  if (!c) { el.hidden = true; return; }
  el.hidden = false;
  el.style.left = (100 * c[0]) + "%"; el.style.top = (100 * c[1]) + "%";
  el.style.width = (100 * (c[2] - c[0])) + "%"; el.style.height = (100 * (c[3] - c[1])) + "%";
}

// ------------------------------------------------------------------ arranque e janelas
async function iniciar() {
  E = await get("/api/estado");
  E.posDe = {}; E.blocos = {};                     // as caixas de cada janela vêm seguidas na lista
  E.itens.forEach((x, i) => { E.posDe[x[0]] = i; const b = E.blocos[x[1]]; if (b) b[1] = i; else E.blocos[x[1]] = [i, i]; });
  const s = $("jsel"); s.innerHTML = "";
  E.janelas.forEach(j => { const o = document.createElement("option"); o.value = j.j; s.appendChild(o); });
  $("modo").textContent = `revisor ${E.revisor} · ` + (E.ajustar ? "modo ajustar" : "modo só marcar");
  $("bmal").hidden = !E.ajustar;
  listaJanelas();
  pos = E.atual;
  await mostrarItem();
  if (E.respondidos === E.total) aviso("Todas as caixas têm resposta ✓");
}
function respondidasJanela(j) { const [a, b] = E.blocos[j]; let r = 0; for (let k = a; k <= b; k++) r += E.itens[k][2]; return r; }
function opcao(j) {
  const w = E.janelas[j], r = respondidasJanela(j);
  $("jsel").options[j].textContent = `${j + 1}/${E.janelas.length} · caso ${w.caso} · seed ${w.seed} · ${r}/${w.n}` + (r === w.n ? " ✓" : "");
}
function listaJanelas() { E.janelas.forEach(w => opcao(w.j)); }
function carregarJanela(j) {
  if (cacheJ[j]) return Promise.resolve(cacheJ[j]);
  if (!aCarregar[j]) aCarregar[j] = get("/api/janela?j=" + j).then(w => { cacheJ[j] = w; delete aCarregar[j]; return w; },
                                                                   e => { delete aCarregar[j]; throw e; });
  return aCarregar[j];
}
function preCarregar() {
  // as próximas caixas desta janela e a janela seguinte (dados, seed, relatório e tira), em segundo plano
  const [, b] = E.blocos[J.j];
  for (let k = pos + 1, n = 0; k <= b && n < 3; k++, n++) { const it = J.itens.find(x => x.item === E.itens[k][0]); if (it) new Image().src = img(it.frame); }
  const jn = J.j + 1;
  if (jn >= E.janelas.length || cacheJ[jn] || aCarregar[jn]) return;
  carregarJanela(jn).then(w => {
    new Image().src = imgDe(w, w.seed);
    w.relatorios.forEach(k => { new Image().src = `/rel?${Wd(w)}&k=${k}`; });
    for (const s of w.seguidos) new Image().src = imgDe(w, s.f, true);
    const it = w.itens.find(x => !x.decisao) || w.itens[0];
    new Image().src = imgDe(w, it.frame);
  }).catch(() => {});
}
async function abrirJanela(j) {
  J = await carregarJanela(j);
  $("jsel").value = j;
  $("jprev").disabled = j === 0; $("jnext").disabled = j === E.janelas.length - 1;
  $("seedimg").src = img(J.seed);
  caixa($("seedbox"), J.caixa);
  $("seedtxt").textContent = `caso ${J.caso} · seed ${J.seed}` + (J.caixa ? "" : " · SEM CAIXA");
  $("rel").innerHTML = J.relatorios.length ? J.relatorios.map(k => `<img src="/rel?${W()}&k=${k}" alt="">`).join("") : '<span class="mut">(sem imagem do relatório)</span>';
  $("desc").innerHTML = J.descricao ? J.descricao.split(" | ").map(d => esc(d.replace(/_/g, " "))).join("<br>") : '<span class="mut">(sem descrição)</span>';
  $("cat").textContent = J.categoria ? "categoria (do texto): " + J.categoria.replace(/_/g, " ") : "";
  construirTira();
  if (J.avisos.length) aviso("Atenção: " + J.avisos.join("; "));
}

// ------------------------------------------------------------------ o frame à vista e a tira
async function mostrarItem() {
  // vai para a caixa pos: abre a janela se for preciso, zoom normal e caixas à vista
  const meu = ++seq, j = E.itens[pos][1];
  if (!J || J.j !== j) {
    carregando = true;
    try { await abrirJanela(j); } finally { if (meu === seq) carregando = false; }
    if (meu !== seq) return;                     // entretanto pediu-se outra caixa
  }
  repor();
  if (!mostrarCaixa) { mostrarCaixa = true; $("aviso").textContent = ""; }   // ao mudar de caixa, as caixas voltam
  mostrarVista(itemAtual().frame, true);
  preCarregar();
}
function mostrarVista(f, nova) {
  desenharOff();
  vista = f;
  const it = itemDoFrame(f), o = f - J.seed;
  if (it) {
    const p = E.posDe[it.item];
    if (nova || p !== pos) { pos = p; mostrado = performance.now(); }
  }
  $("atual").src = img(f);
  desenharCaixas();
  if (it) {
    $("fr").textContent = `caixa ${pos + 1} de ${E.itens.length}`;
    $("off").textContent = `offset ${offTxt(o)} · janela ${J.j + 1}/${E.janelas.length}`;
    emblema(it.decisao || "");
    $("ctxtxt").textContent = "";
    $("nota").disabled = false; $("nota").value = it.nota || "";
    $("nota").placeholder = "nota desta caixa (opcional; Enter grava)";
  } else {
    $("fr").textContent = o === 0 ? "seed" : `offset ${offTxt(o)}`;
    $("off").textContent = `janela ${J.j + 1}/${E.janelas.length}`;
    emblema(null);
    $("ctxtxt").textContent = o === 0 ? "o seed: a caixa azul é o que ele segue" : "frame sem caixa para rever";
    $("nota").disabled = true; $("nota").value = "";
    $("nota").placeholder = "(este frame não tem caixa para rever)";
  }
  atualizarTira();
  contagens();
  const i = J.seguidos.findIndex(s => s.f === f);   // os vizinhos em grande, para as setas não esperarem
  for (const d of [-1, 1]) { const s = J.seguidos[i + d]; if (s) new Image().src = img(s.f); }
}
function desenharCaixas() {
  const s = seguido(vista), it = itemDoFrame(vista), v = mostrarCaixa;
  caixa($("origbox"), v ? (vista === J.seed ? J.caixa : s.o) : null);
  caixa($("modbox"), v && !it ? s.m : null);
  caixa($("propbox"), v && it ? it.box : null);
  caixa($("ajbox"), v && it ? it.box_aj : null);
}
function emblema(d) {
  const b = $("badge");
  if (d === null) { b.hidden = true; return; }
  b.hidden = false; b.className = "badge" + (d ? " " + d : ""); b.textContent = d ? NOME[d] : "sem resposta";
}
function mini(b, cls) {
  return b ? `<i class="mb ${cls}" style="left:${100 * b[0]}%;top:${100 * b[1]}%;width:${100 * (b[2] - b[0])}%;height:${100 * (b[3] - b[1])}%"></i>` : "";
}
function construirTira() {
  $("thumbs").innerHTML = J.seguidos.map(s => {
    const it = itemDoFrame(s.f), o = s.f - J.seed;
    return `<div class="th${o === 0 ? " seed" : ""}${it ? " item" : ""}" data-f="${s.f}">` +
      `<img src="${img(s.f, true)}" alt="">` + mini(s.o, "orig") + (it ? mini(it.box, "prop") : mini(s.m, "mod")) +
      `<span class="lab">${o === 0 ? "★" : offTxt(o)}</span>` + (it ? '<b class="res"></b>' : "") + "</div>";
  }).join("");
}
function atualizarTira() {
  const c = $("thumbs");
  for (const t of c.querySelectorAll(".th")) {
    const f = +t.dataset.f, v = f === vista, r = t.querySelector(".res");
    t.classList.toggle("vista", v);
    if (r) { const it = itemDoFrame(f); r.textContent = it.decisao ? LETRA[it.decisao] : "?"; r.className = "res " + (it.decisao || "falta"); }
    if (v) c.scrollLeft = Math.max(0, t.offsetLeft - (c.clientWidth - t.offsetWidth) / 2);
  }
  const n = {aceitar: 0, rejeitar: 0, nao_avaliavel: 0, mal_posta: 0};
  let falta = 0;
  for (const it of J.itens) { if (it.decisao) n[it.decisao]++; else falta++; }
  $("tally").innerHTML = `nesta janela: <b class="t-a">A ${n.aceitar}</b> · <b class="t-r">R ${n.rejeitar}</b> · <b class="t-n">N ${n.nao_avaliavel}</b>` +
    (n.mal_posta ? ` · M ${n.mal_posta}` : "") + " · " + (falta ? `faltam ${falta}` : "todas ✓") +
    ' &nbsp;|&nbsp; <span class="cor orig">azul</span> original · <span class="cor prop">roxo cheio</span> a rever · <span class="cor mod">roxo tracejado</span> modelo noutro frame';
}
function contagens() {
  const n = E.itens.filter(x => x[2]).length;
  $("cont").innerHTML = `<b>${n}/${E.itens.length}</b> respondidas` + (n === E.itens.length ? ' · <span style="color:var(--ac)">todas ✓</span>' : ` · faltam ${E.itens.length - n}`);
  $("progbar").style.width = (100 * n / E.itens.length) + "%";
}

// ------------------------------------------------------------------ navegação
function passo(d) {
  const i = J.seguidos.findIndex(s => s.f === vista), n = i + d;
  if (i < 0 || n < 0 || n >= J.seguidos.length) {
    aviso(d < 0 ? "É o primeiro frame seguido desta janela (PgUp: janela anterior)." : "É o último frame seguido desta janela (PgDn: janela seguinte).");
    return;
  }
  mostrarVista(J.seguidos[n].f);
}
function voltarItem() { mostrarVista(itemAtual().frame); }
function verSeed() { if (vista === J.seed) voltarItem(); else mostrarVista(J.seed); }
function irPara(p) { pos = clamp(p, 0, E.itens.length - 1); mostrarItem().catch(e => alert(e.message)); }
function proximaPorResponder(p) {
  // primeiro nesta janela (as seguintes, depois as anteriores), depois nas janelas seguintes
  const N = E.itens.length, [a, b] = E.blocos[E.itens[p][1]];
  for (let k = p + 1; k <= b; k++) if (!E.itens[k][2]) return k;
  for (let k = a; k < p; k++) if (!E.itens[k][2]) return k;
  for (let s = 1; s <= N; s++) { const k = (b + s) % N; if (!E.itens[k][2]) return k; }
  return null;
}
function proximaSem() {
  if (!E.itens[pos][2] && vista !== itemAtual().frame) { voltarItem(); return; }
  const k = proximaPorResponder(pos);
  if (k === null) aviso("Todas as caixas têm resposta ✓"); else irPara(k);
}
function primeiraDaJanela(j) { const [a, b] = E.blocos[j]; for (let k = a; k <= b; k++) if (!E.itens[k][2]) return k; return a; }
function janelaRel(d) {
  const j = J.j + d;
  if (j < 0 || j >= E.janelas.length) { aviso(d < 0 ? "É a primeira janela." : "É a última janela."); return; }
  irPara(primeiraDaJanela(j));
}

// ------------------------------------------------------------------ respostas
function responder(d, extra, box) {
  if (carregando) { aviso("A carregar a janela…"); return; }
  const it = itemDoFrame(vista);
  if (!it) {
    aviso((vista === J.seed ? "O seed não é para rever" : "Este frame não tem caixa para rever") + ": ← → andam, U vai à próxima por responder.");
    return;
  }
  const p = E.posDe[it.item], dur = Math.round(performance.now() - mostrado), nota = notaPara(it.nota, extra);
  const corpo = {item: it.item, decisao: d, nota, dur_ms: dur};
  if (box) corpo.box_aj = box;
  it.decisao = d; it.nota = nota; it.box_aj = box || null;
  E.itens[p][2] = 1;
  post("/api/decidir", corpo);
  pilha.push(p);
  opcao(J.j);
  const k = proximaPorResponder(p);
  if (k === null) { pos = p; mostrarVista(vista, true); aviso("Fim: todas as caixas têm resposta ✓"); }
  else irPara(k);
}
function malPosta() {
  if (!itemDoFrame(vista)) { aviso("Este frame não tem caixa para rever."); return; }
  if (!E.ajustar) { responder("mal_posta"); return; }
  desenhar = true; $("palco").classList.add("desenhar");
  aviso("Desenha a caixa certa com o rato · Esc: mal posta sem caixa");
}
function desenharOff() { desenhar = false; ini = null; $("palco").classList.remove("desenhar"); $("desenho").hidden = true; }
async function desfazer() {
  if (!pilha.length) { aviso("Nada para desfazer nesta sessão."); return; }
  const p = pilha.pop(), item = E.itens[p][0];
  await fila;
  await post("/api/apagar", {item});
  E.itens[p][2] = 0;
  const w = cacheJ[E.itens[p][1]];
  if (w) { const it = w.itens.find(x => x.item === item); it.decisao = ""; it.nota = notaPara(it.nota); it.box_aj = null; }
  opcao(E.itens[p][1]);
  pos = p; await mostrarItem();
  aviso("Resposta desfeita.");
}
function gravarNota() {
  const it = itemDoFrame(vista);
  if (!it) return;
  const t = $("nota").value.trim();
  if ((it.nota || "") === t) return;
  it.nota = t;
  post("/api/nota", {item: it.item, nota: t});
}

// ------------------------------------------------------------------ zoom, mover e desenhar
function aplicar() { $("zoom").style.transform = `translate(${zoom.x}px, ${zoom.y}px) scale(${zoom.s})`; }
function repor() { zoom = {s: 1, x: 0, y: 0}; aplicar(); }
function limitar() {
  const r = $("palco").getBoundingClientRect();
  zoom.x = clamp(zoom.x, r.width - r.width * zoom.s, 0); zoom.y = clamp(zoom.y, r.height - r.height * zoom.s, 0);
}
function pontoImagem(ev) {
  const r = $("atual").getBoundingClientRect();
  return [clamp((ev.clientX - r.left) / r.width, 0, 1), clamp((ev.clientY - r.top) / r.height, 0, 1)];
}
$("palco").addEventListener("wheel", ev => {
  ev.preventDefault();
  const r = $("palco").getBoundingClientRect(), mx = ev.clientX - r.left, my = ev.clientY - r.top;
  const s2 = clamp(zoom.s * (ev.deltaY < 0 ? 1.25 : 0.8), 1, 8);
  zoom.x = mx - (mx - zoom.x) * (s2 / zoom.s); zoom.y = my - (my - zoom.y) * (s2 / zoom.s); zoom.s = s2;
  limitar(); aplicar();
}, {passive: false});
$("palco").addEventListener("mousedown", ev => {
  if (ev.button !== 0) return;
  ev.preventDefault();
  if (desenhar) { ini = pontoImagem(ev); caixa($("desenho"), [ini[0], ini[1], ini[0], ini[1]]); return; }
  if (zoom.s > 1) arrasto = {x: ev.clientX, y: ev.clientY, zx: zoom.x, zy: zoom.y};
});
window.addEventListener("mousemove", ev => {
  if (desenhar && ini) {
    const p = pontoImagem(ev);
    caixa($("desenho"), [Math.min(ini[0], p[0]), Math.min(ini[1], p[1]), Math.max(ini[0], p[0]), Math.max(ini[1], p[1])]);
  } else if (arrasto) {
    zoom.x = arrasto.zx + ev.clientX - arrasto.x; zoom.y = arrasto.zy + ev.clientY - arrasto.y; limitar(); aplicar();
  }
});
window.addEventListener("mouseup", ev => {
  if (desenhar && ini) {
    const p = pontoImagem(ev);
    const b = [Math.min(ini[0], p[0]), Math.min(ini[1], p[1]), Math.max(ini[0], p[0]), Math.max(ini[1], p[1])];
    ini = null;
    if (b[2] - b[0] < 0.01 || b[3] - b[1] < 0.01) { $("desenho").hidden = true; aviso("Caixa demasiado pequena: desenha outra vez (Esc: sem caixa)."); return; }
    desenharOff();
    responder("mal_posta", undefined, b.map(x => Math.round(x * 1e6) / 1e6));
  }
  arrasto = null;
});
$("palco").addEventListener("dblclick", () => repor());

// ------------------------------------------------------------------ botões e teclas
document.querySelectorAll("button[data-d]").forEach(b => b.addEventListener("click", () => {
  b.blur(); if (b.dataset.d === "mal_posta") malPosta(); else responder(b.dataset.d);
}));
$("jsel").addEventListener("change", () => { $("jsel").blur(); irPara(primeiraDaJanela(+$("jsel").value)); });
$("jprev").onclick = () => { $("jprev").blur(); janelaRel(-1); };
$("jnext").onclick = () => { $("jnext").blur(); janelaRel(1); };
$("thumbs").addEventListener("click", ev => { const t = ev.target.closest(".th"); if (t) mostrarVista(+t.dataset.f); });
$("nota").addEventListener("keydown", ev => { if (ev.key === "Enter" || ev.key === "Escape") { ev.preventDefault(); $("nota").blur(); } });
$("nota").addEventListener("change", gravarNota);
function alternarCaixa() {
  mostrarCaixa = !mostrarCaixa;
  aviso(mostrarCaixa ? "Caixas à vista." : "Caixas escondidas (H mostra outra vez).");
  desenharCaixas();
}
document.addEventListener("keydown", ev => {
  const a = document.activeElement;
  if (a && a.tagName === "INPUT") { if (ev.key === "Escape") a.blur(); return; }
  if (a && a.tagName === "SELECT") a.blur();
  if (ev.ctrlKey || ev.metaKey || ev.altKey || !J) return;
  const k = ev.key.toLowerCase();
  if (desenhar && k === "escape") { desenharOff(); responder("mal_posta"); ev.preventDefault(); return; }
  if (k === "a") responder("aceitar");
  else if (k === "r") responder("rejeitar");
  else if (k === "n") responder("nao_avaliavel");
  else if (k === "m" && E.ajustar) malPosta();
  else if (k === "s") verSeed();
  else if (k === "z" || k === "backspace") desfazer();
  else if (k === "arrowleft") passo(-1);
  else if (k === "arrowright") passo(1);
  else if (k === "u") proximaSem();
  else if (k === "h") alternarCaixa();
  else if (k === "pageup") janelaRel(-1);
  else if (k === "pagedown") janelaRel(1);
  else if (k === "escape") voltarItem();
  else return;
  ev.preventDefault();
});
window.addEventListener("beforeunload", ev => { if (pendentes) { ev.preventDefault(); ev.returnValue = ""; } });
iniciar().catch(e => { document.body.innerHTML = "<p style='padding:20px'>Erro ao arrancar: " + esc(e.message) + "</p>"; });
</script>
</body>
</html>
"""

if __name__ == "__main__":
    try:
        main()
    except ErroRever as e:
        print(f"ERRO: {e}", file=sys.stderr)
        sys.exit(2)
