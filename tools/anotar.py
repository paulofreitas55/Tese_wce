#!/usr/bin/env python3
"""
Passo 0.7 — Ferramenta de anotação de referência PF v2 (três estados: 1 / 0 / D).

Abre um servidor local (só nesta máquina) com uma página onde se anota, frame a frame, se a lesão
marcada pelo médico no seed está visível. Segue docs/criterio_anotacao.md.

  - À esquerda, sempre visível: o frame do seed com a caixa do médico, a imagem do relatório e a
    descrição clínica. Só é lido o ficheiro da caixa do SEED; as caixas dos vizinhos nunca são lidas.
  - Ao centro: o frame atual ampliado, com o nº do frame e a distância ao seed.
  - Teclas: 1 = visível · 0 = não visível · D = dúvida (marcam e avançam na direção atual) ·
    ← → andam (e mudam a direção) · Shift+← → saltam 10 · S = seed · U = próximo sem rótulo ·
    Del = apagar o rótulo.
  - Em baixo: a janela inteira em miniaturas. Clica num frame e Shift+clica noutro para selecionar
    um intervalo e marcar de uma vez como 0 (ou D) os que ainda não têm rótulo.
  - O seed fica marcado 1 automaticamente. Uma janela só fica completa quando todos os frames têm
    rótulo (nada fica 0 por omissão).
  - Grava a cada decisão em <saida>/<split>__<caso>__<seed>.csv (Case_ID, Seed_Frame, Frame_Number,
    Label ∈ {1, 0, D}, Notes, Timestamp) e num registo de todas as alterações. Uma janela nova abre no
    seed; uma janela começada abre no frame sem rótulo mais próximo do seed (onde o trabalho ficou).
  - NUNCA lê as anotações antigas: não recebe a pasta delas e recusa gravar numa pasta que já tenha
    CSV que não foram criados por esta ferramenta.

Uso, a partir da pasta do projeto (no tmux, porque fica a correr):
    python tools/anotar.py --dataset "$DS" --janelas docs/janelas_reanotar.csv
e abrir http://localhost:8765 no browser. Ctrl+C para parar (o que está marcado já está gravado).

Quando todas as janelas estiverem completas:
    python tools/anotar.py --dataset "$DS" --consolidar
grava data/derived/paulo_frame_labels.csv (split, caso, seed, frame, offset, visivel ∈ {1, 0, duvidoso},
nota), depois de verificar que todos os frames de todas as janelas têm rótulo.
"""
import argparse
import csv
import datetime
import io
import json
import os
import re
import sys
import threading
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

try:
    from PIL import Image
except ImportError:                                            # sem Pillow: miniaturas = imagem inteira
    Image = None

COLUNAS = ["Case_ID", "Seed_Frame", "Frame_Number", "Label", "Notes", "Timestamp"]
ROTULOS = ("1", "0", "D")
MARCA = "_pf_v2_ferramenta.txt"
REGISTO = "_registo.csv"
NOTAS_JANELAS = "_notas_janelas.csv"
RE_MATCH = re.compile(r"match_at_videoframe(\d+)", re.I)
RE_IMAGEM_N = re.compile(r"^imagem(\d+)_", re.I)
RE_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def agora():
    return datetime.datetime.now().isoformat(timespec="seconds")


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


def ler_lista(caminho):
    p = Path(caminho)
    if p.suffix.lower() == ".json":
        d = json.loads(p.read_text(encoding="utf-8"))
        linhas = d["janelas"] if isinstance(d, dict) else d
    else:
        with open(p, encoding="utf-8", newline="") as fh:
            linhas = list(csv.DictReader(fh))
        if linhas and "ordem" in linhas[0]:
            linhas.sort(key=lambda r: int(r["ordem"]))
    chaves = []
    for r in linhas:
        k = (str(r["split"]), str(r["caso"]).zfill(4), str(int(float(r["seed"]))))
        if k not in chaves:
            chaves.append(k)
    return chaves


def ler_tabela(caminho):
    if not Path(caminho).is_file():
        print(f"Aviso: não encontrei {caminho}; a descrição clínica não vai aparecer.")
        return {}
    with open(caminho, encoding="utf-8", newline="") as fh:
        return {(r["split"], r["caso"], r["seed"]): r for r in csv.DictReader(fh)}


def preparar_saida(saida, criar=True):
    marca = saida / MARCA
    if saida.exists():
        alheios = [f.name for f in saida.glob("*.csv") if not f.name.startswith("_")]
        if alheios and not marca.exists():
            sys.exit(f"A pasta {saida} já tem CSV que não foram criados por esta ferramenta (ex.: {alheios[0]}).\n"
                     "Não a uso, para nunca abrir anotações antigas. Escolhe outra pasta com --saida.")
    elif not criar:
        sys.exit(f"Não existe a pasta {saida}: ainda não há nada anotado.")
    saida.mkdir(parents=True, exist_ok=True)
    if not marca.exists():
        marca.write_text("Pasta criada por tools/anotar.py (anotação PF v2). Não pôr aqui outros CSV.\n",
                         encoding="utf-8")
    return saida


@lru_cache(maxsize=2048)
def miniatura(caminho, lado=160):
    with Image.open(caminho) as im:
        im = im.convert("RGB")
        im.thumbnail((lado, lado))
        b = io.BytesIO()
        im.save(b, "JPEG", quality=82)
        return b.getvalue()


class Janela:
    def __init__(self, i, ds, saida, split, caso, seed, info):
        self.i, self.split, self.caso, self.seed = i, split, caso, seed
        self.seed_n = int(seed)
        self.pasta = Path(ds) / split / caso / "GT_com_classes" / seed
        self.problemas, self.avisos = [], []
        rx = re.compile(rf"^{re.escape(caso)}_(\d+)_{re.escape(seed)}\.jpe?g$", re.I)
        self.ficheiros = {}
        if self.pasta.is_dir():
            for f in self.pasta.iterdir():
                m = rx.match(f.name)
                if m:
                    self.ficheiros[int(m.group(1))] = f
        self.frames = sorted(self.ficheiros)
        if not self.frames:
            self.problemas.append("não encontrei frames nesta janela")
        elif self.seed_n not in self.ficheiros:
            self.problemas.append("não existe o frame do seed")
        gt = self.pasta.parent
        rel = []
        if gt.is_dir():
            for f in gt.iterdir():
                m = RE_MATCH.search(f.name)
                if f.is_file() and m and int(m.group(1)) == self.seed_n and f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    n = RE_IMAGEM_N.match(f.name)
                    rel.append((int(n.group(1)) if n else 0, f.name, f))
        self.relatorios = [f for _, _, f in sorted(rel)]
        if not self.relatorios:
            self.avisos.append("sem imagem do relatório")
        self.caixa = self._caixa_seed()
        self.descricao = info.get("descricao", "")
        self.categoria = info.get("categoria", "")
        self.f_rot = Path(saida) / f"{split}__{caso}__{seed}.csv"
        self.rot = {}                                          # frame -> [rótulo, nota, timestamp]
        if self.f_rot.exists():
            with open(self.f_rot, encoding="utf-8", newline="") as fh:
                for r in csv.DictReader(fh):
                    self.rot[int(r["Frame_Number"])] = [r["Label"], r.get("Notes", ""), r.get("Timestamp", "")]

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
        if Image is not None and self.seed_n in self.ficheiros:
            try:
                with Image.open(self.ficheiros[self.seed_n]) as im:
                    w, h = im.size
            except OSError:
                pass
        if not w:
            self.avisos.append("não consegui ler o tamanho da imagem do seed")
            return None
        x1, x2 = sorted((v[0], v[2]))
        y1, y2 = sorted((v[1], v[3]))
        return [x1 / w, y1 / h, x2 / w, y2 / h]

    def n_marcados(self):
        return sum(1 for f in self.frames if self.rot.get(f, [""])[0] in ROTULOS)

    def resumo(self):
        n = self.n_marcados()
        return {"i": self.i, "split": self.split, "caso": self.caso, "seed": self.seed_n,
                "n_frames": len(self.frames), "n_marcados": n, "completa": n == len(self.frames)}


class Estado:
    def __init__(self, ds, saida, chaves, tabela):
        self.lock = threading.Lock()
        self.saida = Path(saida)
        self.janelas = []
        self.removidas = []
        for k in chaves:
            j = Janela(len(self.janelas), ds, saida, *k, tabela.get(k, {}))
            if j.problemas:
                self.removidas.append((k, j.problemas))
            else:
                self.janelas.append(j)
        self.notas_jan = {}
        f = self.saida / NOTAS_JANELAS
        if f.exists():
            with open(f, encoding="utf-8", newline="") as fh:
                for r in csv.DictReader(fh):
                    self.notas_jan[(r["split"], r["caso"], r["seed"])] = [r["nota"], r["timestamp"]]

    def _gravar(self, j):
        linhas = [[j.caso, j.seed, fr, lab, nota, ts] for fr, (lab, nota, ts) in sorted(j.rot.items()) if lab or nota]
        gravar_csv(j.f_rot, COLUNAS, linhas)

    def _registar(self, j, frame, antes, depois):
        f = self.saida / REGISTO
        novo = not f.exists()
        with open(f, "a", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            if novo:
                w.writerow(["timestamp", "split", "caso", "seed", "frame", "antes", "depois"])
            w.writerow([agora(), j.split, j.caso, j.seed, frame, antes, depois])

    def estado(self):
        lista = [j.resumo() for j in self.janelas]
        atual = next((r["i"] for r in lista if not r["completa"]), 0)
        return {"janelas": lista, "atual": atual}

    def janela(self, i):
        j = self.janelas[i]
        with self.lock:
            if j.rot.get(j.seed_n, [""])[0] != "1":            # o seed é sempre 1
                antes = j.rot.get(j.seed_n, ["", "", ""])
                j.rot[j.seed_n] = ["1", antes[1] or "seed", agora()]
                self._registar(j, j.seed_n, antes[0], "1")
                self._gravar(j)
        falta = [f for f in j.frames if j.rot.get(f, [""])[0] not in ROTULOS]
        comecada = any(v[0] in ROTULOS for f, v in j.rot.items() if f != j.seed_n)   # janela nova abre no seed
        d = j.resumo()
        d.update({"frames": j.frames, "caixa": j.caixa, "relatorios": list(range(len(j.relatorios))),
                  "descricao": j.descricao, "categoria": j.categoria,
                  "rotulos": {str(f): v[0] for f, v in j.rot.items() if v[0] in ROTULOS},
                  "notas": {str(f): v[1] for f, v in j.rot.items() if v[1] and f != j.seed_n},
                  "nota_janela": self.notas_jan.get((j.split, j.caso, j.seed), [""])[0],
                  "retomar": min(falta, key=lambda f: (abs(f - j.seed_n), f)) if falta and comecada else None, "avisos": j.avisos})
        return d

    def marcar(self, i, frames, rotulo, so_vazios=False):
        j = self.janelas[i]
        if rotulo not in ROTULOS and rotulo != "":
            raise ValueError(f"rótulo inválido: {rotulo!r}")
        frames = [int(f) for f in frames]
        fora = [f for f in frames if f not in j.ficheiros]
        if fora:
            raise ValueError(f"frames que não existem nesta janela: {fora[:5]}")
        with self.lock:
            mudou = False
            for f in frames:
                if f == j.seed_n:
                    continue
                antes = j.rot.get(f, ["", "", ""])
                if (so_vazios and antes[0] in ROTULOS) or antes[0] == rotulo:
                    continue
                j.rot[f] = [rotulo, antes[1], agora()]
                self._registar(j, f, antes[0], rotulo)
                mudou = True
            if mudou:
                self._gravar(j)
        return j.resumo()

    def nota(self, i, frame, texto):
        j = self.janelas[i]
        frame = int(frame)
        if frame not in j.ficheiros:
            raise ValueError("frame inexistente")
        with self.lock:
            antes = j.rot.get(frame, ["", "", ""])
            j.rot[frame] = [antes[0], " ".join(str(texto).split()), antes[2] or agora()]
            self._gravar(j)
        return {"ok": True}

    def nota_janela(self, i, texto):
        j = self.janelas[i]
        with self.lock:
            self.notas_jan[(j.split, j.caso, j.seed)] = [" ".join(str(texto).split()), agora()]
            gravar_csv(self.saida / NOTAS_JANELAS, ["split", "caso", "seed", "nota", "timestamp"],
                       [[*k, n, ts] for k, (n, ts) in sorted(self.notas_jan.items()) if n])
        return {"ok": True}


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
                return self._json(self.est.janela(int(q["i"])))
            if u.path in ("/img", "/rel"):
                j = self.est.janelas[int(q["i"])]
                if q.get("w", f"{j.caso}_{j.seed}") != f"{j.caso}_{j.seed}":
                    raise KeyError("a janela não corresponde")       # nunca servir a imagem de outra janela
            if u.path == "/img":
                f = j.ficheiros[int(q["f"])]
                if q.get("t") and Image is not None:
                    return self._enviar(miniatura(str(f)), "image/jpeg", cache=True)
                return self._enviar(f.read_bytes(), "image/jpeg", cache=True)
            if u.path == "/rel":
                f = j.relatorios[int(q["k"])]
                tipo = "image/png" if f.suffix.lower() == ".png" else "image/jpeg"
                return self._enviar(f.read_bytes(), tipo, cache=True)
            self._json({"erro": "não existe"}, 404)
        except (KeyError, ValueError, IndexError) as e:
            self._json({"erro": f"pedido inválido ({type(e).__name__}: {e})"}, 400)
        except OSError as e:
            self._json({"erro": f"não consegui ler o ficheiro ({e})"}, 500)

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            d = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/api/marcar":
                return self._json(self.est.marcar(int(d["i"]), d["frames"], d["rotulo"], bool(d.get("so_vazios"))))
            if self.path == "/api/nota":
                return self._json(self.est.nota(int(d["i"]), d["frame"], d.get("nota", "")))
            if self.path == "/api/nota_janela":
                return self._json(self.est.nota_janela(int(d["i"]), d.get("nota", "")))
            self._json({"erro": "não existe"}, 404)
        except (KeyError, ValueError, IndexError, json.JSONDecodeError) as e:
            self._json({"erro": f"pedido inválido ({type(e).__name__}: {e})"}, 400)
        except OSError as e:
            self._json({"erro": f"não consegui gravar ({e})"}, 500)


def consolidar(ds, saida, chaves, destino):
    """Junta todas as janelas anotadas num só ficheiro, só se estiverem todas completas."""
    if not chaves:
        chaves = []
        for f in sorted(saida.glob("*.csv")):
            p = f.stem.split("__")
            if not f.name.startswith("_") and len(p) == 3:
                chaves.append(tuple(p))
    if not chaves:
        sys.exit(f"Não há janelas anotadas em {saida}.")
    linhas, problemas, cont = [], [], {"1": 0, "0": 0, "D": 0}
    nome = {"1": "1", "0": "0", "D": "duvidoso"}
    for k in chaves:
        j = Janela(0, ds, saida, *k, {})
        if j.problemas:
            problemas.append(f"{'/'.join(k)}: {'; '.join(j.problemas)}")
            continue
        if not j.f_rot.exists():
            problemas.append(f"{'/'.join(k)}: ainda não foi anotada")
            continue
        faltam = [f for f in j.frames if j.rot.get(f, [""])[0] not in ROTULOS]
        extra = [f for f, v in j.rot.items() if f not in j.ficheiros and v[0]]
        if faltam:
            problemas.append(f"{'/'.join(k)}: {len(faltam)} frames sem rótulo (ex.: {faltam[:5]})")
        if extra:
            problemas.append(f"{'/'.join(k)}: rótulos em frames que não existem na janela ({extra[:5]})")
        if j.rot.get(j.seed_n, [""])[0] != "1":
            problemas.append(f"{'/'.join(k)}: o seed não está marcado 1")
        for f in j.frames:
            lab, nota, _ = j.rot.get(f, ["", "", ""])
            if lab in ROTULOS:
                cont[lab] += 1
                linhas.append([j.split, j.caso, j.seed, f, f - j.seed_n, nome[lab], nota])
    if problemas:
        print("Não gravei nada — há janelas por completar:")
        for p in problemas:
            print("  " + p)
        sys.exit(1)
    destino.parent.mkdir(parents=True, exist_ok=True)
    gravar_csv(destino, ["split", "caso", "seed", "frame", "offset", "visivel", "nota"], linhas)
    n_tr = sum(k[0].endswith("Treino") for k in chaves)
    tot = sum(cont.values())
    print(f"Consolidado: {len(chaves)} janelas (treino {n_tr} · teste {len(chaves) - n_tr}) · {tot} frames · "
          f"1: {cont['1']} · 0: {cont['0']} · duvidoso: {cont['D']} ({100 * cont['D'] / max(tot, 1):.1f} %)")
    print(f"Gravado em {destino}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="pasta Dataset_WCE (só leitura)")
    ap.add_argument("--janelas", help="lista de janelas: CSV (split, caso, seed[, ordem]) ou JSON")
    ap.add_argument("--tabela", default="data/derived/janelas_v1.csv", help="para a descrição clínica")
    ap.add_argument("--saida", default="data/derived/pf_v2")
    ap.add_argument("--porta", type=int, default=8765)
    ap.add_argument("--consolidar", action="store_true", help="juntar tudo em paulo_frame_labels.csv")
    ap.add_argument("--consolidado", default="data/derived/paulo_frame_labels.csv")
    args = ap.parse_args()
    if not Path(args.dataset).is_dir():
        sys.exit(f"Não encontrei a pasta do dataset: {args.dataset}")

    if args.consolidar:
        saida = preparar_saida(Path(args.saida), criar=False)
        return consolidar(args.dataset, saida, ler_lista(args.janelas) if args.janelas else [], Path(args.consolidado))

    if not args.janelas:
        sys.exit("Falta --janelas (ex.: docs/janelas_reanotar.csv).")
    saida = preparar_saida(Path(args.saida))
    est = Estado(args.dataset, saida, ler_lista(args.janelas), ler_tabela(args.tabela))
    for k, p in est.removidas:
        print(f"FORA: {'/'.join(k)} · {'; '.join(p)}")
    for j in est.janelas:
        for a in j.avisos:
            print(f"Aviso: {j.split}/{j.caso}/{j.seed} · {a}")
    if not est.janelas:
        sys.exit("Nenhuma janela para anotar.")
    feitas = sum(j.resumo()["completa"] for j in est.janelas)
    print(f"{len(est.janelas)} janelas · {sum(len(j.frames) for j in est.janelas)} frames · "
          f"{feitas} já completas · a gravar em {saida}/")
    if Image is None:
        print("Aviso: sem Pillow, as miniaturas são as imagens inteiras (mais lentas).")
    Pedidos.est = est
    try:
        srv = ThreadingHTTPServer(("127.0.0.1", args.porta), Pedidos)
    except OSError:
        sys.exit(f"A porta {args.porta} está ocupada (a ferramenta já está a correr noutra janela?). "
                 f"Usa --porta {args.porta + 1}.")
    print(f"Pronto: abre http://localhost:{args.porta} no browser. Ctrl+C para parar.", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nParado. Tudo o que foi marcado está gravado.")


PAGINA = r"""<!doctype html>
<html lang="pt-PT">
<head>
<meta charset="utf-8">
<title>Anotação PF v2</title>
<style>
:root{--bg:#14171c;--card:#1d2229;--card2:#262c35;--fg:#e7eaee;--mut:#9aa4b1;--line:#353d48;
  --v1:#3fb950;--v0:#6e7681;--vd:#f0883e;--sel:#388bfd;--seed:#e3b341;--erro:#ff6b6b}
*{box-sizing:border-box}
html,body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,Ubuntu,sans-serif}
header{position:sticky;top:0;z-index:10;display:flex;flex-wrap:wrap;gap:8px 12px;align-items:center;padding:8px 14px;background:var(--card);border-bottom:1px solid var(--line)}
button,select,input,textarea{font:inherit;color:var(--fg);background:var(--card2);border:1px solid var(--line);border-radius:6px;padding:5px 10px}
button{cursor:pointer}
button:hover:not(:disabled){border-color:var(--mut)}
button:disabled{opacity:.4;cursor:default}
#jsel{max-width:min(520px,60vw)}
.prog{flex:1;min-width:120px;height:8px;background:var(--card2);border-radius:4px;overflow:hidden}
.prog>div{height:100%;background:var(--v1);width:0;transition:width .2s}
#gravado{color:var(--mut);font-size:13px;min-width:130px;text-align:right}
#gravado.erro{color:var(--erro);font-weight:700}
main{display:grid;grid-template-columns:330px minmax(0,1fr);gap:14px;padding:14px}
.ref,.ver,section.grelha{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}
.ref{display:flex;flex-direction:column;gap:8px;align-self:start}
.ref h3{margin:4px 0 0;font-size:11.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--mut);font-weight:600}
.img-caixa{position:relative;line-height:0;overflow:hidden;border-radius:6px}
.img-caixa img{display:block;width:100%;border-radius:6px;background:#000}
.bbox{position:absolute;border:2px solid #00e5ff;box-shadow:0 0 0 1px #000c;pointer-events:none}
#rel img{width:100%;border-radius:6px;margin-bottom:4px}
.desc{font-size:15px;font-weight:600}
.mut{color:var(--mut)}
textarea{width:100%;min-height:50px;resize:vertical}
.ver{display:flex;flex-direction:column;align-items:center;gap:10px;min-width:0}
.info{display:flex;gap:14px;align-items:baseline;flex-wrap:wrap;justify-content:center}
.info .fr{font-size:20px;font-weight:700}
.info .off{font-size:20px;color:var(--mut)}
.badge{padding:3px 12px;border-radius:999px;font-weight:700;border:1px solid var(--line)}
.badge.r1{background:var(--v1);color:#04130a;border-color:var(--v1)}
.badge.r0{background:var(--v0);color:#fff;border-color:var(--v0)}
.badge.rD{background:var(--vd);color:#1a0d00;border-color:var(--vd)}
.palco{position:relative;line-height:0;overflow:hidden;border-radius:8px}
.palco img{display:block;width:min(60vh,56vw);height:auto;border-radius:8px;background:#000}
.botoes{display:flex;gap:8px;flex-wrap:wrap;justify-content:center}
.b1{border-color:var(--v1)!important}.b0{border-color:var(--v0)!important}.bD{border-color:var(--vd)!important}
.ajuda{color:var(--mut);font-size:12.5px;text-align:center;max-width:760px}
.aviso{min-height:20px;color:var(--seed);font-weight:600;text-align:center}
.completa{color:var(--v1);font-weight:700}
section.grelha{margin:0 14px 24px}
.gbar{display:flex;flex-wrap:wrap;gap:8px 12px;align-items:center;margin-bottom:10px}
#thumbs{display:grid;grid-template-columns:repeat(auto-fill,minmax(var(--tw,104px),1fr));gap:6px}
.th{position:relative;border:3px dashed var(--line);border-radius:6px;cursor:pointer;line-height:0;overflow:hidden;background:#000;user-select:none}
.th img{width:100%;aspect-ratio:1/1;object-fit:cover;display:block}
.th span{position:absolute;left:3px;top:3px;line-height:1;font-size:11px;font-weight:700;padding:2px 4px;border-radius:4px;background:#000b;color:#fff}
.th.r1{border:3px solid var(--v1)}.th.r0{border:3px solid var(--v0)}.th.rD{border:3px solid var(--vd)}
.th.r0 img{opacity:.5}
.th.cur{outline:3px solid #fff;outline-offset:1px}
.th.sel::after{content:"";position:absolute;inset:0;background:#388bfd66}
.th.seed span{background:var(--seed);color:#000}
@media (max-width:900px){main{grid-template-columns:1fr}}
</style>
</head>
<body>
<header>
  <button id="jprev" title="janela anterior">◀</button>
  <select id="jsel" title="janela"></select>
  <button id="jnext" title="janela seguinte">▶</button>
  <span id="cont"></span>
  <div class="prog"><div id="progbar"></div></div>
  <span id="gravado">—</span>
</header>
<main>
  <aside class="ref">
    <h3>Seed com a caixa do médico</h3>
    <div class="img-caixa"><img id="seedimg" alt=""><div class="bbox" id="seedbox" hidden></div></div>
    <div class="mut" id="seedtxt"></div>
    <h3>Imagem do relatório</h3>
    <div id="rel"></div>
    <h3>Descrição clínica</h3>
    <div class="desc" id="desc"></div>
    <div class="mut" id="cat"></div>
    <h3>Nota da janela</h3>
    <textarea id="notajan" placeholder="opcional (grava ao sair da caixa)"></textarea>
  </aside>
  <section class="ver">
    <div class="info"><span class="fr" id="fr"></span><span class="off" id="off"></span>
      <span class="badge" id="badge">sem rótulo</span><span class="mut" id="dirtxt"></span></div>
    <div class="palco"><img id="atual" alt=""><div class="bbox" id="atualbox" hidden></div></div>
    <div class="aviso" id="aviso"></div>
    <div class="botoes">
      <button class="b1" data-r="1">1 · visível</button>
      <button class="b0" data-r="0">0 · não visível</button>
      <button class="bD" data-r="D">D · dúvida</button>
      <button id="bprev">← anterior</button>
      <button id="bnext">seguinte →</button>
      <button id="bseed">S · seed</button>
      <button id="bvazio">U · próximo sem rótulo</button>
    </div>
    <input id="notafr" placeholder="nota deste frame (opcional; Enter grava)" style="width:min(560px,100%)">
    <div class="ajuda">1 / 0 / D marcam e avançam na direção atual · ← → andam e mudam a direção ·
      Shift+← → saltam 10 · S vai ao seed · U vai ao próximo sem rótulo · Del apaga o rótulo</div>
  </section>
</main>
<section class="grelha">
  <div class="gbar">
    <strong>Janela inteira</strong>
    <span class="mut">clica num frame para ir lá · clica num e Shift+clica noutro para selecionar um intervalo</span>
    <span id="seltxt"></span>
    <button id="sel0" disabled>0 nos não marcados da seleção</button>
    <button id="selD" disabled>D nos não marcados da seleção</button>
    <button id="sellimpa" disabled>limpar seleção</button>
    <span style="flex:1"></span>
    <label class="mut">miniaturas <select id="tam"><option value="72">pequenas</option>
      <option value="104" selected>médias</option><option value="150">grandes</option></select></label>
  </div>
  <div id="thumbs"></div>
</section>
<script>
const $ = id => document.getElementById(id);
const NOME = {"1": "1 · visível", "0": "0 · não visível", "D": "D · dúvida"};
let E = null, J = null, pos = 0, dir = -1, ancora = null, sel = null;
let fila = Promise.resolve(), pendentes = 0;

const rot = f => (J.rotulos[f] || "");
const offTxt = f => { const o = f - J.seed; return o === 0 ? "seed" : (o > 0 ? "+" + o : "−" + (-o)); };
const W = () => `i=${J.i}&w=${J.caso}_${J.seed}`;
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));

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
  return p.then(d => { pendentes--; mostrarGravacao(); return d; },
                e => { pendentes--; falhou(e); });
}
function mostrarGravacao() {
  const g = $("gravado");
  g.classList.remove("erro");
  g.textContent = pendentes ? "a gravar…" : "gravado ✓ " + new Date().toLocaleTimeString("pt-PT");
}
function falhou(e) {
  const g = $("gravado");
  g.classList.add("erro"); g.textContent = "ERRO ao gravar";
  alert("Não consegui gravar: " + e.message + "\n\nA janela vai ser recarregada a partir do que está gravado no disco.");
  abrirJanela(J.i, J.frames[pos]).catch(x => alert(x.message));
}
function aviso(t) {
  const a = $("aviso"); a.textContent = t;
  clearTimeout(aviso.tm); aviso.tm = setTimeout(() => a.textContent = "", 2600);
}
function caixa(el, c) {
  if (!c) { el.hidden = true; return; }
  el.hidden = false;
  el.style.left = (100 * c[0]) + "%"; el.style.top = (100 * c[1]) + "%";
  el.style.width = (100 * (c[2] - c[0])) + "%"; el.style.height = (100 * (c[3] - c[1])) + "%";
}

async function iniciar() {
  E = await get("/api/estado");
  const s = $("jsel"); s.innerHTML = "";
  E.janelas.forEach(j => { const o = document.createElement("option"); o.value = j.i; s.appendChild(o); });
  listaJanelas();
  await abrirJanela(E.atual);
}
function listaJanelas() {
  E.janelas.forEach(j => {
    $("jsel").options[j.i].textContent = `${j.i + 1}/${E.janelas.length} · ${j.split.replace("Dataset_", "")} · caso ${j.caso} · seed ${j.seed} · ${j.n_marcados}/${j.n_frames}` + (j.completa ? " ✓" : "");
  });
}
async function abrirJanela(i, frame) {
  J = await get("/api/janela?i=" + i);
  $("jsel").value = i;
  $("jprev").disabled = i === 0; $("jnext").disabled = i === E.janelas.length - 1;
  ancora = null; sel = null;
  const alvo = frame != null ? frame : (J.retomar != null ? J.retomar : J.seed);
  pos = J.frames.indexOf(alvo); if (pos < 0) pos = J.frames.indexOf(J.seed);
  dir = J.frames[pos] > J.seed ? 1 : -1;
  $("seedimg").src = `/img?${W()}&f=${J.seed}`;
  caixa($("seedbox"), J.caixa);
  $("seedtxt").textContent = `frame ${J.seed} · janela ${J.frames[0]}–${J.frames[J.frames.length - 1]} (${J.frames.length} frames)` + (J.caixa ? "" : " · SEM CAIXA");
  $("rel").innerHTML = J.relatorios.length ? J.relatorios.map(k => `<img src="/rel?${W()}&k=${k}" alt="">`).join("") : '<span class="mut">(sem imagem do relatório)</span>';
  $("desc").innerHTML = J.descricao ? J.descricao.split(" | ").map(d => esc(d.replace(/_/g, " "))).join("<br>") : '<span class="mut">(sem descrição)</span>';
  $("cat").textContent = J.categoria ? "categoria (do texto): " + J.categoria.replace(/_/g, " ") : "";
  $("notajan").value = J.nota_janela || "";
  if (J.avisos.length) aviso("Atenção: " + J.avisos.join("; "));
  grelha();
  mostrar();
}
function mostrar() {
  const f = J.frames[pos], r = rot(f);
  $("atual").src = `/img?${W()}&f=${f}`;
  $("fr").textContent = "frame " + f;
  $("off").textContent = f === J.seed ? "(seed)" : "offset " + offTxt(f);
  const b = $("badge"); b.className = "badge" + (r ? " r" + r : ""); b.textContent = r ? NOME[r] : "sem rótulo";
  caixa($("atualbox"), f === J.seed ? J.caixa : null);
  $("dirtxt").textContent = dir < 0 ? "◀ a andar para trás" : "a andar para a frente ▶";
  $("notafr").value = J.notas[f] || "";
  $("bprev").disabled = pos === 0; $("bnext").disabled = pos === J.frames.length - 1;
  contagens();
  const ts = $("thumbs").children;
  for (const t of document.querySelectorAll(".th.cur")) t.classList.remove("cur");
  if (ts[pos]) ts[pos].classList.add("cur");
  for (const d of [1, 2, 3, -1, -2, -3]) {
    const k = pos + d;
    if (k >= 0 && k < J.frames.length) new Image().src = `/img?${W()}&f=${J.frames[k]}`;
  }
}
function contagens() {
  let n1 = 0, n0 = 0, nD = 0;
  for (const f of J.frames) { const r = rot(f); if (r === "1") n1++; else if (r === "0") n0++; else if (r === "D") nD++; }
  const n = n1 + n0 + nD, N = J.frames.length, completa = n === N;
  $("cont").innerHTML = `<b>${n}/${N}</b> marcados · 1: ${n1} · 0: ${n0} · D: ${nD}` +
    (completa ? ' · <span class="completa">janela completa ✓</span>' : ` · faltam ${N - n}`);
  $("progbar").style.width = (100 * n / N) + "%";
  const e = E.janelas[J.i];
  if (completa && !e.completa) aviso("Janela completa ✓ — podes passar à seguinte (▶ lá em cima).");
  if (e.n_marcados !== n || e.completa !== completa) { e.n_marcados = n; e.completa = completa; listaJanelas(); }
}
function grelha() {
  const g = $("thumbs"); g.innerHTML = "";
  const frag = document.createDocumentFragment();
  J.frames.forEach((f, k) => {
    const d = document.createElement("div");
    d.className = "th" + (f === J.seed ? " seed" : ""); d.dataset.k = k;
    d.innerHTML = `<img loading="lazy" src="/img?${W()}&f=${f}&t=1" alt=""><span>${f === J.seed ? "★ seed" : offTxt(f)}</span>`;
    frag.appendChild(d);
  });
  g.appendChild(frag);
  J.frames.forEach((f, k) => pintar(k));
  selecao();
}
function pintar(k) {
  const t = $("thumbs").children[k]; if (!t) return;
  const r = rot(J.frames[k]);
  t.classList.remove("r1", "r0", "rD"); if (r) t.classList.add("r" + r);
}
function selecao() {
  const ts = $("thumbs").children;
  for (let k = 0; k < ts.length; k++) ts[k].classList.toggle("sel", !!sel && k >= sel[0] && k <= sel[1]);
  const livres = sel ? J.frames.slice(sel[0], sel[1] + 1).filter(f => !rot(f) && f !== J.seed).length : 0;
  $("seltxt").textContent = sel ? `seleção ${offTxt(J.frames[sel[0]])} a ${offTxt(J.frames[sel[1]])}: ${sel[1] - sel[0] + 1} frames, ${livres} sem rótulo` : "";
  $("sel0").disabled = $("selD").disabled = !livres;
  $("sellimpa").disabled = !sel;
}
function marcar(r) {
  const f = J.frames[pos];
  if (f === J.seed) { aviso("O seed fica sempre 1 (é a referência)."); andar(dir); return; }
  J.rotulos[f] = r; pintar(pos);
  post("/api/marcar", {i: J.i, frames: [f], rotulo: r});
  andar(dir);
}
function apagar() {
  const f = J.frames[pos];
  if (f === J.seed) { aviso("O seed não se apaga."); return; }
  if (!rot(f)) return;
  delete J.rotulos[f]; pintar(pos); mostrar();
  post("/api/marcar", {i: J.i, frames: [f], rotulo: ""});
}
function andar(d) {
  const novo = Math.min(J.frames.length - 1, Math.max(0, pos + d));
  if (novo === pos) aviso(d < 0 ? "Início da janela." : "Fim da janela.");
  pos = novo; mostrar();
}
function proximoVazio() {
  const N = J.frames.length;
  for (let s = 1; s <= N; s++) {
    const k = ((pos + s * dir) % N + N) % N;
    if (!rot(J.frames[k])) { pos = k; mostrar(); return; }
  }
  aviso("Todos os frames têm rótulo ✓");
}
function marcarSelecao(r) {
  if (!sel) return;
  const fs = J.frames.slice(sel[0], sel[1] + 1).filter(f => !rot(f) && f !== J.seed);
  if (!fs.length || !confirm(`Marcar ${fs.length} frames sem rótulo como ${r}?`)) return;
  fs.forEach(f => J.rotulos[f] = r);
  for (let k = sel[0]; k <= sel[1]; k++) pintar(k);
  post("/api/marcar", {i: J.i, frames: fs, rotulo: r, so_vazios: true});
  sel = null; ancora = null; selecao(); mostrar();
}
function gravarNota() {
  const f = J.frames[pos], t = $("notafr").value.trim();
  if ((J.notas[f] || "") === t) return;
  if (t) J.notas[f] = t; else delete J.notas[f];
  post("/api/nota", {i: J.i, frame: f, nota: t});
}
function irPara(i) { fila.then(() => abrirJanela(i)).catch(e => alert(e.message)); }

$("thumbs").addEventListener("click", ev => {
  const t = ev.target.closest(".th"); if (!t) return;
  const k = +t.dataset.k;
  if (ev.shiftKey && ancora !== null) { sel = [Math.min(ancora, k), Math.max(ancora, k)]; }
  else { ancora = k; sel = null; pos = k; mostrar(); }
  selecao();
});
document.querySelectorAll("button[data-r]").forEach(b => b.addEventListener("click", () => { b.blur(); marcar(b.dataset.r); }));
$("bprev").onclick = () => { $("bprev").blur(); dir = -1; andar(-1); };
$("bnext").onclick = () => { $("bnext").blur(); dir = 1; andar(1); };
$("bseed").onclick = () => { $("bseed").blur(); pos = J.frames.indexOf(J.seed); mostrar(); };
$("bvazio").onclick = () => { $("bvazio").blur(); proximoVazio(); };
$("sel0").onclick = () => { $("sel0").blur(); marcarSelecao("0"); };
$("selD").onclick = () => { $("selD").blur(); marcarSelecao("D"); };
$("sellimpa").onclick = () => { $("sellimpa").blur(); sel = null; ancora = null; selecao(); };
$("jsel").addEventListener("change", () => { $("jsel").blur(); irPara(+$("jsel").value); });
$("jprev").onclick = () => { $("jprev").blur(); if (J.i > 0) irPara(J.i - 1); };
$("jnext").onclick = () => { $("jnext").blur(); if (J.i < E.janelas.length - 1) irPara(J.i + 1); };
$("tam").addEventListener("change", () => { $("thumbs").style.setProperty("--tw", $("tam").value + "px"); $("tam").blur(); });
$("notafr").addEventListener("keydown", ev => { if (ev.key === "Enter" || ev.key === "Escape") { ev.preventDefault(); $("notafr").blur(); } });
$("notafr").addEventListener("change", gravarNota);
$("notajan").addEventListener("change", () => post("/api/nota_janela", {i: J.i, nota: $("notajan").value.trim()}));
document.addEventListener("keydown", ev => {
  const a = document.activeElement;
  if (a && (a.tagName === "INPUT" || a.tagName === "TEXTAREA")) { if (ev.key === "Escape") a.blur(); return; }
  if (a && a.tagName === "SELECT") a.blur();
  if (ev.ctrlKey || ev.metaKey || ev.altKey || !J) return;
  const k = ev.key;
  if (k === "1" || k === "0") marcar(k);
  else if (k === "d" || k === "D") marcar("D");
  else if (k === "ArrowLeft") { dir = -1; andar(ev.shiftKey ? -10 : -1); }
  else if (k === "ArrowRight") { dir = 1; andar(ev.shiftKey ? 10 : 1); }
  else if (k === "s" || k === "S" || k === "Home") { pos = J.frames.indexOf(J.seed); mostrar(); }
  else if (k === "u" || k === "U") proximoVazio();
  else if (k === "Delete" || k === "Backspace") apagar();
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
    main()
