"""
Testes da ferramenta de revisão do Guião C (tools/rever.py). Correm de duas maneiras, a partir da pasta do projeto:
    python tests/test_rever.py          (não precisa de pytest)
    python -m pytest tests/test_rever.py -q
Usam um dataset inventado (imagens geradas na hora, com o formato de pastas real); nunca leem dados reais.
Versão 1.0 (24 out 2026).
"""
import csv
import importlib.util
import json
import os
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("rever", RAIZ / "tools" / "rever.py")
rv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rv)

# (split, caso, seed, frames das caixas a rever)
JANELAS = [("Dataset_Teste", "0201", 600, (597, 603, 605)), ("Dataset_Treino", "0101", 500, (499,)),
           ("Dataset_Teste", "0202", 900, (893, 902))]
CAIXA = (0.30, 0.30, 0.50, 0.50)


class Saltar(Exception):
    pass


def saltar(motivo):
    if "pytest" in sys.modules:
        import pytest
        pytest.skip(motivo)
    raise Saltar(motivo)


def erro(fn, *a, contem=None, tipo=None, **k):
    tipo = tipo or rv.ErroRever
    try:
        fn(*a, **k)
    except tipo as e:
        assert contem is None or contem in str(e), f"a mensagem não diz '{contem}': {e}"
        return str(e)
    raise AssertionError("devia ter dado erro")


def _projeto(raiz, extra_col=None):
    """Dataset com 3 janelas (frames seed−15…seed+15 em JPEG 64×64), caixa do seed e de um vizinho, imagem do
    relatório e janelas_v1.csv; a lista com 6 caixas."""
    if rv.Image is None:
        saltar("Pillow não instalado")
    raiz = Path(raiz)
    ds = raiz / "ds"
    for sp, caso, seed, _ in JANELAS:
        pasta = ds / sp / caso / "GT_com_classes" / str(seed)
        (pasta / f"Findings_{seed}").mkdir(parents=True)
        for f in range(seed - 15, seed + 16):
            rv.Image.new("RGB", (64, 64), ((f * 7) % 255, 80, 120)).save(pasta / f"{caso}_{f}_{seed}.jpg")
            (pasta / f"Findings_{seed}" / f"{caso}_{f}_{seed}.txt").write_text("")
        # caixa do seed (cantos pela ordem de desenho) e de um vizinho, que a ferramenta nunca pode ler
        (pasta / f"Findings_{seed}" / f"{caso}_{seed}_{seed}.txt").write_text("32 32 16 16")
        (pasta / f"Findings_{seed}" / f"{caso}_{seed + 1}_{seed}.txt").write_text("0 0 60 60")
        rv.Image.new("RGB", (40, 30), (200, 200, 200)).save(pasta.parent / f"imagem1_1match_at_videoframe{seed}_x.jpg")
    with open(raiz / "janelas.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["split", "caso", "seed", "categoria", "descricao"])
        for sp, caso, seed, _ in JANELAS:
            w.writerow([sp, caso, seed, "erosao", "Erosao_no_jejuno"])
    lista = raiz / "itens_v1.csv"
    with open(lista, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(rv.COL_LISTA + ([extra_col] if extra_col else []))
        n = 0
        for sp, caso, seed, fs in JANELAS:
            for f in fs:
                n += 1
                w.writerow([n, sp, caso, seed, f, *CAIXA] + (["0.9"] if extra_col else []))
    return raiz, ds, lista


def _estado(raiz, ds, lista, revisor="paulo", ajustar=False):
    itens = rv.ler_lista(lista)
    pasta = rv.preparar_saida(raiz / "revisao", revisor, str(lista), rv.sha256(lista))
    return rv.Estado(str(ds), itens, pasta, rv.ler_tabela(raiz / "janelas.csv"), revisor, ajustar)


def _linhas(f):
    with open(f, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


# ------------------------------------------------------------------ às cegas
def test_lista_tem_de_ser_cega_e_bem_formada():
    with tempfile.TemporaryDirectory() as tmp:
        raiz, ds, lista = _projeto(Path(tmp) / "a", extra_col="score")
        erro(rv.ler_lista, lista, contem="colunas a mais (score)")
        raiz, ds, lista = _projeto(Path(tmp) / "b")
        cab = lista.read_text().splitlines()[0]
        lista.write_text(cab + "\n1,Dataset_Teste,0201,600,597,0.3,0.3,0.5,0.5\n2,Dataset_Treino,0101,500,499,0.3,0.3,0.5,0.5"
                         "\n3,Dataset_Teste,0201,600,603,0.3,0.3,0.5,0.5\n")
        erro(rv.ler_lista, lista, contem="não estão seguidas")
        lista.write_text(cab + "\n1,Dataset_Teste,0201,600,597,0.5,0.3,0.4,0.5\n")
        erro(rv.ler_lista, lista, contem="caixa inválida")
        lista.write_text(cab + "\n1,Dataset_Teste,0201,600,597,0.3,0.3,0.5,0.5\n1,Dataset_Teste,0201,600,603,0.3,0.3,0.5,0.5\n")
        erro(rv.ler_lista, lista, contem="repetidos")


def test_api_nunca_mostra_score_nem_caixas_dos_vizinhos():
    with tempfile.TemporaryDirectory() as tmp:
        raiz, ds, lista = _projeto(tmp)
        est = _estado(raiz, ds, lista)
        txt = json.dumps([est.estado(), est.janela(0)])
        for proibido in ("score", "nivel", "caixa_original", "tipo"):
            assert proibido not in txt
        j = est.janela(0)
        assert [round(x, 3) for x in j["caixa"]] == [0.25, 0.25, 0.5, 0.5]       # a do seed, com os cantos ordenados
        assert [x["frame"] for x in j["itens"]] == [597, 603, 605] and j["itens"][0]["offset"] == -3
        assert j["descricao"] == "Erosao_no_jejuno" and j["relatorios"] == [0] and not j["avisos"]


# ------------------------------------------------------------------ respostas, desfazer e retoma
def test_responder_desfazer_e_retomar():
    with tempfile.TemporaryDirectory() as tmp:
        raiz, ds, lista = _projeto(tmp)
        est = _estado(raiz, ds, lista)
        assert est.estado()["atual"] == 0 and est.estado()["respondidos"] == 0
        est.decidir(1, "aceitar", "", dur_ms=1500)
        est.decidir(2, "rejeitar", "outra lesão", dur_ms=900)
        est.decidir(3, "mal_posta", "")
        dec = _linhas(raiz / "revisao/paulo/decisoes.csv")
        assert [(d["item"], d["decisao"], d["nota"]) for d in dec] == [("1", "aceitar", ""), ("2", "rejeitar", "outra lesão"),
                                                                      ("3", "mal_posta", "")]
        assert dec[0]["dur_ms"] == "1500" and dec[0]["modo"] == "marcar" and dec[0]["revisor"] == "paulo"
        assert list(dec[0]) == rv.COL_DEC
        est.apagar(3)                                                      # Z: desfaz
        assert [d["item"] for d in _linhas(raiz / "revisao/paulo/decisoes.csv")] == ["1", "2"]
        reg = _linhas(raiz / "revisao/paulo/_registo.csv")
        assert [(r["item"], r["antes"], r["depois"]) for r in reg][-1] == ("3", "mal_posta", "")
        # mudar de ideias fica registado; a retoma abre na primeira sem resposta
        est.decidir(1, "mal_posta", "")
        novo = _estado(raiz, ds, lista)
        e = novo.estado()
        assert e["respondidos"] == 2 and e["atual"] == 2 and e["itens"][0] == [1, 0, 1]
        assert novo.janela(0)["itens"][0]["decisao"] == "mal_posta"


def test_validacoes_das_respostas():
    with tempfile.TemporaryDirectory() as tmp:
        raiz, ds, lista = _projeto(tmp)
        est = _estado(raiz, ds, lista)
        erro(est.decidir, 1, "talvez", tipo=ValueError, contem="resposta inválida")
        erro(est.decidir, 99, "aceitar", tipo=ValueError, contem="não está na lista")
        erro(est.decidir, 1, "mal_posta", box_aj=[0.1, 0.1, 0.2, 0.2], tipo=ValueError, contem="modo ajustar")
        aj = _estado(raiz, ds, lista, revisor="paulo_aj", ajustar=True)
        erro(aj.decidir, 1, "aceitar", box_aj=[0.1, 0.1, 0.2, 0.2], tipo=ValueError, contem="só uma caixa mal posta")
        erro(aj.decidir, 1, "mal_posta", box_aj=[0.3, 0.1, 0.2, 0.2], tipo=ValueError, contem="inválida")
        aj.decidir(1, "mal_posta", "", box_aj=[0.1, 0.12, 0.25, 0.3])
        d = _linhas(raiz / "revisao/paulo_aj/decisoes.csv")[0]
        assert (d["box_aj_x1"], d["box_aj_y2"], d["modo"]) == ("0.1", "0.3", "ajustar")


def test_notas_rapidas_e_botao_do_seed():
    assert rv.nota_com("outra lesão · vi sangue") == "vi sangue"
    assert rv.nota_com("", "dúvida") == "dúvida" and rv.nota_com("vi sangue", "dúvida") == "dúvida · vi sangue"
    with tempfile.TemporaryDirectory() as tmp:
        raiz, ds, lista = _projeto(tmp)
        est = _estado(raiz, ds, lista)
        est.nota(1, "  bolhas   no canto ")
        est.decidir(1, "rejeitar", "outra lesão · bolhas no canto")
        est.apagar(1)                                                      # a nota escrita à mão fica, sem resposta
        d = _linhas(raiz / "revisao/paulo/decisoes.csv")
        assert [(x["item"], x["decisao"], x["nota"]) for x in d] == [("1", "", "bolhas no canto")]
        est.nao_identifico(0)
        d = {x["item"]: x for x in _linhas(raiz / "revisao/paulo/decisoes.csv")}
        assert [d[i]["decisao"] for i in ("1", "2", "3")] == ["nao_avaliavel"] * 3 and "4" not in d
        assert d["1"]["nota"] == "não identifico a lesão no seed · bolhas no canto"


# ------------------------------------------------------------------ pastas
def test_cada_revisor_fica_ligado_a_uma_lista():
    with tempfile.TemporaryDirectory() as tmp:
        raiz, ds, lista = _projeto(tmp)
        _estado(raiz, ds, lista).decidir(1, "aceitar", "")
        outra = raiz / "outra.csv"
        outra.write_text(lista.read_text().replace("597", "596"))
        erro(rv.preparar_saida, raiz / "revisao", "paulo", str(outra), rv.sha256(outra), contem="outra lista de itens")
        (raiz / "revisao/alheia").mkdir()
        (raiz / "revisao/alheia/coisas.csv").write_text("a,b\n")
        erro(rv.preparar_saida, raiz / "revisao", "alheia", str(lista), rv.sha256(lista), contem="não foram criados")
        erro(rv.preparar_saida, raiz / "revisao", "Paulo Freitas", str(lista), rv.sha256(lista), contem="inválido")


# ------------------------------------------------------------------ servidor
def _servidor(est):
    rv.Pedidos.est = est
    srv = ThreadingHTTPServer(("127.0.0.1", 0), rv.Pedidos)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def _get(url):
    try:
        with urllib.request.urlopen(url) as r:
            return r.status, r.headers.get("Content-Type"), r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type"), e.read()


def _post(url, corpo):
    req = urllib.request.Request(url, data=json.dumps(corpo).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_servidor_responde_e_nunca_serve_imagens_de_outra_janela():
    with tempfile.TemporaryDirectory() as tmp:
        raiz, ds, lista = _projeto(tmp)
        srv, base = _servidor(_estado(raiz, ds, lista))
        try:
            s, tipo, corpo = _get(base + "/")
            assert s == 200 and "text/html" in tipo and "Revisão das caixas".encode() in corpo
            s, _, corpo = _get(base + "/api/estado")
            e = json.loads(corpo)
            assert s == 200 and e["total"] == 6 and len(e["janelas"]) == 3 and e["ajustar"] is False
            assert _get(base + "/img?j=0&w=0201_600&f=603")[:2] == (200, "image/jpeg")
            assert _get(base + "/img?j=0&w=0201_600&f=612&t=1")[0] == 200           # contexto e miniatura
            assert _get(base + "/rel?j=0&w=0201_600&k=0")[0] == 200
            assert _get(base + "/img?j=0&w=0101_500&f=603")[0] == 400                # outra janela: recusado
            assert _get(base + "/img?j=0&w=0201_600&f=700")[0] == 400                # fora da janela
            assert _post(base + "/api/decidir", {"item": 4, "decisao": "aceitar", "nota": "", "dur_ms": 800}) == \
                (200, {"ok": True, "respondidos": 1})
            assert _post(base + "/api/decidir", {"item": 4, "decisao": "talvez"})[0] == 400
            assert _post(base + "/api/apagar", {"item": 4})[1]["respondidos"] == 0
            assert _post(base + "/api/nao_identifico", {"j": 2})[1]["respondidos"] == 2
        finally:
            srv.shutdown()
            srv.server_close()


# ------------------------------------------------------------------ consolidar
def test_consolidar_so_com_tudo_respondido_e_nao_reescreve():
    with tempfile.TemporaryDirectory() as tmp:
        raiz, ds, lista = _projeto(tmp)
        est = _estado(raiz, ds, lista)
        args = ["--itens", str(lista), "--revisor", "paulo", "--saida", str(raiz / "revisao"), "--consolidar"]
        est.decidir(1, "aceitar", "")
        erro(rv.main, args, contem="5 caixas ainda não têm resposta")
        for item, d in ((2, "mal_posta"), (3, "rejeitar"), (4, "nao_avaliavel"), (5, "aceitar"), (6, "aceitar")):
            est.decidir(item, d, "")
        r = rv.main(args)
        destino = raiz / "decisoes_paulo.csv"
        assert r["destino"] == destino and r["estado"] == "gravado" and not os.stat(destino).st_mode & 0o222
        assert r["contagens"] == {"aceitar": 3, "mal_posta": 1, "rejeitar": 1, "nao_avaliavel": 1}
        assert rv.main(args)["estado"].startswith("igual")
        est.decidir(6, "rejeitar", "")
        erro(rv.main, args, contem="já existe com outro conteúdo")
        erro(rv.main, ["--itens", str(lista), "--revisor", "ninguem", "--saida", str(raiz / "revisao"), "--consolidar"],
             contem="ainda não respondeu")


# ------------------------------------------------------------------ correr sem pytest
if __name__ == "__main__":
    testes = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    falhas = saltados = 0
    for nome, f in testes:
        try:
            f()
            print(f"  ok      {nome}")
        except Saltar as e:
            saltados += 1
            print(f"  saltado {nome} ({e})")
        except Exception as e:                                          # noqa: BLE001
            falhas += 1
            print(f"  FALHOU  {nome}: {type(e).__name__}: {e}")
    print(f"\n{len(testes) - falhas - saltados} passaram · {falhas} falharam · {saltados} saltados")
    sys.exit(1 if falhas else 0)
