"""
Testes do scripts/23_yolo_dados.py (v1.0, 26 out 2026): num mini-projeto inventado (o do avaliador, mais um caixas_v2.csv
e imagens falsas), confere as partes (treino, validação = casos de CAL, teste), a regra de juntar caixas por frame
físico, os rótulos YOLO, as ligações para as imagens, os YAML, os subconjuntos das 6 classes (aceites pelo avaliador) e
as recusas. Nunca leem dados reais.
    python tests/test_yolo_dados.py          ou          python -m pytest tests/ -q
"""
import importlib.util
import sys
import tempfile
from pathlib import Path

import pandas as pd

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
from test_evaluate import Saltar, _projeto, ev, saltar  # noqa: E402

A = (0.30, 0.30, 0.50, 0.50)
A2 = (0.33, 0.33, 0.55, 0.55)        # IoU 0,49 com A: a mesma lesão (≥ 0,3), embora abaixo de 0,7
B = (0.60, 0.60, 0.80, 0.80)
B2 = (0.62, 0.62, 0.84, 0.84)        # IoU 0,58 com B
C = (0.05, 0.05, 0.15, 0.15)         # longe de A: outra lesão
D = (0.45, 0.45, 0.62, 0.62)         # IoU 0,49 com E
E = (0.40, 0.40, 0.60, 0.60)
T, S = "Dataset_Treino", "Dataset_Teste"
# (split, caso, seed, classe_v2, all_seedbox, [(offset, estado, caixa)])
JANELAS = [
    (T, "0101", 500, "angiectasia", 1, [(o, "original", A) for o in (-2, -1, 0, 1, 2)] + [(5, "nova_aceite", A2)]),
    (T, "0102", 800, "ulcera", 1, [(o, "original", A) for o in (-1, 0, 1)]),
    (T, "0103", 300, "outras", 1, [(0, "original", A)]),
    (T, "0110", 1000, "angiectasia", 1, [(0, "original", A), (1, "original", A), (20, "original", A),
                                         (30, "original", A), (40, "original", E), (25, "nova_aceite", B)]),
    (T, "0110", 1050, "angiectasia", 1, [(0, "original", A), (-30, "nova_aceite", A2), (-25, "nova_aceite", B2)]),
    (T, "0110", 1030, "ulcera", 1, [(0, "original", C), (10, "original", D)]),
    (T, "0111", 200, "outras", 1, [(0, "original", A)]),
    (T, "0112", 400, "xantelasma", 1, [(0, "original", A), (3, "nova_aceite", A2)]),
    (T, "0113", 700, "angiectasia", 0, [(0, "original", A)]),                      # fora de ALL-SEEDBOX
    (S, "0201", 600, "polipo", 1, [(o, "original", A) for o in (-1, 0, 1)] + [(4, "nova_aceite", A2)]),
    (S, "0202", 900, "erosao", 1, [(0, "original", A)]),
]
SEM_IMAGEM = {(T, "0110", 1000, 1020), (T, "0112", 400, 403)}   # 1020 existe na janela 1050; 403 em lado nenhum


def _carregar():
    spec = importlib.util.spec_from_file_location("yolo_dados", AQUI.parent / "scripts" / "23_yolo_dados.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _montar(tmp):
    try:
        import yaml  # noqa: F401
        from PIL import Image
    except ImportError:
        saltar("falta o PyYAML ou o Pillow")
    raiz, conj, pred = _projeto(tmp)
    linhas = []
    for sp, caso, seed, cl, asb, cxs in JANELAS:
        for o, est, b in cxs:
            linhas.append({"split": sp, "caso": caso, "seed": seed, "frame": seed + o, "offset": o, "estado": est,
                           "categoria_v2": cl, "classe_v2": cl, "lesao": 1, "excluida": 0, "all_seedbox": asb,
                           "box_x1": b[0], "box_y1": b[1], "box_x2": b[2], "box_y2": b[3], "img_w": 320, "img_h": 320})
            if (sp, caso, seed, seed + o) not in SEM_IMAGEM:
                p = raiz / "DS" / sp / caso / "GT_com_classes" / str(seed) / f"{caso}_{seed + o}_{seed}.jpg"
                p.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (32, 32), (90, 60, 40)).save(p)
    (raiz / "DS" / T / "0110" / "GT_com_classes" / "1050").mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), (90, 60, 40)).save(raiz / "DS" / T / "0110" / "GT_com_classes" / "1050" / "0110_1020_1050.jpg")
    pasta = raiz / "data/derived/Dataset_WCE_v2"
    pasta.mkdir(parents=True)
    pd.DataFrame(linhas).to_csv(pasta / "caixas_v2.csv", index=False, float_format="%.6f")
    args = ["--config", str(raiz / "configs/avaliacao.yaml"), "--caixas", str(pasta / "caixas_v2.csv"),
            "--ds", str(raiz / "DS"), "--saida", str(raiz / "data/derived/yolo"),
            "--resumo", str(raiz / "outputs/23_yolo_dados_resumo.txt"), "--sha-caixas", ""]
    return raiz, conj, pred, args


def _rotulos(raiz, variante, parte):
    pasta = raiz / "data/derived/yolo" / variante / "labels" / parte
    return {f.stem: [ln.split() for ln in f.read_text().splitlines()] for f in sorted(pasta.glob("*.txt"))}


def _caixa(ln):
    c, xc, yc, w, h = (float(x) for x in ln)
    return int(c), (round(xc - w / 2, 4), round(yc - h / 2, 4), round(xc + w / 2, 4), round(yc + h / 2, 4))


def _erro(m, args, contem):
    try:
        m.main(args)
    except ev.ErroAvaliacao as e:
        assert contem in str(e), f"a mensagem não diz '{contem}': {e}"
        return
    raise AssertionError("devia ter parado")


def test_partes_e_contagens():
    m = _carregar()
    with tempfile.TemporaryDirectory() as tmp:
        raiz, conj, pred, args = _montar(tmp)
        r = m.main(args)
        c = r["contagens"]
        # original: treino 0110 (frames 1000, 1001, 1020, 1030, 1040, 1050) e 0112 (400); validação = casos de CAL
        assert (c["original"]["train"]["imagens"], c["original"]["train"]["caixas"]) == (7, 8)
        assert (c["original"]["val"]["imagens"], c["original"]["val"]["caixas"]) == (8, 8)       # sem «outras»
        assert (c["original"]["test"]["imagens"], c["original"]["test"]["caixas"]) == (4, 4)
        # v2: mais o 1025 (treino), o 505 (validação) e o 604 (teste); o 403 não tem imagem
        assert (c["v2"]["train"]["imagens"], c["v2"]["train"]["caixas"]) == (8, 9)
        assert (c["v2"]["val"]["imagens"], c["v2"]["test"]["imagens"]) == (9, 5)
        assert r["sem_imagem"]["v2"] == [f"{T}/0112/frame 403"] and not r["sem_imagem"]["original"]
        assert r["casos"]["val"] == {"0101@" + T, "0102@" + T}                              # 0103 só tem «outras»
        assert "0113@" + T not in r["casos"]["train"]                                       # fora de ALL-SEEDBOX
        j = r["juntadas"]
        assert (j["original"]["train"]["juntadas"], j["original"]["train"]["classes_diferentes"]) == (1, 1)
        assert (j["v2"]["train"]["juntadas"], j["v2"]["train"]["nova_em_original"]) == (3, 1)
        texto = r["resumo"].read_text(encoding="utf-8")
        assert r["linha"].startswith("- ") and "experiência B, dados do YOLO" in r["linha"] and r["linha"] in texto
        assert "validação 8 imagens / 8 caixas" in r["linha"]


def test_regra_de_juntar_e_rotulos():
    m = _carregar()
    with tempfile.TemporaryDirectory() as tmp:
        raiz, conj, pred, args = _montar(tmp)
        m.main(args)
        o, v = _rotulos(raiz, "original", "train"), _rotulos(raiz, "v2", "train")
        # 1030: duas lesões em sítios diferentes, de janelas diferentes → ficam as duas
        assert sorted(_caixa(ln) for ln in o["tr_0110_1030"]) == sorted([(0, A), (2, C)])
        # 1040: a mesma lesão em duas janelas (IoU 0,49) → fica a do seed mais perto (1030, úlcera)
        assert [_caixa(ln) for ln in o["tr_0110_1040"]] == [(2, D)]
        # 1020: a nova (janela 1050) cai sobre a original (janela 1000) → fica a original
        assert [_caixa(ln) for ln in v["tr_0110_1020"]] == [(0, A)] and o["tr_0110_1020"] == v["tr_0110_1020"]
        # 1025: duas novas à mesma distância → a do seed mais pequeno (1000)
        assert [_caixa(ln) for ln in v["tr_0110_1025"]] == [(0, B)] and "tr_0110_1025" not in o
        # a v2 só acrescenta: tudo o que está na original está igual na v2
        assert all(v[k] == o[k] for k in o)
        man = pd.read_csv(raiz / "data/derived/yolo/v2/manifesto.csv", dtype={"caso": str})
        r1020 = man[(man["caso"] == "0110") & (man["frame"] == 1020)].iloc[0]
        assert r1020["seed_imagem"] == 1050                                   # a cópia da janela 1000 não existe
        img = raiz / "data/derived/yolo/v2/images/train/tr_0110_1020.jpg"
        assert img.is_symlink() and img.resolve() == (raiz / "DS" / T / "0110/GT_com_classes/1050/0110_1020_1050.jpg").resolve()
        for ln in sum(v.values(), []):
            assert len(ln) == 5 and 0 <= int(ln[0]) <= 5 and all(0 <= float(x) <= 1 for x in ln[1:])
        for parte in ("train", "val", "test"):
            ims = {f.stem for f in (raiz / "data/derived/yolo/v2/images" / parte).glob("*.jpg")}
            assert ims == set(_rotulos(raiz, "v2", parte)) and all(
                f.resolve().is_file() for f in (raiz / "data/derived/yolo/v2/images" / parte).glob("*.jpg"))


def test_yaml_e_subconjuntos_no_avaliador():
    import yaml
    m = _carregar()
    with tempfile.TemporaryDirectory() as tmp:
        raiz, conj, pred, args = _montar(tmp)
        r = m.main(args)
        base = raiz / "data/derived/yolo"
        y = yaml.safe_load((base / "original/treino.yaml").read_text(encoding="utf-8"))
        assert y["path"] == str((base / "original").resolve()) and "test" not in y
        assert (y["train"], y["val"]) == ("images/train", "images/val")
        assert y["names"] == {0: "angiectasia", 1: "erosao", 2: "ulcera", 3: "polipo", 4: "eritema", 5: "xantelasma"}
        yt = yaml.safe_load((base / "v2/teste.yaml").read_text(encoding="utf-8"))
        assert yt["test"] == "images/test" and yt["path"] == str((base / "v2").resolve())
        assert r["subconjuntos"] == {"CAL": (2, 3), "VAL_EX": (2, 2), "TEST_BOX": (2, 2)}
        cal = base / "conjuntos/CAL_6classes.csv"
        assert ev.ler_conjunto(cal) == [(T, "0101", 500), (T, "0102", 800)]
        js = [x for x in conj["CAL"] if x[1] != "0103"]
        assert ev.main(["calibrar", "--metodo", "YOLO_TESTE", "--pred", pred("YOLO_CAL", js, {1, 2}),
                        "--config", str(raiz / "configs/avaliacao.yaml"), "--conjunto-ficheiro", str(cal)]) == 0


def test_recusas_e_forcar():
    m = _carregar()
    with tempfile.TemporaryDirectory() as tmp:
        raiz, conj, pred, args = _montar(tmp)
        r1 = m.main(args)
        _erro(m, args, "já existe")
        r2 = m.main(args + ["--forcar"])
        assert r2["impressoes"] == r1["impressoes"]                         # o mesmo conteúdo outra vez
        _erro(m, args + ["--forcar", "--sha-caixas", "deadbeef"], "não é o do Dataset WCE v2")
        _erro(m, [x if x != str(raiz / "DS") else str(raiz / "nao_existe") for x in args] + ["--forcar"],
              "Não encontrei a pasta do dataset")
        assert (raiz / "data/derived/yolo/v2/labels/train").is_dir()       # as tentativas falhadas não estragaram nada
        assert not list((raiz / "data/derived").glob(".yolo.*"))           # nem deixaram pastas temporárias


if __name__ == "__main__":
    testes = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    falhas = 0
    for nome, f in testes:
        try:
            f()
            print(f"  ok      {nome}")
        except Saltar as e:
            print(f"  saltado {nome} ({e})")
        except Exception as e:                                              # noqa: BLE001
            falhas += 1
            print(f"  FALHOU  {nome}: {type(e).__name__}: {e}")
    sys.exit(1 if falhas else 0)
