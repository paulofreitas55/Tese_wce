"""
Testes do scripts/24_yolo_prever.py (v1.0, 27 out 2026): num mini-projeto inventado (o do avaliador, com imagens falsas e
os subconjuntos das 6 classes), com um detetor falso, confere a regra do score (a caixa mais confiante da classe da
janela; as outras classes não contam), o formato das previsões (aceites pelo calibrar e pelo avaliar), a marca e as
recusas. Um teste corre o YOLOv5 de verdade, com um modelo pequeno ao acaso: só corre onde houver o torch e o
repositório do YOLOv5 (no PC, no ambiente yolo); nos outros sítios fica «saltado». Nunca leem dados reais.
    python tests/test_yolo_prever.py          ou          python -m pytest tests/ -q
"""
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
from test_evaluate import Saltar, _projeto, ev, saltar  # noqa: E402

PRINCIPAIS = ["angiectasia", "erosao", "ulcera", "polipo", "eritema", "xantelasma"]
A = (0.30, 0.30, 0.50, 0.50)
B = (0.60, 0.60, 0.80, 0.80)
C = (0.05, 0.05, 0.15, 0.15)
JANELAS = {"CAL": [("Dataset_Treino", "0101", 500, "angiectasia"), ("Dataset_Treino", "0102", 800, "ulcera")],
           "VAL_EX": [("Dataset_Teste", "0201", 600, "polipo")],
           "TEST_BOX": [("Dataset_Teste", "0201", 600, "polipo"), ("Dataset_Teste", "0202", 900, "erosao")]}
SEM_IMAGEM = ("0202", 905)


def _carregar():
    spec = importlib.util.spec_from_file_location("yolo_prever", AQUI.parent / "scripts" / "24_yolo_prever.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class Falso:
    """Detetor falso: lê o caso, o frame e o seed do nome da imagem. Perto do seed (|offset| ≤ 2) devolve a lesão da
    classe da janela (0,9) e, mais confiante, uma caixa de outra classe (0,97); a 3–5 frames, duas caixas da classe
    (0,3 e 0,6); mais longe, nada."""
    nomes, commit, versao_torch, dispositivo = PRINCIPAIS, "falso", "-", "cpu"

    def __init__(self):
        self.classe = {(c, s): cl for js in JANELAS.values() for _, c, s, cl in js}

    def __call__(self, caminhos, progresso=None):
        out = []
        for p in caminhos:
            caso, f, s = Path(p).stem.split("_")
            o, c = int(f) - int(s), PRINCIPAIS.index(self.classe[(caso, int(s))])
            outra = (c + 1) % 6
            if abs(o) <= 2:
                out.append(np.array([[*A, 0.9, c], [*C, 0.97, outra]]))
            elif abs(o) <= 5:
                out.append(np.array([[*B, 0.3, c], [*A, 0.6, c]]))
            else:
                out.append(np.zeros((0, 6)))
        if progresso:
            progresso(len(caminhos), len(caminhos), 1.0)
        return out


def _montar(tmp):
    try:
        import yaml  # noqa: F401
        from PIL import Image
    except ImportError:
        saltar("falta o PyYAML ou o Pillow")
    raiz, conj, pred = _projeto(tmp)
    pasta = raiz / "data/derived/yolo/conjuntos"
    pasta.mkdir(parents=True)
    for n, js in JANELAS.items():
        pd.DataFrame([{"split": s, "caso": c, "seed": e, "classe": cl} for s, c, e, cl in js]).to_csv(
            pasta / f"{n}_6classes.csv", index=False)
    for sp, caso, seed, _ in {x for js in JANELAS.values() for x in js}:
        d = raiz / "DS" / sp / caso / "GT_com_classes" / str(seed)
        d.mkdir(parents=True, exist_ok=True)
        for f in range(seed - 10, seed + 11):
            if (caso, f) != SEM_IMAGEM:
                Image.new("RGB", (32, 32), (90, 60, 40)).save(d / f"{caso}_{f}_{seed}.jpg")
    pesos = raiz / "best.pt"
    pesos.write_bytes(b"pesos falsos")
    cfg = str(raiz / "configs/avaliacao.yaml")
    args = ["--modelo", "YOLO_TESTE", "--pesos", str(pesos), "--config", cfg, "--conjuntos", str(pasta),
            "--ds", str(raiz / "DS"), "--saida-prev", str(raiz / "outputs/preds"), "--saida", str(raiz / "outputs/yolo")]
    return raiz, cfg, pasta, args


def _erro(m, args, contem, detetor=None):
    try:
        m.main(args, detetor=detetor or Falso())
    except ev.ErroAvaliacao as e:
        assert contem in str(e), f"a mensagem não diz '{contem}': {e}"
        return
    raise AssertionError("devia ter parado")


def test_score_da_classe_da_janela_e_formato():
    m = _carregar()
    with tempfile.TemporaryDirectory() as tmp:
        raiz, cfg, pasta, args = _montar(tmp)
        r = m.main(args, detetor=Falso())
        cal = pd.read_csv(r["previsoes"]["CAL"], dtype={"caso": str})
        tb = pd.read_csv(r["previsoes"]["TEST_BOX"], dtype={"caso": str})
        ve = pd.read_csv(r["previsoes"]["VAL_EX"], dtype={"caso": str})
        assert (len(cal), len(tb), len(ve)) == (42, 41, 21) and r["sem_imagem"] == 1
        assert not ((tb["caso"] == "0202") & (tb["frame"] == 905)).any()
        perto, meio, longe = (cal["distancia"] <= 2), cal["distancia"].between(3, 5), cal["distancia"] > 5
        assert (cal.loc[perto, "score"] == 0.9).all()                      # não a de outra classe, com 0,97
        assert np.allclose(cal.loc[perto, ["box_x1", "box_y1", "box_x2", "box_y2"]].to_numpy(), A)
        assert (cal.loc[meio, "score"] == 0.6).all()                        # a mais confiante da classe
        assert np.allclose(cal.loc[meio, ["box_x1", "box_y1", "box_x2", "box_y2"]].to_numpy(), A)
        assert (cal.loc[longe, "score"] == 0).all() and cal.loc[longe, "box_x1"].isna().all()
        assert set(cal["classe"]) == {"angiectasia", "ulcera"} and set(ve["classe"]) == {"polipo"}
        marca = (r["pasta"] / "configuracao.txt").read_text(encoding="utf-8")
        assert ev._sha256(raiz / "best.pt") in marca and "conf 0.001" in marca and "iou 0.6" in marca
        assert "calibrar --metodo YOLO_TESTE" in r["texto"]


def test_previsoes_aceites_pelo_avaliador():
    m = _carregar()
    with tempfile.TemporaryDirectory() as tmp:
        raiz, cfg, pasta, args = _montar(tmp)
        r = m.main(args, detetor=Falso())
        p = r["previsoes"]
        assert ev.main(["calibrar", "--metodo", "YOLO_TESTE", "--pred", str(p["CAL"]), "--config", cfg,
                        "--conjunto-ficheiro", str(pasta / "CAL_6classes.csv")]) == 0
        for n in ("VAL_EX", "TEST_BOX"):
            assert ev.main(["avaliar", "--metodo", "YOLO_TESTE", "--conjunto", n, "--pred", str(p[n]), "--config", cfg,
                            "--conjunto-ficheiro", str(pasta / f"{n}_6classes.csv")]) == 0
        j = ev.ler_config(raiz / "outputs/eval/TEST_BOX/YOLO_TESTE.json")
        assert j["contagens"]["janelas"] == 2


def test_recusas():
    m = _carregar()
    with tempfile.TemporaryDirectory() as tmp:
        raiz, cfg, pasta, args = _montar(tmp)
        m.main(args, detetor=Falso())
        _erro(m, args, "Já existem previsões")
        outro = Falso()
        outro.nomes = ["a", "b", "c", "d", "e", "f"]
        _erro(m, args + ["--forcar"], "não são as 6 do desenho", detetor=outro)
        _erro(m, [x if x != str(raiz / "best.pt") else str(raiz / "nao.pt") for x in args] + ["--forcar"],
              "Não encontrei os pesos")


def test_yolov5_de_verdade():
    """O detetor verdadeiro, com um YOLOv5n ao acaso de 6 classes: uma matriz n×6 por imagem, caixas em 0–1."""
    repo = Path(os.environ.get("YOLOV5", AQUI.parent / "ml_models" / "yolov5"))
    try:
        import torch
    except ImportError:
        saltar("falta o torch (corre no ambiente yolo)")
    if not (repo / "models" / "yolo.py").is_file():
        saltar(f"não há o YOLOv5 em {repo}")
    m = _carregar()
    if not m.tem_ultralytics():
        # sem o ultralytics, importar o YOLOv5 faria «pip install -U ultralytics» neste ambiente (ex.: o sam_tese)
        saltar("falta o ultralytics (corre no ambiente yolo)")
    antes, caminho = set(sys.modules), list(sys.path)
    sys.path.insert(0, str(repo))
    try:
        try:
            from models.yolo import Model
        except ImportError as e:
            saltar(f"o YOLOv5 não importa aqui ({e})")
        _correr_yolov5(m, repo, Model, torch)
    finally:                                   # não deixa os módulos models/ e utils/ do YOLOv5 para os outros testes
        for k in set(sys.modules) - antes:
            if k.split(".")[0] in ("models", "utils"):
                del sys.modules[k]
        sys.path[:] = caminho


def _correr_yolov5(m, repo, Model, torch):
    with tempfile.TemporaryDirectory() as tmp:
        raiz, cfg, pasta, args = _montar(tmp)
        torch.manual_seed(0)
        mod = Model(str(repo / "models" / "yolov5n.yaml"), ch=3, nc=6)
        mod.names = dict(enumerate(PRINCIPAIS))
        torch.save({"model": mod.half(), "epoch": -1}, raiz / "aleatorio.pt")
        det = m.DetetorYOLOv5(raiz / "aleatorio.pt", repo, imgsz=64, lote=4, workers=0)
        assert det.nomes == PRINCIPAIS
        ims = sorted(str(p) for p in (raiz / "DS").rglob("*.jpg"))[:6]
        out = det(ims)
        assert len(out) == 6
        for d in out:
            assert d.ndim == 2 and d.shape[1] == 6
            if len(d):
                assert d[:, :4].min() >= -1e-6 and d[:, :4].max() <= 1 + 1e-6 and (d[:, 4] >= 0.001 - 1e-6).all()


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
