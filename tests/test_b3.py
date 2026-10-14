"""
Testes do B3 (methods/b3_csrt.py): NCC, conversão de caixas, seguimento de um quadrado em movimento com o CSRT
verdadeiro do OpenCV e uma corrida completa num mini-projeto com imagens inventadas (variantes de L_max, paragens,
imagem em falta e avaliador). Os testes com o tracker precisam do OpenCV com contrib; sem ele, saltam.
Nunca leem dados reais.
    python tests/test_b3.py          ou          python -m pytest tests/ -q
"""
import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
from test_evaluate import Saltar, _projeto, ev, saltar  # noqa: E402


def _carregar():
    spec = importlib.util.spec_from_file_location("b3_csrt", AQUI.parent / "methods" / "b3_csrt.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _cv2():
    try:
        import cv2
    except ImportError:
        saltar("falta o OpenCV")
    b3 = _carregar()
    try:
        b3.criar_csrt(cv2)
    except ev.ErroAvaliacao:
        saltar("o OpenCV não tem o CSRT")
    return cv2, b3


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def _fundo(cv2):
    """Fundo com textura suave (o mesmo em todos os frames), 320×320 a cores."""
    ruido = (np.random.default_rng(0).random((20, 20, 3)) * 120 + 60).astype(np.uint8)
    return cv2.resize(ruido, (320, 320), interpolation=cv2.INTER_CUBIC)


def _objeto(lado=64):
    """Quadrado com um padrão suave e próprio (mancha vermelha descentrada sobre gradientes), fácil de seguir e
    diferente do fundo; suave como uma lesão, para a NCC não depender de um alinhamento ao píxel."""
    y, x = np.mgrid[0:lado, 0:lado] / (lado - 1)
    mancha = np.exp(-((x - 0.4) ** 2 + (y - 0.45) ** 2) / 0.08)
    obj = np.zeros((lado, lado, 3))
    obj[..., 2] = 60 + 190 * mancha                                         # BGR: vermelho na mancha
    obj[..., 1] = 40 + 150 * x                                              # gradiente horizontal
    obj[..., 0] = 40 + 120 * (1 - y)                                        # gradiente vertical
    return obj.astype(np.uint8)


def _frame(cv2, x, y, visivel=True):
    img = _fundo(cv2).copy()
    if visivel:
        img[y:y + 64, x:x + 64] = _objeto()
    return img


def test_ncc_e_caixas():
    b3 = _carregar()
    a = (np.random.default_rng(1).random((32, 32, 3)) * 255).astype(np.uint8)
    assert abs(b3.ncc(a, a) - 1) < 1e-9 and abs(b3.ncc(a, 255 - a) + 1) < 1e-9
    assert b3.ncc(a, np.full_like(a, 7)) == 0.0                             # recorte liso
    assert b3.caixa_px((0.30, 0.30, 0.50, 0.50), 320, 320) == (96, 96, 64, 64)
    assert b3.caixa_px((0.5, 0.5, 0.5001, 0.5001), 320, 320)[2:] == (1, 1)  # pelo menos 1 px
    assert b3.ret_para_caixa((96, 96, 64, 64), 320, 320) == (0.3, 0.3, 0.5, 0.5)
    assert b3.ret_para_caixa((-10, 300, 40, 40), 320, 320) == (0.0, 0.9375, 0.09375, 1.0)   # cortada à imagem


def test_seguir_quadrado_em_movimento():
    cv2, b3 = _cv2()
    imgs = [_frame(cv2, 96 + 3 * i, 96, visivel=i <= 6) for i in range(10)]
    seguidos, motivo = b3.seguir(cv2, imgs, (96, 96, 64, 64), 32)
    por_i = {i: (sc, cx) for i, sc, cx in seguidos}
    for i in range(1, 7):                                                   # visível: segue bem e NCC alta
        sc, cx = por_i[i]
        assert sc > 0.8 and _iou(cx, ((96 + 3 * i) / 320, 0.3, (160 + 3 * i) / 320, 0.5)) > 0.7
    assert all(por_i[i][0] < 0.5 for i in range(7, 10) if i in por_i)       # desapareceu: NCC baixa
    assert motivo in ("fim", "falha", "fora")


def _mini_projeto(cv2, tmp):
    """Mini-projeto do avaliador (caixa do seed 0,30–0,50); o objeto está na caixa no seed, anda 2 px por frame e só
    se vê nos offsets −6 a +6. Falta a imagem 0102/800/804."""
    raiz, conj, _ = _projeto(tmp)
    for sp, caso, seed, _ in conj["CAL"]:
        pasta = raiz / "DS" / sp / caso / "GT_com_classes" / str(seed)
        pasta.mkdir(parents=True)
        for o in range(-10, 11):
            if (caso, o) != ("0102", 4):
                cv2.imwrite(str(pasta / f"{caso}_{seed + o}_{seed}.jpg"), _frame(cv2, 96 + 2 * o, 96, abs(o) <= 6),
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
    (raiz / "configs/b3.yaml").write_text(
        f"versao: 1\nl_max: [3, 100]\nncc_lado: 32\nsaida: {raiz / 'outputs/preds'}\n", encoding="utf-8")
    return raiz


def test_b3_ponta_a_ponta():
    cv2, b3 = _cv2()
    with tempfile.TemporaryDirectory() as tmp:
        raiz = _mini_projeto(cv2, tmp)
        cfg = str(raiz / "configs/avaliacao.yaml")
        r = b3.main(["--conjunto", "CAL", "--ds", str(raiz / "DS"), "--config", cfg,
                     "--config-b3", str(raiz / "configs/b3.yaml"), "--sem-progresso"])
        assert sorted(r["csv"]) == [3, 100] and r["n_frames"] == 60 and not r["sem_caixa"] and not r["sem_seed"]
        p3, p100 = (pd.read_csv(r["csv"][k], dtype={"caso": str}) for k in (3, 100))
        assert len(p3) == len(p100) == 60 and p100["score"].between(0, 1).all()
        for p in (p3, p100):
            p["o"] = p["frame"] - p["seed"]
        j = p100[p100["caso"] == "0101"].set_index("o")
        assert (j.loc[[-6, -3, -1, 1, 3, 6], "score"] > 0.8).all()           # visível e seguido
        assert (j.loc[[-10, -8, 8, 10], "score"] < 0.5).all()               # já não se vê
        for o in (-3, -1, 1, 3):
            caixa = tuple(j.loc[o, ["box_x1", "box_y1", "box_x2", "box_y2"]].astype(float))
            assert _iou(caixa, ((96 + 2 * o) / 320, 0.3, (160 + 2 * o) / 320, 0.5)) > 0.7
        fora = p3[p3["o"].abs() > 3]                                         # L_max = 3: nada para lá de 3 frames
        assert (fora["score"] == 0).all() and fora["box_x1"].isna().all()
        assert (p3.loc[p3["o"].abs() <= 3, "score"] > 0).all()
        k = p100[(p100["caso"] == "0102") & (p100["o"] >= 4)]                # falta a imagem 804: pára antes dela
        assert (k["score"] == 0).all()
        par = pd.read_csv(r["paragens"], dtype={"caso": str})
        assert len(par) == 6
        linha = par[(par["caso"] == "0102") & (par["direcao"] == "depois")].iloc[0]
        assert linha["motivo"] == "sem_imagem" and linha["frames_seguidos"] == 3
        for lm in (3, 100):
            assert ev.main(["calibrar", "--metodo", f"B3_csrt_L{lm}", "--pred", str(r["csv"][lm]), "--config", cfg]) == 0


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
