"""
Testes da extração DINOv2 (methods/extrair_dinov2.py) e do B1 com esses vetores, com imagens inventadas e um modelo
falso (só numpy): testam a preparação das imagens, as versões (cls, mean, max, std), a gravação, a retoma e a
ligação ao B1 e ao avaliador. O modelo real (torch.hub) não é carregado. Nunca leem dados reais.
    python tests/test_dinov2.py          ou          python -m pytest tests/ -q
"""
import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
from test_b1 import _torch  # noqa: E402
from test_evaluate import Saltar, _projeto, ev, perto, saltar  # noqa: E402


def _carregar(nome, ficheiro):
    spec = importlib.util.spec_from_file_location(nome, AQUI.parent / "methods" / ficheiro)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class _ModeloFalso:
    """Faz de DINOv2: 256 quadrados de 14×14 px, cada um com a cor média projetada em 768 números."""

    def __init__(self, torch, dino):
        self.torch, self.disp, self.dino, self.chamadas = torch, "cpu", dino, 0
        self.W = np.random.default_rng(0).normal(size=(3, 768)).astype(np.float32)

    def __call__(self, lote):
        self.chamadas += 1
        x = np.stack(lote)                                          # B×3×224×224
        assert x.shape[1:] == (3, 224, 224)
        b = x.shape[0]
        q = x.reshape(b, 3, 16, 14, 16, 14).mean(axis=(3, 5)).reshape(b, 3, 256).transpose(0, 2, 1) @ self.W
        return self.dino.versoes(q.mean(axis=1) + 0.5, q)

    def pesos_sha256(self):
        return "falso"


def _imagens(raiz, janelas, falta=()):
    from PIL import Image, ImageDraw
    for sp, caso, seed, _ in janelas:
        pasta = raiz / "DS" / sp / caso / "GT_com_classes" / str(seed)
        pasta.mkdir(parents=True)
        for o in range(-10, 11):
            if (caso, o) in falta:
                continue
            im = Image.new("RGB", (320, 320), (120 + 6 * abs(o), 80, 60))
            ImageDraw.Draw(im).rectangle([100, 100, 180 + 4 * o, 200], fill=(200, 40 + 5 * abs(o), 40))
            im.save(pasta / f"{caso}_{seed + o}_{seed}.jpg", quality=95)


def _config_b1(raiz, lote=8):
    (raiz / "configs/b1.yaml").write_text(
        "versao: 2\nvetores:\n  dinov2:\n"
        f"    pasta: {raiz / 'vec_dino'}\n    chaves: [cls, mean, max, std]\n    modelo: dinov2_vitb14\n"
        f"    lado: 224\n    lote: {lote}\nsaida: {raiz / 'outputs/preds'}\n", encoding="utf-8")


def _com_torch(teste):
    """Corre o teste com o PyTorch real ou, sem ele, com o substituto de test_b1."""
    def corre():
        try:
            import yaml  # noqa: F401
            import PIL  # noqa: F401
        except ImportError:
            saltar("falta o PyYAML ou o Pillow")
        torch, _, falso = _torch()
        if falso is not None:
            falso.from_numpy = lambda a: np.asarray(a)
            sys.modules["torch"] = falso
        try:
            teste(torch)
        finally:
            if falso is not None:
                sys.modules.pop("torch", None)
    corre.__name__ = teste.__name__
    return corre


def _preparar_e_versoes(torch):
    dino = _carregar("extrair_dinov2", "extrair_dinov2.py")
    with tempfile.TemporaryDirectory() as tmp:
        from PIL import Image
        f = Path(tmp) / "branca.jpg"
        Image.new("RGB", (320, 320), (255, 255, 255)).save(f, quality=100)
        a = dino.preparar(f, 224)
        assert a.shape == (3, 224, 224) and a.dtype == np.float32
        assert np.allclose(a[:, 100, 100], (1 - dino.MEDIA) / dino.DESVIO, atol=0.02)
    q = np.array([[[1.0, 4.0], [3.0, 0.0]]])                       # 1 imagem, 2 quadrados, 2 números
    v = dino.versoes(np.array([[9.0, 9.0]]), q)
    assert v["cls"].tolist() == [[9.0, 9.0]] and v["mean"].tolist() == [[2.0, 2.0]]
    assert v["max"].tolist() == [[3.0, 4.0]] and np.allclose(v["std"], [[1.0, 2.0]])


def _extracao_retoma_e_b1(torch):
    dino = _carregar("extrair_dinov2", "extrair_dinov2.py")
    b1 = _carregar("b1_global", "b1_global.py")
    with tempfile.TemporaryDirectory() as tmp:
        raiz, conj, _ = _projeto(tmp)
        _imagens(raiz, conj["CAL"], falta={("0102", 4)})
        _config_b1(raiz)
        cfg = str(raiz / "configs/avaliacao.yaml")
        base = ["--conjunto", "CAL", "--ds", str(raiz / "DS"), "--config", cfg,
                "--config-b1", str(raiz / "configs/b1.yaml"), "--sem-progresso"]
        modelo = _ModeloFalso(torch, dino)
        r = dino.main(base, modelo=modelo)
        assert r["janelas"] == 3 and r["feitas"] == 3 and r["escritos"] == 62
        assert r["sem_imagem"] == ["Dataset_Treino/0102/800/804"]
        assert modelo.chamadas == 9                                   # 3 janelas × 3 lotes de até 8 imagens
        f = raiz / "vec_dino/Dataset_Treino/0101/GT_com_classes/500/0101_495_500.pt"
        o = torch.load(f, map_location="cpu")
        assert sorted(o) == ["cls", "max", "mean", "std"]
        assert all(np.asarray(o[k]).shape == (768,) for k in o)
        assert not list((raiz / "vec_dino").rglob("*.tmp"))
        # retoma: as janelas completas são saltadas; a que tem uma imagem em falta volta a ser vista
        de_novo = dino.main(base, modelo=modelo)
        assert de_novo["saltadas"] == 2 and de_novo["feitas"] == 1 and de_novo["escritos"] == 20
        # B1 com estes vetores: 4 versões, scores em [0, 1], e o avaliador aceita o CSV
        rb = b1.main(["--emb", "dinov2", "--conjunto", "CAL", "--config", cfg,
                      "--config-b1", str(raiz / "configs/b1.yaml"), "--sem-progresso"])
        assert sorted(rb["csv"]) == ["cls", "max", "mean", "std"] and rb["sem_vetor"] == ["Dataset_Treino/0102/800/804"]
        p = pd.read_csv(rb["csv"]["cls"], dtype={"caso": str})
        assert len(p) == 59 and p["score"].between(0, 1).all()
        p["dist"] = (p["frame"] - p["seed"]).abs()
        perto_seed, longe = p.loc[p["dist"] == 1, "score"].mean(), p.loc[p["dist"] == 10, "score"].mean()
        assert perto_seed > longe                                     # imagens mais parecidas → score maior
        assert ev.main(["calibrar", "--metodo", "B1_dinov2_cls", "--pred", str(rb["csv"]["cls"]), "--config", cfg]) == 0


test_preparar_e_versoes = _com_torch(_preparar_e_versoes)
test_extracao_retoma_e_b1 = _com_torch(_extracao_retoma_e_b1)


if __name__ == "__main__":
    testes = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    falhas = 0
    for nome, f in testes:
        try:
            f()
            print(f"  ok      {nome}")
        except Saltar as e:
            print(f"  saltado {nome} ({e})")
        except Exception as e:                                          # noqa: BLE001
            falhas += 1
            print(f"  FALHOU  {nome}: {type(e).__name__}: {e}")
    sys.exit(1 if falhas else 0)
