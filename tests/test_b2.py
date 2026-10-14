"""
Testes do B2 (methods/b2_local.py): protótipo, scores, caixa prevista e uma corrida completa num mini-projeto com
imagens inventadas e um modelo falso (cada quadrado da imagem vira um vetor feito a partir da sua cor média). Testam
também a retoma e que o avaliador aceita os CSV. A corrida completa precisa do PyTorch; sem ele, salta.
Nunca leem dados reais.
    python tests/test_b2.py          ou          python -m pytest tests/ -q
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

CAIXA_0101 = (0.10, 0.55, 0.40, 0.75)          # caixa assimétrica, para apanhar trocas entre x e y


def _carregar():
    spec = importlib.util.spec_from_file_location("b2_local", AQUI.parent / "methods" / "b2_local.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def test_prototipo_dentro_da_caixa():
    b2 = _carregar()
    idx, recurso = b2.indices_prototipo((0.30, 0.30, 0.50, 0.50), 32)
    lin, col = np.divmod(idx, 32)
    assert not recurso and len(idx) == 36 and set(lin) == set(range(10, 16)) and set(col) == set(range(10, 16))
    idx, recurso = b2.indices_prototipo(CAIXA_0101, 32)
    lin, col = np.divmod(idx, 32)
    assert not recurso and (lin.min(), lin.max(), col.min(), col.max()) == (18, 23, 3, 12)
    idx, recurso = b2.indices_prototipo((0.501, 0.501, 0.51, 0.51), 32)     # nenhum centro lá dentro
    assert recurso and idx.tolist() == [16 * 32 + 16]                        # o quadrado do centro da caixa


def test_caixa_do_mapa_componente_do_maximo():
    b2 = _carregar()
    s = np.full((32, 32), 0.2)
    s[18:24, 3:13] = 0.95
    s[20, 5] = 1.0                                                          # o máximo está neste bloco
    s[2:5, 25:30] = 0.97                                                    # bloco alto mas separado: fica de fora
    assert b2.caixa_do_mapa(s, 0.9) == (3 / 32, 18 / 32, 13 / 32, 24 / 32)
    assert b2.caixa_do_mapa(np.full((4, 4), -0.5), 0.9) == (0.0, 0.0, 0.25, 0.25)   # máximo ≤ 0: só esse quadrado


def test_scores():
    b2 = _carregar()
    s = np.full((32, 32), 0.6)
    s[0, :4] = [0.9, 0.8, 0.7, 0.7]
    sm = np.full((32, 32), 0.5)
    sm[3, 3] = 0.65
    assert b2.scores(s, sm) == {"bruto": 0.9, "corrigido": 0.3, "bruto_top4": 0.775,
                                "bruto_media": 0.65, "corrigido_media": 0.15}
    fora = b2.scores(np.full((2, 2), -0.3), np.full((2, 2), 1.4))           # cortados a [0, 1]
    assert fora["bruto"] == 0.0 and fora["corrigido"] == 0.0 and fora["bruto_media"] == 1.0


class _ModeloFalso:
    """Faz de DINOv2: cada quadrado de 14×14 px dá 768 números (a cor média, projetada), com norma 1."""

    def __init__(self, torch):
        self.torch, self.disp, self.chamadas = torch, "cpu", 0
        self.W = torch.from_numpy(np.random.default_rng(0).normal(size=(3, 768)).astype(np.float32))

    def quadrados(self, lote):
        self.chamadas += 1
        x = np.stack(lote)                                                  # B×3×448×448
        b = x.shape[0]
        cor = x.reshape(b, 3, 32, 14, 32, 14).mean(axis=(3, 5)).reshape(b, 3, 1024).transpose(0, 2, 1)
        q = self.torch.from_numpy(np.ascontiguousarray(cor, dtype=np.float32)) @ self.W
        return self.torch.nn.functional.normalize(q, dim=-1)

    def pesos_sha256(self):
        return "falso"


def _mini_projeto(tmp):
    """Mini-projeto do avaliador; a janela 0101 passa a ter a caixa CAIXA_0101 (offsets −2 a +2). Imagens: fundo cor
    de mucosa e, nos offsets −3 a +3, um retângulo vermelho na caixa da janela; falta a imagem 0102/800/804."""
    from PIL import Image, ImageDraw
    raiz, conj, _ = _projeto(tmp)
    f = raiz / "data/derived/annotations_v1.csv"
    a = pd.read_csv(f, dtype={"caso": str}, keep_default_na=False)
    m = (a["caso"] == "0101") & (a["box_src"].isin(["medico_seed", "medico_vizinho"]))
    for c, v in zip(["box_x1", "box_y1", "box_x2", "box_y2"], CAIXA_0101):
        a.loc[m, c] = str(v)
    a.to_csv(f, index=False)
    for sp, caso, seed, _ in conj["CAL"]:
        cx = CAIXA_0101 if caso == "0101" else (0.30, 0.30, 0.50, 0.50)
        pasta = raiz / "DS" / sp / caso / "GT_com_classes" / str(seed)
        pasta.mkdir(parents=True)
        for o in range(-10, 11):
            if (caso, o) == ("0102", 4):
                continue
            im = Image.new("RGB", (320, 320), (120 + 2 * o, 80, 60))
            if abs(o) <= 3:
                ImageDraw.Draw(im).rectangle([cx[0] * 320 + o, cx[1] * 320, cx[2] * 320 + o - 1, cx[3] * 320 - 1],
                                             fill=(200, 40, 40))
            im.save(pasta / f"{caso}_{seed + o}_{seed}.jpg", quality=95)
    return raiz


def _config_b2(raiz, limiar=0.9):
    (raiz / "configs/b2.yaml").write_text(
        f"versao: 1\nmodelo: dinov2_vitb14\nlado: 448\nlote: 8\nlimiar_caixa: {limiar}\n"
        f"partes: {raiz / 'b2_partes'}\nsaida: {raiz / 'outputs/preds'}\n", encoding="utf-8")


def test_b2_ponta_a_ponta():
    try:
        import torch
        import yaml  # noqa: F401
        import PIL  # noqa: F401
    except ImportError:
        saltar("falta o PyTorch, o PyYAML ou o Pillow")
    b2 = _carregar()
    with tempfile.TemporaryDirectory() as tmp:
        raiz = _mini_projeto(tmp)
        _config_b2(raiz)
        cfg = str(raiz / "configs/avaliacao.yaml")
        base = ["--conjunto", "CAL", "--ds", str(raiz / "DS"), "--config", cfg,
                "--config-b2", str(raiz / "configs/b2.yaml"), "--sem-progresso"]
        modelo = _ModeloFalso(torch)
        r = b2.main(base, modelo=modelo)
        assert r["janelas"] == 3 and r["feitas"] == 3 and r["calculados"] == 59 and not r["sem_caixa"]
        assert r["sem_imagem"] == ["Dataset_Treino/0102/800/804"]
        assert modelo.chamadas == 3 * (1 + 3)                               # por janela: o seed + 3 lotes de até 8
        p = {v: pd.read_csv(r["csv"][v], dtype={"caso": str}) for v in b2.VARIANTES}
        assert all(len(x) == 59 and x["score"].between(0, 1).all() for x in p.values())
        assert list(p["bruto"].columns) == ["split", "caso", "seed", "frame", "score",
                                            "box_x1", "box_y1", "box_x2", "box_y2"]
        j = p["bruto"][p["bruto"]["caso"] == "0101"].assign(o=lambda x: x["frame"] - 500)
        perto_seed, longe = j[j["o"].abs() <= 3], j[j["o"].abs() >= 5]
        assert perto_seed["score"].min() > 0.99 > longe["score"].max()      # lesão presente → score máximo
        c = p["corrigido"][p["corrigido"]["caso"] == "0101"].assign(o=lambda x: x["frame"] - 500)
        assert c.loc[c["o"].abs() <= 3, "score"].min() > c.loc[c["o"].abs() >= 5, "score"].max()
        caixas = perto_seed[["box_x1", "box_y1", "box_x2", "box_y2"]].to_numpy()
        assert all(_iou(tuple(b), CAIXA_0101) >= 0.6 for b in caixas)      # caixa no sítio certo (x e y)
        for v in ("bruto", "corrigido"):
            assert ev.main(["calibrar", "--metodo", f"B2_{v}", "--pred", str(r["csv"][v]), "--config", cfg]) == 0
        # retoma: tudo já feito; os CSV saem iguais
        de_novo = b2.main(base, modelo=modelo)
        assert de_novo["saltadas"] == 3 and de_novo["feitas"] == 0
        assert pd.read_csv(r["csv"]["bruto"], dtype={"caso": str}).equals(p["bruto"])
        # partes feitas com outra configuração não se misturam
        _config_b2(raiz, limiar=0.8)
        try:
            b2.main(base, modelo=modelo)
        except ev.ErroAvaliacao as e:
            assert "outra configuração" in str(e)
        else:
            raise AssertionError("devia ter recusado as partes de outra configuração")


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
