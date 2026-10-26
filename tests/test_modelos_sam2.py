"""Testes do scripts/20_modelos_sam2.py (v1.0, 26 out 2026). Dados inventados.

Os dois últimos precisam do sam2 instalado (correm no sam_tese, no CPU, com
pesos aleatórios; não usam os checkpoints verdadeiros).
"""

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("modelos_sam2", RAIZ / "scripts" / "20_modelos_sam2.py")
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)


def test_video_inventado(tmp_path):
    from PIL import Image

    caixas = m.video_inventado(tmp_path, n=5)
    nomes = sorted(p.name for p in tmp_path.glob("*.jpg"))
    assert nomes == [f"{i:05d}.jpg" for i in range(5)]
    assert len(caixas) == 5
    for i, (x1, y1, x2, y2) in enumerate(caixas):
        assert 0 <= x1 < x2 <= 319 and 0 <= y1 < y2 <= 319
        img = np.asarray(Image.open(tmp_path / f"{i:05d}.jpg")).astype(int)
        centro = img[int((y1 + y2) / 2), int((x1 + x2) / 2)]
        canto = img[5, 5]
        assert centro[0] < canto[0] - 30  # o disco é mais escuro do que o fundo
    assert caixas[1][0] > caixas[0][0]  # o disco mexe-se


def test_caixa_e_iou():
    mascara = np.zeros((20, 30), bool)
    mascara[4:10, 7:15] = True
    assert m.caixa_da_mascara(mascara) == (7.0, 4.0, 14.0, 9.0)
    assert m.caixa_da_mascara(np.zeros((5, 5), bool)) is None
    a = (0.0, 0.0, 10.0, 10.0)
    assert m.iou(a, a) == pytest.approx(1.0)
    assert m.iou(a, (20.0, 20.0, 30.0, 30.0)) == 0.0
    assert m.iou(a, (5.0, 0.0, 15.0, 10.0)) == pytest.approx(50 / 150)
    assert m.iou(None, a) == 0.0


def test_score_do_frame():
    torch = pytest.importorskip("torch")
    estado = {"output_dict_per_obj": {0: {
        "cond_frame_outputs": {0: {"object_score_logits": torch.tensor([[2.0]])}},
        "non_cond_frame_outputs": {3: {"object_score_logits": torch.tensor([[0.0]])}},
    }}}
    assert m.score_do_frame(estado, 0) == pytest.approx(1 / (1 + math.exp(-2.0)))
    assert m.score_do_frame(estado, 3) == pytest.approx(0.5)
    assert math.isnan(m.score_do_frame(estado, 7))


def _ckpt_aleatorio(tmp_path, nome_config, ficheiro):
    """Tiny com pesos aleatórios, gravado como os checkpoints do sam2."""
    torch = pytest.importorskip("torch")
    pred = m.construir_preditor("pacote", nome_config, None, "cpu")
    caminho = tmp_path / ficheiro
    torch.save({"model": pred.state_dict()}, caminho)
    return caminho


def test_config_em_ficheiro_carrega_e_repoe_o_hydra(tmp_path):
    pytest.importorskip("sam2")
    import sam2

    # um config fora do pacote, como o do MedSAM2: o do Tiny com 512 px
    original = Path(sam2.__file__).parent / "configs" / "sam2.1" / "sam2.1_hiera_t.yaml"
    txt = original.read_text()
    txt = txt.replace("feat_sizes: [64, 64]", "feat_sizes: [32, 32]").replace("image_size: 1024", "image_size: 512")
    pasta = tmp_path / "outro_repo" / "configs"
    pasta.mkdir(parents=True)
    (pasta / "tiny512.yaml").write_text(txt)

    ckpt = _ckpt_aleatorio(tmp_path, "configs/sam2.1/sam2.1_hiera_t.yaml", "tiny.pt")
    pred = m.construir_preditor("ficheiro", str(pasta / "tiny512.yaml"), ckpt, "cpu")
    assert pred.image_size == 512
    # depois, os configs do pacote continuam a funcionar
    pred2 = m.construir_preditor("pacote", "configs/sam2.1/sam2.1_hiera_t.yaml", ckpt, "cpu")
    assert pred2.image_size == 1024


def test_correr_modelo_e_falha_com_chaves_erradas(tmp_path):
    pytest.importorskip("sam2")
    torch = pytest.importorskip("torch")
    caixas = m.video_inventado(tmp_path / "video", n=3)
    ckpt = _ckpt_aleatorio(tmp_path, "configs/sam2.1/sam2.1_hiera_t.yaml", "tiny.pt")

    r = m.correr_modelo("tiny", "pacote", "configs/sam2.1/sam2.1_hiera_t.yaml", ckpt,
                        tmp_path / "video", caixas, "cpu")
    assert r["carregou"] and r["correu"], r.get("erro")
    assert r["image_size"] == 1024 and r["obj_scores"]
    assert 0.0 <= r["score_min"] <= 1.0 and 0.0 <= r["iou_mediana"] <= 1.0

    # um checkpoint com uma chave a mais não carrega, e o diagnóstico diz o que tem
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)["model"]
    sd["chave_inventada"] = torch.zeros(1)
    mau = tmp_path / "mau.pt"
    torch.save({"model": sd}, mau)
    r2 = m.correr_modelo("tiny", "pacote", "configs/sam2.1/sam2.1_hiera_t.yaml", mau,
                         tmp_path / "video", caixas, "cpu")
    assert not r2["carregou"]
    assert "chave_inventada" in r2["erro"]
    assert "'model'" in r2["diagnostico"]
