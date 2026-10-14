"""
Testes do A4, SAM 2 (methods/sam2_prop.py): RLE das máscaras, caixa da maior componente, e uma corrida completa num
mini-projeto com um preditor falso que se comporta como o do SAM 2 (pasta de frames numerados, caixa no seed,
propagação para a frente e para trás, score de objeto). Testam também as variantes de L_max, a área (area_px), a
retoma, o modo de teste e que o avaliador aceita os CSV. Precisam do PyTorch e do OpenCV; sem eles, saltam.
O SAM 2 verdadeiro não é carregado. Nunca leem dados reais.
    python tests/test_sam2.py          ou          python -m pytest tests/ -q
"""
import contextlib
import gzip
import importlib.util
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
from test_evaluate import Saltar, _projeto, ev, saltar  # noqa: E402


def _carregar():
    spec = importlib.util.spec_from_file_location("sam2_prop", AQUI.parent / "methods" / "sam2_prop.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _requisitos():
    try:
        import cv2  # noqa: F401
        import torch
        import yaml  # noqa: F401
        from PIL import Image  # noqa: F401
    except ImportError:
        saltar("falta o PyTorch, o OpenCV, o PyYAML ou o Pillow")
    return torch


def test_rle_e_caixa():
    try:
        import cv2
    except ImportError:
        saltar("falta o OpenCV")
    s2 = _carregar()
    m = np.zeros((6, 5), dtype=bool)
    m[1:3, 1:4] = True                                                      # componente maior (6 px)
    m[5, 4] = True                                                          # componente pequena (1 px)
    r = s2.rle(m)
    assert r["size"] == [6, 5] and sum(r["counts"]) == 30 and np.array_equal(s2.rle_para_mascara(r), m)
    assert s2.rle(np.ones((2, 2), bool))["counts"] == [0, 4]                # começa pelos zeros, mesmo que sejam 0
    assert s2.caixa_maior_componente(cv2, m) == (0.2, round(1 / 6, 6), 0.8, 0.5)
    assert s2.caixa_maior_componente(cv2, np.zeros((4, 4), bool)) is None
    assert abs(s2.sigmoide(0) - 0.5) < 1e-12 and abs(s2.sigmoide(-800) - 0.0) < 1e-12 and s2.sigmoide(800) == 1.0


class _PreditorFalso:
    """Faz de SAM 2: lê a pasta de frames numerados; a máscara é a caixa do seed deslocada 2 px por frame para a
    direita (visível até 6 frames do seed); o score de objeto é 4 − distância (sigmoide alta perto, baixa longe)."""

    def __init__(self, torch):
        self.torch, self.lidos, self.direcoes = torch, None, []

    def init_state(self, video_path, offload_video_to_cpu, offload_state_to_cpu):
        from PIL import Image
        nomes = sorted(Path(video_path).glob("*.jpg"), key=lambda p: int(p.stem))
        assert [int(p.stem) for p in nomes] == list(range(len(nomes)))      # 0.jpg, 1.jpg, … sem buracos
        self.lidos = [p.resolve().name for p in nomes]
        with Image.open(nomes[0]) as im:
            w, h = im.size
        return {"video_height": h, "video_width": w, "num_frames": len(nomes)}

    def reset_state(self, estado):
        estado["output_dict_per_obj"] = {0: {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}}
        estado.pop("caixa", None)

    def add_new_points_or_box(self, inference_state, frame_idx, obj_id, box):
        inference_state["caixa"] = (frame_idx, [float(v) for v in box])
        inference_state["output_dict_per_obj"][0]["cond_frame_outputs"][frame_idx] = {
            "object_score_logits": self.torch.tensor([[5.0]])}

    def propagate_in_video(self, estado, start_frame_idx, max_frame_num_to_track, reverse):
        torch = self.torch
        i0, (x1, y1, x2, y2) = estado["caixa"]
        assert start_frame_idx == i0 and not estado["output_dict_per_obj"][0]["non_cond_frame_outputs"]
        self.direcoes.append("antes" if reverse else "depois")
        n, h, w = estado["num_frames"], estado["video_height"], estado["video_width"]
        if reverse:
            ordem = range(i0, max(i0 - max_frame_num_to_track, 0) - 1, -1)
        else:
            ordem = range(i0, min(i0 + max_frame_num_to_track, n - 1) + 1)
        for fi in ordem:
            d = fi - i0
            m = torch.full((1, 1, h, w), -10.0)
            if abs(d) <= 6:
                m[0, 0, int(y1):int(y2), int(x1) + 2 * d:int(x2) + 2 * d] = 10.0
            if fi != i0:
                estado["output_dict_per_obj"][0]["non_cond_frame_outputs"][fi] = {
                    "object_score_logits": torch.tensor([[4.0 - abs(d)]])}
            yield fi, [1], m


class _ModeloFalso:
    def __init__(self, torch):
        self.torch, self.disp, self.p = torch, "cpu", _PreditorFalso(torch)
        self.info = {"codigo": "falso", "checkpoint": "falso", "precisao": "float32"}

    def contexto(self):
        return contextlib.nullcontext()


def _mini_projeto(tmp):
    """Mini-projeto do avaliador (caixa do seed 0,30–0,50 em 320 px = 96–160); imagens lisas (o preditor falso não
    olha para elas); falta a imagem 0102/800/804."""
    from PIL import Image
    raiz, conj, _ = _projeto(tmp)
    for sp, caso, seed, _ in conj["CAL"]:
        pasta = raiz / "DS" / sp / caso / "GT_com_classes" / str(seed)
        pasta.mkdir(parents=True)
        for o in range(-10, 11):
            if (caso, o) != ("0102", 4):
                Image.new("RGB", (320, 320), (100 + 5 * o, 80, 60)).save(pasta / f"{caso}_{seed + o}_{seed}.jpg")
    (raiz / "configs/sam2.yaml").write_text(
        "versao: 1\nrepo: sem_repo\ncheckpoint: sem_checkpoint\nmodelo_cfg: x\nl_max: [3, 100]\n"
        "offload_video_to_cpu: true\noffload_state_to_cpu: false\n"
        f"partes: {raiz / 'partes'}\nmascaras: {raiz / 'masks'}\nsaida: {raiz / 'outputs/preds'}\n", encoding="utf-8")
    return raiz


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def test_sam2_ponta_a_ponta():
    torch = _requisitos()
    s2 = _carregar()
    with tempfile.TemporaryDirectory() as tmp:
        raiz = _mini_projeto(tmp)
        cfg = str(raiz / "configs/avaliacao.yaml")
        base = ["--conjunto", "CAL", "--ds", str(raiz / "DS"), "--config", cfg,
                "--config-sam2", str(raiz / "configs/sam2.yaml"), "--sem-progresso"]
        modelo = _ModeloFalso(torch)
        # modo de teste: só a 1.ª janela, sem CSV finais
        t = s2.main(base + ["--max-janelas", "1"], modelo=modelo)
        assert t["feitas"] == 1 and not t["csv"] and t["resumo"].name == "sam2_teste_CAL_resumo.txt"
        assert modelo.p.direcoes == ["depois", "antes"]
        # corrida completa: a janela do teste é aproveitada
        r = s2.main(base, modelo=modelo)
        assert r["saltadas"] == 1 and r["feitas"] == 2 and r["sem_imagem"] == ["Dataset_Treino/0102/800/804"]
        assert r["com_saltos"] == ["Dataset_Treino/0102/800"]
        assert modelo.p.lidos[0].startswith("0103_290_")                    # frames pela ordem numérica
        p3, p100 = (pd.read_csv(r["csv"][k], dtype={"caso": str}) for k in (3, 100))
        assert list(p100.columns) == ["split", "caso", "seed", "frame", "score",
                                      "box_x1", "box_y1", "box_x2", "box_y2", "area_px"]
        assert len(p100) == len(p3) == 59 and p100["score"].between(0, 1).all()
        j = p100[p100["caso"] == "0101"].assign(o=lambda x: x["frame"] - 500).set_index("o")
        for o in (-6, -2, 1, 6):                                            # dos dois lados do seed
            assert abs(j.loc[o, "score"] - 1 / (1 + math.exp(-(4 - abs(o))))) < 1e-6
            caixa = tuple(j.loc[o, ["box_x1", "box_y1", "box_x2", "box_y2"]].astype(float))
            assert _iou(caixa, ((96 + 2 * o) / 320, 0.3, (160 + 2 * o) / 320, 0.5)) > 0.95
            assert j.loc[o, "area_px"] == 64 * 64
        assert j.loc[8, "area_px"] == 0 and pd.isna(j.loc[8, "box_x1"])     # já sem máscara
        q = p3.assign(o=lambda x: x["frame"] - x["seed"])
        assert (q.loc[q["o"].abs() > 3, "score"] == 0).all() and (q.loc[q["o"].abs() > 3, "area_px"] == 0).all()
        assert (q.loc[q["o"].abs() <= 3, "score"] > 0).all()
        mas = json.loads(gzip.decompress((raiz / "masks/CAL/Dataset_Treino__0101__500.json.gz").read_bytes()))
        m = s2.rle_para_mascara(mas["frames"]["501"])
        assert mas["tamanho"] == [320, 320] and m.sum() == 64 * 64 and m[96, 98] and not m[96, 97]
        for lm in (3, 100):
            assert ev.main(["calibrar", "--metodo", f"SAM2_L{lm}", "--pred", str(r["csv"][lm]), "--config", cfg]) == 0


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
