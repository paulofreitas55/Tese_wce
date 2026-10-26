"""Experiência A, passo 3: os três modelos SAM 2 carregam e correm?

Versão 1.0 (26 out 2026).

Confirma que o SAM 2.1 base+, o SAM 2.1 Tiny e o MedSAM2 carregam no sam2 já
instalado (o clone em ml_models/sam2), sem chaves em falta nem a mais, e que
seguem um disco num vídeo inventado (8 frames de 320 px), com a caixa do disco
no frame 0 e propagação para a frente, como no SAM2_L12. Não lê dados reais.

O config do MedSAM2 está fora do pacote sam2 (no clone do MedSAM2). Para o
carregar sem mexer no clone do sam2, o Hydra lê-o da pasta dele durante a
construção do modelo e volta logo a seguir aos configs do pacote.

Uso (na raiz do projeto, com o sam_tese ativo):
    python scripts/20_modelos_sam2.py
    python scripts/20_modelos_sam2.py --modelos medsam2 --device cpu

Grava outputs/20_modelos_sam2_resumo.txt. Sai com código 1 se algum modelo
não carregar ou não correr.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

VERSAO = "1.0"

# nome -> (tipo do config, config, checkpoint); caminhos relativos à raiz do projeto
MODELOS = {
    "base_plus": (
        "pacote",
        "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "ml_models/sam2/checkpoints/sam2.1_hiera_base_plus.pt",
    ),
    "tiny": (
        "pacote",
        "configs/sam2.1/sam2.1_hiera_t.yaml",
        "ml_models/sam2/checkpoints/sam2.1_hiera_tiny.pt",
    ),
    "medsam2": (
        "ficheiro",
        "ml_models/MedSAM2/sam2/configs/sam2.1_hiera_t512.yaml",
        "ml_models/MedSAM2/checkpoints/MedSAM2_latest.pt",
    ),
}


# ---------------------------------------------------------------- vídeo inventado
def video_inventado(pasta: Path, n: int = 8, tam: int = 320, raio: int = 28,
                    semente: int = 0) -> list[tuple[float, float, float, float]]:
    """Grava n frames JPEG (00000.jpg, ...) com um disco escuro a mexer-se num
    fundo rosado com textura. Devolve a caixa verdadeira do disco em cada frame
    (x1, y1, x2, y2, em píxeis)."""
    from PIL import Image

    rng = np.random.default_rng(semente)
    pasta.mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[0:tam, 0:tam].astype(np.float32)
    # textura suave: soma de algumas ondas de baixa frequência
    textura = np.zeros((tam, tam), np.float32)
    for _ in range(6):
        fx, fy = rng.uniform(0.005, 0.03, size=2)
        fase = rng.uniform(0, 2 * np.pi)
        textura += np.sin(2 * np.pi * (fx * xx + fy * yy) + fase)
    textura = textura / np.abs(textura).max()
    base = np.stack([175 + 25 * textura, 95 + 15 * textura, 85 + 15 * textura], axis=-1)

    caixas = []
    for i in range(n):
        cx, cy = 90 + 6 * i, 110 + 4 * i
        img = base + rng.normal(0, 4, size=base.shape)
        dentro = (xx - cx) ** 2 + (yy - cy) ** 2 <= raio ** 2
        img[dentro] = (120, 30, 45)
        img = np.clip(img, 0, 255).astype(np.uint8)
        Image.fromarray(img).save(pasta / f"{i:05d}.jpg", quality=95)
        caixas.append((max(cx - raio, 0.0), max(cy - raio, 0.0),
                       min(cx + raio, tam - 1.0), min(cy + raio, tam - 1.0)))
    return caixas


def caixa_da_mascara(mascara: np.ndarray):
    """Caixa (x1, y1, x2, y2) dos píxeis a True, ou None se a máscara está vazia."""
    ys, xs = np.nonzero(mascara)
    if len(xs) == 0:
        return None
    return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))


def iou(a, b) -> float:
    """IoU de duas caixas xyxy (0 se alguma for None)."""
    if a is None or b is None:
        return 0.0
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    uniao = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / uniao if uniao > 0 else 0.0


def score_do_frame(estado: dict, frame: int, obj_idx: int = 0) -> float:
    """Sigmoide do object_score_logits guardado no estado do preditor para um
    frame (o do prompt ou um propagado); nan se não houver."""
    import torch

    saidas = estado.get("output_dict_per_obj", {}).get(obj_idx, {})
    for chave in ("cond_frame_outputs", "non_cond_frame_outputs"):
        out = saidas.get(chave, {}).get(frame)
        if out is not None and out.get("object_score_logits") is not None:
            return float(torch.sigmoid(out["object_score_logits"].float()).flatten()[0])
    return float("nan")


# ---------------------------------------------------------------- modelos
def construir_preditor(tipo: str, config: str, checkpoint, device: str):
    """Constrói o SAM2VideoPredictor do sam2 instalado.

    tipo "pacote": config é um nome dentro do pacote sam2
    (ex.: configs/sam2.1/sam2.1_hiera_t.yaml).
    tipo "ficheiro": config é o caminho de um .yaml fora do pacote (ex.: o do
    MedSAM2); o Hydra lê dessa pasta só durante a construção."""
    from hydra import initialize_config_dir, initialize_config_module
    from hydra.core.global_hydra import GlobalHydra

    import sam2  # noqa: F401  (inicializa o Hydra com os configs do pacote)
    from sam2.build_sam import build_sam2_video_predictor

    ckpt = None if checkpoint is None else str(checkpoint)
    if tipo == "pacote":
        return build_sam2_video_predictor(config, ckpt, device=device)
    if tipo != "ficheiro":
        raise ValueError(f"tipo de config desconhecido: {tipo}")
    caminho = Path(config).resolve()
    if not caminho.is_file():
        raise FileNotFoundError(f"config não encontrado: {caminho}")
    GlobalHydra.instance().clear()
    initialize_config_dir(config_dir=str(caminho.parent), version_base="1.2")
    try:
        return build_sam2_video_predictor(caminho.name, ckpt, device=device)
    finally:
        GlobalHydra.instance().clear()
        initialize_config_module("sam2", version_base="1.2")


def sha256(caminho: Path) -> str:
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def diagnosticar(checkpoint: Path) -> str:
    """Quando um modelo não carrega: o que há no topo do checkpoint."""
    import torch

    try:
        ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except Exception as e:  # noqa: BLE001
        return f"o checkpoint nem abre com weights_only=True: {type(e).__name__}: {e}"
    if isinstance(ck, dict):
        chaves = list(ck.keys())
        txt = f"chaves de topo do checkpoint: {chaves[:10]}"
        if "model" in ck and isinstance(ck["model"], dict):
            txt += f"; 'model' tem {len(ck['model'])} tensores"
        return txt
    return f"o checkpoint não é um dicionário: {type(ck).__name__}"


def correr_modelo(nome: str, tipo: str, config: str, checkpoint: Path,
                  pasta_video: Path, caixas: list, device: str) -> dict:
    """Carrega um modelo, põe a caixa do disco no frame 0, propaga para a frente
    e mede o score e o IoU da caixa da máscara com a caixa verdadeira."""
    import torch

    r = {"modelo": nome, "carregou": False, "correu": False}
    if not checkpoint.is_file():
        r["erro"] = f"checkpoint não encontrado: {checkpoint}"
        return r
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    try:
        t0 = time.time()
        pred = construir_preditor(tipo, config, checkpoint, device)
        r["t_carregar_s"] = time.time() - t0
    except Exception as e:  # noqa: BLE001
        r["erro"] = f"{type(e).__name__}: {str(e)[:600]}"
        r["diagnostico"] = diagnosticar(checkpoint)
        return r
    r["carregou"] = True
    r["image_size"] = int(pred.image_size)
    r["obj_scores"] = bool(getattr(pred, "pred_obj_scores", False))
    r["parametros_M"] = sum(p.numel() for p in pred.parameters()) / 1e6

    autocast = (torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda"
                else torch.autocast("cpu", enabled=False))
    try:
        with torch.inference_mode(), autocast:
            estado = pred.init_state(video_path=str(pasta_video))
            pred.add_new_points_or_box(estado, frame_idx=0, obj_id=1,
                                       box=np.array(caixas[0], dtype=np.float32))
            scores, ious = [], []
            t0 = time.time()
            for f, _ids, logits in pred.propagate_in_video(estado):
                if f == 0:
                    continue  # o frame do prompt não conta
                mascara = (logits[0, 0] > 0).cpu().numpy()
                scores.append(score_do_frame(estado, f))
                ious.append(iou(caixa_da_mascara(mascara), caixas[f]))
            dt = time.time() - t0
    except Exception as e:  # noqa: BLE001
        r["erro"] = f"a correr: {type(e).__name__}: {str(e)[:600]}"
        r["traceback"] = traceback.format_exc()[-1500:]
        return r
    r["correu"] = True
    r["score_min"] = float(np.nanmin(scores)) if scores else float("nan")
    r["iou_mediana"] = float(np.median(ious)) if ious else float("nan")
    r["frames_s"] = len(scores) / dt if dt > 0 else float("nan")
    if device == "cuda":
        r["gpu_gb"] = torch.cuda.max_memory_allocated() / 1e9
    del pred, estado
    return r


def linha_tabela(r: dict) -> str:
    def num(chave, fmt):
        v = r.get(chave)
        return format(v, fmt) if isinstance(v, (int, float)) and not (
            isinstance(v, float) and math.isnan(v)) else "-"

    return (f"{r['modelo']:<10} {'sim' if r['carregou'] else 'NÃO':<9}"
            f"{'sim' if r['correu'] else 'não':<7}{num('image_size', 'd'):>6}  "
            f"{('sim' if r.get('obj_scores') else 'não') if r['carregou'] else '-':<10}"
            f"{num('parametros_M', '.1f'):>8}{num('score_min', '.3f'):>11}"
            f"{num('iou_mediana', '.3f'):>13}{num('frames_s', '.1f'):>10}"
            f"{num('gpu_gb', '.2f'):>9}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--modelos", nargs="+", default=list(MODELOS),
                    choices=list(MODELOS))
    ap.add_argument("--raiz", default=".", help="raiz do projeto (por omissão, a pasta atual)")
    ap.add_argument("--device", default=None, help="cuda ou cpu (por omissão, cuda se houver)")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--saida", default="outputs/20_modelos_sam2_resumo.txt")
    a = ap.parse_args(argv)

    import torch

    import sam2

    raiz = Path(a.raiz).resolve()
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else "-"

    linhas = [
        f"20_modelos_sam2.py v{VERSAO} · {datetime.now():%Y-%m-%d %H:%M}",
        f"torch {torch.__version__} · device {device} ({gpu})",
        f"sam2 em uso: {Path(sam2.__file__).parent}",
        f"vídeo inventado: {a.frames} frames de 320 px, um disco a mexer-se; "
        "caixa no frame 0; propagação para a frente",
        "",
    ]
    for l in linhas:
        print(l)

    resultados = []
    with tempfile.TemporaryDirectory() as tmp:
        pasta = Path(tmp) / "video"
        caixas = video_inventado(pasta, n=a.frames)
        for nome in a.modelos:
            tipo, config, ckpt = MODELOS[nome]
            config_usado = str(raiz / config) if tipo == "ficheiro" else config
            print(f"-> {nome}: a carregar ...", flush=True)
            r = correr_modelo(nome, tipo, config_usado, raiz / ckpt, pasta, caixas, device)
            if (raiz / ckpt).is_file():
                r["sha256_ckpt"] = sha256(raiz / ckpt)
            if tipo == "ficheiro" and Path(config_usado).is_file():
                r["sha256_config"] = sha256(Path(config_usado))
            resultados.append(r)
            if not r["carregou"] or not r["correu"]:
                print(f"   ERRO: {r.get('erro')}")
                if r.get("diagnostico"):
                    print(f"   {r['diagnostico']}")

    cab = (f"{'modelo':<10} {'carregou':<9}{'correu':<7}{'img':>6}  {'obj_score':<10}"
           f"{'parâm(M)':>8}{'score_mín':>11}{'IoU_mediana':>13}{'frames/s':>10}{'GPU(GB)':>9}")
    corpo = [cab] + [linha_tabela(r) for r in resultados]
    extra = []
    for r in resultados:
        for chave in ("sha256_ckpt", "sha256_config", "erro", "diagnostico"):
            if r.get(chave):
                extra.append(f"{r['modelo']}: {chave} {r[chave]}")
    falhou = [r["modelo"] for r in resultados if not (r["carregou"] and r["correu"])]
    fim = [("TODOS OK: carregam sem chaves em falta nem a mais e correm."
            if not falhou else f"FALHOU: {', '.join(falhou)}")]

    print()
    for l in corpo + [""] + extra + [""] + fim:
        print(l)

    saida = raiz / a.saida
    saida.parent.mkdir(parents=True, exist_ok=True)
    saida.write_text("\n".join(linhas + corpo + [""] + extra + [""] + fim) + "\n",
                     encoding="utf-8")
    print(f"\nresumo gravado em {saida}")
    return 1 if falhou else 0


if __name__ == "__main__":
    sys.exit(main())
