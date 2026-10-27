#!/usr/bin/env python3
"""
24 — Experiência B (detetor YOLO): previsões de um modelo YOLO nas janelas das 6 classes de CAL, VAL_EX e TEST_BOX, no
formato do avaliador (eval/evaluate.py), pelo desenho fixado no docs/decisoes.md a 26 out 2026.

Para cada janela (as de data/derived/yolo/conjuntos/, do script 23) e cada frame da janela (os da annotations_v1.csv):
  - o YOLO corre na imagem do frame, a 416 px, como o val.py do YOLOv5: confiança ≥ 0,001, NMS com IoU 0,6, várias
    classes por caixa (multi_label) e no máximo 300 caixas;
  - o score do frame é a confiança da caixa mais confiante da classe da janela (0 se não houver nenhuma) e a caixa do
    frame é essa (normalizada 0–1, xyxy, como na annotations_v1.csv). As caixas das outras classes não contam.
O YOLO não usa o seed: vê cada imagem sozinha. A janela só diz que frames ver e que classe procurar.

Grava (fora do git):
  outputs/preds/pred_<modelo>_<CONJUNTO>.csv   uma linha por frame com imagem: split, caso, seed, frame, distancia,
                                               score, box_x1..box_y2 e classe (a da janela)
  outputs/yolo/prever/<modelo>/configuracao.txt  a marca da geração (script, pesos e sha256, parâmetros, YOLOv5,
                                               conjuntos): o congelamento (script 25) confere-a
  outputs/yolo/prever/<modelo>/resumo.txt      contagens (só números agregados: pode colar-se no chat)
Não reescreve previsões que já existam (--forcar substitui-as; depois de calibrar, só com linha no docs/decisoes.md).

Corre no ambiente yolo (precisa do torch e do repositório do YOLOv5), a partir da pasta do projeto:
    conda activate /mnt/paulo/tese_2026/envs/yolo
    python scripts/24_yolo_prever.py --modelo YOLO_ORIG --pesos outputs/yolo/runs/YOLO_ORIG/weights/best.pt
Versão 1.0 (27 out 2026).
"""
import argparse
import datetime
import hashlib
import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "eval"))
import evaluate as ev  # noqa: E402

VERSAO = "1.0"
CONJUNTOS = ("CAL", "VAL_EX", "TEST_BOX")
K3 = ["split", "caso", "seed"]
CAIXA = ["box_x1", "box_y1", "box_x2", "box_y2"]
IMGSZ, CONF, IOU, MAX_DET = 416, 0.001, 0.6, 300


def _n(x):
    return f"{int(x):,}".replace(",", " ")


# ============================================================================ janelas e frames
def ler_conjuntos(pasta, principais):
    """{conjunto: DataFrame(split, caso, seed, classe)} a partir de <pasta>/<CONJUNTO>_6classes.csv (script 23)."""
    out = {}
    for n in CONJUNTOS:
        p = Path(pasta) / f"{n}_6classes.csv"
        if not p.is_file():
            raise ev.ErroAvaliacao(f"Não encontrei {p} (é gerado pelo scripts/23_yolo_dados.py).")
        t = pd.read_csv(p, dtype=str, keep_default_na=False)
        if not set(K3 + ["classe"]) <= set(t.columns):
            raise ev.ErroAvaliacao(f"{p.name} tem de ter as colunas split, caso, seed e classe.")
        t["caso"] = t["caso"].str.strip().str.zfill(4)
        t["seed"] = t["seed"].astype(int)
        fora = sorted(set(t["classe"]) - set(principais))
        if fora:
            raise ev.ErroAvaliacao(f"{p.name} tem classes que o YOLO não conhece: {fora}.")
        out[n] = t[K3 + ["classe"]]
    em_tb = set(zip(out["TEST_BOX"]["split"], out["TEST_BOX"]["caso"], out["TEST_BOX"]["seed"]))
    if not set(zip(out["VAL_EX"]["split"], out["VAL_EX"]["caso"], out["VAL_EX"]["seed"])) <= em_tb:
        raise ev.ErroAvaliacao("Há janelas de VAL_EX que não estão em TEST_BOX.")
    return out


def frames_das_janelas(cfg, janelas):
    """Os frames de cada janela (split, caso, seed, frame, offset), da annotations_v1.csv, lida aos bocados."""
    p = Path(cfg["caminhos"]["anotacoes"])
    if not p.is_file():
        raise ev.ErroAvaliacao(f"Não encontrei {p}.")
    alvo, casos = set(janelas), {k[1] for k in janelas}
    partes = []
    for b in pd.read_csv(p, dtype=str, keep_default_na=False, usecols=K3 + ["frame", "offset"], chunksize=200_000):
        b = b[b["caso"].str.strip().str.zfill(4).isin(casos)]
        if len(b):
            b = b.assign(caso=b["caso"].str.strip().str.zfill(4), seed=b["seed"].astype(int))
            b = b[[k in alvo for k in zip(b["split"], b["caso"], b["seed"])]]
            partes.append(b)
    fr = pd.concat(partes, ignore_index=True) if partes else pd.DataFrame(columns=K3 + ["frame", "offset"])
    fr["frame"] = fr["frame"].astype(int)
    fr["offset"] = fr["offset"].astype(int)
    faltam = alvo - set(zip(fr["split"], fr["caso"], fr["seed"]))
    if faltam:
        raise ev.ErroAvaliacao(f"{len(faltam)} janelas não estão em {p} (ex.: {ev._txt(sorted(faltam)[0])}).")
    return fr.sort_values(K3 + ["frame"]).reset_index(drop=True)


def caminho_imagem(ds, sp, caso, frame, seed):
    return Path(ds) / sp / caso / "GT_com_classes" / str(seed) / f"{caso}_{frame}_{seed}.jpg"


# ============================================================================ de deteções a previsões
def reduzir(fr, deteccoes, idx):
    """fr: frames com imagem (uma linha por imagem, com a classe da janela); deteccoes: uma matriz n×6 por linha de fr
    (x1, y1, x2, y2 normalizados 0–1, confiança, índice da classe). Devolve as previsões no formato do avaliador."""
    score = np.zeros(len(fr))
    caixa = np.full((len(fr), 4), np.nan)
    alvo = fr["classe"].map(idx).to_numpy()
    for i, (d, c) in enumerate(zip(deteccoes, alvo)):
        d = np.asarray(d, dtype=float).reshape(-1, 6)
        d = d[d[:, 5].astype(int) == c]
        if len(d):
            j = int(np.argmax(d[:, 4]))
            score[i] = float(np.clip(d[j, 4], 0, 1))
            x1, y1, x2, y2 = np.clip(d[j, :4], 0, 1)
            caixa[i] = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
    out = fr[K3 + ["frame"]].copy()
    out["distancia"] = fr["offset"].abs().to_numpy()
    out["score"] = score
    for k, col in enumerate(CAIXA):
        out[col] = caixa[:, k]
    out["classe"] = fr["classe"].to_numpy()
    return out


# ============================================================================ o YOLOv5
def tem_ultralytics():
    """Há o pacote ultralytics neste ambiente? Vê-se sem o importar."""
    try:
        return importlib.util.find_spec("ultralytics") is not None
    except (ImportError, ValueError):
        return False


class DetetorYOLOv5:
    """Corre um modelo do YOLOv5 em lotes de imagens, como o val.py (letterbox a 416, conf 0,001, NMS 0,6,
    multi_label, 300 caixas). Devolve, por imagem, uma matriz n×6 com as caixas normalizadas 0–1."""

    def __init__(self, pesos, repo, imgsz=IMGSZ, lote=32, workers=2):
        repo = Path(repo).resolve()
        if not (repo / "models" / "common.py").is_file():
            raise ev.ErroAvaliacao(f"Não encontrei o YOLOv5 em {repo}.")
        if not tem_ultralytics():
            # sem isto, o YOLOv5 faria «pip install -U ultralytics» no ambiente ativo (por exemplo, no sam_tese)
            raise ev.ErroAvaliacao("Este script corre no ambiente yolo (aqui não há o ultralytics). Faz primeiro: "
                                   "conda activate /mnt/paulo/tese_2026/envs/yolo")
        sys.path.insert(0, str(repo))
        import torch
        from models.common import DetectMultiBackend
        from utils.augmentations import letterbox
        from utils.general import non_max_suppression, scale_boxes
        from utils.torch_utils import select_device
        self.torch, self.letterbox, self.nms, self.scale = torch, letterbox, non_max_suppression, scale_boxes
        self.dev = select_device("")
        self.half = self.dev.type != "cpu"
        self.m = DetectMultiBackend(str(pesos), device=self.dev, fp16=self.half)
        self.m.eval()
        self.imgsz, self.lote, self.workers = int(imgsz), int(lote), int(workers)
        self.m.warmup(imgsz=(1, 3, self.imgsz, self.imgsz))
        nomes = self.m.names
        self.nomes = [nomes[i] for i in sorted(nomes)] if isinstance(nomes, dict) else list(nomes)
        try:
            self.commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short=8", "HEAD"], capture_output=True,
                                         text=True, timeout=20).stdout.strip() or "?"
        except Exception:                                                   # noqa: BLE001
            self.commit = "?"
        self.versao_torch = torch.__version__
        self.dispositivo = str(self.dev)

    def _ler(self, caminho):
        import cv2
        im0 = cv2.imread(str(caminho))
        if im0 is None:
            raise ev.ErroAvaliacao(f"Não consegui ler a imagem {caminho}.")
        im = self.letterbox(im0, self.imgsz, auto=False, scaleup=True)[0]
        return np.ascontiguousarray(im.transpose((2, 0, 1))[::-1]), im0.shape[:2]

    def __call__(self, caminhos, progresso=None):
        torch = self.torch
        det = self

        class _Imagens(torch.utils.data.Dataset):
            def __len__(self):
                return len(caminhos)

            def __getitem__(self, i):
                im, hw = det._ler(caminhos[i])
                return torch.from_numpy(im), torch.tensor(hw)

        dl = torch.utils.data.DataLoader(_Imagens(), batch_size=self.lote, shuffle=False, num_workers=self.workers)
        out, feitas, t0 = [], 0, time.time()
        with torch.no_grad():
            for ims, hws in dl:
                x = ims.to(self.dev)
                x = (x.half() if self.half else x.float()) / 255
                pred = self.m(x)
                pred = pred[0] if isinstance(pred, (list, tuple)) else pred
                for k, (h, w) in enumerate(hws.tolist()):
                    # NMS imagem a imagem: o do YOLOv5 tem um limite de tempo por chamada e, se o passar, deixa as
                    # imagens que faltam do lote sem caixas; uma imagem de cada vez nunca chega lá
                    d = self.nms(pred[k:k + 1], CONF, IOU, multi_label=True, max_det=MAX_DET)[0].float().cpu()
                    if len(d):
                        d[:, :4] = self.scale(x.shape[2:], d[:, :4], (h, w))
                        d[:, [0, 2]] /= w
                        d[:, [1, 3]] /= h
                    out.append(d.numpy())
                feitas += len(ims)
                if progresso:
                    progresso(feitas, len(caminhos), time.time() - t0)
        return out


def _progresso(feitas, total, dt):
    if feitas == total or feitas % 4000 < 32:
        print(f"  {_n(feitas)} / {_n(total)} imagens · {feitas / max(dt, 1e-9):.0f} imagens/s", flush=True)


# ============================================================================ principal
def main(argv=None, detetor=None):
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--modelo", required=True, help="nome do método no avaliador (YOLO_ORIG ou YOLO_V2)")
    a.add_argument("--pesos", required=True, help="pesos do treino (…/weights/best.pt)")
    a.add_argument("--config", default="configs/avaliacao.yaml")
    a.add_argument("--conjuntos", default="data/derived/yolo/conjuntos", help="pasta dos <CONJUNTO>_6classes.csv")
    a.add_argument("--ds", default=os.environ.get("DS", ""), help="pasta do Dataset WCE (por omissão, a variável DS)")
    a.add_argument("--yolov5", default="ml_models/yolov5", help="repositório do YOLOv5")
    a.add_argument("--saida-prev", default="outputs/preds")
    a.add_argument("--saida", default="outputs/yolo/prever")
    a.add_argument("--lote", type=int, default=32)
    a.add_argument("--workers", type=int, default=2)
    a.add_argument("--forcar", action="store_true", help="substitui previsões que já existam")
    args = a.parse_args(argv)
    if not args.modelo.replace("_", "").isalnum():
        raise ev.ErroAvaliacao(f"Nome de modelo inválido: {args.modelo}.")
    cfg = ev.ler_config(args.config)
    principais = list(cfg["estratos"]["classes_principais"])
    idx = {cl: i for i, cl in enumerate(principais)}
    if not args.ds:
        raise ev.ErroAvaliacao("Falta a pasta do dataset: faz export DS=/mnt/paulo/tese_2026/Update_Final/raw/Dataset_WCE "
                               "(ou passa --ds).")
    pesos = Path(args.pesos)
    if not pesos.is_file():
        raise ev.ErroAvaliacao(f"Não encontrei os pesos {pesos}.")
    prev = Path(args.saida_prev)
    destinos = {n: prev / f"pred_{args.modelo}_{n}.csv" for n in CONJUNTOS}
    ja = [str(p) for p in destinos.values() if p.exists()]
    if ja and not args.forcar:
        raise ev.ErroAvaliacao(f"Já existem previsões de {args.modelo} ({ja[0]}). Não as reescrevo: se for mesmo para as "
                               "refazer, corre com --forcar (depois de calibrar, só com linha no docs/decisoes.md).")

    # 1. janelas, frames e imagens
    conj = ler_conjuntos(args.conjuntos, principais)
    todas = pd.concat([conj["CAL"], conj["TEST_BOX"]], ignore_index=True).drop_duplicates(K3)
    fr = frames_das_janelas(cfg, list(zip(todas["split"], todas["caso"], todas["seed"])))
    fr = fr.merge(todas, on=K3, how="left", validate="many_to_one")
    fr["imagem"] = [str(caminho_imagem(args.ds, s, c, f, e)) for s, c, e, f in
                    zip(fr["split"], fr["caso"], fr["seed"], fr["frame"])]
    existe = np.array([os.path.isfile(p) for p in fr["imagem"]])
    sem = fr[~existe]
    fr = fr[existe].reset_index(drop=True)

    # 2. o YOLO, uma vez por imagem
    if detetor is None:
        detetor = DetetorYOLOv5(pesos, args.yolov5, lote=args.lote, workers=args.workers)
    if list(getattr(detetor, "nomes", principais)) != principais:
        raise ev.ErroAvaliacao(f"As classes do modelo ({list(detetor.nomes)}) não são as 6 do desenho ({principais}).")
    print(f"{args.modelo}: {_n(len(fr))} imagens de {_n(len(todas))} janelas ({_n(len(sem))} frames sem imagem)", flush=True)
    t0 = time.time()
    dets = detetor(list(fr["imagem"]), progresso=_progresso)
    dt = time.time() - t0
    if len(dets) != len(fr):
        raise ev.ErroAvaliacao("O detetor não devolveu uma resposta por imagem.")
    p = reduzir(fr, dets, idx)

    # 3. gravar: um ficheiro por conjunto e a marca
    sha_pesos = ev._sha256(pesos)
    sha_conj = hashlib.sha256("".join(ev._sha256(Path(args.conjuntos) / f"{n}_6classes.csv") for n in CONJUNTOS)
                              .encode()).hexdigest()
    marca = (f"24_yolo_prever.py v{VERSAO} · pesos {pesos.as_posix()} sha256 {sha_pesos} · imgsz {IMGSZ} · conf {CONF} · "
             f"iou {IOU} · multi_label · max_det {MAX_DET} · yolov5 {getattr(detetor, 'commit', '?')} · conjuntos sha256 "
             f"{sha_conj[:16]}")
    prev.mkdir(parents=True, exist_ok=True)
    pasta = Path(args.saida) / args.modelo
    pasta.mkdir(parents=True, exist_ok=True)
    contagens = {}
    for n in CONJUNTOS:
        js = set(zip(conj[n]["split"], conj[n]["caso"], conj[n]["seed"]))
        x = p[[k in js for k in zip(p["split"], p["caso"], p["seed"])]]
        x.to_csv(destinos[n], index=False, float_format="%.6f")
        ns = x["distancia"] != 0
        contagens[n] = {"janelas": len(js), "frames": int(ns.sum()), "com_caixa": int(x.loc[ns, "box_x1"].notna().sum()),
                        "score_ge_05": int((x.loc[ns, "score"] >= 0.5).sum()), "sha256": ev._sha256(destinos[n])}
    (pasta / "configuracao.txt").write_text(marca + "\n", encoding="utf-8")
    L = [f"24 — Experiência B: previsões de {args.modelo} · {datetime.datetime.now():%Y-%m-%d %H:%M}",
         marca,
         f"torch {getattr(detetor, 'versao_torch', '?')} · {getattr(detetor, 'dispositivo', '?')} · {_n(len(fr))} imagens "
         f"em {dt / 60:.1f} min ({len(fr) / max(dt, 1e-9):.0f} imagens/s)",
         f"frames sem imagem (sem previsão: o avaliador conta-os como ausentes): {_n(len(sem))}"
         + (f" (ex.: {sem.iloc[0]['split']}/{sem.iloc[0]['caso']}/{sem.iloc[0]['seed']}, frame {sem.iloc[0]['frame']})"
            if len(sem) else ""),
         ""]
    for n in CONJUNTOS:
        c = contagens[n]
        L.append(f"  {n:<9} {_n(c['janelas']):>4} janelas · {_n(c['frames']):>6} frames sem o seed · com caixa da classe "
                 f"{_n(c['com_caixa']):>6} · score ≥ 0,5 em {_n(c['score_ge_05']):>5} · {destinos[n].name} sha256 "
                 f"{c['sha256'][:16]}")
    L += ["", "Calibração em CAL (no sam_tese, a partir da pasta do projeto):",
          f"  python eval/evaluate.py calibrar --metodo {args.modelo} --pred {destinos['CAL'].as_posix()} "
          f"--conjunto-ficheiro {(Path(args.conjuntos) / 'CAL_6classes.csv').as_posix()}"]
    texto = "\n".join(L)
    (pasta / "resumo.txt").write_text(texto + "\n", encoding="utf-8")
    print(texto)
    return {"previsoes": destinos, "contagens": contagens, "marca": marca, "sem_imagem": len(sem), "texto": texto,
            "pasta": pasta}


if __name__ == "__main__":
    try:
        main()
    except ev.ErroAvaliacao as e:
        print(f"ERRO: {e}", file=sys.stderr)
        sys.exit(2)
