#!/usr/bin/env python3
"""
Extrai os vetores DINOv2 (ViT-B/14) de cada frame das janelas de um conjunto, para o B1 (Guião A, A1).

Grava um .pt por frame com a mesma estrutura dos vetores EndoFM v2:
    <pasta>/<split>/<caso>/GT_com_classes/<seed>/<caso>_<frame>_<seed>.pt
cada um um dict com as versões (768 números cada):
    cls   token CLS (o resumo da imagem que o modelo dá por omissão)
    mean · max · std   média, máximo e desvio-padrão dos 256 vetores dos quadrados de 14×14 px
Imagem: 320×320 → 224×224 (bicúbica) e normalização ImageNet, a resolução padrão do modelo.
Parâmetros e pasta de saída em configs/b1.yaml (vetores.dinov2). As imagens vêm do dataset em $DS (inicio.sh).
Retoma: as janelas já completas são saltadas; cada ficheiro é gravado de forma atómica.

Uso, a partir da pasta do projeto (na primeira vez, o torch.hub descarrega o modelo da internet):
    python methods/extrair_dinov2.py --conjunto CAL
Depois: python methods/b1_global.py --emb dinov2 --conjunto CAL
"""
import argparse
import datetime
import os
import sys
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "eval"))
sys.path.insert(0, str(RAIZ / "methods"))
import evaluate as ev  # noqa: E402
from b1_global import frames_das_janelas  # noqa: E402

VERSAO = "1.0"
MEDIA = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DESVIO = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preparar(caminho, lado):
    """Imagem → array C×H×W normalizado (ImageNet), já no tamanho do modelo."""
    from PIL import Image
    with Image.open(caminho) as im:
        im = im.convert("RGB").resize((lado, lado), Image.BICUBIC)
        a = np.asarray(im, dtype=np.float32) / 255.0
    return ((a - MEDIA) / DESVIO).transpose(2, 0, 1)


def versoes(cls, quadrados):
    """cls: B×D; quadrados: B×N×D → {versão: B×D}."""
    return {"cls": cls, "mean": quadrados.mean(axis=1), "max": quadrados.max(axis=1), "std": quadrados.std(axis=1)}


class Modelo:
    def __init__(self, nome):
        import torch
        self.torch, self.nome = torch, nome
        self.disp = "cuda" if torch.cuda.is_available() else "cpu"
        self.m = torch.hub.load("facebookresearch/dinov2", nome, trust_repo=True)   # sem pergunta interativa
        self.m.eval()
        self.m = self.m.to(self.disp)

    def __call__(self, lote):
        with self.torch.no_grad():
            x = self.torch.from_numpy(np.ascontiguousarray(np.stack(lote))).to(self.disp)
            o = self.m.forward_features(x)
            cls = o["x_norm_clstoken"].float().cpu().numpy()
            quad = o["x_norm_patchtokens"].float().cpu().numpy()
        return versoes(cls, quad)

    def pesos_sha256(self):
        try:
            d = Path(self.torch.hub.get_dir()) / "checkpoints"
            f = sorted(d.glob(f"{self.nome}*.pth"))
            return f"{f[0].name} {ev._sha256(f[0])[:16]}" if f else "desconhecido"
        except Exception:                                               # noqa: BLE001
            return "desconhecido"


def gravar(torch, vetores, destino):
    destino.parent.mkdir(parents=True, exist_ok=True)
    tmp = destino.with_name(destino.name + ".tmp")
    torch.save({k: torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32)) for k, v in vetores.items()}, tmp)
    tmp.replace(destino)


def extrair(fr, ds, pasta, modelo, lado, lote, progresso=True):
    feitas = saltadas = escritos = 0
    sem_imagem = []
    grupos = list(fr.groupby(["split", "caso", "seed"], sort=True))
    t0 = time.time()
    for i, ((sp, caso, seed), g) in enumerate(grupos, 1):
        w = Path(sp) / caso / "GT_com_classes" / str(seed)
        frames = sorted(g["frame"].tolist())
        destinos = {f: pasta / w / f"{caso}_{f}_{seed}.pt" for f in frames}
        if all(d.is_file() for d in destinos.values()):
            saltadas += 1
        else:
            pendentes = []
            for f in frames:
                img = ds / w / f"{caso}_{f}_{seed}.jpg"
                if img.is_file():
                    pendentes.append((f, img))
                else:
                    sem_imagem.append(f"{sp}/{caso}/{seed}/{f}")
            for k in range(0, len(pendentes), lote):
                parte = pendentes[k:k + lote]
                v = modelo([preparar(img, lado) for _, img in parte])
                for j, (f, _) in enumerate(parte):
                    gravar(modelo.torch, {c: v[c][j] for c in v}, destinos[f])
                    escritos += 1
            feitas += 1
        if progresso:
            dt = time.time() - t0
            print(f"\r  janelas {i}/{len(grupos)} · {escritos} frames · {dt:.0f} s"
                  + (f" · {escritos / dt:.0f} frames/s" if dt > 0 and escritos else ""), end="", flush=True)
    if progresso:
        print()
    return {"janelas": len(grupos), "feitas": feitas, "saltadas": saltadas, "escritos": escritos,
            "sem_imagem": sem_imagem, "segundos": time.time() - t0}


def main(argv=None, modelo=None):
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--conjunto", required=True, choices=("CAL", "VAL_EX", "TEST_BOX"))
    a.add_argument("--ds", default=os.environ.get("DS"), help="pasta do dataset (por omissão, $DS do inicio.sh)")
    a.add_argument("--config", default="configs/avaliacao.yaml")
    a.add_argument("--config-b1", default="configs/b1.yaml")
    a.add_argument("--sem-progresso", action="store_true")
    args = a.parse_args(argv)
    if not args.ds or not Path(args.ds).is_dir():
        raise ev.ErroAvaliacao("Falta a pasta do dataset: corre primeiro o source …/inicio.sh (define $DS) ou usa --ds.")
    cfg = ev.ler_config(args.config)
    conf = ev.ler_config(args.config_b1)["vetores"]["dinov2"]
    janelas = ev.ler_conjunto(Path(cfg["caminhos"]["conjuntos"]) / f"{args.conjunto}.json")
    fr = frames_das_janelas(cfg, janelas)
    modelo = modelo or Modelo(conf["modelo"])
    pasta = Path(conf["pasta"])
    r = extrair(fr, Path(args.ds), pasta, modelo, int(conf["lado"]), int(conf["lote"]), not args.sem_progresso)
    git = ev._estado_git(RAIZ)
    estado = ("sem git" if git["commit"] is None else
              f"commit {git['commit']}" + (" (com alterações por guardar)" if git["alteracoes_por_guardar"] else ""))
    L = [f"DINOv2 ({conf['modelo']}, {conf['lado']}×{conf['lado']}) em {args.conjunto} · "
         f"{datetime.datetime.now():%Y-%m-%d %H:%M} · {estado}",
         f"extrair_dinov2.py v{VERSAO} · torch {modelo.torch.__version__} · {modelo.disp} · pesos {modelo.pesos_sha256()}",
         f"janelas: {r['janelas']} (extraídas agora {r['feitas']}, já completas {r['saltadas']}) · frames gravados: {r['escritos']}"
         f" · {r['segundos']:.0f} s",
         f"frames sem imagem: {len(r['sem_imagem'])}" + (f" (ex.: {', '.join(r['sem_imagem'][:3])})" if r["sem_imagem"] else ""),
         f"vetores em {pasta}"]
    texto = "\n".join(L)
    saida = Path(ev.ler_config(args.config_b1)["saida"])
    saida.mkdir(parents=True, exist_ok=True)
    (saida / f"extrair_dinov2_{args.conjunto}_resumo.txt").write_text(texto + "\n", encoding="utf-8")
    print(texto)
    return r


if __name__ == "__main__":
    try:
        main()
    except ev.ErroAvaliacao as e:
        print(f"ERRO: {e}", file=sys.stderr)
        sys.exit(2)
