#!/usr/bin/env python3
"""
14 — A7: inspeção dos FP aparentes (Guião A). Prepara para classificares à mão a amostra de FP aparentes do método
principal em TEST_BOX: os frames que o método marcou como lesão (no OP-F1) e que não têm caixa do médico. A amostra
(50, semente 0) foi sorteada pelo avaliador no A7: coluna amostra = 1 de outputs/eval/TEST_BOX/<método>_fp_aparentes.csv.
Não avalia nada e não mexe nos resultados.

Para cada FP grava uma imagem com quatro quadros:
  em cima    o frame do seed com a caixa do médico (verde) | o frame do FP com a caixa do método (magenta) e o
             contorno da máscara (amarelo), se houver máscara gravada
  em baixo   as mesmas duas zonas ampliadas
e uma folha, classificacao.csv, com uma linha por FP e as colunas classificacao e nota vazias. Em classificacao
escreve um destes valores:
  lesao_nao_anotada   é a lesão do seed, num frame que o médico não anotou
  outra_lesao         é uma lesão, mas não a do seed
  fundo_mucosa        não é lesão (mucosa normal, pregas, bolhas, resíduos, reflexos)
  nao_avaliavel       a imagem não deixa decidir (desfocada, escura, tapada)
O 13_a7_tabelas.py lê a folha e resume as classes. Se a folha já tiver alguma classificação, não é reescrita (as
imagens são).
Antes de começar, confirma que as previsões são as avaliadas no A7 (sha256 igual ao registado pelo avaliador).

Uso, a partir da pasta do projeto, depois do source …/inicio.sh:
    python scripts/14_fp_inspecao.py --ds "$DS"
Grava em outputs/14_fp_inspecao/ (configs/a7.yaml): NN_<caso>_<seed>_<frame>.png e classificacao.csv.
"""
import argparse
import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "eval"))
sys.path.insert(0, str(RAIZ / "methods"))
import evaluate as ev  # noqa: E402
from b2_local import caixas_do_seed  # noqa: E402
from sam2_prop import rle_para_mascara  # noqa: E402

VERSAO = "1.0"
CAIXA = ["box_x1", "box_y1", "box_x2", "box_y2"]
CLASSES = ("lesao_nao_anotada", "outra_lesao", "fundo_mucosa", "nao_avaliavel")
VERDE, MAGENTA, AMARELO, PRETO = (0, 170, 0), (255, 0, 255), (0, 215, 255), (20, 20, 20)       # BGR (OpenCV)


def caminho_imagem(ds, sp, caso, seed, frame):
    return Path(ds) / sp / caso / "GT_com_classes" / str(seed) / f"{caso}_{frame}_{seed}.jpg"


def ler_mascara(pasta, sp, caso, seed, frame):
    """Máscara do frame (booleana, no tamanho do vídeo) ou None se não houver ficheiro ou máscara para esse frame."""
    f = Path(pasta) / f"{sp}__{caso}__{seed}.json.gz"
    if not f.is_file():
        return None
    r = json.loads(gzip.decompress(f.read_bytes()).decode("utf-8"))["frames"].get(str(int(frame)))
    return None if r is None else rle_para_mascara(r)


def _px(caixa, w, h):
    return tuple(int(round(v * s)) for v, s in zip(caixa, (w, h, w, h)))


def _desenhar(cv2, img, caixa_px, cor, mascara=None):
    out = img.copy()
    if mascara is not None and mascara.any():
        contornos = cv2.findContours(mascara.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
        cv2.drawContours(out, contornos, -1, AMARELO, 1)
    if caixa_px is not None:
        cv2.rectangle(out, caixa_px[:2], caixa_px[2:], cor, 2)
    return out


def recorte(caixa_px, w, h, minimo=48, fator=2.5):
    """Quadrado à volta da caixa: 2,5 vezes o maior lado (pelo menos 48 px), dentro da imagem. Devolve (x, y, lado)."""
    x1, y1, x2, y2 = caixa_px
    lado = int(min(max(minimo, fator * max(x2 - x1, y2 - y1)), w, h))
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return int(round(min(max(cx - lado / 2, 0), w - lado))), int(round(min(max(cy - lado / 2, 0), h - lado))), lado


def quadro_inteiro(cv2, img, caixa, cor, mascara, lado):
    """O frame inteiro reduzido a lado × lado, com a caixa (0–1) e o contorno da máscara."""
    m = None if mascara is None else cv2.resize(mascara.astype(np.uint8), (lado, lado), interpolation=cv2.INTER_NEAREST)
    return _desenhar(cv2, cv2.resize(img, (lado, lado), interpolation=cv2.INTER_AREA),
                     None if caixa is None else _px(caixa, lado, lado), cor, m)


def quadro_zoom(cv2, img, caixa, cor, mascara, lado):
    """A zona da caixa ampliada (a partir da imagem original) para lado × lado."""
    h, w = img.shape[:2]
    if caixa is None:
        return quadro_inteiro(cv2, img, None, cor, mascara, lado)
    cx = _px(caixa, w, h)
    a, b, q = recorte(cx, w, h)
    esc = lado / q
    rec = cv2.resize(img[b:b + q, a:a + q], (lado, lado), interpolation=cv2.INTER_CUBIC)
    m = None
    if mascara is not None:
        m = cv2.resize(mascara[b:b + q, a:a + q].astype(np.uint8), (lado, lado), interpolation=cv2.INTER_NEAREST)
    return _desenhar(cv2, rec, tuple(int(round((v - o) * esc)) for v, o in zip(cx, (a, b, a, b))), cor, m)


def _cabecalho(cv2, largura, texto, alto=26):
    c = np.full((alto, largura, 3), 255, np.uint8)
    cv2.putText(c, texto, (6, alto - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, PRETO, 1, cv2.LINE_AA)
    return c


def montar(cv2, seed_img, seed_cx, fp_img, fp_cx, mascara, titulo, rotulos, lado=400):
    """Imagem final: título; seed | FP; zoom do seed | zoom do FP (texto em ASCII: as fontes do OpenCV não têm
    acentos)."""
    if mascara is not None and mascara.shape != fp_img.shape[:2]:
        mascara = cv2.resize(mascara.astype(np.uint8), (fp_img.shape[1], fp_img.shape[0]),
                             interpolation=cv2.INTER_NEAREST).astype(bool)
    q = [quadro_inteiro(cv2, seed_img, seed_cx, VERDE, None, lado),
         quadro_inteiro(cv2, fp_img, fp_cx, MAGENTA, mascara, lado),
         quadro_zoom(cv2, seed_img, seed_cx, VERDE, None, lado),
         quadro_zoom(cv2, fp_img, fp_cx, MAGENTA, mascara, lado)]
    q = [np.vstack([_cabecalho(cv2, lado, r), x]) for r, x in zip(rotulos, q)]
    sep_v = np.full((q[0].shape[0], 6, 3), 255, np.uint8)
    linhas = [np.hstack([q[0], sep_v, q[1]]), np.hstack([q[2], sep_v, q[3]])]
    largura = linhas[0].shape[1]
    return np.vstack([_cabecalho(cv2, largura, titulo, 30), linhas[0], np.full((6, largura, 3), 255, np.uint8), linhas[1]])


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--ds", default=os.environ.get("DS"), help="pasta do dataset (por omissão, $DS)")
    a.add_argument("--config", default="configs/avaliacao.yaml")
    a.add_argument("--config-a7", default="configs/a7.yaml")
    a.add_argument("--lado", type=int, default=400, help="lado de cada quadro, em píxeis")
    args = a.parse_args(argv)
    if not args.ds or not Path(args.ds).is_dir():
        raise ev.ErroAvaliacao('Falta a pasta do dataset: corre primeiro o source …/inicio.sh e usa --ds "$DS".')
    import cv2
    cfg, a7 = ev.ler_config(args.config), ev.ler_config(args.config_a7)
    conf = a7["fp_inspecao"]
    metodo = conf["metodo"]
    base = Path(cfg["caminhos"]["saida"]) / "TEST_BOX"
    f_fp, f_res = base / f"{metodo}_fp_aparentes.csv", base / f"{metodo}.json"
    for f in (f_fp, f_res):
        if not f.is_file():
            raise ev.ErroAvaliacao(f"Não encontrei {f}: o {metodo} tem de estar avaliado em TEST_BOX (A7).")
    prev = Path(conf["previsoes"])
    if not prev.is_file():
        raise ev.ErroAvaliacao(f"Não encontrei as previsões {prev} (configs/a7.yaml).")
    registado = json.loads(f_res.read_text(encoding="utf-8")).get("registo", {}).get("previsoes_sha256")
    if registado and ev._sha256(prev) != registado:
        raise ev.ErroAvaliacao(f"{prev} não é o ficheiro avaliado no A7 (o sha256 não bate com o de {f_res}).")

    fp = pd.read_csv(f_fp, dtype={"caso": str, "split": str})
    fp["caso"] = fp["caso"].str.zfill(4)
    am = fp[fp["amostra"] == 1].reset_index(drop=True)
    if am.empty:
        raise ev.ErroAvaliacao(f"{f_fp} não tem FP sorteados (coluna amostra = 1).")
    p = pd.read_csv(prev, dtype={"caso": str, "split": str}, keep_default_na=False)
    p["caso"] = p["caso"].str.zfill(4)
    caixas_prev = {}
    for r in p.merge(am[["caso", "seed", "frame"]], on=["caso", "seed", "frame"]).itertuples(index=False):
        v = [getattr(r, c) for c in CAIXA]
        caixas_prev[(r.caso, int(r.seed), int(r.frame))] = None if any(x == "" for x in v) else tuple(float(x) for x in v)
    janelas = sorted({(r.split, r.caso, int(r.seed)) for r in am.itertuples(index=False)})
    seeds = caixas_do_seed(cfg, janelas)

    saida = Path(conf["saida"])
    saida.mkdir(parents=True, exist_ok=True)
    linhas, sem_imagem, sem_caixa, com_mascara = [], [], 0, 0
    for n, r in enumerate(am.itertuples(index=False), 1):
        sp, caso, seed, frame = r.split, r.caso, int(r.seed), int(r.frame)
        nome = f"{n:02d}_{caso}_{seed}_{frame}.png"
        img_s, img_f = (cv2.imread(str(caminho_imagem(args.ds, sp, caso, seed, x))) for x in (seed, frame))
        if img_s is None or img_f is None:
            sem_imagem.append(f"{sp}/{caso}/{seed}/{frame}")
            nome = ""
        else:
            cx = caixas_prev.get((caso, seed, frame))
            sem_caixa += cx is None
            m = ler_mascara(conf["mascaras"], sp, caso, seed, frame)
            com_mascara += m is not None
            titulo = f"{n:02d}/{len(am)}  {sp}  caso {caso}  seed {seed}  frame {frame} ({int(r.offset):+d})"
            rot = ["seed: caixa do medico", f"FP: frame {int(r.offset):+d}, score {float(r.score):.2f}"
                   + ("" if cx is not None else " (sem caixa)"), "seed (zoom)", "FP (zoom)"]
            cv2.imwrite(str(saida / nome), montar(cv2, img_s, seeds.get((sp, caso, seed)), img_f, cx, m, titulo, rot,
                                                  args.lado))
        linhas.append({"n": n, "imagem": nome, "split": sp, "caso": caso, "seed": seed, "frame": frame,
                       "offset": int(r.offset), "score": float(r.score), "classificacao": "", "nota": ""})

    folha = saida / "classificacao.csv"
    ja = (folha.is_file() and pd.read_csv(folha, dtype=str, keep_default_na=False).get(
        "classificacao", pd.Series(dtype=str)).str.strip().ne("").any())
    if not ja:
        pd.DataFrame(linhas).to_csv(folha, index=False)
    L = [f"14 — inspeção dos FP aparentes · 14_fp_inspecao.py v{VERSAO} · {metodo} em TEST_BOX (OP-F1)",
         f"FP aparentes: {len(fp)} no total · sorteados para inspeção: {len(am)} (em {len(janelas)} janelas)",
         f"imagens gravadas: {len(am) - len(sem_imagem)} em {saida}/ · sem imagem: {len(sem_imagem)}"
         + (f" (ex.: {', '.join(sem_imagem[:3])})" if sem_imagem else "")
         + f" · sem caixa prevista: {sem_caixa} · com contorno da máscara: {com_mascara}",
         (f"folha: {folha} já tem classificações: não foi reescrita." if ja else
          f"folha: {folha} (colunas classificacao e nota por preencher)."),
         "",
         "Em classificacao, um destes valores por linha:",
         "  lesao_nao_anotada   é a lesão do seed, num frame que o médico não anotou",
         "  outra_lesao         é uma lesão, mas não a do seed",
         "  fundo_mucosa        não é lesão (mucosa normal, pregas, bolhas, resíduos, reflexos)",
         "  nao_avaliavel       a imagem não deixa decidir (desfocada, escura, tapada)",
         "Depois: python scripts/13_a7_tabelas.py (resume as classes)."]
    print("\n".join(L))
    return {"folha": folha, "imagens": [saida / x["imagem"] for x in linhas if x["imagem"]], "sem_imagem": sem_imagem,
            "reescrita": not ja}


if __name__ == "__main__":
    try:
        main()
    except ev.ErroAvaliacao as e:
        print(f"ERRO: {e}", file=sys.stderr)
        sys.exit(2)
