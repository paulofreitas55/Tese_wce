#!/usr/bin/env python3
"""
A3 — B3, tracker clássico (Guião A). Segue a caixa do seed com o tracker CSRT do OpenCV, para a frente e para trás.
Usa só a caixa do próprio seed, que é a entrada do método; as caixas dos outros frames (o alvo) nunca são lidas.

Para cada janela (frames pela ordem numérica de annotations_v1.csv):
  1. Inicializa o CSRT no seed, com a caixa do seed.
  2. Segue para a frente, frame a frame, até o tracker dizer que falhou, a caixa sair da imagem, faltar uma imagem ou
     chegar a L_max frames; depois, com um tracker novo, faz o mesmo para trás (frames por ordem inversa).
  3. Em cada frame seguido: score = correlação normalizada (NCC) entre o recorte do seed e o recorte atual, os dois
     reduzidos a 32×32 px e a cores, cortada a [0, 1]; caixa = a do tracker, em 0–1. Nos frames não seguidos: score 0
     e sem caixa. Presença = tracker ativo e NCC ≥ τ, com o τ calibrado em CAL como nos outros métodos.
  4. L_max: segue até ao maior valor de configs/b3.yaml e grava uma variante por valor, cortando aí (o tracker não
     depende dos frames seguintes, por isso é o mesmo que correr com cada L_max): B3_csrt_L12, B3_csrt_L29,
     B3_csrt_L100. O protocolo começa em 12; os três comparam-se em CAL e o escolhido fica fixo antes do A7.
Regista onde cada seguimento parou e porquê, por distância ao seed (o guião pede as falhas por distância).
Parâmetros em configs/b3.yaml. Corre em poucos minutos, por isso não tem retoma.

Uso, a partir da pasta do projeto, depois do source …/inicio.sh:
    python methods/b3_csrt.py --conjunto CAL --ds "$DS"
Gera em outputs/preds/: pred_B3_csrt_L<L>_<CONJUNTO>.csv (split, caso, seed, frame, score, box_x1 … box_y2),
b3_paragens_<CONJUNTO>.csv (onde parou cada seguimento) e pred_B3_<CONJUNTO>_resumo.txt. Depois, por variante:
    python eval/evaluate.py calibrar --metodo B3_csrt_L<L> --pred outputs/preds/pred_B3_csrt_L<L>_CAL.csv
VAL_EX e TEST_BOX só no A7.
"""
import argparse
import datetime
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "eval"))
sys.path.insert(0, str(RAIZ / "methods"))
import evaluate as ev  # noqa: E402
from b1_global import frames_das_janelas  # noqa: E402
from b2_local import caixas_do_seed  # noqa: E402

VERSAO = "1.0"
CAIXA = ["box_x1", "box_y1", "box_x2", "box_y2"]
DISTANCIAS = (("1-3", 1, 3), ("4-6", 4, 6), ("7-12", 7, 12), (">12", 13, 10 ** 9))


# ============================================================================ peças
def criar_csrt(cv2):
    """Um tracker CSRT novo, pela API que o OpenCV deste ambiente tiver."""
    for f in (getattr(getattr(cv2, "TrackerCSRT", None), "create", None), getattr(cv2, "TrackerCSRT_create", None),
              getattr(getattr(cv2, "legacy", None), "TrackerCSRT_create", None)):
        if f is not None:
            return f()
    raise ev.ErroAvaliacao("Este OpenCV não tem o tracker CSRT: instala o opencv-contrib-python-headless.")


def caixa_px(caixa, w, h):
    """Caixa 0–1 (x1, y1, x2, y2) → retângulo do OpenCV (x, y, largura, altura) em píxeis, com pelo menos 1 px."""
    x1, y1, x2, y2 = caixa
    x, y = int(round(x1 * w)), int(round(y1 * h))
    return x, y, max(1, int(round(x2 * w)) - x), max(1, int(round(y2 * h)) - y)


def ret_para_caixa(ret, w, h):
    """Retângulo do OpenCV em píxeis → caixa 0–1, cortada aos limites da imagem."""
    x, y, rw, rh = (float(v) for v in ret)

    def c(v):
        return round(min(1.0, max(0.0, v)), 6)
    return c(x / w), c(y / h), c((x + rw) / w), c((y + rh) / h)


def recorte(cv2, img, ret, lado):
    """O recorte do retângulo, cortado aos limites da imagem e reduzido a lado×lado; None se ficar vazio."""
    altura, largura = img.shape[:2]
    x, y, w, h = (float(v) for v in ret)
    xa, ya = max(0, int(round(x))), max(0, int(round(y)))
    xb, yb = min(largura, int(round(x + w))), min(altura, int(round(y + h)))
    if xb - xa < 1 or yb - ya < 1:
        return None
    return cv2.resize(img[ya:yb, xa:xb], (lado, lado), interpolation=cv2.INTER_LINEAR)


def ncc(a, b):
    """Correlação normalizada entre dois recortes do mesmo tamanho, a cores: de −1 a 1; 0 se um deles for liso."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    a = a - a.mean(axis=(0, 1))
    b = b - b.mean(axis=(0, 1))
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else 0.0


def seguir(cv2, imagens, ret0, lado):
    """imagens: por ordem de seguimento, a primeira é o seed (None = imagem em falta). Segue até ao fim da lista ou
    até parar. Devolve ([(i, score, caixa 0–1)] dos frames seguidos, motivo): motivo ∈ falha · fora · sem_imagem · fim."""
    seed = imagens[0]
    h, w = seed.shape[:2]
    ref = recorte(cv2, seed, ret0, lado)
    t = criar_csrt(cv2)
    t.init(seed, tuple(int(v) for v in ret0))
    seguidos = []
    for i in range(1, len(imagens)):
        if imagens[i] is None:
            return seguidos, "sem_imagem"
        ok, ret = t.update(imagens[i])
        if not ok:
            return seguidos, "falha"
        atual = recorte(cv2, imagens[i], ret, lado)
        if atual is None:
            return seguidos, "fora"
        seguidos.append((i, round(max(0.0, min(1.0, ncc(ref, atual))), 6), ret_para_caixa(ret, w, h)))
    return seguidos, "fim"


def processar_janela(cv2, frames, caminhos, i_seed, caixa, lado, l_max):
    """frames/caminhos: toda a janela por ordem numérica; i_seed: posição do seed. Segue até l_max frames em cada
    direção. Devolve ({frame: (distância, score, caixa)}, [(direção, frames seguidos, motivo)])."""
    def ler(p):
        return cv2.imread(str(p), cv2.IMREAD_COLOR) if p.is_file() else None
    seed = ler(caminhos[i_seed])
    altura, largura = seed.shape[:2]
    ret0 = caixa_px(caixa, largura, altura)
    res, paragens = {}, []
    for direcao, passo in (("depois", 1), ("antes", -1)):
        idx = list(range(i_seed, len(frames))) if passo == 1 else list(range(i_seed, -1, -1))
        cortado = len(idx) > l_max + 1
        idx = idx[:l_max + 1]
        seguidos, motivo = seguir(cv2, [seed] + [ler(caminhos[k]) for k in idx[1:]], ret0, lado)
        if motivo == "fim" and cortado:
            motivo = "l_max"
        for i, sc, cx in seguidos:
            res[frames[idx[i]]] = (i, sc, cx)
        paragens.append((direcao, seguidos[-1][0] if seguidos else 0, motivo))
    return res, paragens


# ============================================================================ janelas e resumo
def _percentis(s, ps=(5, 25, 50, 75, 95, 99)):
    return " · ".join(f"P{p} {v:.3f}" for p, v in zip(ps, np.percentile(s, ps))) if len(s) else "sem valores"


def _exemplos(lista):
    return f" (ex.: {', '.join(lista[:3])})" if lista else ""


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--conjunto", required=True, choices=("CAL", "VAL_EX", "TEST_BOX"))
    a.add_argument("--ds", default=os.environ.get("DS"), help="pasta do dataset (por omissão, $DS)")
    a.add_argument("--config", default="configs/avaliacao.yaml")
    a.add_argument("--config-b3", default="configs/b3.yaml")
    a.add_argument("--sem-progresso", action="store_true")
    args = a.parse_args(argv)
    if not args.ds or not Path(args.ds).is_dir():
        raise ev.ErroAvaliacao('Falta a pasta do dataset: corre primeiro o source …/inicio.sh e usa --ds "$DS".')
    try:
        import cv2
    except ImportError:
        raise ev.ErroAvaliacao("Falta o OpenCV neste ambiente (opencv-contrib-python-headless).")
    criar_csrt(cv2)                                                         # pára já se não houver CSRT
    cfg = ev.ler_config(args.config)
    conf = ev.ler_config(args.config_b3)
    l_maxs, lado = sorted({int(x) for x in conf["l_max"]}), int(conf["ncc_lado"])
    janelas = ev.ler_conjunto(Path(cfg["caminhos"]["conjuntos"]) / f"{args.conjunto}.json")
    fr = frames_das_janelas(cfg, janelas)
    caixas = caixas_do_seed(cfg, janelas)
    ds, t0 = Path(args.ds), time.time()
    linhas = {lm: [] for lm in l_maxs}
    paragens, sem_caixa, sem_seed = [], [], []
    grupos = list(fr.groupby(["split", "caso", "seed"], sort=True))
    n_frames = 0
    for n, ((sp, caso, seed), g) in enumerate(grupos, 1):
        w = ds / sp / caso / "GT_com_classes" / str(seed)
        frames = sorted(int(f) for f in g["frame"])
        if (sp, caso, seed) not in caixas:
            sem_caixa.append(f"{sp}/{caso}/{seed}")
        elif seed not in frames or not (w / f"{caso}_{seed}_{seed}.jpg").is_file():
            sem_seed.append(f"{sp}/{caso}/{seed}")
        else:
            caminhos = [w / f"{caso}_{f}_{seed}.jpg" for f in frames]
            res, par = processar_janela(cv2, frames, caminhos, frames.index(seed), caixas[(sp, caso, seed)], lado,
                                        max(l_maxs))
            paragens += [{"split": sp, "caso": caso, "seed": int(seed), "direcao": d, "frames_seguidos": k,
                          "motivo": m} for d, k, m in par]
            for f in frames:
                if f == seed:
                    continue
                n_frames += 1
                dist, sc, cx = res.get(f, (None, 0.0, None))
                for lm in l_maxs:
                    dentro = dist is not None and dist <= lm
                    linhas[lm].append({"split": sp, "caso": caso, "seed": int(seed), "frame": f,
                                       "score": sc if dentro else 0.0,
                                       **{c: (v if dentro else "") for c, v in zip(CAIXA, cx or ("",) * 4)}})
        if not args.sem_progresso:
            dt = time.time() - t0
            print(f"\r  janelas {n}/{len(grupos)} · {dt:.0f} s", end="", flush=True)
    if not args.sem_progresso:
        print()
    if not paragens:
        raise ev.ErroAvaliacao("Nenhuma janela do conjunto ficou com previsões (ver caixas do seed e imagens).")
    segundos = time.time() - t0

    saida = Path(conf["saida"])
    saida.mkdir(parents=True, exist_ok=True)
    csv = {}
    for lm in l_maxs:
        csv[lm] = saida / f"pred_B3_csrt_L{lm}_{args.conjunto}.csv"
        pd.DataFrame(linhas[lm], columns=["split", "caso", "seed", "frame", "score"] + CAIXA).to_csv(csv[lm], index=False)
    par = pd.DataFrame(paragens)
    f_par = saida / f"b3_paragens_{args.conjunto}.csv"
    par.to_csv(f_par, index=False)

    # resumo
    git = ev._estado_git(RAIZ)
    estado = ("sem git" if git["commit"] is None else
              f"commit {git['commit']}" + (" (com alterações por guardar)" if git["alteracoes_por_guardar"] else ""))
    maior = pd.DataFrame(linhas[max(l_maxs)])
    seguidos = maior[maior["box_x1"] != ""]
    area = ((seguidos["box_x2"].astype(float) - seguidos["box_x1"].astype(float))
            * (seguidos["box_y2"].astype(float) - seguidos["box_y1"].astype(float)) * 100).to_numpy()
    parou = par[par["motivo"].isin(["falha", "fora", "sem_imagem"])]
    dist_paragem = parou["frames_seguidos"].to_numpy() + 1                  # o frame onde deixou de seguir
    L = [f"B3 (CSRT, OpenCV {cv2.__version__}) em {args.conjunto} · {datetime.datetime.now():%Y-%m-%d %H:%M} · {estado}",
         f"b3_csrt.py v{VERSAO} · sha256 {ev._sha256(__file__)[:16]} · {Path(args.config_b3).name} sha256 "
         f"{ev._sha256(args.config_b3)[:16]} · L_max {', '.join(map(str, l_maxs))} · NCC a {lado}×{lado} px",
         "",
         f"janelas: {len(grupos)} · com previsões: {len(par) // 2} · frames sem o seed: {n_frames} · {segundos:.0f} s",
         f"janelas sem caixa do seed (ficam sem previsões): {len(sem_caixa)}{_exemplos(sem_caixa)}",
         f"janelas sem imagem do seed (ficam sem previsões): {len(sem_seed)}{_exemplos(sem_seed)}",
         "",
         f"Seguimento (até {max(l_maxs)} frames para cada lado), frames seguidos por direção:"]
    for d in ("depois", "antes"):
        k = par.loc[par["direcao"] == d, "frames_seguidos"].to_numpy()
        L.append(f"  {d:>6}: mediana {np.median(k):.0f} · P90 {np.percentile(k, 90):.0f} · máximo {k.max()}")
    L += ["Onde parou: " + " · ".join(f"{m} {int((par['motivo'] == m).sum())}"
                                      for m in ("falha", "fora", "sem_imagem", "l_max", "fim")),
          "  (falha = o tracker desistiu; fora = a caixa saiu da imagem; l_max = chegou ao limite; fim = fim da janela)",
          "Paragens antes do fim, por distância ao seed: " + " · ".join(
              f"{nome} {int(((dist_paragem >= lo) & (dist_paragem <= hi)).sum())}" for nome, lo, hi in DISTANCIAS),
          "",
          "Distribuição do score (NCC) nos frames seguidos, sem rótulos (a grelha de τ tem passo 0,02):",
          f"  {_percentis(seguidos['score'].to_numpy(dtype=float))}",
          "Frames com score > 0 por variante: " + " · ".join(
              f"L{lm} {sum(r['score'] > 0 for r in linhas[lm])}" for lm in l_maxs),
          f"caixa do tracker, área em % do frame: {_percentis(area, (5, 50, 95))}" if len(area) else
          "caixa do tracker: sem frames seguidos"]
    if args.conjunto == "CAL":
        L += ["", "Calibração, uma por variante de L_max (comparam-se em CAL):"]
        L += [f"  python eval/evaluate.py calibrar --metodo B3_csrt_L{lm} --pred {csv[lm]}" for lm in l_maxs]
    L += ["", f"3 linhas de exemplo (L{max(l_maxs)}, frames seguidos):",
          seguidos.head(3).to_string(index=False) if len(seguidos) else "  (nenhum)"]
    texto = "\n".join(L)
    f_resumo = saida / f"pred_B3_{args.conjunto}_resumo.txt"
    f_resumo.write_text(texto + "\n", encoding="utf-8")
    print(texto)
    print(f"\ngravado em {saida}/: {len(l_maxs)} CSV de previsões, {f_par.name} e {f_resumo.name}")
    return {"csv": csv, "paragens": f_par, "resumo": f_resumo, "sem_caixa": sem_caixa, "sem_seed": sem_seed,
            "n_frames": n_frames}


if __name__ == "__main__":
    try:
        main()
    except ev.ErroAvaliacao as e:
        print(f"ERRO: {e}", file=sys.stderr)
        sys.exit(2)
