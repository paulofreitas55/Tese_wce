#!/usr/bin/env python3
"""
A4/A5 — Propagação com SAM 2 (Guião A). Dá ao SAM 2 o prompt do seed e deixa-o seguir a lesão pela janela, com a
memória dos frames anteriores. Usa só a caixa do próprio seed (a entrada do método); as caixas dos outros frames não
são lidas, exceto no V5.

Prompt (--prompt):
  caixa         a caixa do seed (método base: SAM2_L<L>)
  ponto         V1: um ponto positivo no centro da caixa (SAM2_ponto_L<L>)
  ponto_caixa   V2: a caixa e o ponto do centro (SAM2_ponto_caixa_L<L>)
  multi         V5: a caixa do seed e, como prompts noutros frames, todas as caixas originais dos vizinhos na janela
                (box_src medico_vizinho, válidas, em frames com imagem) (SAM2_multi_L<L>). Usa parte da referência
                como entrada: é só um limite superior (quanto melhoraria com mais anotação) e só corre em CAL e VAL_EX
                (em TEST_BOX as caixas seriam a própria resposta). As métricas de caixas em CAL não contam.

Para cada janela (frames pela ordem numérica de annotations_v1.csv):
  1. Cria uma pasta temporária com ligações 0.jpg, 1.jpg, … para as imagens da janela (o SAM 2 lê frames numerados);
     o seed fica no índice que tiver nessa lista.
  2. Para a frente: o prompt (em píxeis) no índice do seed e propagate_in_video até L_max frames. Para trás: o
     estado é limpo, o prompt volta a ser dado no seed e propaga-se com reverse=True. Cada direção usa só os seus
     próprios frames; por isso, cortar uma corrida longa em L_max = 12 ou 29 dá o mesmo que correr com esse L_max.
  3. Em cada frame: score = sigmoide do score de objeto do SAM 2 (a probabilidade de o objeto estar visível);
     máscara = logits > 0; área em píxeis (area_px, para o A_min do avaliador); caixa = a da maior componente da
     máscara, em 0–1. Nos frames para lá de L_max: score 0, sem caixa, área 0.
  4. Grava as máscaras em RLE (formato COCO, por colunas) em outputs/masks/<CONJUNTO>/, uma por janela.
Variantes de L_max (configs/sam2.yaml): SAM2_L12, SAM2_L29, SAM2_L100 (o protocolo começa em 12; comparam-se em CAL).
Modelo: sam2.1_hiera_base_plus (configs/sam2.yaml), em bfloat16 na GPU, como nos exemplos do SAM 2; os frames ficam na
memória do processador (offload_video_to_cpu) para poupar a da GPU. Uma janela de cada vez.
Retoma: cada janela grava um CSV em outputs/preds/sam2_partes/<CONJUNTO>/ e as janelas já feitas são saltadas. As
variantes de prompt têm partes e máscaras em pastas próprias (sam2_partes_ponto/, masks_ponto/, …).
Sem a extensão CUDA do SAM 2 (_C), o SAM 2 não preenche os buracos pequenos das máscaras e avisava em cada janela; o
aviso é silenciado e fica uma linha no resumo.

Uso, a partir da pasta do projeto, depois do source …/inicio.sh:
    python methods/sam2_prop.py --conjunto CAL --ds "$DS" --max-janelas 1     (teste: só a 1.ª janela)
    python methods/sam2_prop.py --conjunto CAL --ds "$DS"                     (caixa)
    python methods/sam2_prop.py --conjunto CAL --ds "$DS" --prompt ponto      (V1; ponto_caixa para o V2; multi, V5)
Gera em outputs/preds/: pred_<NOME>_L<L>_<CONJUNTO>.csv (NOME = SAM2, SAM2_ponto, SAM2_ponto_caixa ou SAM2_multi;
colunas split, caso, seed, frame, score, box_x1 … box_y2, area_px), no formato do avaliador, e
pred_<NOME>_<CONJUNTO>_resumo.txt.
Depois, por variante:
    python eval/evaluate.py calibrar --metodo SAM2_L<L> --pred outputs/preds/pred_SAM2_L<L>_CAL.csv
VAL_EX e TEST_BOX só no A7.
"""
import argparse
import contextlib
import datetime
import gzip
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "eval"))
sys.path.insert(0, str(RAIZ / "methods"))
import evaluate as ev  # noqa: E402
from b1_global import frames_das_janelas  # noqa: E402
from b2_local import caixas_do_seed  # noqa: E402

VERSAO = "1.2"
CAIXA = ["box_x1", "box_y1", "box_x2", "box_y2"]
PROMPTS = {"caixa": "SAM2", "ponto": "SAM2_ponto", "ponto_caixa": "SAM2_ponto_caixa",   # prompt → nome do método
           "multi": "SAM2_multi"}
SO_SEM_REFERENCIA_DE_CAIXAS = ("CAL", "VAL_EX")                             # onde o V5 (multi) pode correr


# ============================================================================ máscaras
def rle(mascara):
    """Máscara booleana H×W → RLE não comprimido do COCO: {"size": [H, W], "counts": [...]}, por colunas, a começar
    pelos zeros."""
    v = np.asarray(mascara, dtype=np.uint8).ravel(order="F")
    if not len(v):
        return {"size": list(np.shape(mascara)), "counts": []}
    mudancas = np.flatnonzero(v[1:] != v[:-1]) + 1
    counts = np.diff(np.concatenate([[0], mudancas, [len(v)]])).tolist()
    if v[0] == 1:
        counts = [0] + counts
    return {"size": [int(x) for x in np.shape(mascara)], "counts": [int(c) for c in counts]}


def rle_para_mascara(r):
    h, w = r["size"]
    v = np.zeros(h * w, dtype=bool)
    pos, valor = 0, False
    for c in r["counts"]:
        v[pos:pos + c] = valor
        pos, valor = pos + c, not valor
    return v.reshape((h, w), order="F")


def caixa_maior_componente(cv2, mascara):
    """Caixa 0–1 da maior componente ligada (vizinhança de 8) da máscara; None se a máscara estiver vazia."""
    h, w = mascara.shape
    n, _, stats, _ = cv2.connectedComponentsWithStats(mascara.astype(np.uint8), connectivity=8)
    if n <= 1:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, bw, bh = (int(stats[k, c]) for c in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH,
                                                 cv2.CC_STAT_HEIGHT))
    return round(x / w, 6), round(y / h, 6), round((x + bw) / w, 6), round((y + bh) / h, 6)


def sigmoide(x):
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def argumentos_prompt(caixa_px, prompt):
    """Prompt do seed para o add_new_points_or_box: a caixa (x1, y1, x2, y2 em píxeis), um ponto positivo no centro
    da caixa, ou os dois."""
    if prompt not in PROMPTS:
        raise ev.ErroAvaliacao(f"Prompt desconhecido: {prompt} (opções: {', '.join(PROMPTS)}).")
    x1, y1, x2, y2 = (float(v) for v in caixa_px)
    caixa = {"box": np.array([x1, y1, x2, y2], dtype=np.float32)}
    ponto = {"points": np.array([[(x1 + x2) / 2, (y1 + y2) / 2]], dtype=np.float32),
             "labels": np.array([1], dtype=np.int32)}
    return {"caixa": caixa, "ponto": ponto, "ponto_caixa": {**caixa, **ponto}, "multi": caixa}[prompt]


def caixas_vizinhas(cfg, janelas):
    """V5: {(split, caso, seed): {frame: (x1, y1, x2, y2)}} em 0–1, com as caixas originais dos outros frames da janela
    (box_src medico_vizinho), válidas pela regra do avaliador (4 coordenadas e x2 > x1, y2 > y1). As caixas copiadas de
    janelas sobrepostas (boxc_*) não entram."""
    alvo, casos, caixas = set(janelas), {k[1] for k in janelas}, {}
    for b in pd.read_csv(cfg["caminhos"]["anotacoes"], dtype=str, keep_default_na=False,
                         usecols=["split", "caso", "seed", "frame", "box_src"] + CAIXA, chunksize=200_000):
        b = b[b["caso"].isin(casos) & (b["box_src"] == "medico_vizinho")]
        for _, r in b.iterrows():
            k = (r["split"], r["caso"], int(r["seed"]))
            v = pd.to_numeric(pd.Series([r[c] for c in CAIXA]), errors="coerce")
            if k in alvo and v.notna().all() and v[2] > v[0] and v[3] > v[1]:
                caixas.setdefault(k, {})[int(r["frame"])] = tuple(float(x) for x in v)
    return caixas


# ============================================================================ modelo
def silenciar_aviso_sem_c():
    """Sem a extensão CUDA (_C), o SAM 2 avisa em cada janela que salta o preenchimento de buracos das máscaras."""
    warnings.filterwarnings("ignore", message=r"(?s).*Skipping the post-processing step", category=UserWarning)


def _commit(pasta):
    if not (Path(pasta) / ".git").exists():
        return "sem git"
    try:
        r = subprocess.run(["git", "-C", str(pasta), "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=10)
        return f"commit {r.stdout.strip()}" if r.returncode == 0 and r.stdout.strip() else "sem git"
    except Exception:                                                          # noqa: BLE001
        return "sem git"


class ModeloSAM2:
    """O preditor de vídeo do SAM 2 (pacote sam2 instalado a partir de ml_models/sam2), com o checkpoint do yaml."""

    def __init__(self, conf):
        import torch
        self.torch = torch
        self.disp = "cuda" if torch.cuda.is_available() else "cpu"
        ck = Path(conf["checkpoint"])
        if not ck.is_file():
            raise ev.ErroAvaliacao(f"Não encontrei o checkpoint {ck} (configs/sam2.yaml).")
        try:
            import sam2
            import sam2.sam2_video_predictor as preditor_video
            import sam2.utils.misc as misc
            from sam2.build_sam import build_sam2_video_predictor
        except ImportError:
            raise ev.ErroAvaliacao("Falta o pacote sam2 neste ambiente (ver a instalação em ml_models/sam2).")

        def sem_barra(iteravel=None, *a, **k):                              # as barras internas do SAM 2 enchiam o log
            return iteravel
        preditor_video.tqdm = misc.tqdm = sem_barra
        silenciar_aviso_sem_c()
        try:
            from sam2 import _C  # noqa: F401
            extensao = "com a extensão _C do SAM 2"
        except ImportError:
            extensao = "sem a extensão _C: o SAM 2 não preenche os buracos pequenos das máscaras (aviso silenciado)"
        if self.disp == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True                     # como nos exemplos do SAM 2
            torch.backends.cudnn.allow_tf32 = True
        self.p = build_sam2_video_predictor(conf["modelo_cfg"], str(ck), device=self.disp)
        self.info = {"codigo": f"{conf['repo']} ({_commit(conf['repo'])}) · pacote {Path(sam2.__file__).parent}",
                     "checkpoint": f"{ck.name} sha256 {ev._sha256(ck)[:16]}", "precisao": "bfloat16"
                     if self.disp == "cuda" else "float32", "extensao": extensao}

    def contexto(self):
        """inference_mode e, na GPU, bfloat16 (como nos exemplos do SAM 2)."""
        pilha = contextlib.ExitStack()
        pilha.enter_context(self.torch.inference_mode())
        if self.disp == "cuda":
            pilha.enter_context(self.torch.autocast("cuda", dtype=self.torch.bfloat16))
        return pilha


def _logit_objeto(estado, frame_idx):
    d = estado["output_dict_per_obj"][0]
    out = d["non_cond_frame_outputs"].get(frame_idx) or d["cond_frame_outputs"].get(frame_idx)
    return float(out["object_score_logits"].float().reshape(-1)[0])


# ============================================================================ janelas
def processar_janela(modelo, cv2, imagens, i_seed, caixa, l_max, conf, prompt="caixa", vizinhas=None):
    """imagens: caminhos da janela por ordem numérica (sem buracos); i_seed: posição do seed; vizinhas (só V5):
    {índice: caixa 0–1} dadas como prompts noutros frames, nas duas passagens. Devolve
    ({índice: (distância, score, caixa, área)}, máscaras {índice: rle}, área da máscara do seed, tamanho)."""
    p = modelo.p
    with tempfile.TemporaryDirectory() as tmp:
        for i, img in enumerate(imagens):
            os.symlink(Path(img).resolve(), Path(tmp) / f"{i}.jpg")
        with modelo.contexto():
            estado = p.init_state(video_path=tmp, offload_video_to_cpu=bool(conf["offload_video_to_cpu"]),
                                  offload_state_to_cpu=bool(conf["offload_state_to_cpu"]))
            h, w = estado["video_height"], estado["video_width"]
            x1, y1, x2, y2 = caixa
            kw = argumentos_prompt((x1 * w, y1 * h, x2 * w, y2 * h), prompt)
            res, mascaras, area_seed = {}, {}, None
            for reverso in (False, True):
                p.reset_state(estado)
                p.add_new_points_or_box(inference_state=estado, frame_idx=i_seed, obj_id=1, **kw)
                for i, (a1, b1, a2, b2) in sorted((vizinhas or {}).items()):   # V5: caixas dos vizinhos
                    p.add_new_points_or_box(inference_state=estado, frame_idx=i, obj_id=1,
                                            **argumentos_prompt((a1 * w, b1 * h, a2 * w, b2 * h), "caixa"))
                for fi, _, logits in p.propagate_in_video(estado, start_frame_idx=i_seed,
                                                          max_frame_num_to_track=l_max, reverse=reverso):
                    m = (logits[0, 0] > 0).cpu().numpy()
                    if fi == i_seed:
                        area_seed = int(m.sum())
                        mascaras.setdefault("seed", rle(m))
                        continue
                    score = round(sigmoide(_logit_objeto(estado, fi)), 6)
                    res[fi] = (abs(fi - i_seed), score, caixa_maior_componente(cv2, m), int(m.sum()))
                    if m.any():
                        mascaras[fi] = rle(m)
            p.reset_state(estado)
    return res, mascaras, area_seed, (h, w)


def correr(fr, caixas, ds, partes, pasta_mascaras, modelo, conf, max_janelas=None, progresso=True, prompt="caixa",
           vizinhas=None):
    import cv2
    l_max = max(int(x) for x in conf["l_max"])
    feitas = saltadas = calculados = 0
    sem_caixa, sem_imagem, com_saltos = [], [], []
    grupos = list(fr.groupby(["split", "caso", "seed"], sort=True))
    if max_janelas:
        grupos = grupos[:max_janelas]
    t0, t_modelo = time.time(), 0.0
    for n, ((sp, caso, seed), g) in enumerate(grupos, 1):
        destino = partes / f"{sp}__{caso}__{seed}.csv"
        w = ds / sp / caso / "GT_com_classes" / str(seed)
        if destino.is_file():
            saltadas += 1
        elif (sp, caso, seed) not in caixas:
            sem_caixa.append(f"{sp}/{caso}/{seed}")
        elif not (w / f"{caso}_{seed}_{seed}.jpg").is_file():
            sem_imagem.append(f"{sp}/{caso}/{seed}/{seed} (seed)")
        else:
            frames = []
            for f in sorted(int(x) for x in g["frame"]):
                if (w / f"{caso}_{f}_{seed}.jpg").is_file():
                    frames.append(f)
                else:
                    sem_imagem.append(f"{sp}/{caso}/{seed}/{f}")
            if any(b - a != 1 for a, b in zip(frames, frames[1:])):
                com_saltos.append(f"{sp}/{caso}/{seed}")                     # a propagação passa por cima do buraco
            i_seed = frames.index(int(seed))
            viz = None
            if vizinhas is not None:                                        # V5: só frames com imagem
                pos = {f: i for i, f in enumerate(frames)}
                viz = {pos[f]: cx for f, cx in vizinhas.get((sp, caso, seed), {}).items()
                       if f in pos and f != int(seed)}
            t1 = time.time()
            res, mascaras, area_seed, (h, wd) = processar_janela(
                modelo, cv2, [w / f"{caso}_{f}_{seed}.jpg" for f in frames], i_seed, caixas[(sp, caso, seed)],
                l_max, conf, prompt, viz)
            t_modelo += time.time() - t1
            linhas = []
            for i, f in enumerate(frames):
                if i == i_seed:
                    continue
                dist, score, cx, area = res.get(i, (None, 0.0, None, 0))
                linhas.append({"split": sp, "caso": caso, "seed": int(seed), "frame": f,
                               "distancia": dist if dist is not None else -1, "score": score,
                               **{c: (v if cx else "") for c, v in zip(CAIXA, cx or ("",) * 4)}, "area_px": area})
            d = pd.DataFrame(linhas)
            d["prompts_vizinhos"] = len(viz or {})
            d["area_seed_px"] = area_seed
            d["area_caixa_seed_px"] = round((caixas[(sp, caso, seed)][2] - caixas[(sp, caso, seed)][0])
                                            * (caixas[(sp, caso, seed)][3] - caixas[(sp, caso, seed)][1]) * h * wd)
            (pasta_mascaras / f"{sp}__{caso}__{seed}.json.gz").write_bytes(gzip.compress(json.dumps(
                {"tamanho": [h, wd], "frames": {str(frames[k]) if k != "seed" else "seed": v
                                                for k, v in mascaras.items()}}).encode("utf-8")))
            tmp = destino.with_name(destino.name + ".tmp")
            d.to_csv(tmp, index=False)
            tmp.replace(destino)
            calculados += len(res)
            feitas += 1
        if progresso:
            dt = time.time() - t0
            print(f"\r  janelas {n}/{len(grupos)} · {calculados} frames · {dt:.0f} s"
                  + (f" · {calculados / t_modelo:.1f} frames/s" if t_modelo > 0 and calculados else ""),
                  end="", flush=True)
    if progresso:
        print()
    return {"janelas": len(grupos), "feitas": feitas, "saltadas": saltadas, "calculados": calculados,
            "sem_caixa": sem_caixa, "sem_imagem": sem_imagem, "com_saltos": com_saltos,
            "segundos": time.time() - t0, "segundos_modelo": t_modelo}


def _marca(pasta, config, prompt):
    pasta.mkdir(parents=True, exist_ok=True)
    marca = pasta / "configuracao.txt"
    atual = f"sam2_prop.py v{VERSAO} · sam2.yaml sha256 {ev._sha256(config)} · prompt {prompt}"
    if marca.is_file() and marca.read_text(encoding="utf-8").strip() != atual:
        raise ev.ErroAvaliacao(f"As partes em {pasta} foram feitas com outra configuração; apaga essa pasta para "
                               "recalcular.")
    marca.write_text(atual + "\n", encoding="utf-8")


def _percentis(s, ps=(5, 25, 50, 75, 95, 99)):
    return " · ".join(f"P{p} {v:.3f}" for p, v in zip(ps, np.percentile(s, ps))) if len(s) else "sem valores"


def _exemplos(lista):
    return f" (ex.: {', '.join(lista[:3])})" if lista else ""


def main(argv=None, modelo=None):
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--conjunto", required=True, choices=("CAL", "VAL_EX", "TEST_BOX", "ALL_SEEDBOX"))
    a.add_argument("--ds", default=os.environ.get("DS"), help="pasta do dataset (por omissão, $DS)")
    a.add_argument("--config", default="configs/avaliacao.yaml")
    a.add_argument("--config-sam2", default="configs/sam2.yaml")
    a.add_argument("--prompt", default="caixa", choices=tuple(PROMPTS),
                   help="caixa (base), ponto (V1), ponto_caixa (V2) ou multi (V5, só CAL e VAL_EX)")
    a.add_argument("--max-janelas", type=int, help="só as primeiras N janelas (teste); não junta os CSV finais")
    a.add_argument("--sem-progresso", action="store_true")
    args = a.parse_args(argv)
    if args.prompt == "multi" and args.conjunto not in SO_SEM_REFERENCIA_DE_CAIXAS:
        raise ev.ErroAvaliacao(f"O V5 (multi) usa as caixas dos vizinhos como entrada: só corre em "
                               f"{' e '.join(SO_SEM_REFERENCIA_DE_CAIXAS)} (em {args.conjunto} seriam a própria "
                               "resposta).")
    if not args.ds or not Path(args.ds).is_dir():
        raise ev.ErroAvaliacao('Falta a pasta do dataset: corre primeiro o source …/inicio.sh e usa --ds "$DS".')
    cfg = ev.ler_config(args.config)
    conf = ev.ler_config(args.config_sam2)
    l_maxs = sorted({int(x) for x in conf["l_max"]})
    nome = PROMPTS[args.prompt]
    janelas = ev.ler_conjunto(Path(cfg["caminhos"]["conjuntos"]) / f"{args.conjunto}.json")
    fr = frames_das_janelas(cfg, janelas)
    caixas = caixas_do_seed(cfg, janelas)
    sufixo = "" if args.prompt == "caixa" else f"_{args.prompt}"
    partes = Path(f"{conf['partes']}{sufixo}") / args.conjunto
    pasta_mascaras = Path(f"{conf['mascaras']}{sufixo}") / args.conjunto
    _marca(partes, args.config_sam2, args.prompt)
    pasta_mascaras.mkdir(parents=True, exist_ok=True)
    modelo = modelo or ModeloSAM2(conf)
    if modelo.disp == "cuda":
        modelo.torch.cuda.reset_peak_memory_stats()
    vizinhas = caixas_vizinhas(cfg, janelas) if args.prompt == "multi" else None
    r = correr(fr, caixas, Path(args.ds), partes, pasta_mascaras, modelo, conf, args.max_janelas,
               not args.sem_progresso, args.prompt, vizinhas)
    memoria = (f"{modelo.torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB" if modelo.disp == "cuda" else "—")

    grupos = sorted(set(janelas))[:args.max_janelas] if args.max_janelas else sorted(set(janelas))
    ficheiros = [f for f in (partes / f"{sp}__{caso}__{seed}.csv" for (sp, caso, seed) in grupos) if f.is_file()]
    if not ficheiros:
        raise ev.ErroAvaliacao("Nenhuma janela ficou com previsões (ver caixas do seed e imagens).")
    d = pd.concat([pd.read_csv(f, dtype={"caso": str}, keep_default_na=False) for f in ficheiros], ignore_index=True)
    saida = Path(conf["saida"])
    saida.mkdir(parents=True, exist_ok=True)
    csv = {}
    if not args.max_janelas:
        for lm in l_maxs:
            dentro = (d["distancia"] >= 1) & (d["distancia"] <= lm)
            p = d[["split", "caso", "seed", "frame"]].copy()
            p["score"] = d["score"].where(dentro, 0.0)
            for c in CAIXA:
                p[c] = d[c].where(dentro, "")
            p["area_px"] = d["area_px"].where(dentro, 0)
            csv[lm] = saida / f"pred_{nome}_L{lm}_{args.conjunto}.csv"
            p.sort_values(["split", "caso", "seed", "frame"]).to_csv(csv[lm], index=False)

    # resumo
    git = ev._estado_git(RAIZ)
    estado = ("sem git" if git["commit"] is None else
              f"commit {git['commit']}" + (" (com alterações por guardar)" if git["alteracoes_por_guardar"] else ""))
    seguidos = d[d["distancia"] >= 1]
    com_mascara = seguidos[seguidos["area_px"] > 0]
    por_janela = d.drop_duplicates(["split", "caso", "seed"])
    razao = (por_janela["area_seed_px"] / por_janela["area_caixa_seed_px"].clip(lower=1)).to_numpy(dtype=float)
    area = com_mascara["area_px"].to_numpy(dtype=float)
    fps = r["calculados"] / r["segundos_modelo"] if r["segundos_modelo"] > 0 else 0.0
    inf = modelo.info
    L = [f"SAM 2 ({Path(conf['checkpoint']).name}) em {args.conjunto} · prompt: {args.prompt} ({nome})"
         + (f" · TESTE com {args.max_janelas} janela(s)" if args.max_janelas else "")
         + f" · {datetime.datetime.now():%Y-%m-%d %H:%M} · {estado}",
         f"sam2_prop.py v{VERSAO} · sha256 {ev._sha256(__file__)[:16]} · {Path(args.config_sam2).name} sha256 "
         f"{ev._sha256(args.config_sam2)[:16]} · torch {modelo.torch.__version__} · {modelo.disp} · {inf['precisao']}",
         f"código: {inf['codigo']} · checkpoint: {inf['checkpoint']}"
         + (f" · {inf['extensao']}" if inf.get("extensao") else ""),
         "",
         f"janelas: {r['janelas']} (calculadas agora {r['feitas']}, já feitas antes {r['saltadas']}) · frames "
         f"calculados agora: {r['calculados']} · {r['segundos']:.0f} s no total · {fps:.1f} frames/s no modelo · "
         f"memória máxima da GPU: {memoria}",
         f"janelas sem caixa do seed (ficam sem previsões): {len(r['sem_caixa'])}{_exemplos(r['sem_caixa'])}",
         f"frames sem imagem: {len(r['sem_imagem'])}{_exemplos(r['sem_imagem'])} · janelas com saltos na numeração: "
         f"{len(r['com_saltos'])}{_exemplos(r['com_saltos'])}",
         f"máscara do seed ÷ área da caixa do seed: {_percentis(razao, (5, 50, 95))} (a máscara deve ocupar boa "
         "parte da caixa)",
         "",
         f"Frames seguidos (até {max(l_maxs)} para cada lado): {len(seguidos)} · com máscara: {len(com_mascara)} "
         f"({100 * len(com_mascara) / max(1, len(seguidos)):.0f} %)",
         "Score (sigmoide do score de objeto) nos frames seguidos, sem rótulos (a grelha de τ tem passo 0,02):",
         f"  {_percentis(seguidos['score'].to_numpy(dtype=float))}",
         "Por distância ao seed, fração de frames com máscara: " + " · ".join(
             f"{nome} {100 * (seguidos.loc[seguidos['distancia'].between(lo, hi), 'area_px'] > 0).mean():.0f} %"
             for nome, lo, hi in (("1-3", 1, 3), ("4-6", 4, 6), ("7-12", 7, 12), (">12", 13, 10 ** 9))
             if (seguidos["distancia"].between(lo, hi)).any()),
         f"área da máscara (px, frames com máscara): {_percentis(area, (5, 50, 95))}"]
    if args.prompt == "multi":
        n_viz = por_janela["prompts_vizinhos"].to_numpy(dtype=int)
        L += ["", f"V5: caixas dos vizinhos dadas como prompt: {n_viz.sum()} no total · por janela: mediana "
                  f"{np.median(n_viz):g}, máximo {n_viz.max()} · janelas sem nenhuma: {(n_viz == 0).sum()}",
              "É um limite superior (usa parte da referência como entrada). Em CAL, as métricas de caixas (R_box, "
              "R_loc, mIoU) não contam: essas caixas são ao mesmo tempo a entrada e a referência."]
    if csv and args.conjunto == "CAL":
        L += ["", "Calibração, uma por variante de L_max (comparam-se em CAL):"]
        L += [f"  python eval/evaluate.py calibrar --metodo {nome}_L{lm} --pred {csv[lm]}" for lm in l_maxs]
    elif args.max_janelas:
        L += ["", "Teste: os CSV finais não foram gravados. Para o conjunto inteiro, corre sem --max-janelas "
                  "(a janela já feita é aproveitada)."]
    texto = "\n".join(L)
    f_resumo = saida / (f"pred_{nome}_{args.conjunto}_resumo.txt" if not args.max_janelas
                        else f"{nome.lower()}_teste_{args.conjunto}_resumo.txt")
    f_resumo.write_text(texto + "\n", encoding="utf-8")
    print(texto)
    return {"csv": csv, "resumo": f_resumo, **r}


if __name__ == "__main__":
    try:
        main()
    except ev.ErroAvaliacao as e:
        print(f"ERRO: {e}", file=sys.stderr)
        sys.exit(2)
