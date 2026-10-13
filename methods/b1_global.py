#!/usr/bin/env python3
"""
A1 — B1, semelhança global (Guião A). Em cada janela, o score de um frame é o cosseno entre o vetor do seed e
o vetor do frame, cortado a [0, 1]: score = max(0, cos). Não usa caixas nem rótulos (só a lista de frames de
cada janela, de annotations_v1.csv). É o controlo: o método final tem de o superar.

Vetores (configs/b1.yaml): um .pt por frame em <pasta>/<split>/<caso>/GT_com_classes/<seed>/<caso>_<frame>_<seed>.pt,
cada um um dict com várias versões do vetor do frame (cls, mean, max, std, ...). Calcula todas as versões pedidas
numa só passagem (cada ficheiro é lido uma vez) e grava um CSV por versão; o método chama-se B1_<emb>_<chave>.
  endofm   EndoFM v2 (vectors_v2)
  dinov2   DINOv2 ViT-B/14, extraído antes por methods/extrair_dinov2.py para a mesma estrutura

Uso, a partir da pasta do projeto:
    python methods/b1_global.py --emb endofm --conjunto CAL              (as versões de configs/b1.yaml)
    python methods/b1_global.py --emb endofm --conjunto VAL_EX --chaves cls   (só a escolhida, no A7)
Gera em outputs/preds/: pred_B1_<emb>_<chave>_<CONJUNTO>.csv (split, caso, seed, frame, score), no formato do
avaliador, e pred_B1_<emb>_<CONJUNTO>_resumo.txt (contagens, distribuição dos scores, proveniência).
Depois, para cada versão: python eval/evaluate.py calibrar --metodo B1_<emb>_<chave> --pred <csv de CAL>
Regra de escolha (configs/b1.yaml): maior AP em CAL; em empate, cls. VAL_EX e TEST_BOX só no A7.
"""
import argparse
import datetime
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "eval"))
import evaluate as ev  # noqa: E402

VERSAO = "2.1"
CONJUNTOS = ("CAL", "VAL_EX", "TEST_BOX")


def frames_das_janelas(cfg, janelas):
    """Frames de cada janela do conjunto (só split, caso, seed, frame, offset; nenhuma coluna de rótulos)."""
    alvo, casos, partes = set(janelas), {k[1] for k in janelas}, []
    for b in pd.read_csv(cfg["caminhos"]["anotacoes"], dtype=str, keep_default_na=False,
                         usecols=["split", "caso", "seed", "frame", "offset"], chunksize=200_000):
        b = b[b["caso"].isin(casos)]
        if len(b):
            b = b[[k in alvo for k in zip(b["split"], b["caso"], b["seed"].astype(int))]]
            if len(b):
                partes.append(b)
    if not partes:
        raise ev.ErroAvaliacao("Nenhuma janela do conjunto está em annotations_v1.csv.")
    a = pd.concat(partes, ignore_index=True)
    for c in ("seed", "frame", "offset"):
        a[c] = a[c].astype(int)
    faltam = set(janelas) - set(zip(a["split"], a["caso"], a["seed"]))
    if faltam:
        raise ev.ErroAvaliacao(f"{len(faltam)} janelas do conjunto não estão em annotations_v1.csv.")
    return a


class VetoresPt:
    """Lê os vetores de um frame (um .pt com um dict de versões) e devolve as versões pedidas."""

    def __init__(self, pasta, chaves):
        try:
            import torch
        except ImportError:
            raise ev.ErroAvaliacao("Falta o PyTorch neste ambiente (os vetores são .pt).")
        self.torch, self.pasta, self.chaves, self.lidos = torch, Path(pasta), list(chaves), 0
        if not self.pasta.is_dir():
            raise ev.ErroAvaliacao(f"Não encontrei a pasta dos vetores {self.pasta} (configs/b1.yaml).")
        if not self.chaves:
            raise ev.ErroAvaliacao("Não há versões do vetor para calcular (chaves vazias).")

    def ficheiro(self, split, caso, seed, frame):
        return self.pasta / split / caso / "GT_com_classes" / str(seed) / f"{caso}_{frame}_{seed}.pt"

    def vetores(self, split, caso, seed, frame):
        f = self.ficheiro(split, caso, seed, frame)
        if not f.is_file():
            return None
        try:
            o = self.torch.load(f, map_location="cpu", weights_only=True)
        except TypeError:                                               # PyTorch antigo, sem weights_only
            o = self.torch.load(f, map_location="cpu")
        except Exception:                                               # noqa: BLE001
            o = self.torch.load(f, map_location="cpu", weights_only=False)  # ficheiros do próprio projeto
        falta = [c for c in self.chaves if not isinstance(o, dict) or c not in o]
        if falta:
            tem = ", ".join(map(str, o)) if isinstance(o, dict) else type(o).__name__
            raise ev.ErroAvaliacao(f"{f.name} não tem a versão '{falta[0]}' (tem: {tem}); vê configs/b1.yaml.")
        self.lidos += 1
        out = {}
        for c in self.chaves:
            v = o[c]
            v = v.float().numpy() if hasattr(v, "numpy") else np.asarray(v, dtype=np.float32)
            out[c] = np.asarray(v, dtype=np.float64).reshape(-1)
        return out


def _cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def prever(fr, vet, progresso=True):
    """Devolve {chave: DataFrame de previsões}, frames sem vetor, janelas sem vetor do seed e {chave: n.º de negativos}."""
    linhas = {c: [] for c in vet.chaves}
    negativos = {c: 0 for c in vet.chaves}
    sem_vetor, sem_seed = [], []
    grupos = list(fr.groupby(["split", "caso", "seed"], sort=True))
    t0 = time.time()
    for i, ((sp, caso, seed), g) in enumerate(grupos, 1):
        s = vet.vetores(sp, caso, seed, seed)
        if s is None:
            sem_seed.append(f"{sp}/{caso}/{seed}")
        else:
            for f in g.loc[g["offset"] != 0, "frame"].tolist():
                v = vet.vetores(sp, caso, seed, f)
                if v is None:
                    sem_vetor.append(f"{sp}/{caso}/{seed}/{f}")
                    continue
                for c in vet.chaves:
                    cos = _cos(v[c], s[c])
                    negativos[c] += cos < 0
                    linhas[c].append({"split": sp, "caso": caso, "seed": int(seed), "frame": int(f),
                                      "score": round(min(1.0, max(0.0, cos)), 6)})
        if progresso and (i % 5 == 0 or i == len(grupos)):
            print(f"\r  janelas {i}/{len(grupos)} · {time.time() - t0:.0f} s", end="", flush=True)
    if progresso:
        print()
    cols = ["split", "caso", "seed", "frame", "score"]
    return {c: pd.DataFrame(linhas[c], columns=cols) for c in vet.chaves}, sem_vetor, sem_seed, negativos


def _linha_distribuicao(chave, s, neg):
    if not len(s):
        return f"  {chave:>8}: sem scores"
    q = np.percentile(s, [5, 25, 50, 75, 95, 99])
    n = int(np.floor(q[5] / 0.02) - np.ceil(q[0] / 0.02)) + 1
    return (f"  {chave:>8}: P5 {q[0]:.3f} · P25 {q[1]:.3f} · P50 {q[2]:.3f} · P75 {q[3]:.3f} · P95 {q[4]:.3f}"
            f" · P99 {q[5]:.3f} · negativos {neg} · limiares da grelha entre P5 e P99: {n}")


def resumo(prev, fr, sem_vetor, sem_seed, negativos, args, vet, t0):
    git = ev._estado_git(RAIZ)
    estado = ("sem git" if git["commit"] is None else
              f"commit {git['commit']}" + (" (com alterações por guardar)" if git["alteracoes_por_guardar"] else ""))
    n_esperados = int((fr["offset"] != 0).sum())
    c0 = vet.chaves[0]
    L = [f"B1 ({args.emb}) em {args.conjunto} · versões: {', '.join(vet.chaves)} · {datetime.datetime.now():%Y-%m-%d %H:%M} · {estado}",
         f"b1_global.py v{VERSAO} · sha256 {ev._sha256(__file__)[:16]} · {Path(args.config_b1).name} sha256 "
         f"{ev._sha256(args.config_b1)[:16]} · torch {vet.torch.__version__}",
         "",
         f"janelas: {fr.groupby(['split', 'caso', 'seed']).ngroups} · frames sem o seed: {n_esperados} · "
         f"com score: {len(prev[c0])} · {time.time() - t0:.0f} s",
         f"janelas sem vetor do seed (ficam sem previsões): {len(sem_seed)}" + (f" (ex.: {', '.join(sem_seed[:3])})" if sem_seed else ""),
         f"frames sem vetor (contam como ausentes no avaliador): {len(sem_vetor)}" + (f" (ex.: {', '.join(sem_vetor[:3])})" if sem_vetor else ""),
         "",
         "Distribuição dos scores, sem rótulos (a grelha de τ do avaliador tem passo 0,02):"]
    L += [_linha_distribuicao(c, prev[c]["score"].to_numpy(dtype=float), negativos[c]) for c in vet.chaves]
    if args.conjunto == "CAL":
        L += ["", "Calibração (só depois de o protocolo estar aprovado), uma por versão:"]
        L += [f"  python eval/evaluate.py calibrar --metodo B1_{args.emb}_{c} --pred {Path(args.saida_csv[c])}"
              for c in vet.chaves]
    L += ["", f"3 linhas de exemplo ({c0}):", prev[c0].head(3).to_string(index=False)]
    return "\n".join(L)


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--emb", required=True, choices=["endofm", "dinov2"], help="vetores a usar (configs/b1.yaml)")
    a.add_argument("--conjunto", required=True, choices=CONJUNTOS)
    a.add_argument("--chaves", help="versões separadas por vírgulas (por omissão, as de configs/b1.yaml)")
    a.add_argument("--config", default="configs/avaliacao.yaml", help="caminhos dos conjuntos e de annotations_v1")
    a.add_argument("--config-b1", default="configs/b1.yaml")
    a.add_argument("--sem-progresso", action="store_true")
    args = a.parse_args(argv)
    t0 = time.time()
    cfg = ev.ler_config(args.config)
    cfg_b1 = ev.ler_config(args.config_b1)
    conf = cfg_b1["vetores"][args.emb]
    chaves = [c.strip() for c in args.chaves.split(",") if c.strip()] if args.chaves else list(conf["chaves"])
    janelas = ev.ler_conjunto(Path(cfg["caminhos"]["conjuntos"]) / f"{args.conjunto}.json")
    fr = frames_das_janelas(cfg, janelas)
    vet = VetoresPt(conf["pasta"], chaves)
    prev, sem_vetor, sem_seed, negativos = prever(fr, vet, not args.sem_progresso)
    saida = Path(cfg_b1["saida"])
    saida.mkdir(parents=True, exist_ok=True)
    args.saida_csv = {c: saida / f"pred_B1_{args.emb}_{c}_{args.conjunto}.csv" for c in chaves}
    for c in chaves:
        prev[c].to_csv(args.saida_csv[c], index=False)
    texto = resumo(prev, fr, sem_vetor, sem_seed, negativos, args, vet, t0)
    f_resumo = saida / f"pred_B1_{args.emb}_{args.conjunto}_resumo.txt"
    f_resumo.write_text(texto + "\n", encoding="utf-8")
    print(texto)
    print(f"\ngravado em {saida}/: {len(chaves)} CSV de previsões e {f_resumo.name}")
    return {"previsoes": prev, "sem_vetor": sem_vetor, "sem_seed": sem_seed, "negativos": negativos,
            "csv": args.saida_csv, "resumo": f_resumo}


if __name__ == "__main__":
    try:
        main()
    except ev.ErroAvaliacao as e:
        print(f"ERRO: {e}", file=sys.stderr)
        sys.exit(2)
