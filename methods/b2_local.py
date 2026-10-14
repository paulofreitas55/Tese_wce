#!/usr/bin/env python3
"""
A2 — B2, semelhança local com DINOv2 (Guião A). Compara cada frame da janela com a região da caixa do seed, em vez do
frame inteiro (B1). Usa só a caixa do próprio seed, que é a entrada do método; as caixas dos outros frames (o alvo)
nunca são lidas.

Para cada janela:
  1. O seed passa no DINOv2 (dinov2_vitb14, sem fine-tune) com a imagem a 448×448, o que dá uma grelha de 32×32
     quadrados de 14 px (≈ 10 px do frame original). O protótipo são os vetores dos quadrados cujo centro cai dentro
     da caixa do seed; se nenhum cair (caixa muito pequena), usa-se o quadrado onde está o centro da caixa.
     Guarda-se também o protótipo médio (a média desses vetores).
  2. Cada frame passa no DINOv2 e dá dois mapas de 32×32:
       S(i,j)       = max_k cos(quadrado_ij, protótipo_k)
       S_media(i,j) = cos(quadrado_ij, protótipo médio)
  3. Scores, cortados a [0, 1] sem reescalar (como no B1):
       bruto            max(S)
       corrigido        max(S) − mediana(S)          a mediana tira a parte "mesma cena" do frame
       bruto_top4       média dos 4 maiores valores de S
       bruto_media      max(S_media)
       corrigido_media  max(S_media) − mediana(S_media)
     bruto e corrigido são as duas variantes do guião e do protocolo (linhas "B2 local bruto" e "B2 local corrigido"
     da tabela mestre). As outras três são os testes que o guião pede, para comparar só em CAL.
  4. Caixa prevista: os quadrados com S ≥ 0,9·max(S) ligados ao máximo (vizinhança de 4), em coordenadas 0–1 como
     em annotations_v1; as variantes _media usam o mapa S_media.
Imagem: 320×320 → 448×448 (bicúbica) e normalização ImageNet, como em methods/extrair_dinov2.py.
Parâmetros em configs/b2.yaml. Retoma: cada janela grava um ficheiro em outputs/preds/b2_partes/<CONJUNTO>/ e as
janelas já feitas são saltadas; no fim junta tudo num CSV por variante.

Uso, a partir da pasta do projeto, depois do source …/inicio.sh:
    python methods/b2_local.py --conjunto CAL --ds "$DS"
Gera em outputs/preds/: pred_B2_<variante>_<CONJUNTO>.csv (split, caso, seed, frame, score, box_x1 … box_y2), no
formato do avaliador, e pred_B2_<CONJUNTO>_resumo.txt. Depois, para cada variante:
    python eval/evaluate.py calibrar --metodo B2_<variante> --pred outputs/preds/pred_B2_<variante>_CAL.csv
VAL_EX e TEST_BOX só no A7.
"""
import argparse
import datetime
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "eval"))
sys.path.insert(0, str(RAIZ / "methods"))
import evaluate as ev  # noqa: E402
from b1_global import frames_das_janelas  # noqa: E402
from extrair_dinov2 import Modelo, preparar  # noqa: E402

VERSAO = "1.0"
VARIANTES = ("bruto", "corrigido", "bruto_top4", "bruto_media", "corrigido_media")
CAIXA = ["box_x1", "box_y1", "box_x2", "box_y2"]
LADO_QUADRADO = 14                                       # px do DINOv2 ViT-B/14


# ============================================================================ partes sem modelo (testadas à parte)
def caixas_do_seed(cfg, janelas):
    """{(split, caso, seed): (x1, y1, x2, y2)} em 0–1: só a caixa do próprio seed (offset 0, box_src medico_seed)."""
    alvo, casos, caixas = set(janelas), {k[1] for k in janelas}, {}
    for b in pd.read_csv(cfg["caminhos"]["anotacoes"], dtype=str, keep_default_na=False,
                         usecols=["split", "caso", "seed", "offset", "box_src"] + CAIXA, chunksize=200_000):
        b = b[b["caso"].isin(casos) & (pd.to_numeric(b["offset"], errors="coerce") == 0)
              & (b["box_src"] == "medico_seed")]
        for _, r in b.iterrows():
            k = (r["split"], r["caso"], int(r["seed"]))
            if k in alvo and all(r[c] != "" for c in CAIXA):
                x1, y1, x2, y2 = (float(r[c]) for c in CAIXA)
                caixas[k] = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
    return caixas


def indices_prototipo(caixa, g):
    """Índices (0 … g²−1, linha a linha) dos quadrados da grelha g×g cujo centro cai dentro da caixa (0–1). Se nenhum
    cair, o quadrado onde está o centro da caixa. Devolve (índices, recurso), com recurso=True no segundo caso."""
    x1, y1, x2, y2 = caixa
    centros = (np.arange(g) + 0.5) / g
    cy, cx = np.meshgrid(centros, centros, indexing="ij")
    idx = np.flatnonzero(((cx >= x1) & (cx <= x2) & (cy >= y1) & (cy <= y2)).reshape(-1))
    if len(idx):
        return idx, False
    lin = min(g - 1, max(0, int((y1 + y2) / 2 * g)))
    col = min(g - 1, max(0, int((x1 + x2) / 2 * g)))
    return np.array([lin * g + col]), True


def caixa_do_mapa(s, fracao):
    """s: mapa g×g. Componente ligada (vizinhança de 4) dos quadrados com s ≥ fracao·max(s) que contém o máximo →
    caixa (x1, y1, x2, y2) em 0–1. Se max(s) ≤ 0, só o quadrado do máximo."""
    g = s.shape[0]
    l0, c0 = (int(v) for v in np.unravel_index(int(np.argmax(s)), s.shape))
    ok = s >= fracao * s[l0, c0] if s[l0, c0] > 0 else np.zeros(s.shape, dtype=bool)
    ok[l0, c0] = True
    visto = np.zeros(s.shape, dtype=bool)
    visto[l0, c0] = True
    fila, lins, cols = deque([(l0, c0)]), [l0], [c0]
    while fila:
        lin, col = fila.popleft()
        for dl, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ll, cc = lin + dl, col + dc
            if 0 <= ll < g and 0 <= cc < g and ok[ll, cc] and not visto[ll, cc]:
                visto[ll, cc] = True
                fila.append((ll, cc))
                lins.append(ll)
                cols.append(cc)
    return min(cols) / g, min(lins) / g, (max(cols) + 1) / g, (max(lins) + 1) / g


def scores(s, s_media):
    """Os cinco scores de um frame a partir dos dois mapas, cortados a [0, 1]."""
    v = np.sort(np.asarray(s, dtype=np.float64).reshape(-1))[::-1]
    vm = np.asarray(s_media, dtype=np.float64).reshape(-1)

    def corta(x):
        return round(float(min(1.0, max(0.0, x))), 6)
    return {"bruto": corta(v[0]), "corrigido": corta(v[0] - np.median(v)), "bruto_top4": corta(v[:4].mean()),
            "bruto_media": corta(vm.max()), "corrigido_media": corta(vm.max() - np.median(vm))}


# ============================================================================ modelo
class ModeloDino(Modelo):
    """O DINOv2 de methods/extrair_dinov2.py (torch.hub, sem fine-tune); aqui devolve os vetores dos quadrados."""

    def quadrados(self, lote):
        """lote: imagens C×H×W já preparadas → B×N×D na GPU (se houver), com norma 1."""
        torch = self.torch
        with torch.no_grad():
            x = torch.from_numpy(np.ascontiguousarray(np.stack(lote))).to(self.disp)
            q = self.m.forward_features(x)["x_norm_patchtokens"].float()
            return torch.nn.functional.normalize(q, dim=-1)


def mapas(torch, q, prot, media):
    """q: B×N×D, prot: K×D, media: D (todos com norma 1) → (S, S_media) em numpy, B×N."""
    with torch.no_grad():
        return (q @ prot.T).amax(dim=-1).cpu().numpy(), (q @ media).cpu().numpy()


# ============================================================================ janelas
def processar_janela(modelo, frames, seed_img, caixa, lado, lote, fracao):
    """frames: [(frame, imagem)] sem o seed. Devolve (linhas, nº de quadrados do protótipo, recurso)."""
    torch = modelo.torch
    g = lado // LADO_QUADRADO
    q_seed = modelo.quadrados([preparar(seed_img, lado)])[0]
    if q_seed.shape[0] != g * g:
        raise ev.ErroAvaliacao(f"O modelo devolveu {q_seed.shape[0]} quadrados; esperava {g}×{g}.")
    idx, recurso = indices_prototipo(caixa, g)
    prot = q_seed[torch.as_tensor(idx, device=q_seed.device)]
    media = torch.nn.functional.normalize(prot.mean(dim=0), dim=0)
    linhas = []
    for k0 in range(0, len(frames), lote):
        parte = frames[k0:k0 + lote]
        s, sm = mapas(torch, modelo.quadrados([preparar(img, lado) for _, img in parte]), prot, media)
        for j, (f, _) in enumerate(parte):
            mapa, mapa_m = s[j].reshape(g, g), sm[j].reshape(g, g)
            sc = scores(mapa, mapa_m)
            cx, cxm = caixa_do_mapa(mapa, fracao), caixa_do_mapa(mapa_m, fracao)
            for v in VARIANTES:
                b = cxm if v.endswith("_media") else cx
                linhas.append({"frame": int(f), "variante": v, "score": sc[v],
                               **{c: round(x, 6) for c, x in zip(CAIXA, b)}})
    return linhas, len(idx), recurso


def correr(fr, caixas, ds, partes, modelo, conf, progresso=True):
    lado, lote, fracao = int(conf["lado"]), int(conf["lote"]), float(conf["limiar_caixa"])
    feitas = saltadas = calculados = 0
    sem_imagem, sem_caixa = [], []
    grupos = list(fr.groupby(["split", "caso", "seed"], sort=True))
    t0 = time.time()
    for i, ((sp, caso, seed), g) in enumerate(grupos, 1):
        destino = partes / f"{sp}__{caso}__{seed}.csv"
        w = Path(sp) / caso / "GT_com_classes" / str(seed)
        if destino.is_file():
            saltadas += 1
        elif (sp, caso, seed) not in caixas:
            sem_caixa.append(f"{sp}/{caso}/{seed}")
        elif not (ds / w / f"{caso}_{seed}_{seed}.jpg").is_file():
            sem_imagem.append(f"{sp}/{caso}/{seed}/{seed} (seed)")
        else:
            frames = []
            for f in sorted(int(x) for x in g.loc[g["offset"] != 0, "frame"]):
                img = ds / w / f"{caso}_{f}_{seed}.jpg"
                if img.is_file():
                    frames.append((f, img))
                else:
                    sem_imagem.append(f"{sp}/{caso}/{seed}/{f}")
            linhas, n_prot, recurso = processar_janela(modelo, frames, ds / w / f"{caso}_{seed}_{seed}.jpg",
                                                       caixas[(sp, caso, seed)], lado, lote, fracao)
            d = pd.DataFrame(linhas, columns=["frame", "variante", "score"] + CAIXA)
            d.insert(0, "split", sp)
            d.insert(1, "caso", caso)
            d.insert(2, "seed", int(seed))
            d["n_prototipo"], d["recurso"] = n_prot, int(recurso)
            tmp = destino.with_name(destino.name + ".tmp")
            d.to_csv(tmp, index=False)
            tmp.replace(destino)
            calculados += len(frames)
            feitas += 1
        if progresso:
            dt = time.time() - t0
            print(f"\r  janelas {i}/{len(grupos)} · {calculados} frames · {dt:.0f} s"
                  + (f" · {calculados / dt:.1f} frames/s" if dt > 0 and calculados else ""), end="", flush=True)
    if progresso:
        print()
    return {"janelas": len(grupos), "feitas": feitas, "saltadas": saltadas, "calculados": calculados,
            "sem_imagem": sem_imagem, "sem_caixa": sem_caixa, "segundos": time.time() - t0}


def _marca_partes(partes, config_b2):
    """As partes de uma configuração não se misturam com as de outra: a pasta guarda o sha256 do b2.yaml."""
    partes.mkdir(parents=True, exist_ok=True)
    marca, atual = partes / "configuracao.txt", f"b2_local.py v{VERSAO} · b2.yaml sha256 {ev._sha256(config_b2)}"
    if marca.is_file() and marca.read_text(encoding="utf-8").strip() != atual:
        raise ev.ErroAvaliacao(f"As partes em {partes} foram feitas com outra configuração; apaga essa pasta "
                               "para recalcular.")
    marca.write_text(atual + "\n", encoding="utf-8")


def _distribuicao(nome, s):
    if not len(s):
        return f"  {nome:>16}: sem scores"
    q = np.percentile(s, [5, 25, 50, 75, 95, 99])
    n = int(np.floor(q[5] / 0.02) - np.ceil(q[0] / 0.02)) + 1
    return (f"  {nome:>16}: P5 {q[0]:.3f} · P25 {q[1]:.3f} · P50 {q[2]:.3f} · P75 {q[3]:.3f} · P95 {q[4]:.3f}"
            f" · P99 {q[5]:.3f} · limiares da grelha entre P5 e P99: {n}")


def _exemplos(lista):
    return f" (ex.: {', '.join(lista[:3])})" if lista else ""


def main(argv=None, modelo=None):
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--conjunto", required=True, choices=("CAL", "VAL_EX", "TEST_BOX"))
    a.add_argument("--ds", default=os.environ.get("DS"), help="pasta do dataset (por omissão, $DS)")
    a.add_argument("--config", default="configs/avaliacao.yaml")
    a.add_argument("--config-b2", default="configs/b2.yaml")
    a.add_argument("--sem-progresso", action="store_true")
    args = a.parse_args(argv)
    if not args.ds or not Path(args.ds).is_dir():
        raise ev.ErroAvaliacao('Falta a pasta do dataset: corre primeiro o source …/inicio.sh e usa --ds "$DS".')
    cfg = ev.ler_config(args.config)
    conf = ev.ler_config(args.config_b2)
    janelas = ev.ler_conjunto(Path(cfg["caminhos"]["conjuntos"]) / f"{args.conjunto}.json")
    fr = frames_das_janelas(cfg, janelas)
    caixas = caixas_do_seed(cfg, janelas)
    partes = Path(conf["partes"]) / args.conjunto
    _marca_partes(partes, args.config_b2)
    modelo = modelo or ModeloDino(conf["modelo"])
    r = correr(fr, caixas, Path(args.ds), partes, modelo, conf, not args.sem_progresso)

    # juntar as partes das janelas do conjunto num CSV por variante
    ficheiros = [f for f in (partes / f"{sp}__{caso}__{seed}.csv" for (sp, caso, seed) in sorted(set(janelas)))
                 if f.is_file()]
    if not ficheiros:
        raise ev.ErroAvaliacao("Nenhuma janela do conjunto ficou com previsões (ver caixas do seed e imagens).")
    d = pd.concat([pd.read_csv(f, dtype={"caso": str}) for f in ficheiros], ignore_index=True)
    saida = Path(conf["saida"])
    saida.mkdir(parents=True, exist_ok=True)
    csv = {}
    for v in VARIANTES:
        p = d[d["variante"] == v].sort_values(["split", "caso", "seed", "frame"])
        csv[v] = saida / f"pred_B2_{v}_{args.conjunto}.csv"
        p[["split", "caso", "seed", "frame", "score"] + CAIXA].to_csv(csv[v], index=False)

    # resumo
    git = ev._estado_git(RAIZ)
    estado = ("sem git" if git["commit"] is None else
              f"commit {git['commit']}" + (" (com alterações por guardar)" if git["alteracoes_por_guardar"] else ""))
    lado = int(conf["lado"])
    por_janela = d.drop_duplicates(["split", "caso", "seed"])
    b = d[d["variante"] == "bruto"]
    area = ((b["box_x2"] - b["box_x1"]) * (b["box_y2"] - b["box_y1"]) * 100).to_numpy()
    n_janelas = len(por_janela)
    L = [f"B2 (DINOv2 {conf['modelo']}, {lado}×{lado}, grelha {lado // LADO_QUADRADO}×{lado // LADO_QUADRADO}) em "
         f"{args.conjunto} · {datetime.datetime.now():%Y-%m-%d %H:%M} · {estado}",
         f"b2_local.py v{VERSAO} · sha256 {ev._sha256(__file__)[:16]} · {Path(args.config_b2).name} sha256 "
         f"{ev._sha256(args.config_b2)[:16]} · torch {modelo.torch.__version__} · {modelo.disp} · pesos "
         f"{modelo.pesos_sha256()}",
         "",
         f"janelas: {r['janelas']} (calculadas agora {r['feitas']}, já feitas antes {r['saltadas']}) · com previsões: "
         f"{n_janelas} · frames calculados agora: {r['calculados']} · {r['segundos']:.0f} s"
         + (f" ({r['calculados'] / r['segundos']:.1f} frames/s)" if r["calculados"] and r["segundos"] > 0 else ""),
         f"janelas sem caixa do seed (ficam sem previsões): {len(r['sem_caixa'])}{_exemplos(r['sem_caixa'])}",
         f"frames sem imagem (contam como ausentes no avaliador): {len(r['sem_imagem'])}{_exemplos(r['sem_imagem'])}",
         ("protótipo: quadrados por seed: mínimo {} · mediana {:.0f} · máximo {} · seeds sem nenhum centro de quadrado "
          "na caixa (usado o do centro): {}").format(
             int(por_janela["n_prototipo"].min()) if n_janelas else 0,
             float(por_janela["n_prototipo"].median()) if n_janelas else 0,
             int(por_janela["n_prototipo"].max()) if n_janelas else 0, int(por_janela["recurso"].sum())),
         "",
         "Distribuição dos scores, sem rótulos (a grelha de τ do avaliador tem passo 0,02):"]
    L += [_distribuicao(v, d.loc[d["variante"] == v, "score"].to_numpy(dtype=float)) for v in VARIANTES]
    L += [f"caixa prevista (bruto e corrigido), área em % do frame: P5 {np.percentile(area, 5):.2f} · "
          f"mediana {np.percentile(area, 50):.2f} · P95 {np.percentile(area, 95):.2f}" if len(area) else
          "caixa prevista: sem frames"]
    if args.conjunto == "CAL":
        L += ["", "Calibração, uma por variante (bruto e corrigido são as do guião; as outras só se comparam em CAL):"]
        L += [f"  python eval/evaluate.py calibrar --metodo B2_{v} --pred {csv[v]}" for v in VARIANTES]
    L += ["", "3 linhas de exemplo (bruto):",
          pd.read_csv(csv["bruto"], dtype={"caso": str}).head(3).to_string(index=False)]
    texto = "\n".join(L)
    f_resumo = saida / f"pred_B2_{args.conjunto}_resumo.txt"
    f_resumo.write_text(texto + "\n", encoding="utf-8")
    print(texto)
    print(f"\ngravado em {saida}/: {len(VARIANTES)} CSV de previsões e {f_resumo.name}")
    return {"csv": csv, "resumo": f_resumo, **r}


if __name__ == "__main__":
    try:
        main()
    except ev.ErroAvaliacao as e:
        print(f"ERRO: {e}", file=sys.stderr)
        sys.exit(2)
