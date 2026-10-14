#!/usr/bin/env python3
"""
Extrai os vetores EndoFM-LV de cada frame das janelas de um conjunto, para o B1 (Guião A, A1). Refeito do zero a
14 out 2026; substitui a extração antiga (vectors_v2), que lia mal os vetores dos quadrados e ordenava os frames pelo
nome do ficheiro (ver docs/decisoes.md).

Cada frame f entra num clip de 8 frames seguidos, de f−4 a f+3 (o modelo foi treinado com clips de 8), pela ordem
numérica dos frames da janela (a de annotations_v1.csv). Nas pontas da janela, o clip completa-se repetindo o primeiro
ou o último frame, para o frame ficar sempre na posição 4 do clip. Grava um .pt por frame, na mesma estrutura dos
outros vetores:
    <pasta>/<split>/<caso>/GT_com_classes/<seed>/<caso>_<frame>_<seed>.pt
cada um um dict com as versões (768 números cada):
    cls               token CLS do clip (o resumo que o modelo dá por omissão; é a versão que o B1 usa)
    mean · max · std  média, máximo e desvio-padrão dos 196 vetores dos quadrados de 16×16 px do próprio frame
Imagem: 320×320 → 224×224 (bilinear) e normalização ImageNet, como no pré-treino do EndoFM-LV.
Modelo: o código do EndoFM-LV (pasta local) e o checkpoint, montados como no código de avaliação dos autores
(get_vit_base_patch16_224 e os pesos do "teacher" sem o prefixo "backbone."). Antes de extrair, o script pára se
alguma chave do modelo ficar por carregar ou se o ramo temporal ficar a zero, e confirma, com o próprio código do
modelo, a ordem em que saem os vetores dos quadrados.
Parâmetros e pastas em configs/b1.yaml (vetores.endofm), relativas à pasta do projeto. Retoma: as janelas já
completas são saltadas; cada ficheiro é gravado de forma atómica. Recusa escrever numa pasta com ficheiros de outra
extração (ex.: vectors_v2).

Uso, a partir da pasta do projeto, depois do source …/inicio.sh:
    python methods/extrair_endofm.py --conjunto CAL --ds "$DS"
Depois: python methods/b1_global.py --emb endofm --conjunto CAL
"""
import argparse
import datetime
import importlib
import os
import subprocess
import sys
import time
import types
from functools import partial
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "eval"))
sys.path.insert(0, str(RAIZ / "methods"))
import evaluate as ev  # noqa: E402
from b1_global import frames_das_janelas  # noqa: E402

VERSAO = "1.0"
MARCA = "extrator.txt"                  # ficheiro que identifica a pasta de saída como sendo desta extração
MEDIA = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DESVIO = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ============================================================================ imagens, clips e versões do vetor
def preparar(caminho, lado):
    """Imagem → array C×H×W normalizado (ImageNet), já no tamanho do modelo (bilinear, como o torchvision)."""
    from PIL import Image
    with Image.open(caminho) as im:
        im = im.convert("RGB").resize((lado, lado), Image.BILINEAR)
        a = np.asarray(im, dtype=np.float32) / 255.0
    return ((a - MEDIA) / DESVIO).transpose(2, 0, 1)


def indices_clip(n, k, t):
    """Índices (0 … n−1) do clip do frame k: de k−t//2 a k+t//2−1; nas pontas repete o primeiro ou o último.
    O frame fica sempre na posição t//2 do clip."""
    return [min(max(j, 0), n - 1) for j in range(k - t // 2, k - t // 2 + t)]


def quadrados_do_frame(tok, t, pos, ordem):
    """tok: B×(1+N·t)×D, a saída do modelo (CLS e vetores dos quadrados) → B×N×D, os quadrados do frame que está na
    posição pos do clip. ordem 'nt': para cada quadrado, os t frames seguidos (a do EndoFM-LV); 'tn': frame a frame."""
    q = tok[:, 1:]
    b, n_vet, d = q.shape
    if n_vet % t:
        raise ev.ErroAvaliacao(f"O modelo devolveu {n_vet} vetores de quadrados, que não se dividem por {t} frames.")
    if ordem == "nt":
        return q.reshape(b, n_vet // t, t, d)[:, :, pos]
    if ordem == "tn":
        return q.reshape(b, t, n_vet // t, d)[:, pos]
    raise ValueError(f"ordem desconhecida: {ordem}")


def versoes(tok, t, pos, ordem):
    """{versão: B×D}: o CLS do clip e a média, o máximo e o desvio-padrão dos quadrados do próprio frame."""
    q = quadrados_do_frame(tok, t, pos, ordem)
    return {"cls": tok[:, 0], "mean": q.mean(axis=1), "max": q.max(axis=1), "std": q.std(axis=1)}


# ============================================================================ modelo
def importar_timesformer(repo):
    """O módulo models/timesformer.py do EndoFM-LV, importado da pasta local. Carrega só esse módulo e os dois de que
    ele depende (models/helpers.py e models/vit_utils.py), sem passar pelo models/__init__.py, que carrega também
    outros modelos (Swin, S3D) e pediria o pacote timm. Fora do PyTorch, só precisa do pacote einops."""
    pasta = Path(repo).resolve() / "models"
    if not (pasta / "timesformer.py").is_file():
        raise ev.ErroAvaliacao(f"Não encontrei o código do EndoFM-LV em {Path(repo)} (falta models/timesformer.py); "
                               "vê configs/b1.yaml.")
    pacote = sys.modules.get("models")
    if pacote is None or [str(pasta)] != list(getattr(pacote, "__path__", [])):
        for nome in [n for n in sys.modules if n == "models" or n.startswith("models.")]:
            del sys.modules[nome]
        pacote = types.ModuleType("models")
        pacote.__path__ = [str(pasta)]                       # pacote "models" sem executar o __init__.py
        sys.modules["models"] = pacote
    try:
        return importlib.import_module("models.timesformer")
    except ModuleNotFoundError as e:
        raise ev.ErroAvaliacao(f"Falta o pacote {e.name} para o código do EndoFM-LV: instala-o no ambiente com "
                               f"python -m pip install {e.name}")


def vit(torch, tf, lado, t, depth=12, dim=768, cabecas=12):
    """O ViT-B/16 do EndoFM-LV, com os parâmetros de get_vit_base_patch16_224 (o código de avaliação dos autores).
    depth, dim e cabecas só mudam nos testes e na verificação da ordem dos vetores."""
    return tf.VisionTransformer(img_size=lado, num_classes=0, patch_size=16, embed_dim=dim, depth=depth,
                                num_heads=cabecas, mlp_ratio=4, qkv_bias=True,
                                norm_layer=partial(torch.nn.LayerNorm, eps=1e-6), drop_rate=0.,
                                attn_drop_rate=0., drop_path_rate=0.1, num_frames=t,
                                attention_type="divided_space_time")


class _Captura:
    """Guarda a saída da última normalização do modelo: o CLS e todos os vetores dos quadrados, B×(1+N·t)×D."""

    def __init__(self, modulo):
        self.saida = None
        self.gancho = modulo.register_forward_hook(self._guardar)

    def _guardar(self, modulo, entrada, saida):
        self.saida = saida[0] if isinstance(saida, tuple) else saida

    def fechar(self):
        self.gancho.remove()


def ordem_dos_tokens(torch, tf, lado, t):
    """Confirma, com o próprio código do modelo, a ordem em que saem os vetores dos quadrados. Usa uma cópia pequena
    sem blocos de atenção (que misturariam os vetores) e compara a saída com a conta feita à mão, quadrado a quadrado
    e frame a frame. Devolve 'nt' ou 'tn' (ver quadrados_do_frame); se nenhuma bater, pára."""
    g = torch.Generator().manual_seed(0)
    m = vit(torch, tf, lado, t, depth=0, dim=48, cabecas=1).eval()
    with torch.no_grad():
        m.time_embed.normal_(generator=g)
        x = torch.randn(1, 3, t, lado, lado, generator=g)
        cap = _Captura(m.norm)
        try:
            m.forward_features(x)
        finally:
            cap.fechar()
        tok = cap.saida[0, 1:]                                                  # (N·t)×D, pela ordem do modelo
        e = m.patch_embed(x)[0]                                                 # t×N×D, frame a frame
        e = m.norm(e + m.pos_embed[:, 1:] + m.time_embed[0][:, None, :])
        n = e.shape[1]
        if tuple(tok.shape) == (n * t, e.shape[2]):
            if torch.allclose(tok, e.transpose(0, 1).reshape(n * t, -1), atol=1e-5):
                return "nt"
            if torch.allclose(tok, e.reshape(t * n, -1), atol=1e-5):
                return "tn"
    raise ev.ErroAvaliacao("Não consegui confirmar a ordem dos vetores dos quadrados no código do EndoFM-LV; "
                           "não extraio nada.")


def carregar_pesos(torch, m, checkpoint):
    """Carrega os pesos do "teacher" sem o prefixo "backbone.", como no código de avaliação dos autores. Pára se
    alguma chave do modelo ficar por carregar ou tiver outra forma, ou se o ramo temporal ficar a zero (pesos
    temporais por treinar: não seria o modelo de vídeo). Devolve um resumo para o relatório."""
    try:
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)    # ficheiro dos autores do modelo
    except TypeError:                                                          # PyTorch antigo, sem weights_only
        ck = torch.load(checkpoint, map_location="cpu")
    sd = ck.get("teacher", ck.get("model", ck)) if isinstance(ck, dict) else None
    if not isinstance(sd, dict):
        raise ev.ErroAvaliacao(f"{Path(checkpoint).name} não tem um dicionário de pesos.")
    sd = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in sd.items()}
    if any(k.startswith("backbone.") for k in sd):
        sd = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
    alvo = m.state_dict()
    certas = {k: v for k, v in sd.items() if k in alvo and tuple(v.shape) == tuple(alvo[k].shape)}
    erradas = [k for k, v in sd.items() if k in alvo and tuple(v.shape) != tuple(alvo[k].shape)]
    faltam = [k for k in alvo if k not in certas]
    if faltam or erradas:
        raise ev.ErroAvaliacao(
            f"O checkpoint não encaixa no modelo: {len(certas)}/{len(alvo)} chaves carregadas"
            + (f"; em falta, ex.: {', '.join(faltam[:3])}" if faltam else "")
            + (f"; com outra forma, ex.: {', '.join(erradas[:3])}" if erradas else "") + ".")
    m.load_state_dict(certas, strict=True)
    temporal = [float(b.temporal_fc.weight.detach().abs().max()) for b in m.blocks if hasattr(b, "temporal_fc")]
    if not temporal or min(temporal) == 0:
        raise ev.ErroAvaliacao("O ramo temporal do modelo ficou a zero depois de carregar o checkpoint (pesos "
                               "temporais por treinar); confirma que o checkpoint é o do EndoFM-LV.")
    return {"chaves": f"{len(certas)}/{len(alvo)}", "sobras": len(sd) - len(certas),
            "ramo_temporal": min(temporal) if temporal else None}


def _commit(pasta):
    """Commit do repositório git do próprio código do modelo (só se a pasta tiver o seu .git; senão, "sem git")."""
    if not (Path(pasta) / ".git").exists():
        return "sem git"
    try:
        r = subprocess.run(["git", "-C", str(pasta), "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=10)
        return f"commit {r.stdout.strip()}" if r.returncode == 0 and r.stdout.strip() else "sem git"
    except Exception:                                                          # noqa: BLE001
        return "sem git"


class ModeloEndoFM:
    """EndoFM-LV pronto a usar: recebe clips (t×3×H×W, numpy) e devolve a saída completa, B×(1+N·t)×D."""

    def __init__(self, repo, checkpoint, t, lado):
        try:
            import torch
        except ImportError:
            raise ev.ErroAvaliacao("Falta o PyTorch neste ambiente.")
        if not Path(checkpoint).is_file():
            raise ev.ErroAvaliacao(f"Não encontrei o checkpoint {checkpoint} (configs/b1.yaml).")
        self.torch, self.t = torch, t
        self.disp = "cuda" if torch.cuda.is_available() else "cpu"
        tf = importar_timesformer(repo)
        self.ordem = ordem_dos_tokens(torch, tf, lado, t)
        m = vit(torch, tf, lado, t)
        carga = carregar_pesos(torch, m, checkpoint)
        self.m = m.eval().to(self.disp)
        self.cap = _Captura(self.m.norm)
        self.info = {"codigo": f"{repo} ({_commit(repo)})",
                     "checkpoint": f"{Path(checkpoint).name} sha256 {ev._sha256(checkpoint)[:16]}", **carga}

    def __call__(self, clips):
        torch = self.torch
        with torch.no_grad():
            x = torch.from_numpy(np.ascontiguousarray(np.stack(clips)))        # B×t×3×H×W
            x = x.permute(0, 2, 1, 3, 4).contiguous().to(self.disp)             # B×3×t×H×W, a entrada do modelo
            self.m.forward_features(x)
            return self.cap.saida.float().cpu().numpy()


# ============================================================================ extração
def gravar(torch, vetores, destino):
    destino.parent.mkdir(parents=True, exist_ok=True)
    tmp = destino.with_name(destino.name + ".tmp")
    torch.save({k: torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32)) for k, v in vetores.items()}, tmp)
    tmp.replace(destino)


def preparar_pasta(pasta, texto):
    """Cria a pasta de saída com a marca desta extração; recusa uma pasta que já tenha ficheiros de outra extração."""
    marca = pasta / MARCA
    if pasta.is_dir() and not marca.is_file() and any(pasta.iterdir()):
        raise ev.ErroAvaliacao(f"A pasta {pasta} já tem ficheiros de outra extração; escolhe uma pasta nova "
                               "em configs/b1.yaml (vetores.endofm.pasta).")
    pasta.mkdir(parents=True, exist_ok=True)
    if not marca.is_file():
        marca.write_text(texto + "\n", encoding="utf-8")


def extrair(fr, ds, pasta, modelo, lado, lote, progresso=True):
    t = modelo.t
    pos = t // 2
    feitas = saltadas = escritos = repetidos = 0
    sem_imagem, com_saltos = [], []
    grupos = list(fr.groupby(["split", "caso", "seed"], sort=True))
    t0 = time.time()
    for i, ((sp, caso, seed), g) in enumerate(grupos, 1):
        w = Path(sp) / caso / "GT_com_classes" / str(seed)
        frames = sorted(int(f) for f in g["frame"])                         # ordem numérica, não a do nome
        destinos = {f: pasta / w / f"{caso}_{f}_{seed}.pt" for f in frames}
        if all(d.is_file() for d in destinos.values()):
            saltadas += 1
        else:
            disp = []
            for f in frames:
                img = ds / w / f"{caso}_{f}_{seed}.jpg"
                if img.is_file():
                    disp.append((f, img))
                else:
                    sem_imagem.append(f"{sp}/{caso}/{seed}/{f}")
            if any(b - a != 1 for (a, _), (b, _) in zip(disp, disp[1:])):
                com_saltos.append(f"{sp}/{caso}/{seed}")                   # o clip passa por cima do buraco
            if disp:
                imgs = np.stack([preparar(p, lado) for _, p in disp])       # n×3×H×W, já pela ordem dos frames
                vistos = set()
                for k0 in range(0, len(disp), lote):
                    ks = list(range(k0, min(len(disp), k0 + lote)))
                    v = versoes(modelo([imgs[indices_clip(len(disp), k, t)] for k in ks]), t, pos, modelo.ordem)
                    for j, k in enumerate(ks):
                        assinatura = np.asarray(v["cls"][j], dtype=np.float32).tobytes()
                        repetidos += assinatura in vistos
                        vistos.add(assinatura)
                        gravar(modelo.torch, {c: v[c][j] for c in v}, destinos[disp[k][0]])
                        escritos += 1
            feitas += 1
        if progresso:
            dt = time.time() - t0
            print(f"\r  janelas {i}/{len(grupos)} · {escritos} frames · {dt:.0f} s"
                  + (f" · {escritos / dt:.1f} frames/s" if dt > 0 and escritos else ""), end="", flush=True)
    if progresso:
        print()
    return {"janelas": len(grupos), "feitas": feitas, "saltadas": saltadas, "escritos": escritos,
            "sem_imagem": sem_imagem, "com_saltos": com_saltos, "repetidos": repetidos,
            "segundos": time.time() - t0}


def _exemplos(lista):
    return f" (ex.: {', '.join(lista[:3])})" if lista else ""


def main(argv=None, modelo=None):
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--conjunto", required=True, choices=("CAL", "VAL_EX", "TEST_BOX"))
    a.add_argument("--ds", default=os.environ.get("DS"), help="pasta do dataset (por omissão, $DS)")
    a.add_argument("--config", default="configs/avaliacao.yaml")
    a.add_argument("--config-b1", default="configs/b1.yaml")
    a.add_argument("--sem-progresso", action="store_true")
    args = a.parse_args(argv)
    if not args.ds or not Path(args.ds).is_dir():
        raise ev.ErroAvaliacao('Falta a pasta do dataset: corre primeiro o source …/inicio.sh e usa --ds "$DS".')
    cfg = ev.ler_config(args.config)
    cfg_b1 = ev.ler_config(args.config_b1)
    conf = cfg_b1["vetores"]["endofm"]
    janelas = ev.ler_conjunto(Path(cfg["caminhos"]["conjuntos"]) / f"{args.conjunto}.json")
    fr = frames_das_janelas(cfg, janelas)
    t, lado, pasta = int(conf["frames"]), int(conf["lado"]), Path(conf["pasta"])
    git = ev._estado_git(RAIZ)
    estado = ("sem git" if git["commit"] is None else
              f"commit {git['commit']}" + (" (com alterações por guardar)" if git["alteracoes_por_guardar"] else ""))
    preparar_pasta(pasta, f"vetores EndoFM-LV de methods/extrair_endofm.py v{VERSAO} · "
                          f"{datetime.datetime.now():%Y-%m-%d %H:%M} · {estado}")
    modelo = modelo or ModeloEndoFM(conf["repo"], conf["checkpoint"], t, lado)
    r = extrair(fr, Path(args.ds), pasta, modelo, lado, int(conf["lote"]), not args.sem_progresso)
    inf = modelo.info
    ordem = {"nt": "por posição (para cada quadrado, os frames do clip seguidos)", "tn": "frame a frame"}
    L = [f"EndoFM-LV (clips de {t} frames, f−{t // 2} a f+{t - t // 2 - 1}, {lado}×{lado}) em {args.conjunto} · "
         f"{datetime.datetime.now():%Y-%m-%d %H:%M} · {estado}",
         f"extrair_endofm.py v{VERSAO} · torch {modelo.torch.__version__} · {modelo.disp}",
         f"código: {inf['codigo']} · checkpoint: {inf['checkpoint']}",
         f"chaves do modelo carregadas: {inf['chaves']} (do checkpoint por usar: {inf['sobras']}) · ramo temporal: "
         + ("ativo" if inf["ramo_temporal"] else "DESLIGADO (pesos temporais a zero)"),
         f"ordem dos vetores dos quadrados: {ordem[modelo.ordem]}, confirmada com o código do modelo",
         f"janelas: {r['janelas']} (extraídas agora {r['feitas']}, já completas {r['saltadas']}) · "
         f"frames gravados: {r['escritos']} · {r['segundos']:.0f} s",
         f"frames sem imagem: {len(r['sem_imagem'])}{_exemplos(r['sem_imagem'])} · "
         f"janelas com saltos na numeração: {len(r['com_saltos'])}{_exemplos(r['com_saltos'])}",
         f"frames com o cls igual ao de outro frame da mesma janela: {r['repetidos']} (deve ser 0)",
         f"vetores em {pasta}"]
    texto = "\n".join(L)
    saida = Path(cfg_b1["saida"])
    saida.mkdir(parents=True, exist_ok=True)
    (saida / f"extrair_endofm_{args.conjunto}_resumo.txt").write_text(texto + "\n", encoding="utf-8")
    print(texto)
    return r


if __name__ == "__main__":
    try:
        main()
    except ev.ErroAvaliacao as e:
        print(f"ERRO: {e}", file=sys.stderr)
        sys.exit(2)
