#!/usr/bin/env python3
"""
Documenta a máquina e o ambiente Python do projeto. Só lê: não instala nem altera nada.

Uso (a partir da raiz do projeto, com o ambiente do projeto ativo):
    python scripts/00_ambiente.py           # relatório no ecrã + ficheiros em outputs/
    python scripts/00_ambiente.py --rede    # inclui teste de acesso à internet

Gera:
    outputs/ambiente.txt              relatório legível (serve para a secção de hardware/software da tese)
    outputs/ambiente.json             o mesmo, estruturado (reprodutibilidade)
    outputs/requirements_freeze.txt   versões exatas de todos os pacotes (pip freeze)
"""
import argparse
import datetime
import importlib.metadata as md
import json
import os
import platform
import re
import resource
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# (nome no pip, para que fase é preciso, versão mínima ou None)
PACOTES = [
    ("numpy", "Etapa 0", None),
    ("pandas", "Etapa 0", None),
    ("Pillow", "Etapa 0", None),
    ("PyYAML", "Etapa 0", None),
    ("tqdm", "Etapa 0", None),
    ("matplotlib", "Etapa 0", None),
    ("openpyxl", "Etapa 0", None),
    ("scikit-learn", "Etapa 0", None),
    ("scipy", "Etapa 0", None),
    ("pytest", "Etapa 0", None),
    ("streamlit", "Etapa 0 (ferramenta de anotação)", None),
    ("torch", "Guião A", "2.5.1"),
    ("torchvision", "Guião A", "0.20.1"),
    ("SAM-2", "Guião A (instala-se na semana 2)", None),
    ("hydra-core", "Guião A (dependência do SAM 2)", None),
    ("iopath", "Guião A (dependência do SAM 2)", None),
]
OPENCV = ["opencv-python", "opencv-python-headless", "opencv-contrib-python", "opencv-contrib-python-headless"]
URLS = [
    "https://pypi.org/simple/",
    "https://github.com",
    "https://download.pytorch.org/whl/",
    "https://dl.fbaipublicfiles.com",   # pesos do SAM 2 e do DINOv2
    "https://huggingface.co",
    "https://api.anthropic.com",        # Claude Code
]


def norm(nome):
    return nome.lower().replace("_", "-")


def versao_num(v):
    partes = re.findall(r"\d+", str(v).split("+")[0])[:3]
    return tuple(int(p) for p in partes) + (0,) * (3 - len(partes))


def correr(cmd, timeout=30):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip()
    except Exception:
        return ""


def gb(n_bytes):
    return round(n_bytes / 1e9, 1)


def info_sistema():
    d = {
        "hostname": socket.gethostname(),
        "plataforma": platform.platform(),
        "cpus": os.cpu_count(),
    }
    try:
        for linha in Path("/etc/os-release").read_text().splitlines():
            if linha.startswith("PRETTY_NAME="):
                d["so"] = linha.split("=", 1)[1].strip('"')
    except Exception:
        pass
    try:
        for linha in Path("/proc/cpuinfo").read_text().splitlines():
            if linha.startswith("model name"):
                d["cpu"] = linha.split(":", 1)[1].strip()
                break
    except Exception:
        pass
    try:
        mem = {}
        for linha in Path("/proc/meminfo").read_text().splitlines():
            chave, valor = linha.split(":", 1)
            mem[chave] = int(valor.strip().split()[0]) * 1024
        d["ram_total_gb"] = gb(mem.get("MemTotal", 0))
        d["ram_disponivel_gb"] = gb(mem.get("MemAvailable", 0))
        d["swap_gb"] = gb(mem.get("SwapTotal", 0))
    except Exception:
        pass
    return d


def info_limites():
    nomes = {
        "ficheiros_abertos (NOFILE)": resource.RLIMIT_NOFILE,
        "tamanho_ficheiro (FSIZE)": resource.RLIMIT_FSIZE,
        "memoria_virtual (AS)": resource.RLIMIT_AS,
        "dados (DATA)": resource.RLIMIT_DATA,
        "stack (STACK)": resource.RLIMIT_STACK,
    }
    d = {}
    for nome, r in nomes.items():
        soft, hard = resource.getrlimit(r)
        f = lambda x: "ilimitado" if x == resource.RLIM_INFINITY else x
        d[nome] = {"atual": f(soft), "maximo": f(hard)}
    return d


def info_disco(caminhos):
    d = {}
    for nome, p in caminhos.items():
        p = Path(p)
        if not p.exists():
            d[nome] = {"caminho": str(p), "existe": False}
            continue
        real = p.resolve()
        u = shutil.disk_usage(real)
        d[nome] = {
            "caminho": str(real),
            "sistema_ficheiros": correr(["stat", "-f", "-c", "%T", str(real)]) or "?",
            "total_gb": gb(u.total),
            "livre_gb": gb(u.free),
        }
    return d


def info_gpu_driver():
    smi = shutil.which("nvidia-smi")
    if not smi:
        return {"nvidia_smi": "não encontrado"}
    texto = correr([smi])
    d = {}
    m = re.search(r"Driver Version:\s*([\d.]+)", texto)
    d["driver"] = m.group(1) if m else None
    m = re.search(r"CUDA Version:\s*([\d.]+)", texto)
    d["cuda_maximo_do_driver"] = m.group(1) if m else None
    q = correr([smi, "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader"])
    d["gpus"] = [l.strip() for l in q.splitlines() if l.strip()]
    p = correr([smi, "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"])
    d["processos_na_gpu"] = [l.strip() for l in p.splitlines() if l.strip()]
    return d


def info_python():
    return {
        "versao": platform.python_version(),
        "executavel": sys.executable,
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "venv": os.environ.get("VIRTUAL_ENV"),
    }


def pacotes_instalados():
    inst = {}
    for dist in md.distributions():
        nome = dist.metadata.get("Name") if dist.metadata else None
        if nome:
            inst[norm(nome)] = dist.version
    return inst


def info_torch():
    try:
        import torch
    except Exception as e:
        return {"instalado": False, "erro": f"{type(e).__name__}: {e}"}
    d = {
        "instalado": True,
        "versao": torch.__version__,
        "cuda_do_torch": torch.version.cuda,
        "cuda_disponivel": torch.cuda.is_available(),
    }
    try:
        d["cudnn"] = torch.backends.cudnn.version()
    except Exception:
        d["cudnn"] = None
    if d["cuda_disponivel"]:
        d["gpus"] = []
        for i in range(torch.cuda.device_count()):
            prop = torch.cuda.get_device_properties(i)
            livre, total = torch.cuda.mem_get_info(i)
            d["gpus"].append({
                "indice": i,
                "nome": prop.name,
                "memoria_total_gb": gb(total),
                "memoria_livre_gb": gb(livre),
                "capacidade": f"{prop.major}.{prop.minor}",
            })
        try:
            d["bf16"] = torch.cuda.is_bf16_supported()
        except Exception:
            d["bf16"] = None
        try:
            t0 = time.time()
            a = torch.randn(2048, 2048, device="cuda")
            (a @ a).sum().item()
            torch.cuda.synchronize()
            d["teste_gpu"] = f"ok ({(time.time() - t0) * 1000:.0f} ms)"
            del a
            torch.cuda.empty_cache()
        except Exception as e:
            d["teste_gpu"] = f"FALHOU: {e}"
    return d


def info_opencv(instalados):
    d = {"distribuicoes": [n for n in OPENCV if n in instalados]}
    try:
        import cv2
        d["versao"] = cv2.__version__
        legacy = getattr(cv2, "legacy", None)
        d["csrt_disponivel"] = bool(
            hasattr(cv2, "TrackerCSRT_create")
            or (legacy is not None and hasattr(legacy, "TrackerCSRT_create"))
            or hasattr(getattr(cv2, "TrackerCSRT", None), "create")
        )
    except Exception as e:
        d["erro"] = f"{type(e).__name__}: {e}"
    return d


def testar_url(url):
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "diagnostico-tese"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return f"acessível ({r.status})"
    except urllib.error.HTTPError as e:
        return f"acessível ({e.code})"
    except Exception as e:
        return f"SEM ACESSO ({type(e).__name__})"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rede", action="store_true", help="testar acesso à internet")
    ap.add_argument("--dataset", default="data/external/Dataset_WCE", help="caminho do dataset")
    ap.add_argument("--saida", default="outputs", help="pasta onde gravar os ficheiros")
    args = ap.parse_args()

    raiz = Path.cwd()
    instalados = pacotes_instalados()
    r = {
        "data": datetime.datetime.now().isoformat(timespec="seconds"),
        "raiz_projeto": str(raiz),
        "commit": correr(["git", "rev-parse", "--short", "HEAD"]) or None,
        "sistema": info_sistema(),
        "limites": info_limites(),
        "disco": info_disco({"projeto": raiz, "dataset": args.dataset, "home": Path.home()}),
        "gpu_driver": info_gpu_driver(),
        "python": info_python(),
        "torch": info_torch(),
        "opencv": info_opencv(instalados),
        "pacotes": {},
        "rede": {},
    }
    for nome, fase, minimo in PACOTES:
        v = instalados.get(norm(nome))
        if v is None:
            estado = "FALTA"
        elif minimo and versao_num(v) < versao_num(minimo):
            estado = f"VERSÃO < {minimo}"
        else:
            estado = "ok"
        r["pacotes"][nome] = {"versao": v, "fase": fase, "estado": estado}
    if args.rede:
        r["rede"] = {u: testar_url(u) for u in URLS}

    # ---------- verificações ----------
    av = []  # (nível, mensagem)
    py = versao_num(r["python"]["versao"])
    if py < (3, 10, 0):
        av.append(("PROBLEMA", f"Python {r['python']['versao']}: o SAM 2 precisa de Python ≥ 3.10."))
    t = r["torch"]
    if not t.get("instalado"):
        av.append(("AVISO", "PyTorch não está instalado neste ambiente (só é preciso a partir da semana 2)."))
    else:
        if not t.get("cuda_disponivel"):
            av.append(("PROBLEMA", "PyTorch instalado mas sem acesso à GPU (cuda_disponivel = False)."))
        if r["pacotes"]["torch"]["estado"] != "ok" or r["pacotes"]["torchvision"]["estado"] != "ok":
            av.append(("AVISO", "O SAM 2 precisa de torch ≥ 2.5.1 e torchvision ≥ 0.20.1. Se este é o ambiente "
                                "antigo do EndoFM, não o atualizes (podes partir o EndoFM): os vetores já estão "
                                "calculados, por isso o mais seguro é um ambiente novo só para o projeto v2."))
        for g in t.get("gpus", []):
            if g["memoria_total_gb"] < 8:
                av.append(("AVISO", f"GPU {g['nome']} com {g['memoria_total_gb']} GB: o SAM 2 base_plus pede ≥ 8 GB "
                                    "(usar o modelo small)."))
    if r["gpu_driver"].get("processos_na_gpu"):
        av.append(("AVISO", f"Há {len(r['gpu_driver']['processos_na_gpu'])} processo(s) a usar a GPU: "
                            "confirma se são teus ou de outras pessoas (memória partilhada)."))
    faltam0 = [n for n, p in r["pacotes"].items() if p["fase"].startswith("Etapa 0") and p["estado"] != "ok"]
    if faltam0:
        av.append(("PROBLEMA", "Faltam pacotes da Etapa 0. Instala com:  pip install " + " ".join(faltam0)))
    cv = r["opencv"]
    if len(cv["distribuicoes"]) > 1:
        av.append(("PROBLEMA", f"Várias versões do OpenCV instaladas ao mesmo tempo ({', '.join(cv['distribuicoes'])}): "
                               "entram em conflito. Deixa só uma variante 'contrib' "
                               "(no servidor, opencv-contrib-python-headless é a melhor)."))
    if not any("contrib" in d for d in cv["distribuicoes"]) or not cv.get("csrt_disponivel"):
        av.append(("AVISO", "Sem OpenCV 'contrib' com o tracker CSRT (preciso no Guião A, baseline B3)."))
    disco = r["disco"].get("projeto", {})
    if disco.get("livre_gb") is not None and disco["livre_gb"] < 50:
        av.append(("AVISO", f"Só {disco['livre_gb']} GB livres no disco do projeto (recomendado ≥ 50 GB)."))
    if not r["disco"].get("dataset", {}).get("existe", True):
        av.append(("AVISO", f"Não encontrei o dataset em {args.dataset} (usa --dataset <caminho>)."))
    lim = r["limites"]
    if lim["tamanho_ficheiro (FSIZE)"]["atual"] != "ilimitado":
        av.append(("PROBLEMA", "Há um limite ao tamanho dos ficheiros (ulimit -f): pode impedir gravar checkpoints."))
    nofile = lim["ficheiros_abertos (NOFILE)"]["atual"]
    if nofile != "ilimitado" and nofile < 4096:
        av.append(("AVISO", f"Limite de ficheiros abertos baixo ({nofile}): pode falhar com DataLoaders com vários workers."))
    for u, est in r["rede"].items():
        if est.startswith("SEM ACESSO"):
            av.append(("AVISO", f"Sem acesso a {u}."))
    r["verificacoes"] = [{"nivel": n, "mensagem": m} for n, m in av]

    # ---------- relatório ----------
    L = []
    sec = lambda titulo: L.append(f"\n== {titulo} " + "=" * max(0, 60 - len(titulo)))
    it = lambda k, v: L.append(f"  {k:<28} {v}")
    L.append(f"Diagnóstico do ambiente — {r['data']}  (projeto: {raiz}, commit: {r['commit']})")
    s = r["sistema"]
    sec("Sistema")
    it("Máquina", s.get("hostname"))
    it("SO", s.get("so", s.get("plataforma")))
    it("CPU", f"{s.get('cpu', '?')}  ({s.get('cpus')} núcleos)")
    it("RAM", f"{s.get('ram_total_gb')} GB (disponível {s.get('ram_disponivel_gb')} GB) · swap {s.get('swap_gb')} GB")
    sec("Limites (ulimit)")
    for k, v in lim.items():
        it(k, f"{v['atual']} (máx. {v['maximo']})")
    sec("Disco")
    for k, v in r["disco"].items():
        if v.get("existe") is False:
            it(k, f"{v['caminho']} — não existe")
        else:
            it(k, f"{v['livre_gb']} GB livres de {v['total_gb']} GB · {v['sistema_ficheiros']} · {v['caminho']}")
    sec("GPU (driver)")
    g = r["gpu_driver"]
    if "nvidia_smi" in g:
        it("nvidia-smi", g["nvidia_smi"])
    else:
        it("Driver", g.get("driver"))
        it("CUDA máximo do driver", g.get("cuda_maximo_do_driver"))
        for linha in g.get("gpus", []):
            it("GPU", linha)
        it("Processos na GPU", "; ".join(g.get("processos_na_gpu", [])) or "nenhum")
    sec("Python")
    p = r["python"]
    it("Versão", p["versao"])
    it("Executável", p["executavel"])
    it("Ambiente", p["conda_env"] or p["venv"] or "(nenhum ativo)")
    sec("PyTorch")
    if t.get("instalado"):
        it("torch", f"{t['versao']} · CUDA do torch {t['cuda_do_torch']} · cuDNN {t.get('cudnn')}")
        it("GPU acessível", t["cuda_disponivel"])
        for gg in t.get("gpus", []):
            it(f"GPU {gg['indice']}", f"{gg['nome']} · {gg['memoria_total_gb']} GB (livre {gg['memoria_livre_gb']} GB) "
                                     f"· capacidade {gg['capacidade']}")
        if t["cuda_disponivel"]:
            it("bfloat16", t.get("bf16"))
            it("Teste de cálculo na GPU", t.get("teste_gpu"))
    else:
        it("torch", "não instalado")
    sec("OpenCV")
    it("Distribuições", ", ".join(cv["distribuicoes"]) or "nenhuma")
    it("cv2", cv.get("versao", cv.get("erro")))
    it("Tracker CSRT", cv.get("csrt_disponivel"))
    sec("Pacotes do projeto")
    for nome, info in r["pacotes"].items():
        L.append(f"  {nome:<16} {str(info['versao'] or '-'):<16} {info['estado']:<14} {info['fase']}")
    if r["rede"]:
        sec("Internet")
        for u, est in r["rede"].items():
            it(u, est)
    sec("Verificações")
    if not av:
        L.append("  Tudo em ordem.")
    for n, m in av:
        L.append(f"  [{n}] {m}")

    texto = "\n".join(L)
    print(texto)

    saida = Path(args.saida)
    saida.mkdir(parents=True, exist_ok=True)
    (saida / "ambiente.txt").write_text(texto + "\n", encoding="utf-8")
    (saida / "ambiente.json").write_text(json.dumps(r, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    freeze = correr([sys.executable, "-m", "pip", "freeze"], timeout=120)
    (saida / "requirements_freeze.txt").write_text((freeze or "# pip freeze falhou neste ambiente") + "\n",
                                                   encoding="utf-8")
    print(f"\nGravado: {saida}/ambiente.txt, {saida}/ambiente.json, {saida}/requirements_freeze.txt")


if __name__ == "__main__":
    main()
