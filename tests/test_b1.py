"""
Testes do B1 (methods/b1_global.py), com vetores inventados em pastas temporárias; nunca leem dados reais.
Usam o PyTorch se existir (no PC); sem ele, um substituto mínimo só para estes testes.
    python tests/test_b1.py          ou          python -m pytest tests/ -q
"""
import importlib.util
import math
import pickle
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
from test_evaluate import Saltar, _projeto, ev, perto, saltar  # noqa: E402

_spec = importlib.util.spec_from_file_location("b1_global", AQUI.parent / "methods" / "b1_global.py")
b1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b1)


class _Tensor:                                   # substituto mínimo de um tensor (só quando não há PyTorch)
    def __init__(self, a):
        self.a = np.asarray(a, dtype=np.float32)

    def float(self):
        return self

    def numpy(self):
        return self.a


def _torch():
    """(módulo torch, função que cria um tensor, a pôr em sys.modules ou None)."""
    try:
        import torch
        return torch, (lambda a: torch.tensor(np.asarray(a, dtype=np.float32))), None
    except ImportError:
        falso = types.ModuleType("torch")
        falso.__version__ = "falso (testes)"

        def save(o, f):
            with open(f, "wb") as fh:
                pickle.dump(o, fh)

        def load(f, map_location=None, weights_only=None):
            with open(f, "rb") as fh:
                return pickle.load(fh)
        falso.save, falso.load = save, load
        return falso, _Tensor, falso


def _cos_esperado(o, chave="cls"):
    """cls: desce 0,05 por frame de distância e a ±10 é negativo (−0,2). mean: desce 0,03 por frame."""
    if chave == "mean":
        return 1 - 0.03 * abs(o)
    return -0.2 if abs(o) == 10 else 1 - 0.05 * abs(o)


def _montar_vetores(raiz, torch, tensor, janelas, sem_ficheiro=(), sem_seed=()):
    rng = np.random.default_rng(0)
    for sp, caso, seed, _ in janelas:
        pasta = raiz / "vetores" / sp / caso / "GT_com_classes" / str(seed)
        pasta.mkdir(parents=True)
        base = np.linalg.qr(rng.normal(size=(768, 768)))[0]               # eixos ortonormados ao acaso
        e1, e2, e3, e4 = base[:, 0], base[:, 1], base[:, 2], base[:, 3]
        for o in range(-10, 11):
            if (caso, o) in sem_ficheiro or (o == 0 and caso in sem_seed):
                continue
            c, m = _cos_esperado(o, "cls"), _cos_esperado(o, "mean")
            d = {"cls": (3.0 + abs(o)) * (c * e1 + math.sqrt(1 - c * c) * e2),   # a norma não deve contar
                 "mean": (1.0 + abs(o)) * (m * e3 + math.sqrt(1 - m * m) * e4)}
            d.update({k: rng.normal(size=768) for k in ("max", "std", "mean_all")})
            torch.save({k: tensor(v) for k, v in d.items()}, pasta / f"{caso}_{seed + o}_{seed}.pt")
    (raiz / "configs/b1.yaml").write_text(
        f"versao: 2\nvetores:\n  endofm:\n    pasta: {raiz / 'vetores'}\n    chaves: [cls, mean, max, std]\n"
        f"saida: {raiz / 'outputs/preds'}\n", encoding="utf-8")


def _correr(raiz, *extra):
    return b1.main(["--emb", "endofm", "--conjunto", "CAL", "--config", str(raiz / "configs/avaliacao.yaml"),
                    "--config-b1", str(raiz / "configs/b1.yaml"), "--sem-progresso", *extra])


def test_b1_cosseno_cortado_e_formato_do_avaliador():
    """3 janelas de CAL: na 0101 falta o vetor do frame +7; na 0103 falta o do seed (janela sem previsões).
    Gera um CSV por versão (cls, mean, max, std); o score é o cosseno cortado a [0, 1]; o CSV passa no avaliador."""
    try:
        import yaml  # noqa: F401
    except ImportError:
        saltar("PyYAML não instalado")
    torch, tensor, falso = _torch()
    if falso is not None:
        sys.modules["torch"] = falso
    try:
        with tempfile.TemporaryDirectory() as tmp:
            raiz, conj, _ = _projeto(tmp)
            _montar_vetores(raiz, torch, tensor, conj["CAL"], sem_ficheiro={("0101", 7)}, sem_seed={"0103"})
            r = _correr(raiz)
            assert sorted(r["csv"]) == ["cls", "max", "mean", "std"]
            for c, f in r["csv"].items():
                assert f.name == f"pred_B1_endofm_{c}_CAL.csv" and f.is_file()
            assert r["sem_vetor"] == ["Dataset_Treino/0101/500/507"] and r["sem_seed"] == ["Dataset_Treino/0103/300"]
            assert r["negativos"]["cls"] == 4 and r["negativos"]["mean"] == 0  # cls: ±10 nas duas janelas
            for c in ("cls", "mean"):
                p = pd.read_csv(r["csv"][c], dtype={"caso": str})
                assert list(p.columns) == ["split", "caso", "seed", "frame", "score"]
                assert len(p) == 19 + 20 and set(p["caso"]) == {"0101", "0102"}
                p["offset"] = p["frame"] - p["seed"]
                esperado = p["offset"].map(lambda o: max(0.0, _cos_esperado(o, c)))
                assert np.allclose(p["score"], esperado, atol=1e-5) and p["score"].between(0, 1).all()
                assert (p["offset"] != 0).all()                             # o seed não tem linha
            texto = r["resumo"].read_text(encoding="utf-8")
            assert "frames sem vetor (contam como ausentes no avaliador): 1" in texto
            assert "--metodo B1_endofm_mean" in texto
            assert ev.main(["calibrar", "--metodo", "B1_endofm_cls", "--pred", str(r["csv"]["cls"]),
                            "--config", str(raiz / "configs/avaliacao.yaml")]) == 0
            so_cls = _correr(raiz, "--chaves", "cls")                      # A7: só a versão escolhida
            assert list(so_cls["csv"]) == ["cls"]
    finally:
        if falso is not None:
            sys.modules.pop("torch", None)


def test_b1_chave_inexistente_da_erro_claro():
    try:
        import yaml  # noqa: F401
    except ImportError:
        saltar("PyYAML não instalado")
    torch, tensor, falso = _torch()
    if falso is not None:
        sys.modules["torch"] = falso
    try:
        with tempfile.TemporaryDirectory() as tmp:
            raiz, conj, _ = _projeto(tmp)
            _montar_vetores(raiz, torch, tensor, conj["CAL"])
            f = raiz / "configs/b1.yaml"
            f.write_text(f.read_text(encoding="utf-8").replace("chaves: [cls, mean, max, std]", "chaves: [cls, token]"),
                         encoding="utf-8")
            try:
                _correr(raiz)
            except ev.ErroAvaliacao as e:
                assert "não tem a versão 'token'" in str(e) and "mean_all" in str(e)
            else:
                raise AssertionError("devia ter dado erro")
    finally:
        if falso is not None:
            sys.modules.pop("torch", None)


if __name__ == "__main__":
    testes = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    falhas = 0
    for nome, f in testes:
        try:
            f()
            print(f"  ok      {nome}")
        except Saltar as e:
            print(f"  saltado {nome} ({e})")
        except Exception as e:                                          # noqa: BLE001
            falhas += 1
            print(f"  FALHOU  {nome}: {type(e).__name__}: {e}")
    sys.exit(1 if falhas else 0)
