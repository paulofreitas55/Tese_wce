"""
Testes da extração EndoFM-LV (methods/extrair_endofm.py), com imagens inventadas e um modelo falso que devolve os
vetores pela mesma ordem do EndoFM-LV verdadeiro. Testam o clip de cada frame (ordem numérica, pontas repetidas), a
leitura dos quadrados do próprio frame (o erro da extração antiga), a gravação, a retoma e a ligação ao B1 e ao
avaliador. Com o PyTorch e o código do EndoFM-LV presentes (a pasta de configs/b1.yaml, no PC), testam também, com o
código verdadeiro do modelo, a ordem dos vetores e a leitura de um checkpoint com pesos inventados.
Nunca leem dados reais.
    python tests/test_endofm.py          ou          python -m pytest tests/ -q
"""
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
from test_dinov2 import _com_torch, _imagens  # noqa: E402
from test_evaluate import Saltar, _projeto, ev, saltar  # noqa: E402


def _carregar(nome, ficheiro):
    spec = importlib.util.spec_from_file_location(nome, AQUI.parent / "methods" / ficheiro)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _projeto_1000(tmp):
    """O mini-projeto dos testes do avaliador, com a janela 0101 movida para o seed 1000 (frames 990 a 1010): atravessa
    999 → 1000, onde a ordem pelo nome do ficheiro ("1000" antes de "990") não é a ordem dos frames."""
    raiz, conj, _ = _projeto(tmp)
    for nome in ("annotations_v1.csv", "paulo_frame_labels.csv", "janelas_v1.csv"):
        f = raiz / "data/derived" / nome
        x = pd.read_csv(f, dtype={"caso": str}, keep_default_na=False)
        m = x["caso"] == "0101"
        x.loc[m, "seed"] = 1000
        if "frame" in x:
            x.loc[m, "frame"] = 1000 + x.loc[m, "offset"].astype(int)
        x.to_csv(f, index=False)
    c = json.loads((raiz / "sets/CAL.json").read_text())
    for j in c["janelas"]:
        if j["caso"] == "0101":
            j["seed"] = 1000
    (raiz / "sets/CAL.json").write_text(json.dumps(c))
    conj["CAL"] = [(s, k, 1000 if k == "0101" else e, cat) for s, k, e, cat in conj["CAL"]]
    return raiz, conj


def _config_b1(raiz, lote=8):
    (raiz / "configs/b1.yaml").write_text(
        "versao: 4\nvetores:\n  endofm:\n"
        f"    pasta: {raiz / 'vec_endofm'}\n    chaves: [cls, mean, max, std]\n"
        f"    repo: {raiz / 'sem_codigo'}\n    checkpoint: {raiz / 'sem_checkpoint.pth'}\n"
        f"    frames: 8\n    lado: 224\n    lote: {lote}\n"
        f"saida: {raiz / 'outputs/preds'}\n", encoding="utf-8")


class _ModeloFalso:
    """Faz de EndoFM-LV: cada quadrado de 16×16 px de cada frame dá 768 números (a cor média projetada); devolve o CLS
    e os quadrados pela ordem do modelo verdadeiro ('nt': para cada quadrado, os 8 frames do clip seguidos). O CLS
    depende dos 8 frames e da posição de cada um no clip."""

    def __init__(self, torch, ordem="nt"):
        self.torch, self.disp, self.t, self.ordem, self.chamadas = torch, "cpu", 8, ordem, 0
        self.W = np.random.default_rng(0).normal(size=(3, 768)).astype(np.float32)
        self.info = {"codigo": "falso", "checkpoint": "falso", "chaves": "0/0", "sobras": 0, "ramo_temporal": 1.0}

    def quadrados(self, imgs):                                      # k×3×224×224 → k×196×768
        k = imgs.shape[0]
        return imgs.reshape(k, 3, 14, 16, 14, 16).mean(axis=(3, 5)).reshape(k, 3, 196).transpose(0, 2, 1) @ self.W

    def cls(self, clip):                                            # 8×3×224×224 → 768
        return (self.quadrados(clip).mean(axis=1) * np.arange(1, 9)[:, None]).sum(axis=0)

    def __call__(self, clips):
        self.chamadas += 1
        x = np.stack(clips)                                         # B×8×3×224×224
        b = x.shape[0]
        q = self.quadrados(x.reshape(b * 8, *x.shape[2:])).reshape(b, 8, 196, 768)
        seq = q.transpose(0, 2, 1, 3).reshape(b, 196 * 8, 768) if self.ordem == "nt" else q.reshape(b, 8 * 196, 768)
        return np.concatenate([np.stack([self.cls(c) for c in x])[:, None], seq], axis=1)


def test_clip_de_cada_frame():
    ex = _carregar("extrair_endofm", "extrair_endofm.py")
    assert ex.indices_clip(201, 100, 8) == list(range(96, 104))             # f−4 … f+3
    assert ex.indices_clip(201, 0, 8) == [0, 0, 0, 0, 0, 1, 2, 3]           # ponta: repete o primeiro
    assert ex.indices_clip(201, 200, 8) == [196, 197, 198, 199, 200, 200, 200, 200]
    assert all(ex.indices_clip(201, k, 8)[4] == k for k in range(201))      # o frame fica sempre na posição 4
    assert len({tuple(ex.indices_clip(201, k, 8)) for k in range(201)}) == 201   # nenhum clip repetido
    assert ex.indices_clip(3, 1, 8) == [0, 0, 0, 0, 1, 2, 2, 2]             # janela mais curta do que o clip


def test_quadrados_do_proprio_frame():
    """O modelo devolve os quadrados por posição (para cada quadrado, os 8 frames). A extração antiga lia-os frame a
    frame e ficava, no "frame 4", com uma faixa da imagem (quadrados 98 a 122) misturada pelos 8 frames."""
    ex = _carregar("extrair_endofm", "extrair_endofm.py")
    t, n = 8, 196
    tok = np.zeros((1, 1 + n * t, 2))
    for q in range(n):
        for f in range(t):
            tok[0, 1 + q * t + f] = (f, q)                                  # ordem 'nt', a do EndoFM-LV
    certo = ex.quadrados_do_frame(tok, t, 4, "nt")[0]
    assert (certo[:, 0] == 4).all() and (certo[:, 1] == np.arange(n)).all()   # só o frame 4, os 196 quadrados
    antigo = tok[:, 1:].reshape(1, t, 14, 14, 2)[0, 4].reshape(-1, 2)        # a leitura da extração antiga
    assert set(antigo[:, 0]) == set(range(t)) and (antigo[:, 1].min(), antigo[:, 1].max()) == (98, 122)
    tn = np.concatenate([tok[:, :1], tok[:, 1:].reshape(1, n, t, 2).transpose(0, 2, 1, 3).reshape(1, n * t, 2)], axis=1)
    assert np.array_equal(ex.quadrados_do_frame(tn, t, 4, "tn"), certo[None])


def _extracao_retoma_e_b1(torch):
    ex = _carregar("extrair_endofm", "extrair_endofm.py")
    b1 = _carregar("b1_global", "b1_global.py")
    with tempfile.TemporaryDirectory() as tmp:
        raiz, conj = _projeto_1000(tmp)
        _imagens(raiz, conj["CAL"], falta={("0102", 4)})
        _config_b1(raiz)
        cfg = str(raiz / "configs/avaliacao.yaml")
        base = ["--conjunto", "CAL", "--ds", str(raiz / "DS"), "--config", cfg,
                "--config-b1", str(raiz / "configs/b1.yaml"), "--sem-progresso"]
        modelo = _ModeloFalso(torch)
        r = ex.main(base, modelo=modelo)
        assert r["janelas"] == 3 and r["feitas"] == 3 and r["escritos"] == 62 and r["repetidos"] == 0
        assert r["sem_imagem"] == ["Dataset_Treino/0102/800/804"] and r["com_saltos"] == ["Dataset_Treino/0102/800"]
        assert modelo.chamadas == 9                                         # 3 janelas × 3 lotes de até 8 clips
        assert (raiz / "vec_endofm/extrator.txt").is_file()
        # janela 0101 (990 a 1010, atravessa 999 → 1000): o clip de cada frame é f−4 … f+3 por ordem numérica, com as
        # pontas repetidas; mean, max e std vêm só dos quadrados do próprio frame
        pasta_img = raiz / "DS/Dataset_Treino/0101/GT_com_classes/1000"
        img = {f: ex.preparar(pasta_img / f"0101_{f}_1000.jpg", 224) for f in range(990, 1011)}
        for f in range(990, 1011):
            o = torch.load(raiz / f"vec_endofm/Dataset_Treino/0101/GT_com_classes/1000/0101_{f}_1000.pt",
                           map_location="cpu")
            assert sorted(o) == ["cls", "max", "mean", "std"] and all(np.asarray(o[k]).shape == (768,) for k in o)
            v = {k: np.asarray(o[k], dtype=np.float64) for k in o}
            clip = np.stack([img[min(max(g, 990), 1010)] for g in range(f - 4, f + 4)])
            q = modelo.quadrados(img[f][None])[0]
            assert np.allclose(v["cls"], modelo.cls(clip), rtol=1e-4, atol=1e-3)
            assert np.allclose(v["mean"], q.mean(axis=0), atol=1e-4) and np.allclose(v["max"], q.max(axis=0), atol=1e-4)
            assert np.allclose(v["std"], q.std(axis=0), atol=1e-4)
        assert not list((raiz / "vec_endofm").rglob("*.tmp"))
        # retoma: as janelas completas são saltadas; a que tem uma imagem em falta volta a ser vista
        de_novo = ex.main(base, modelo=modelo)
        assert de_novo["saltadas"] == 2 and de_novo["feitas"] == 1 and de_novo["escritos"] == 20
        # B1 com estes vetores: 4 versões, scores em [0, 1], e o avaliador aceita o CSV
        rb = b1.main(["--emb", "endofm", "--conjunto", "CAL", "--config", cfg,
                      "--config-b1", str(raiz / "configs/b1.yaml"), "--sem-progresso"])
        assert sorted(rb["csv"]) == ["cls", "max", "mean", "std"] and rb["sem_vetor"] == ["Dataset_Treino/0102/800/804"]
        p = pd.read_csv(rb["csv"]["cls"], dtype={"caso": str})
        assert len(p) == 59 and p["score"].between(0, 1).all()
        assert ev.main(["calibrar", "--metodo", "B1_endofm_cls", "--pred", str(rb["csv"]["cls"]), "--config", cfg]) == 0


def _pasta_de_outra_extracao(torch):
    ex = _carregar("extrair_endofm", "extrair_endofm.py")
    with tempfile.TemporaryDirectory() as tmp:
        raiz, _ = _projeto_1000(tmp)
        _config_b1(raiz)
        (raiz / "vec_endofm/Dataset_Treino").mkdir(parents=True)              # como os vectors_v2 antigos
        try:
            ex.main(["--conjunto", "CAL", "--ds", str(raiz), "--config", str(raiz / "configs/avaliacao.yaml"),
                     "--config-b1", str(raiz / "configs/b1.yaml"), "--sem-progresso"], modelo=_ModeloFalso(torch))
        except ev.ErroAvaliacao as e:
            assert "outra extração" in str(e)
        else:
            raise AssertionError("devia ter recusado a pasta")


test_extracao_retoma_e_b1 = _com_torch(_extracao_retoma_e_b1)
test_pasta_de_outra_extracao = _com_torch(_pasta_de_outra_extracao)


def _repo_endofm():
    """Pasta do código do EndoFM-LV: $ENDOFM_REPO ou a de configs/b1.yaml; None se não existir."""
    r = os.environ.get("ENDOFM_REPO")
    if not r:
        try:
            r = ev.ler_config(AQUI.parent / "configs/b1.yaml")["vetores"]["endofm"].get("repo")
        except Exception:                                                   # noqa: BLE001
            r = None
    if not r:
        return None
    p = Path(r) if Path(r).is_absolute() else AQUI.parent / r              # relativa à pasta do projeto
    return p if (p / "models/timesformer.py").is_file() else None


def test_codigo_verdadeiro_do_endofm():
    """Com o código verdadeiro do EndoFM-LV: a ordem dos vetores sai 'nt', um checkpoint lê-se por inteiro e um
    checkpoint com uma chave em falta ou com o ramo temporal a zero é recusado."""
    try:
        import torch
    except ImportError:
        saltar("sem PyTorch")
    repo = _repo_endofm()
    if repo is None:
        saltar("sem o código do EndoFM-LV")
    ex = _carregar("extrair_endofm", "extrair_endofm.py")
    tf = ex.importar_timesformer(repo)
    assert ex.ordem_dos_tokens(torch, tf, 224, 8) == "nt"
    # checkpoint com pesos inventados, num modelo pequeno com a mesma estrutura (2 blocos)
    pequeno = lambda: ex.vit(torch, tf, 224, 8, depth=2, dim=48, cabecas=4)  # noqa: E731
    pesos = {"backbone." + k: torch.randn_like(v) for k, v in pequeno().state_dict().items()}
    pesos["head.last_layer.weight"] = torch.zeros(3)                        # a cabeça do treino fica de fora
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "ck.pth"
        torch.save({"teacher": pesos, "epoch": 1}, f)
        m = pequeno()
        r = ex.carregar_pesos(torch, m, f)
        n = len(m.state_dict())
        assert r["chaves"] == f"{n}/{n}" and r["sobras"] == 0 and r["ramo_temporal"] > 0
        assert all(torch.equal(v, pesos["backbone." + k]) for k, v in m.state_dict().items())
        torch.save({"teacher": {k: (torch.zeros_like(v) if "temporal_fc" in k else v) for k, v in pesos.items()}}, f)
        try:
            ex.carregar_pesos(torch, pequeno(), f)
        except ev.ErroAvaliacao as e:
            assert "ramo temporal" in str(e)
        else:
            raise AssertionError("devia ter parado com o ramo temporal a zero")
        del pesos["backbone.blocks.1.temporal_fc.weight"]
        torch.save({"teacher": pesos}, f)
        try:
            ex.carregar_pesos(torch, pequeno(), f)
        except ev.ErroAvaliacao as e:
            assert "em falta" in str(e) and "temporal_fc" in str(e)
        else:
            raise AssertionError("devia ter parado com uma chave em falta")


if __name__ == "__main__":
    testes = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    falhas = 0
    for nome, f in testes:
        try:
            f()
            print(f"  ok      {nome}")
        except Saltar as e:
            print(f"  saltado {nome} ({e})")
        except Exception as e:                                              # noqa: BLE001
            falhas += 1
            print(f"  FALHOU  {nome}: {type(e).__name__}: {e}")
    sys.exit(1 if falhas else 0)
