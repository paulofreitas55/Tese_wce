"""
Testes do scripts/25_yolo_congelar.py (v1.0, 27 out 2026): num mini-projeto inventado (o do test_yolo_prever.py, com
dois treinos falsos, as previsões do script 24 com o detetor falso e as calibrações de CAL), confere o congelamento, os
comandos da avaliação única (que correm) e as recusas quando alguma coisa não bate. Nunca leem dados reais.
    python tests/test_yolo_congelar.py          ou          python -m pytest tests/ -q
"""
import importlib.util
import sys
import tempfile
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
from test_evaluate import Saltar, ev, saltar  # noqa: E402
import test_yolo_prever as tp  # noqa: E402

MODELOS = {"YOLO_ORIG": "original", "YOLO_V2": "v2"}
RESULTS = ("               epoch,      train/box_loss,      train/obj_loss,      train/cls_loss,   metrics/precision,"
           "      metrics/recall,     metrics/mAP_0.5,metrics/mAP_0.5:0.95,        val/box_loss,        val/obj_loss,"
           "        val/cls_loss,               x/lr0,               x/lr1,               x/lr2\n"
           "0,0.1,0.02,0.03,0.30,0.20,0.25,0.10,0.1,0.02,0.03,0.07,0.003,0.003\n"
           "1,0.09,0.02,0.03,0.40,0.30,0.35,0.15,0.1,0.02,0.03,0.05,0.005,0.005\n"
           "2,0.08,0.02,0.03,0.42,0.31,0.33,0.14,0.1,0.02,0.03,0.03,0.007,0.007\n")


def _carregar(nome, ficheiro):
    spec = importlib.util.spec_from_file_location(nome, AQUI.parent / "scripts" / ficheiro)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _opt(raiz, variante, **muda):
    o = {"weights": "yolov5x.pt", "cfg": "", "data": str(raiz / "data/derived/yolo" / variante / "treino.yaml"),
         "hyp": {"lr0": 0.01}, "epochs": 100, "batch_size": 8, "imgsz": 416, "rect": False, "cache": None,
         "optimizer": "SGD", "single_cls": False, "workers": 2, "patience": 20, "seed": 0}
    o.update(muda)
    return o


def _montar(tmp, opt_muda=None, calibrar_com_subconjunto=True):
    import yaml
    raiz, cfg, pasta, args24 = tp._montar(tmp)
    m24 = _carregar("yolo_prever", "24_yolo_prever.py")
    for n, v in MODELOS.items():
        (raiz / "data/derived/yolo" / v).mkdir(parents=True, exist_ok=True)
        (raiz / "data/derived/yolo" / v / "treino.yaml").write_text(f"path: {v}\n", encoding="utf-8")
        run = raiz / "outputs/yolo/runs" / n
        (run / "weights").mkdir(parents=True)
        (run / "weights/best.pt").write_bytes(f"pesos de {n}".encode())
        o = _opt(raiz, v, **((opt_muda or {}).get(n, {})))
        (run / "opt.yaml").write_text(yaml.safe_dump(o, sort_keys=False), encoding="utf-8")
        (run / "hyp.yaml").write_text("lr0: 0.01\n", encoding="utf-8")
        (run / "results.csv").write_text(RESULTS, encoding="utf-8")
        a = [x for x in args24]
        a[a.index("--modelo") + 1] = n
        a[a.index("--pesos") + 1] = str(run / "weights/best.pt")
        m24.main(a, detetor=tp.Falso())
        cal = ["calibrar", "--metodo", n, "--pred", str(raiz / f"outputs/preds/pred_{n}_CAL.csv"), "--config", cfg]
        if calibrar_com_subconjunto:
            cal += ["--conjunto-ficheiro", str(pasta / "CAL_6classes.csv")]
        assert ev.main(cal) == 0
    args = ["--config", cfg, "--dados", str(raiz / "data/derived/yolo"), "--conjuntos", str(pasta),
            "--runs", str(raiz / "outputs/yolo/runs"), "--prever", str(raiz / "outputs/yolo"),
            "--previsoes", str(raiz / "outputs/preds"), "--script24", str(AQUI.parent / "scripts/24_yolo_prever.py"),
            "--saida-configs", str(raiz / "configs"), "--saida", str(raiz / "outputs")]
    return raiz, cfg, pasta, args


def _erro(m, args, contem):
    try:
        m.main(args)
    except ev.ErroAvaliacao as e:
        assert contem in str(e), f"a mensagem não diz '{contem}': {e}"
        return
    raise AssertionError("devia ter parado")


def _yaml():
    try:
        import yaml
    except ImportError:
        saltar("falta o PyYAML")
    return yaml


def test_congela_e_os_comandos_correm():
    yaml = _yaml()
    m = _carregar("yolo_congelar", "25_yolo_congelar.py")
    with tempfile.TemporaryDirectory() as tmp:
        raiz, cfg, pasta, args = _montar(tmp)
        r = m.main(args)
        for n in MODELOS:
            f = raiz / "configs" / f"final_{n}.yaml"
            assert f.read_text(encoding="utf-8").startswith("# Congelado na experiência B")
            y = yaml.safe_load(f.read_text(encoding="utf-8"))
            cal = ev.ler_config(raiz / "outputs/eval/CAL" / f"{n}.json")
            assert y["pontos_operacao"]["OP_F1"]["tau"] == cal["pontos_operacao"]["OP_F1"]["tau"]
            assert y["modelo"]["pesos_sha256"] == ev._sha256(raiz / "outputs/yolo/runs" / n / "weights/best.pt")
            assert y["modelo"]["treino"]["melhor_epoca"] == 1 and y["modelo"]["treino"]["epocas_feitas"] == 3
            assert y["previsoes"]["TEST_BOX"]["sha256"] == ev._sha256(raiz / f"outputs/preds/pred_{n}_TEST_BOX.csv")
            assert y["conjuntos_teste"] == ["VAL_EX", "TEST_BOX"] and y["modelo"]["yolov5"] == "falso"
        t = r["texto"]
        assert t.count("evaluate.py avaliar") == 4 and t.count("--conjunto-ficheiro") == 4
        assert t.count("evaluate.py comparar") == 2 and "--a YOLO_V2 --b YOLO_ORIG" in t
        assert t.count("python val.py") == 4 and "git tag yolo-congelado" in t
        assert r["linha"].startswith("- ") and "tag yolo-congelado" in r["linha"] and "YOLO_V2 OP-F1 τ" in r["linha"]
        # os comandos da avaliação correm
        for linha in t.splitlines():
            linha = linha.strip()
            if linha.startswith("python eval/evaluate.py avaliar") or linha.startswith("python eval/evaluate.py comparar"):
                assert ev.main(linha.split()[2:] + ["--config", cfg]) == 0, linha
        # congela-se uma vez: depois de avaliar, nem com --forcar
        _erro(m, args, "Já existem")
        _erro(m, args + ["--forcar"], "Já há avaliações de teste")


def test_recusa_um_treino_fora_do_desenho():
    _yaml()
    m = _carregar("yolo_congelar", "25_yolo_congelar.py")
    with tempfile.TemporaryDirectory() as tmp:
        raiz, cfg, pasta, args = _montar(tmp, opt_muda={"YOLO_V2": {"batch_size": 16}})
        _erro(m, args, "batch_size = 16")
    with tempfile.TemporaryDirectory() as tmp:
        raiz, cfg, pasta, args = _montar(tmp, opt_muda={"YOLO_ORIG": {"data": "x/v2/treino.yaml"}})
        _erro(m, args, "devia ser")
        assert not (raiz / "configs/final_YOLO_V2.yaml").exists()               # nada gravado


def test_recusa_quando_nao_bate():
    _yaml()
    m = _carregar("yolo_congelar", "25_yolo_congelar.py")
    with tempfile.TemporaryDirectory() as tmp:
        raiz, cfg, pasta, args = _montar(tmp)
        # os pesos mudaram depois das previsões
        w = raiz / "outputs/yolo/runs/YOLO_V2/weights/best.pt"
        antes = w.read_bytes()
        w.write_bytes(antes + b"!")
        _erro(m, args, "mudaram depois das previsões")
        w.write_bytes(antes)
        # as previsões de CAL mudaram depois da calibração (e já não são as do resumo do 24)
        p = raiz / "outputs/preds/pred_YOLO_ORIG_CAL.csv"
        antes = p.read_bytes()
        p.write_bytes(antes + b"\n")
        _erro(m, args, "não é o da última corrida")
        p.write_bytes(antes)
        # falta uma previsão de teste
        t = raiz / "outputs/preds/pred_YOLO_V2_TEST_BOX.csv"
        t.rename(t.with_suffix(".bak"))
        _erro(m, args, "Faltam as previsões de YOLO_V2 em TEST_BOX")
        t.with_suffix(".bak").rename(t)
        assert not (raiz / "configs/final_YOLO_ORIG.yaml").exists()
        assert m.main(args)["linha"]                                            # com tudo certo, congela
    with tempfile.TemporaryDirectory() as tmp:
        raiz, cfg, pasta, args = _montar(tmp, calibrar_com_subconjunto=False)
        _erro(m, args, "não foi feita com")


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
