"""
Testes do scripts/26_yolo_tabelas.py (v1.0, 27 out 2026): num mini-projeto inventado (o do test_yolo_congelar.py,
congelado, avaliado uma vez em VAL_EX e TEST_BOX e com as duas comparações), mais quatro logs do val.py inventados (com
as barras de progresso pelo meio), confere a tabela, a comparação, os estratos, o mAP, as figuras e a linha para o
decisoes.md, e as recusas. Nunca leem dados reais.
    python tests/test_yolo_tabelas.py          ou          python -m pytest tests/ -q
"""
import importlib.util
import sys
import tempfile
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
from test_evaluate import Saltar, ev, saltar  # noqa: E402
import test_yolo_congelar as tc  # noqa: E402

PF = "%22s" + "%11i" * 2 + "%11.3g" * 4          # o formato das linhas do val.py


def _log(map50_all):
    linhas = ["\x1b[34m\x1b[1mval: \x1b[0mdata=…, task=test, conf_thres=0.001, iou_thres=0.6",
              "                 Class     Images  Instances          P          R      mAP50   mAP50-95: 50% ━━━━━━ 54/109"
              "\r                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% ━━━━━━ 109/109",
              PF % ("all", 1729, 1734, 0.6, 0.5, map50_all, map50_all / 2),
              PF % ("angiectasia", 1729, 802, 0.7, 0.6, 0.651, 0.3),
              PF % ("eritema", 1729, 89, 0.1, 0.05, 5.2e-05, 1e-05),
              "Speed: 0.1ms pre-process, 7.6ms inference, 0.6ms NMS per image at shape (16, 3, 416, 416)"]
    return "\n".join(linhas) + "\n"


def _carregar(nome, ficheiro):
    spec = importlib.util.spec_from_file_location(nome, AQUI.parent / "scripts" / ficheiro)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _avaliado(tmp):
    try:
        import yaml  # noqa: F401
    except ImportError:
        saltar("falta o PyYAML")
    raiz, cfg, pasta, args = tc._montar(tmp)
    r = _carregar("yolo_congelar", "25_yolo_congelar.py").main(args)
    for linha in r["texto"].splitlines():
        linha = linha.strip()
        if linha.startswith("python eval/evaluate.py avaliar") or linha.startswith("python eval/evaluate.py comparar"):
            assert ev.main(linha.split()[2:] + ["--config", cfg]) == 0, linha
    logs = raiz / "outputs/logs"
    logs.mkdir(parents=True, exist_ok=True)
    for i, (n, ref) in enumerate((n, ref) for n in ("YOLO_ORIG", "YOLO_V2") for ref in ("original", "v2")):
        (logs / f"yolo_map_{n}_{ref}.log").write_text(_log(0.40 + 0.01 * i), encoding="utf-8")
    base = ["--config", cfg, "--configs", str(raiz / "configs"), "--logs", str(logs), "--saida", str(raiz / "out26")]
    return raiz, cfg, base


def test_tabela_comparacao_map_e_linha():
    m = _carregar("yolo_tabelas", "26_yolo_tabelas.py")
    with tempfile.TemporaryDirectory() as tmp:
        raiz, cfg, base = _avaliado(tmp)
        r = m.main(base + ["--sem-figuras"])
        tab = {t["metodo"]: t for t in r["tabela"]}
        assert list(tab) == ["YOLO_ORIG", "YOLO_V2"]
        j = ev.ler_config(raiz / "outputs/eval/TEST_BOX/YOLO_V2.json")
        assert tab["YOLO_V2"]["R_loc"] == j["por_op"]["OP_F1"]["micro"]["R_loc"]["valor"]
        assert tab["YOLO_ORIG"]["janelas_TEST_BOX"] == 2 and tab["YOLO_ORIG"]["janelas_VAL_EX"] == 1
        assert [(c["conjunto"], c["metrica"], c["a"], c["b"]) for c in r["comparacoes"]] == [
            ("VAL_EX", "F1", "YOLO_V2", "YOLO_ORIG"), ("TEST_BOX", "R_loc", "YOLO_V2", "YOLO_ORIG")]
        est = r["estratos"]
        assert set(est.loc[est["estrato"] == "distancia", "grupo"]) == {"1-3", "4-6", "7-12", ">12"}
        assert r["map"][("YOLO_V2", "v2")]["all"]["mAP50"] == 0.43
        assert r["map"][("YOLO_ORIG", "original")]["angiectasia"] == {"imagens": 1729, "caixas": 802, "mAP50": 0.651,
                                                                       "mAP50_95": 0.3}
        assert abs(r["map"][("YOLO_ORIG", "v2")]["eritema"]["mAP50"] - 5.2e-05) < 1e-12
        texto = r["resumo"].read_text(encoding="utf-8")
        assert "| YOLO, anotação original |" in texto and "1 execução(ões) registada(s)" in texto
        assert "0,430 / 0,215" in texto and "| todas |" in texto
        assert r["linha"].startswith("- ") and "experiência B, avaliação única" in r["linha"]
        assert "R_loc@0,3 em TEST-BOX" in r["linha"] and "contra as da v2 original 0,410, v2 0,430" in r["linha"]
        for f in ("tabela.csv", "comparacoes.csv", "estratos.csv", "map.csv", "resumo.md"):
            assert (raiz / "out26" / f).is_file()


def test_figuras():
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        saltar("falta o matplotlib")
    m = _carregar("yolo_tabelas", "26_yolo_tabelas.py")
    with tempfile.TemporaryDirectory() as tmp:
        raiz, cfg, base = _avaliado(tmp)
        m.main(base)
        for f in ("fig_yolo_distancia", "fig_yolo_classes"):
            for ext in ("png", "pdf"):
                p = raiz / "out26" / f"{f}.{ext}"
                assert p.is_file() and p.stat().st_size > 5000, p


def test_recusas():
    m = _carregar("yolo_tabelas", "26_yolo_tabelas.py")
    with tempfile.TemporaryDirectory() as tmp:
        raiz, cfg, base = _avaliado(tmp)
        log = raiz / "outputs/logs/yolo_map_YOLO_V2_v2.log"
        log.rename(log.with_suffix(".bak"))
        _erro(m, base + ["--sem-figuras"], "Falta")
        log.write_text("o val.py foi interrompido\n", encoding="utf-8")
        _erro(m, base + ["--sem-figuras"], "não tem a tabela final")
        log.with_suffix(".bak").replace(log)
        f = raiz / "outputs/eval/TEST_BOX/comparar_YOLO_V2_vs_YOLO_ORIG_R_loc_OP_F1.json"
        f.rename(f.with_suffix(".bak"))
        _erro(m, base + ["--sem-figuras"], "Falta a comparação")
        f.with_suffix(".bak").rename(f)
        g = raiz / "outputs/eval/VAL_EX/YOLO_ORIG.json"
        g.rename(g.with_suffix(".bak"))
        _erro(m, base + ["--sem-figuras"], "não está avaliado")


def _erro(m, args, contem):
    try:
        m.main(args)
    except ev.ErroAvaliacao as e:
        assert contem in str(e), f"a mensagem não diz '{contem}': {e}"
        return
    raise AssertionError("devia ter parado")


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
