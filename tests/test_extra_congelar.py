"""
Testes do scripts/21_extra_congelar.py (v1.0, 26 out 2026): congelamento da experiência A num mini-projeto inventado
(calibrações de CAL feitas pelo avaliador, marcas das partes como as do sam2_prop.py, previsões de teste e a referência
avaliada) e as recusas quando alguma coisa não bate. Nunca leem dados reais.
    python tests/test_extra_congelar.py          ou          python -m pytest tests/ -q
"""
import importlib.util
import sys
import tempfile
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
from test_evaluate import Saltar, _projeto, ev, saltar  # noqa: E402

EXTRA = """versao: 1
etapa: experiência A
tag: extra-congelado
referencia: SAM2_L12
previsoes: {prev}
conjuntos: [VAL_EX, TEST_BOX]
metodos:
  - {{metodo: SAM2T_L12, linha: "SAM2 Tiny", script: {script}, config: {t}, parametros: {{prompt: caixa, l_max: 12}}}}
  - {{metodo: MEDSAM2_L12, linha: "MedSAM2", script: {script}, config: {m}, parametros: {{prompt: caixa, l_max: 12}}}}
comparacoes:
  - [SAM2T_L12, SAM2_L12]
  - [MEDSAM2_L12, SAM2T_L12]
  - [MEDSAM2_L12, SAM2_L12]
"""


def _carregar():
    spec = importlib.util.spec_from_file_location("extra_congelar", AQUI.parent / "scripts" / "21_extra_congelar.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _yaml():
    try:
        import yaml
    except ImportError:
        saltar("falta o PyYAML")
    return yaml


def _montar(tmp):
    """Mini-projeto: os dois modelos calibrados em CAL, com previsões de CAL, VAL_EX e TEST_BOX e as marcas das
    partes; a referência SAM2_L12 com o final_SAM2_L12.yaml e avaliada em VAL_EX e TEST_BOX."""
    raiz, conj, pred = _projeto(tmp)
    cfg = str(raiz / "configs/avaliacao.yaml")
    prev = raiz / "outputs/preds"
    prev.mkdir(parents=True)
    (raiz / "methods").mkdir()
    (raiz / "methods/sam2_prop.py").write_text("# falso\n", encoding="utf-8")
    (raiz / "ck").mkdir()
    ymls = {}
    for nome, fich, fora in (("SAM2T_L12", "sam2_tiny.yaml", False), ("MEDSAM2_L12", "medsam2.yaml", True)):
        ck = raiz / "ck" / f"{nome}.pt"
        ck.write_bytes(nome.encode())
        partes = raiz / f"partes_{nome}"
        texto = f"versao: 1\ncheckpoint: {ck}\npartes: {partes}\n"
        if fora:
            mc = raiz / "medsam2_cfg.yaml"
            mc.write_text("image_size: 512\n", encoding="utf-8")
            texto += f"modelo_cfg: {mc}\nmodelo_cfg_fora_do_pacote: true\n"
        else:
            texto += "modelo_cfg: configs/sam2.1/sam2.1_hiera_t.yaml\n"
        yml = raiz / "configs" / fich
        yml.write_text(texto, encoding="utf-8")
        marca = f"sam2_prop.py v1.3 · {fich} sha256 {ev._sha256(yml)} · prompt caixa"
        for c in ("CAL", "VAL_EX", "TEST_BOX"):
            (partes / c).mkdir(parents=True)
            (partes / c / "configuracao.txt").write_text(marca + "\n", encoding="utf-8")
        for c, deteta in (("CAL", {1, 2}), ("VAL_EX", {1, 2, 3}), ("TEST_BOX", {1, 2})):
            Path(pred(f"{nome}_{c}", conj[c], deteta)).replace(prev / f"pred_{nome}_{c}.csv")
        assert ev.main(["calibrar", "--metodo", nome, "--pred", str(prev / f"pred_{nome}_CAL.csv"),
                        "--config", cfg]) == 0
        ymls[nome] = yml
    (raiz / "configs/final_SAM2_L12.yaml").write_text("metodo: SAM2_L12\n", encoding="utf-8")
    assert ev.main(["calibrar", "--metodo", "SAM2_L12", "--pred", pred("SAM2_L12_CAL", conj["CAL"], {1, 2}),
                    "--config", cfg]) == 0
    for c in ("VAL_EX", "TEST_BOX"):
        assert ev.main(["avaliar", "--metodo", "SAM2_L12", "--conjunto", c, "--pred",
                        pred(f"SAM2_L12_{c}", conj[c], {1}), "--config", cfg]) == 0
    extra = raiz / "configs/extra.yaml"
    extra.write_text(EXTRA.format(prev=prev, script=raiz / "methods/sam2_prop.py", t=ymls["SAM2T_L12"],
                                  m=ymls["MEDSAM2_L12"]), encoding="utf-8")
    args = ["--config", cfg, "--config-extra", str(extra), "--saida-configs", str(raiz / "configs"),
            "--saida", str(raiz / "outputs")]
    return raiz, conj, pred, cfg, args, ymls


def _erro(m, args, contem):
    try:
        m.main(args)
    except ev.ErroAvaliacao as e:
        assert contem in str(e), f"a mensagem não diz '{contem}': {e}"
        return
    raise AssertionError("devia ter parado")


def test_congela_e_escreve_os_comandos():
    yaml = _yaml()
    m = _carregar()
    with tempfile.TemporaryDirectory() as tmp:
        raiz, conj, pred, cfg, args, ymls = _montar(tmp)
        r = m.main(args)
        prev = raiz / "outputs/preds"
        for nome in ("SAM2T_L12", "MEDSAM2_L12"):
            f = raiz / "configs" / f"final_{nome}.yaml"
            assert f.read_text(encoding="utf-8").startswith("# Congelado na experiência A")
            y = yaml.safe_load(f.read_text(encoding="utf-8"))
            cal = ev.ler_config(raiz / "outputs/eval/CAL" / f"{nome}.json")      # JSON é YAML válido
            assert y["pontos_operacao"]["OP_F1"]["tau"] == cal["pontos_operacao"]["OP_F1"]["tau"]
            assert y["calibracao"]["sha256"] == ev._sha256(raiz / "outputs/eval/CAL" / f"{nome}.json")
            for c in ("VAL_EX", "TEST_BOX"):
                assert y["previsoes"][c]["sha256"] == ev._sha256(prev / f"pred_{nome}_{c}.csv")
            assert y["modelo"]["checkpoint_sha256"] == ev._sha256(raiz / "ck" / f"{nome}.pt")
            assert ("modelo_cfg_sha256" in y["modelo"]) == (nome == "MEDSAM2_L12")
            assert y["geracao_previsoes"].endswith("prompt caixa") and y["referencia"]["metodo"] == "SAM2_L12"
            assert y["conjuntos_teste"] == ["VAL_EX", "TEST_BOX"] and y["parametros"]["l_max"] == 12
        texto = r["resumo"].read_text(encoding="utf-8")
        assert texto.count("evaluate.py avaliar") == 4 and texto.count("evaluate.py comparar") == 6
        assert "comparar --conjunto TEST_BOX --a MEDSAM2_L12 --b SAM2T_L12" in texto
        assert "git tag extra-congelado" in texto
        assert r["linha"].startswith("- ") and "SAM2T_L12 OP-F1 τ" in r["linha"] and "tag extra-congelado" in r["linha"]
        # os comandos da avaliação correm (o avaliar aceita as previsões congeladas)
        assert ev.main(["avaliar", "--metodo", "SAM2T_L12", "--conjunto", "VAL_EX", "--pred",
                        str(prev / "pred_SAM2T_L12_VAL_EX.csv"), "--config", cfg]) == 0
        assert ev.main(["comparar", "--conjunto", "VAL_EX", "--a", "SAM2T_L12", "--b", "SAM2_L12",
                        "--config", cfg]) == 0
        # congela-se uma vez: depois de avaliar, nem com --forcar
        _erro(m, args, "Já existem")
        _erro(m, args + ["--forcar"], "Já há avaliações de teste")


def test_recusa_quando_nao_bate():
    _yaml()
    m = _carregar()
    with tempfile.TemporaryDirectory() as tmp:
        raiz, conj, pred, cfg, args, ymls = _montar(tmp)
        prev = raiz / "outputs/preds"
        # TEST_BOX gerado com outra versão do script
        f = raiz / "partes_SAM2T_L12/TEST_BOX/configuracao.txt"
        bom = f.read_text(encoding="utf-8")
        f.write_text(bom.replace("v1.3", "v1.2"), encoding="utf-8")
        _erro(m, args, "não foram geradas da mesma maneira")
        f.write_text(bom, encoding="utf-8")
        # o YAML do modelo mudou depois de gerar as previsões
        y = ymls["MEDSAM2_L12"]
        antes = y.read_text(encoding="utf-8")
        y.write_text(antes + "# mudei\n", encoding="utf-8")
        _erro(m, args, "outro YAML")
        y.write_text(antes, encoding="utf-8")
        # as previsões de CAL mudaram depois da calibração
        p = prev / "pred_MEDSAM2_L12_CAL.csv"
        antes_cal = p.read_bytes()
        p.write_bytes(antes_cal + b"\n")
        _erro(m, args, "mudou depois da calibração")
        p.write_bytes(antes_cal)
        # falta uma previsão de teste
        t = prev / "pred_SAM2T_L12_TEST_BOX.csv"
        t.rename(t.with_suffix(".bak"))
        _erro(m, args, "Faltam as previsões de SAM2T_L12 em TEST_BOX")
        t.with_suffix(".bak").rename(t)
        # nada foi gravado pelas tentativas falhadas; com tudo certo, congela
        assert not (raiz / "configs/final_SAM2T_L12.yaml").exists()
        assert m.main(args)["linha"]


def test_recusa_se_ja_avaliado_ou_sem_referencia():
    _yaml()
    m = _carregar()
    with tempfile.TemporaryDirectory() as tmp:
        raiz, conj, pred, cfg, args, ymls = _montar(tmp)
        (raiz / "outputs/eval/TEST_BOX/SAM2_L12.json").rename(raiz / "outputs/eval/TEST_BOX/SAM2_L12.bak")
        _erro(m, args, "não tem a avaliação de TEST_BOX")
        (raiz / "outputs/eval/TEST_BOX/SAM2_L12.bak").rename(raiz / "outputs/eval/TEST_BOX/SAM2_L12.json")
        assert ev.main(["avaliar", "--metodo", "MEDSAM2_L12", "--conjunto", "TEST_BOX", "--pred",
                        str(raiz / "outputs/preds/pred_MEDSAM2_L12_TEST_BOX.csv"), "--config", cfg]) == 0
        _erro(m, args, "Já há avaliações de teste")
        assert not (raiz / "configs/final_MEDSAM2_L12.yaml").exists()


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
