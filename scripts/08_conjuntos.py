#!/usr/bin/env python3
"""
Passo 0.6 — Sorteio das 38 janelas novas e conjuntos de avaliação congelados.

Regra (protocolo, secção "Conjuntos de avaliação"):
  - semente 0; só janelas de lesão com caixa no seed, não excluídas;
  - cada janela nova vai para a classe com menos janelas até aí nesse conjunto (a de maior falta até
    ao alvo de 4–5 por classe); empates sorteados; dentro da classe, a janela é sorteada;
  - no máximo uma janela por grupo de sobreposição, e nenhum grupo já usado;
  - ficam de fora as janelas das folhas de contacto (o anotador viu lá as caixas dos vizinhos), as 30
    anotadas antes e todas as janelas que partilham grupo (frames) com alguma delas.
CAL = 14 reanotadas + 16 novas (treino) · VAL-EX = 8 reanotadas + 22 novas (teste).

Uso, a partir da pasta do projeto:
    python scripts/08_conjuntos.py --anotadas "$NOVA/annotations"
(da pasta dos CSV antigos só se leem os NOMES dos ficheiros, nunca o conteúdo)

Lê:    data/derived/janelas_v1.csv · docs/janelas_reanotar.csv · nomes em --anotadas e em outputs/folhas/
Gera:  sets/CAL.json · sets/VAL_EX.json · sets/TEST_BOX.json · sets/TRAIN_BOX.json · sets/ALL_SEEDBOX.json
       sets/SHA256SUMS.txt · sets/anotar_60.csv (lista para a ferramenta de anotação)
       outputs/08_conjuntos_resumo.txt   ← é este que se lê
Os conjuntos ficam congelados: o script recusa reescrevê-los (só com --forcar, e isso tem de ser declarado).
"""
import argparse
import csv
import datetime
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

PRINCIPAIS = ["angiectasia", "erosao", "ulcera", "polipo", "eritema", "xantelasma"]
CLASSES = PRINCIPAIS + ["outras"]
TREINO, TESTE = "Dataset_Treino", "Dataset_Teste"
ESPERADO = {"TEST_BOX": 445, "ALL_SEEDBOX": 1368, "TRAIN_BOX": 958}
REGRA = ("semente 0; só janelas de lesão com caixa no seed; cada nova vai para a classe com menos janelas até aí "
         "(empates sorteados), janela sorteada dentro da classe; no máximo uma janela por grupo de sobreposição e "
         "nenhum grupo já usado; fora: folhas de contacto, as 30 anotadas antes, excluídas e janelas do mesmo grupo")


def classe(cat):
    return cat if cat in PRINCIPAIS else "outras"


def nomes(pasta, ext):
    """Janelas (split, caso, seed) pelos NOMES dos ficheiros <split>__<caso>__<seed><ext>."""
    out = set()
    p = Path(pasta)
    if p.is_dir():
        for f in p.glob(f"*{ext}"):
            partes = f.stem.split("__")
            if len(partes) == 3:
                out.add(tuple(partes))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--anotadas", required=True, help="pasta dos CSV antigos (só os nomes são lidos)")
    ap.add_argument("--janelas", default="data/derived/janelas_v1.csv")
    ap.add_argument("--reanotar", default="docs/janelas_reanotar.csv")
    ap.add_argument("--folhas", default="outputs/folhas")
    ap.add_argument("--sets", default="sets")
    ap.add_argument("--resumo", default="outputs/08_conjuntos_resumo.txt")
    ap.add_argument("--total", type=int, default=30, help="janelas por conjunto (CAL e VAL-EX)")
    ap.add_argument("--semente", type=int, default=0)
    ap.add_argument("--forcar", action="store_true", help="reescrever conjuntos já congelados (declarar!)")
    args = ap.parse_args()

    for f in (args.janelas, args.reanotar):
        if not Path(f).is_file():
            sys.exit(f"Não encontrei {f}.")
    if not Path(args.anotadas).is_dir():
        sys.exit(f"Não encontrei a pasta {args.anotadas}.")
    sets = Path(args.sets)
    nomes_sets = ["CAL", "VAL_EX", "TEST_BOX", "TRAIN_BOX", "ALL_SEEDBOX"]
    existentes = [n for n in nomes_sets if (sets / f"{n}.json").exists()]
    if existentes and not args.forcar:
        sys.exit(f"Os conjuntos já estão congelados ({', '.join(existentes)} em {sets}/). Não os reescrevo. "
                 "Uma alteração exige nova versão, declarada no protocolo (--forcar).")

    with open(args.janelas, encoding="utf-8", newline="") as fh:
        jan = list(csv.DictReader(fh))
    falta = [c for c in ("split", "caso", "seed", "excluida", "lesao", "seed_tem_box", "categoria", "group_id", "n_box")
             if c not in jan[0]]
    if falta:
        sys.exit(f"{args.janelas} não tem as colunas {falta} (falta correr o 04 ou o 05?).")
    K = lambda r: (r["split"], r["caso"], r["seed"])                                  # noqa: E731
    info = {K(r): r for r in jan}
    elegivel = lambda r: r["excluida"] == "0" and r["lesao"] == "1" and r["seed_tem_box"] == "1"  # noqa: E731

    antigas = nomes(args.anotadas, ".csv")
    folhas = nomes(args.folhas, ".jpg")
    with open(args.reanotar, encoding="utf-8", newline="") as fh:
        reanot = [(r["split"], r["caso"], r["seed"]) for r in sorted(csv.DictReader(fh), key=lambda r: int(r["ordem"]))]
    problemas = [k for k in reanot if k not in info or not elegivel(info[k]) or k not in antigas]
    if problemas:
        sys.exit(f"Janelas da lista de reanotação que não são válidas: {problemas[:5]}")

    vistas = antigas | folhas
    bloqueados = {info[k]["group_id"] for k in vistas if k in info}
    cand = [r for r in jan if elegivel(r) and K(r) not in vistas and r["group_id"] not in bloqueados]
    rng = random.Random(args.semente)
    usados = set(bloqueados)
    conj, faltas, disponiveis_ini = {}, {}, {}
    for nome, split in (("CAL", TREINO), ("VAL_EX", TESTE)):
        base = [k for k in reanot if k[0] == split]
        cont = Counter(classe(info[k]["categoria"]) for k in base)
        pool = {c: [] for c in CLASSES}
        for r in sorted((r for r in cand if r["split"] == split), key=lambda r: (r["caso"], int(r["seed"]))):
            pool[classe(r["categoria"])].append(r)
        disponiveis_ini[nome] = {c: len({r["group_id"] for r in pool[c]}) for c in CLASSES}
        novas = []
        while len(novas) < args.total - len(base):
            abertas = [c for c in CLASSES if any(r["group_id"] not in usados for r in pool[c])]
            if not abertas:
                faltas[nome] = args.total - len(base) - len(novas)
                break
            m = min(cont[c] for c in abertas)
            c = rng.choice([c for c in abertas if cont[c] == m])
            r = rng.choice([r for r in pool[c] if r["group_id"] not in usados])
            novas.append(K(r))
            usados.add(r["group_id"])
            cont[c] += 1
        conj[nome] = (base, novas)

    # ---------- conjuntos grandes
    grandes = {
        "TEST_BOX": [K(r) for r in jan if r["split"] == TESTE and elegivel(r)],
        "ALL_SEEDBOX": [K(r) for r in jan if elegivel(r)],
        "TRAIN_BOX": [K(r) for r in jan if r["split"] == TREINO and r["excluida"] == "0" and r["lesao"] == "1"
                      and int(r["n_box"] or 0) > 0],
    }

    # ---------- verificações
    V = []
    def verifica(ok, texto):
        V.append(("ok   " if ok else "FALHA") + "  " + texto)
        return ok
    cal = conj["CAL"][0] + conj["CAL"][1]
    val = conj["VAL_EX"][0] + conj["VAL_EX"][1]
    casos_tr = {r["caso"] for r in jan if r["split"] == TREINO}
    casos_te = {r["caso"] for r in jan if r["split"] == TESTE}
    tudo_ok = all([
        verifica(not (casos_tr & casos_te), "nenhum caso aparece no treino e no teste"),
        verifica(all(k[0] == TREINO for k in cal), "CAL só tem janelas de treino"),
        verifica(all(k[0] == TESTE for k in val), "VAL-EX só tem janelas de teste"),
        verifica(all(elegivel(info[k]) for k in cal + val), "todas as janelas de CAL e VAL-EX são de lesão, com caixa no seed"),
        verifica(len({info[k]["group_id"] for k in cal + val}) == len(cal + val), "nenhum par de janelas de CAL e VAL-EX partilha grupo"),
        verifica(not any(k in vistas for k in conj["CAL"][1] + conj["VAL_EX"][1]), "nenhuma janela nova foi vista antes (folhas ou anotação antiga)"),
        verifica(set(val) <= set(grandes["TEST_BOX"]), "VAL-EX está contido em TEST-BOX"),
        verifica(set(cal) <= set(grandes["ALL_SEEDBOX"]), "CAL está contido em ALL-SEEDBOX"),
        verifica(len(cal) == args.total and len(val) == args.total, f"CAL e VAL-EX com {args.total} janelas cada"),
    ])

    # ---------- conteúdo dos conjuntos e ordem de anotação
    hoje = datetime.date.today().isoformat()
    def entrada(k, origem=None):
        r = info[k]
        e = {"split": k[0], "caso": k[1], "seed": int(k[2]), "categoria": r["categoria"], "classe": classe(r["categoria"])}
        if origem:
            e["origem"] = origem
        return e
    conteudo = {}
    for nome in ("CAL", "VAL_EX"):
        base, novas = conj[nome]
        conteudo[nome] = [entrada(k, "reanotada") for k in base] + [entrada(k, "nova") for k in novas]
    for nome, ks in grandes.items():
        conteudo[nome] = [entrada(k) for k in sorted(ks, key=lambda k: (k[0], k[1], int(k[2])))]
    # lista para a ferramenta: as 22 primeiro (já em curso), depois as novas de CAL e as de VAL-EX, baralhadas
    ordem = [(k, "CAL" if k[0] == TREINO else "VAL_EX", "reanotada") for k in reanot]
    for nome in ("CAL", "VAL_EX"):
        novas = list(conj[nome][1])
        rng.shuffle(novas)
        ordem += [(k, nome, "nova") for k in novas]

    # ---------- resumo
    L = ["Passo 0.6 — sorteio das janelas novas e conjuntos congelados", f"  regra: {REGRA}"]
    L.append(f"\n  janelas vistas antes (ficam de fora, com os seus grupos): anotadas {len(antigas)} · folhas de contacto {len(folhas)}"
             f" · grupos bloqueados {len(bloqueados)}")
    L.append(f"  candidatas depois das exclusões: treino {sum(r['split'] == TREINO for r in cand)} · teste {sum(r['split'] == TESTE for r in cand)}")
    L.append("\n  Composição por classe")
    L.append(f"  {'classe':<14}{'CAL ant.':>9}{'novas':>7}{'total':>7}{'VAL-EX ant.':>13}{'novas':>7}{'total':>7}")
    for c in CLASSES:
        v = []
        for nome in ("CAL", "VAL_EX"):
            base, novas = conj[nome]
            a = sum(classe(info[k]["categoria"]) == c for k in base)
            n = sum(classe(info[k]["categoria"]) == c for k in novas)
            v += [a, n, a + n]
        L.append(f"  {c:<14}{v[0]:>9}{v[1]:>7}{v[2]:>7}{v[3]:>13}{v[4]:>7}{v[5]:>7}")
    L.append(f"  {'total':<14}{len(conj['CAL'][0]):>9}{len(conj['CAL'][1]):>7}{len(cal):>7}"
             f"{len(conj['VAL_EX'][0]):>13}{len(conj['VAL_EX'][1]):>7}{len(val):>7}")
    for nome, n in faltas.items():
        L.append(f"  ATENÇÃO: faltaram {n} janelas em {nome} (não havia candidatas suficientes)")
    L.append("\n  Tamanho dos conjuntos")
    for nome in nomes_sets:
        esp = ESPERADO.get(nome, args.total)
        L.append(f"  {nome:<12} {len(conteudo[nome]):>5}   (protocolo: {esp})" + ("" if len(conteudo[nome]) == esp else "  ← DIFERENTE"))
    L.append("\n  Verificações")
    L += ["  " + v for v in V]
    L.append("\n  As 38 janelas novas (pela ordem de anotação)")
    for i, (k, nome, origem) in enumerate(ordem, 1):
        if origem == "nova":
            L.append(f"    {i:>2}. {nome:<6} {k[0]}/{k[1]}/{k[2]} · {info[k]['categoria']}")
    L.append(f"\n  ficheiros: {sets}/*.json · {sets}/SHA256SUMS.txt · {sets}/anotar_60.csv")
    texto = "\n".join(L)
    print(texto)
    Path(args.resumo).parent.mkdir(parents=True, exist_ok=True)
    Path(args.resumo).write_text(texto + "\n", encoding="utf-8")
    # ---------- gravar (só se todas as verificações passaram)
    if not tudo_ok:
        sys.exit("\nHá verificações que falharam: não gravei nenhum conjunto. Manda-me o resumo.")
    sets.mkdir(parents=True, exist_ok=True)
    linhas_sha = []
    for nome in nomes_sets:
        doc = {"nome": nome, "versao": 1, "criado": hoje, "semente": args.semente,
               "regra": REGRA if nome in ("CAL", "VAL_EX") else "definição do protocolo (secção Conjuntos de avaliação)",
               "n": len(conteudo[nome]), "janelas": conteudo[nome]}
        p = sets / f"{nome}.json"
        p.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
        linhas_sha.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}")
    (sets / "SHA256SUMS.txt").write_text("\n".join(linhas_sha) + "\n", encoding="utf-8")
    with open(sets / "anotar_60.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ordem", "split", "caso", "seed", "categoria", "conjunto", "origem"])
        for i, (k, nome, origem) in enumerate(ordem, 1):
            w.writerow([i, *k, info[k]["categoria"], nome, origem])
    print(f"\nConjuntos congelados em {sets}/ (sha256 em {sets}/SHA256SUMS.txt).")


if __name__ == "__main__":
    main()
