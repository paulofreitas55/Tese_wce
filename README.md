# Propagação de anotações de lesões em vídeos de cápsula endoscópica

Código da componente prática da dissertação de mestrado em Engenharia Biomédica (Informática Médica), UTAD.
Autor: Paulo Freitas.

**Tarefa:** dado o frame de um seed, com a sua caixa e a descrição clínica do relatório, encontrar em que frames da janela de ±100 a **mesma lesão** está visível, e onde. Não é deteção genérica de lesões.

> **Os dados não estão neste repositório.** O dataset, as tabelas derivadas, a anotação de referência e os resultados são privados e ficam só no PC de trabalho. Ver [Regras do repositório](#regras-do-repositório).

## Estado

| Fase | Estado |
| --- | --- |
| Etapa 0: dados, conjuntos, anotação de referência, avaliador | concluída (5–10 out 2026) |
| Guião A: baselines B1–B3, SAM 2, calibração, avaliação final | em curso |
| Validação humana das anotações propagadas | por fazer |

## Estrutura

| Pasta | Conteúdo |
| --- | --- |
| `scripts/` | passos da preparação dos dados, numerados por ordem (ver [Pipeline](#pipeline)) |
| `tools/` | ferramenta de anotação (`anotar.py`, servidor local na porta 8765) |
| `eval/` | avaliador único (`evaluate.py`) |
| `methods/` | métodos do Guião A: B1, B2, B3 e SAM 2 |
| `configs/` | parâmetros e caminhos em YAML; nada fica escrito no código |
| `tests/` | testes (`pytest`) |
| `docs/` | critério de anotação, registo de decisões, versão do dataset |
| `sets/` | conjuntos de avaliação congelados (listas de janelas) e os seus sha256 |

Só no PC, fora do git: `data/` (tabelas derivadas e anotação de referência), `outputs/` (resultados), `annotations/` (anotação antiga) e o dataset.

## Instalação

```bash
conda env create -f environment.yml       # cria o ambiente sam_tese
conda activate sam_tese
git config core.hooksPath scripts/hooks   # ativa a verificação antes de cada commit (uma vez por clone)
git config user.email "<id>+<utilizador>@users.noreply.github.com"   # email noreply do GitHub
```

No PC de trabalho, cada terminal começa com `source inicio.sh`: define `$NOVA` (raiz do projeto) e `$DS` (dataset), ativa o ambiente e entra na raiz. Noutra máquina, ajustar os caminhos em `inicio.sh` e `configs/paths.yaml`.

## Dados

Dataset WCE com 320 casos (219 de treino e 101 de teste), só de leitura, no caminho `$DS`:

```
<split>/<caso>/GT_com_classes/
    imagemN_1match_at_videoframe<seed>_*.jpg           imagem do relatório que corresponde ao seed
    imagemN_<descrição clínica>*                       descrição em texto livre, no nome do ficheiro
    <seed>/<caso>_<frame>_<seed>.jpg                   janela de ±100 frames
    <seed>/Findings_<seed>/<caso>_<frame>_<seed>.txt   uma caixa xyxy em píxeis; vazio = sem caixa
```

- `split` ∈ {Dataset_Treino, Dataset_Teste}; imagens de 320×320 (o caso 0601 tem 256×256).
- O relatório clínico (imagens e descrições) vem da médica; as caixas são da anotação original, feita por um aluno anterior.
- Versão: sha256 do conteúdo em `docs/DATASET_VERSION.txt`.

## Pipeline

| Script | Faz | Gera |
| --- | --- | --- |
| `00_ambiente.py` | regista a máquina e os pacotes | `outputs/ambiente.txt`, `requirements_freeze.txt` |
| `00_setup.sh` | estrutura de pastas e ligações | — |
| `01_inventario.py`, `01c_sha256.py` | inventário e sha256 de cada ficheiro | `manifest*.csv`, `docs/DATASET_VERSION.txt` |
| `02_tabela.py`, `02a_ver_caixas.py`, `03_folhas.py` | tabela única e verificação visual | `annotations_v1.csv`, `janelas_v1.csv`, folhas de contacto |
| `04_grupos.py` | janelas sobrepostas e conflitos | `conflitos.csv` |
| `05_rotulos.py` | mapa de rótulos (728 descrições → 14 categorias) | `label_map.csv` |
| `06_estatisticas.py` | estatísticas | `outputs/stats/` |
| `07_reanotar_lista.py` | lista das janelas a reanotar | `docs/janelas_reanotar.csv` |
| `08_conjuntos.py` | sorteio dos conjuntos (semente 0) | `sets/*.json`, `sets/SHA256SUMS.txt` |
| `09_kappa_intra.py` | concordância intra-anotador | `outputs/09_kappa_intra_resumo.txt` |
| `tools/anotar.py` | anotação de referência PF v2 | `data/derived/pf_v2/`, `paulo_frame_labels.csv` |
| `eval/evaluate.py` | avaliador único | `outputs/eval/` |

## Conjuntos de avaliação

| Conjunto | Janelas | Uso |
| --- | --- | --- |
| CAL | 30 (treino) | calibração e desenvolvimento |
| VAL-EX | 30 (teste) | avaliação final da presença: uma execução por método |
| TEST-BOX | 445 (teste) | recall e localização: uma execução por método |
| ALL-SEEDBOX | 1 368 (treino e teste) | anotação expandida, sem métricas |
| TRAIN-BOX | 958 (treino) | reservado para treino de um detetor (Guião B) |

Nada se afina em VAL-EX nem em TEST-BOX.

## Avaliação e testes

```bash
python eval/evaluate.py calibrar --metodo B1_endofm --pred outputs/preds/pred_B1_endofm_CAL.csv
python eval/evaluate.py avaliar  --metodo B1_endofm --conjunto VAL_EX --pred outputs/preds/pred_B1_endofm_VAL_EX.csv
python eval/evaluate.py comparar --conjunto VAL_EX --a SAM2 --b B1_endofm
python -m pytest tests/
```

`calibrar` fixa os pontos de operação em CAL; `avaliar` usa-os. Em VAL-EX e TEST-BOX, uma segunda execução do mesmo método é recusada, a não ser com `--repetir "<motivo>"`, e o motivo fica registado.

## Reprodutibilidade

- Sementes fixas (`random`, `numpy`, `torch`) = 0.
- Conjuntos congelados a 7 out 2026, com sha256 em `sets/SHA256SUMS.txt`; o `08_conjuntos.py` recusa reescrevê-los sem `--forcar`.
- Anotação de referência PF v2 congelada a 9 out 2026 (só no PC, com o sha256 ao lado).
- Cada resultado do avaliador guarda a versão do dataset, os sha256 das referências, das previsões e do avaliador, o commit, a configuração e o tempo.
- Commit antes de cada execução que conta; tag em cada marco (`etapa0`, `protocolo-v1`, `a6-congelado`, `a7-final`).

## Regras do repositório

- **Os dados nunca entram.** O `.gitignore` funciona por lista branca: só entram as pastas acima e os `.md`, `.py` e `.sh` da raiz. Dentro delas, imagens, tabelas (CSV, Excel), PDF e Word, notebooks, bases de dados, arrays e pesos ficam de fora.
- **Verificação antes de cada commit** (`scripts/hooks/pre-commit`): recusa ficheiros que não sejam código ou texto, ficheiros das pastas de dados, ficheiros com mais de 1 MB, alterações aos conjuntos congelados em `sets/` e commits sem o email noreply do GitHub.
- **Email dos commits:** o noreply do GitHub, que liga os commits à conta do autor sem mostrar o email pessoal.
- `bash scripts/git_verificar.sh` mostra o que entraria no git, sem mexer em nada.
- Código: Python ≥ 3.10; parâmetros e caminhos em `configs/*.yaml`; barra de progresso, gravação incremental e retoma nos scripts longos; cada script grava um resumo em `outputs/`.
- Commits pequenos, com mensagens curtas no imperativo (ex.: `Corrige o avaliar em CAL`), no ramo `main`.
- Cada decisão fica datada em `docs/decisoes.md`.

## Licença

Sem licença: todos os direitos reservados. Repositório privado.
