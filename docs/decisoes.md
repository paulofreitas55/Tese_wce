
## 05/10/2026 — Passo 0.3 (rótulos clínicos)
- 14 categorias: 12 de lesão + nao_lesao + qualidade. Critérios em docs/criterios_categorias.md; mapa final em data/derived/label_map.csv (728 descrições).
- Método: regras de palavras + revisão linha a linha com apoio de IA. Não houve novo diagnóstico: só se traduziu o texto do médico. 12 correções e 3 textos sem regra, listados em outputs/05_rotulos_resumo.txt.
- Linfangiectasia tem categoria própria (antes contava como angiectasia).
- Lesao_no_ileon e Congestao_da_mucosa: patologia (outra_patologia).
- Sangue conta como lesão.
- Duas lesões no mesmo texto: fica a primeira mencionada. Lesão de tipo incerto: outra_patologia.
- Texto só com o local (ex.: Ileo_proximal), mesmo com caixa no seed: não-lesão, fora dos conjuntos de lesão (14 janelas).
- Seeds com duas descrições (0167/8391 e 0527/6944): fica a patologia; nenhuma das duas tem caixas.
- As 747 descrições do diagnóstico antigo são 728: 19 só diferiam num ponto ou espaço no fim do nome.
- Diagnóstico antigo → agora: ALL-SEEDBOX 1 346 → 1 368; patologia sem caixa no seed 605 → 634; não-lesão com caixa 35 → 31.

- 2026-10-06: passo 0.1 fechado (01c). Versão do dataset (sha256 do conteúdo): 8ffdf8044e771295b94b24c6802c754e1215861c81a667906383f90da2d435aa. 12 seeds sem janela em docs/seeds_sem_janela.csv: 8 do caso 0118 (sem frames) e 4 sem emparelhamento com o vídeo (0045/6633, 0048/11409, 0048/3021, 0048/6014). 140 imagens vazias: 139 nas 2 janelas excluídas e 1 em Dataset_Treino/0657/GT_com_classes/no_comments, fora de qualquer janela (sem efeito).

- 2026-10-06: passo 0.8 fechado: eval/evaluate.py com os 22 testes a passar.
