# Instruções da revisão — Guião C

Versão 1 (24 out 2026). Valem para toda a revisão. Depois do piloto (C4) não mudam; qualquer mudança dá uma versão nova, registada no `docs/decisoes.md`.

## O que se pede

Em cada caixa, decidir se a caixa proposta marca **a lesão do seed** naquele frame. Não é para diagnosticar nem para procurar outras lesões.

## O que a ferramenta mostra (`tools/rever.py`)

- **À esquerda:** o seed com a caixa da anotação original (azul-claro), a imagem do relatório e a descrição clínica. É a referência: a lesão a seguir é a que está dentro da caixa do seed.
- **À direita:** o frame a rever, com a caixa proposta (magenta). Por baixo, os frames entre o seed e esse frame, sem caixas. Um clique abre-os em grande e o Esc volta.
- **Não mostra:** o score do modelo, se o frame já tinha caixa original e se a caixa é um controlo.

## A mesma lesão

O critério é o da referência PF v2. A lesão do seed tem de estar visível, pelo menos em parte, e tens de a reconhecer com certeza como a mesma. Para isso ajudam:
- a forma e a cor;
- a posição em relação às pregas e ao que está à volta;
- a continuidade nos frames do meio.

## As respostas

| Tecla | Resposta | Quando |
| --- | --- | --- |
| A | aceitar | A caixa apanha a lesão do seed mais ou menos como a desenharias: não corta uma parte importante dela e não é muito maior (à vista, IoU ≥ 0,5). |
| M | mal posta | A lesão do seed está no frame, mas a caixa corta uma parte importante dela, é muito maior ou está ao lado. |
| R | rejeitar | A lesão do seed não está neste frame. A caixa está em mucosa, pregas, bolhas, resíduos ou reflexos. |
| O | rejeitar, com a nota «outra lesão» | A caixa está numa lesão, mas não na do seed. |
| N | não avaliável | A imagem não deixa decidir: desfocada, escura, ou tapada por bolhas ou conteúdo. |
| D | não avaliável, com a nota «dúvida» | A imagem é boa, mas não tens a certeza de que é a mesma lesão. |
| botão | não identifico a lesão no seed | Não consegues ver a lesão dentro da caixa do seed. Todas as caixas da janela ficam «não avaliável». |

**Desempates**
- **Entre aceitar e mal posta:** usa «mal posta» só quando o erro é claro. Uma caixa um pouco maior ou um pouco mais pequena aceita-se.
- **Entre aceitar e dúvida:** sem certeza de que é a mesma lesão, é D. Não se dá o benefício ao método.
- **Entre rejeitar e não avaliável:** se a imagem deixa ver bem o sítio e a lesão do seed não está lá, é R. Se a imagem não deixa ver, é N.

## Como trabalhar

- Sessões de cerca de 1 h, com pausa entre elas. A ferramenta grava cada resposta e, quando a voltas a abrir, retoma na primeira caixa sem resposta.
- Decide à primeira impressão. O Z serve só para enganos de tecla.
- Quando a lesão mudou de aspeto, sobretudo longe do seed, usa os frames de baixo.
- Usa a nota (opcional) só para o que for útil na análise de erros.
- Até consolidares a revisão, não abras o `itens_chave_v1.csv` nem o `expanded_v1.csv`.
- No fim de cada sessão, copia a pasta das respostas para `/mnt/paulo/tese_2026/copias/`.

## Pistas visuais

Servem para reconhecer a lesão, não para a diagnosticar.

- **Angiectasia:** mancha vermelho-viva, pequena e plana, às vezes com pequenos vasos à volta.
- **Erosão:** pequena falha na mucosa, com o centro esbranquiçado ou amarelado e um halo vermelho.
- **Úlcera:** como a erosão, mas maior e mais funda, com o fundo branco-amarelado.
- **Pólipo:** saliência arredondada que sobressai da parede.
- **Eritema:** zona avermelhada, plana, de contornos difusos.
- **Xantelasma:** pequena placa branco-amarelada.
