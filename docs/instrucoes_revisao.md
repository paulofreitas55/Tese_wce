# Instruções da revisão — Guião C

Versão 3 (24 out 2026), para a ferramenta v1.3. Substitui a v2 (a do piloto): as três respostas e a regra não mudam; muda o que se vê. Qualquer mudança dá uma versão nova, registada no `docs/decisoes.md`.

## O que se pede

Em cada caixa a rever (roxa, cheia), decidir se marca **a lesão do seed** naquele frame. Não é para diagnosticar nem para procurar outras lesões.

## O que ele segue

A lesão do seed é **a que está dentro da caixa azul do seed**: a imagem no canto superior esquerdo, ou em grande com a tecla **S**. A descrição clínica ajuda a saber o que procurar, mas o alvo é sempre o que está na caixa azul do seed. Num texto com duas lesões, o alvo é só a que está na caixa.

## O que se vê

- Em baixo, a tira com todos os frames seguidos (do seed −12 ao +12), cada um com as suas caixas:
  - **azul**: caixa da anotação original, desenhada pelo Rodrigo (pode ter erros);
  - **roxo cheio**: a caixa a rever, com a resposta já dada (A, R, N) ou «?» se ainda não tem;
  - **roxo tracejado**: a caixa do modelo noutro frame. Não é para rever e pode estar certa ou errada.
- ← → andam frame a frame dentro da janela; um clique numa miniatura abre-a em grande. Assim segue-se a lesão desde o seed, passando pelos frames com caixa azul, até ao frame a rever.
- No frame que estás a rever nunca aparece caixa azul: nas caixas novas não existe e, nos poucos frames da lista que a têm, fica escondida de propósito.

## As três respostas

| Tecla | Resposta | Quando |
| --- | --- | --- |
| A | aceitar | A caixa roxa cheia marca a lesão do seed, mais ou menos como a desenharias. Uma caixa um pouco maior ou um pouco mais pequena aceita-se. |
| R | rejeitar | A caixa não marca a lesão do seed: está noutra coisa (mucosa, pregas, bolhas, resíduos, reflexos ou outra lesão), ou a lesão não está no frame. Mesmo que a lesão esteja noutro sítio do frame, se a caixa não está nela é R. |
| N | não dá para decidir | A imagem não deixa ver (desfocada, escura, tapada por bolhas ou conteúdo), ou não reconheces a lesão do seed, ou não tens a certeza de que é a mesma lesão. |

**É a mesma lesão** quando a reconheces com certeza pela forma, pela cor, pela posição em relação às pregas e pela continuidade nos frames vizinhos. Sem essa certeza, a resposta é N: não se dá o benefício ao método. As caixas dos outros frames (azuis ou tracejadas) são só contexto: ajudam a seguir a lesão, mas não decidem a resposta.

**Seed duvidoso.** Se a caixa azul do seed não marca nada claro (caixa do Rodrigo mal posta), escreve `seed duvidoso` na nota da primeira caixa dessa janela (Enter grava) e responde N a todas as caixas da janela.

## Como trabalhar

- Sessões de cerca de 1 h, com pausa entre elas. A ferramenta grava cada resposta e, quando a voltas a abrir, retoma na primeira caixa sem resposta.
- Depois de cada resposta, a ferramenta passa à próxima caixa por responder (primeiro as da mesma janela).
- Quando é óbvio, responde logo; quando não é, usa as setas para seguir a lesão.
- Teclas: A, R, N respondem (só nos frames com caixa a rever) · ← → frame a frame · U próxima por responder · Esc volta à caixa · S seed · Z desfaz a última resposta · H esconde as caixas · PgUp/PgDn mudam de janela · roda do rato: zoom; duplo clique: tamanho normal.
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
