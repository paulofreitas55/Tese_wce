# Critérios das categorias clínicas (passo 0.3)

Rascunho gerado por scripts/05_rotulos.py em 05/10/2026. Podes editar à vontade: o script não volta a escrever este ficheiro.

Cada seed tem uma descrição clínica em texto livre (nome do ficheiro imagemN_<descrição>). Cada descrição distinta foi posta numa de 14 categorias: primeiro por regras de palavras (texto em minúsculas, sem acentos), depois revista à mão, linha a linha (docs/label_map_revisao.xlsx). O mapa final está em data/derived/label_map.csv.

## Categorias

| # | Categoria | Lesão | O que entra | Palavras procuradas |
|---|---|---|---|---|
| 1 | Angiectasia (`angiectasia`) | 1 | Lesão vascular: angiectasia/angioectasia, angiodisplasia, telangiectasia, lesão vascular, GAVE (ectasia vascular antral), incluindo gralhas (angeictasia, anfiectasia). Não inclui linfangiectasia nem flebectasia. | angiect…, angioect…, angeict…, anfiect…, angiodisplas…, vascular(es), GAVE |
| 2 | Erosão (`erosao`) | 1 | Erosão, erosões, microerosões, mucosa erosiva. Aftas/lesões aftoides entram aqui, marcadas para confirmar. | eros…, afta/aftoide/aftosa |
| 3 | Úlcera (`ulcera`) | 1 | Úlcera, ulceração, lesão ulcerada. | ulcer… |
| 4 | Pólipo (`polipo`) | 1 | Pólipo, lesão polipoide, polipose, pseudopólipo. | polip… |
| 5 | Eritema (`eritema`) | 1 | Eritema, mucosa eritematosa, hiperemia, enantema. | eritem…, hiperem…, enantem… |
| 6 | Xantelasma (`xantelasma`) | 1 | Xantelasma, xantoma. | xantel…, xantom… |
| 7 | Linfangiectasia (`linfangiectasia`) | 1 | Linfangiectasia (dilatação linfática: pontos ou vilosidades esbranquiçadas). Categoria própria: o texto contém 'angiect' e, no trabalho anterior, era contada como angiectasia. | linf…ect… (linfangiectasia e gralhas) |
| 8 | Sangue (`sangue`) | 1 | Sangue, hemorragia, sangramento, coágulo, hematina, vestígios hemáticos, melena. | sangu…, sangr…, hemorr…, hemat…, coagul…, melen… |
| 9 | Hiperplasia nodular linfoide (`hiperplasia_nodular`) | 1 | Hiperplasia nodular linfoide (HNL), nódulos ou folículos linfoides, mucosa nodular. | hiperplas…, nodul…, HNL, linfoid…, folicul… |
| 10 | Estenose (`estenose`) | 1 | Estenose, estreitamento ou redução do lúmen, anel, diafragma, segmento pouco distensível. | estenos…, estreit…, anel, diafragma, distensív…, redução do lúmen/luminal/de calibre |
| 11 | Lesão subepitelial / massa (`subepitelial`) | 1 | Abaulamento, protrusão, lesão subepitelial ou submucosa, massa, tumor, neoplasia, lesão vegetante. | subepitel…, submucos…, abaul…, protus…/protrus…, massa, tumor…, neopla…, vegetant… |
| 12 | Outra patologia (`outra_patologia`) | 1 | Outros achados patológicos: edema, congestão, fissura, friabilidade, divertículo, atrofia, cicatriz, exsudado/fibrina, flebectasia/varizes, tom azulado/violáceo, inflamação ('-ite': bulbite, duodenite, ileíte…), enteropatia/gastropatia, scalloping, apagamento ou convergência de pregas, deformação, área deprimida, aspeto invaginado, escara de laqueação, parasitas. Também 'lesão'/'achado' sem tipo (ex.: Lesao_no_ileon). | edem…, congest…, fissur…, friab…, divertic…, atrofi…, cicatri…, exsud…, fibrin…, flebect…, variz…, inflam…, espessa…, …ite (gastrite, bulbite…), …patia, scalloping, azulad…, violác…, apagamento, convergência, deform…, deprimid…, invagin…, escara, laqueação, ascaris, oxiúro, parasit…; sem outro termo: lesão, achado, anomalia, alteração, suspeita |
| 13 | Não-lesão (landmark) (`nao_lesao`) | 0 | Sem patologia no texto: marcos anatómicos e localização (primeira imagem gástrica, duodenal ou cecal, piloro, papila, válvula ileocecal, antro, incisura, vértice bulbar, D2…), pregas, mucosa normal, conteúdo ou bílis, tatuagem, clip, cicatriz cirúrgica, ostomia, corpo estranho. | primeira/first, última/last, piloro, válvula, ileocecal, papila, linha Z, cárdia, esófago, estômago, duodeno, bulbo, jejuno, íleo/íleon, cego/cecal, cólon, apêndice, reto, tatuagem, clip, agrafe, sutura, anastomose, bílis, muco, conteúdo, restos alimentares, normal, antro, incisura, recesso, vértice, bulbar, D1–D4, ostomia, bolsa, corpo estranho, cirúrgic…, angulação, prega(s) |
| 14 | Qualidade (`qualidade`) | 0 | O texto diz que não se consegue avaliar: má preparação ou limpeza, resíduos, bolhas ou espuma, observação prejudicada ou inconclusiva. | prepara…, prejudic…, inconclusiv…, impureza…, espuma, bolha…, resíduo…, turv…, fezes/fecal, limpeza, não avaliável |

## Regras de decisão

1. Termos de mais de um tipo de patologia: fica o tipo cujo termo aparece primeiro no texto (o nome da lesão costuma vir antes do adjetivo: "Polipo_ulcerado" → pólipo); a linha é revista à mão.
2. `outra_patologia` só quando não há termo de um tipo definido. "Lesão"/"achado" sem tipo também vão para aqui (ex.: Lesao_no_ileon).
3. Termos de localização ("jejuno", "íleon"…) só decidem quando não há patologia no texto: nesse caso a descrição é não-lesão.
4. Sem patologia, qualidade prevalece sobre não-lesão.
5. "sem X" não conta (ex.: "Erosao_sem_sangue" é erosão).
6. Texto com dúvida ("possível", "suspeita") fica na categoria da lesão e é revisto à mão.
7. Seeds com mais de uma descrição: se só uma é patologia, fica essa; duas patologias diferentes são decididas à mão e registadas em docs/rotulos_janelas_manual.csv.

## Decisões

- Linfangiectasia: categoria própria (o texto contém "angiect"; no trabalho anterior era contada como angiectasia).
- Lesao_no_ileon e Congestao_da_mucosa: patologia (`outra_patologia`), como pede o guião.
- Sangue conta como lesão (como no script antigo).
- Estratificação por classe (guião A): 6 classes principais (angiectasia, erosão, úlcera, pólipo, eritema, xantelasma) + "outras patologias" (as outras 6 categorias com lesão = 1).
- Não-lesão e qualidade ficam fora dos conjuntos de lesão (ALL-SEEDBOX, TEST-BOX, CAL, VAL-EX).
