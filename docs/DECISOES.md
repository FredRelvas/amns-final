# Registro de decisões

O que foi decidido em cada parte do trabalho e por quê. Cada pessoa acrescenta a sua seção ao terminar a issue, com os números que justificaram as escolhas.

## Treino e teste (vale para todos)

"Treinar", aqui, é ajustar sem rótulos: rodar o algoritmo iterativo até convergir (centroides no K-means, EM no GMM). Os rótulos só entram no cálculo das métricas.

| Etapa | Aprende do treino | Aplica no teste |
|---|---|---|
| `Preprocessor` | quais colunas manter, a média e, só no `tumors`, o desvio de cada uma | centraliza (e padroniza, no `tumors`) com esses números |
| PCA | as *d* direções principais | projeta nessas direções |
| K-means | os K centroides | cada ponto vai para o centroide mais próximo |
| GMM | médias, variâncias e pesos das K gaussianas | cada ponto vai para a gaussiana mais provável |

Uma repetição (semente `s`, um valor de *d*), orquestrada pelo `run_sweep`:

1. `split(X, y, seed=s)`.
2. `Preprocessor(scale=dataset in STANDARDIZE).fit(X_train)`, que transforma treino e teste.
3. PCA com *d* componentes ajustado no treino, que projeta treino e teste.
4. `algo.fit(Z_train, K)` e depois `predict` no treino e no teste.
5. ARI, ACC e NMI contra `y_train` e `y_test`.

Por que separar treino e teste mesmo sem rótulos:
- **Sem vazamento:** o PCA e o pré-processamento não veem os dados em que medimos.
- **H4:** com muitas dimensões e poucas amostras, o PCA e os centroides se ajustam ao ruído do treino. A diferença entre `ari_train` e `ari_test` mede esse overfitting.
- **Comparação justa:** as regras escolhem *d* olhando só o treino.

O `ari_train` (ajustar e avaliar nos mesmos dados) equivale à avaliação usual da literatura. O `ari_test` mede se a estrutura generaliza.

**Quando usar todos os dados, sem split (`test_size=0` no `run_sweep`):**
- Para H1, H2, H3 e H5, ajustar e avaliar nos mesmos dados é válido, porque nada antes da avaliação usa rótulos. Só a H4 precisa do split.
- Com muitos dados (imagens completas), o split não custa nada. Nesses casos usamos sempre o split, num protocolo só.
- Com poucos dados, o split dobra o ruído. No `tumors`, com K-means sem PCA e 10 sementes:

  | Avaliação | ARI | Variação entre sementes |
  |---|---|---|
  | treino (121 amostras) | 0,339 | ±0,080 |
  | teste (53 amostras) | 0,289 | ±0,065 |
  | tudo, sem split (174 amostras) | 0,340 | ±0,039 |

- **Por isso o `tumors` tem as duas análises:**
  - A completa, sem split, responde H1, H2, H3 e H5.
  - A com split serve à H4: lá a diferença entre treino e teste (~0,05, já sem PCA) é o próprio resultado.
- O mesmo vale para o MNIST subamostrado com n pequeno (p/n alto).

Só usamos K-means e GMM porque ambos têm `predict` natural. Algoritmos sem `predict` (aglomerativo, DBSCAN, espectral) exigiriam uma regra extra para atribuir pontos novos.

## Issue #1: Estrutura do repositório

- **Estrutura enxuta.** Ficaram de fora, por enquanto, CI, ruff, `configs/`, `notebooks/`, `plots.py` e `scripts/`. Cada item entra quando alguma issue precisar.
- **Dependências mínimas:** `numpy`, `scipy`, `scikit-learn`, `pandas`, `matplotlib` e, para desenvolvimento, `pytest`.
- **Ajustes nas interfaces da seção 6 do contexto:**
  - `run_sweep` ganhou o parâmetro `dataset`, necessário para preencher a coluna de mesmo nome.
  - `rules.py` ganhou `d_elbow` (cotovelo), que a #4 pede e faltava na seção 6.

## Issue #2: Dados e métricas (P1)

### Datasets

| Nome | Tamanho | Classes | Papel |
|---|---|---|---|
| `digits` | 1.797 × 64 (8×8, 17 tons de cinza) | 10 dígitos | Desenvolvimento rápido: vem com o scikit-learn, sem download. É outro dataset, não um pedaço do MNIST. |
| `mnist` | 70.000 × 784 (28×28) | 10 dígitos | Dataset principal. Permite variar p/n de ~0,01 até ~1,6 por subamostragem. |
| `fashion_mnist` | 70.000 × 784 (28×28) | 10 tipos de roupa | Mesmo formato do MNIST, mas mais difícil de agrupar. Testa se as conclusões valem além de dígitos. |
| `tumors` | 174 × 12.533 genes | 11 tipos de câncer | Caso natural de p > n (p/n ≈ 72). |
| `synthetic` | escolhido (padrão n=1000, p=50) | escolhido (padrão 10) | Única verdade conhecida: o sinal ocupa exatamente K−1 dimensões. |

**Carregamento do OpenML** (MNIST, Fashion-MNIST, 11_Tumors). Os arquivos ficam em cache em `data/`, fora do git.
- **`parser="liac-arff"`:** o parser `pandas` do `fetch_openml` (scikit-learn 1.9.1) quebra com o pandas 3 (`TypeError` em `strip_single_quotes`). O `liac-arff` funciona e leva ~6 s para carregar o MNIST. Travar o pandas abaixo da versão 3 foi descartado por ser um remendo para o projeto todo.
- **`as_frame=True`:** o 11_Tumors guarda os rótulos como atributo STRING no ARFF, e o `fetch_openml` não aceita isso com `as_frame=False`.

**Por que o 11_Tumors** (Su et al., 2001):
- Queríamos um p > n natural; com o MNIST subamostrado só chegamos a ~1,6. Os 11 tipos de câncer têm de 6 a 27 amostras cada.
- **Atenção:** Yeung e Ruzzo agrupam *genes* usando as condições como variáveis, e nesse arranjo p < n. Aqui agrupamos *amostras de pacientes* usando os genes como variáveis, e por isso p > n.
- **Outro candidato do OpenML:** `Lung` (203 × 12.600, 5 classes). Foi descartado porque 139 das 203 amostras são de uma única classe, o que confundiria o efeito do PCA com o desbalanceamento.
- **Valores brutos (de −7.346 a 67.796), sem log:**
  - Testamos `log2(clip(X, 20, 16000))`, comum em microarray. O ARI do K-means foi de 0,356 ± 0,040 (bruto) para 0,392 ± 0,054 (log), com 5 sementes, sem split e com padronização. A diferença fica dentro da variação entre sementes, então não vale um pré-processamento só para esse dataset.
- **Consequência:** com 121 amostras de treino, o PCA tem no máximo 120 componentes. O sweep nesse dataset vai de *d* = 1 a 120.

**Sintético** (`_synthetic`):
- **Como é gerado:**
  - Os K centros são `separation` × vetores ortonormais de um subespaço aleatório de dimensão K em ℝ^p. Todo par de centros fica a `separation·√2` de distância, e o sinal ocupa exatamente K−1 dimensões.
  - Há ruído isotrópico N(0, `noise_std`²) em todas as p dimensões.
  - Os rótulos são sorteados uniformemente.
- **Por que centros ortogonais:**
  - A primeira versão sorteava centros gaussianos. Com K=10, só 7 dos 9 autovalores de sinal se destacavam, porque os centros aleatórios deixavam algumas direções quase sem variância.
  - Com centros ortogonais, os 9 autovalores de sinal ficam entre 2,1 e 2,7 e o 10º cai para 1,03 (n=1000, p=50, dados padronizados). Assim a P3 consegue testar se uma regra acha o número certo de dimensões.
- O sinal não cresce com p: dimensões extras só adicionam ruído. Isso é bom para variar p/n.
- É preciso ter p ≥ K.

### Subamostragem e split

- **`subsample`:** sorteia n linhas sem reposição. Não é estratificado: nos datasets que vamos subamostrar (MNIST) as classes são balanceadas.
  - **Cuidado:** subamostrar o `tumors` pode deixar uma classe com menos de 2 amostras, e aí o split estratificado falha.
- **`split`:** estratificado, com 30% para teste por padrão. Mantém a proporção das classes, o que importa no 11_Tumors.

### Pré-processamento e padronização

**`Preprocessor(scale=...)`:**
- Remove as colunas com desvio-padrão zero **no treino** e centraliza com a média **do treino**. Isso evita que o teste vaze para o ajuste (seção 4.3 do contexto).
- Uma coluna constante no treino é removida mesmo que varie no teste. Exemplo: no MNIST com n=1000, 784 colunas viram 592.
- Remover essas colunas também mantém o p correto para o Marchenko-Pastur (γ = p/n): colunas constantes só acrescentariam autovalores zero.
- Com `scale=True`, também divide pelo desvio do treino. Isso é a padronização: cada coluna fica com média 0 e desvio 1.

**Decisão: padronizar só quando as variáveis estão em unidades ou escalas diferentes.** O registro fica em `STANDARDIZE`, em `data.py`.

| Dataset | Variáveis | Padroniza? |
|---|---|---|
| `digits`, `mnist`, `fashion_mnist` | pixels, todos na mesma unidade (intensidade) | não, só centraliza |
| `synthetic` | mesma escala por construção | não, só centraliza |
| `tumors` | genes com escalas de dezenas a dezenas de milhares | sim |

**Por quê:**
- **O critério é clássico e não usa rótulos.** Escolher o pré-processamento pelo ARI de cada dataset seria deixar os rótulos guiarem o desenho. Por isso o Fashion-MNIST também fica sem padronização, mesmo sem diferença nos números.
- **O teste do MNIST confirma o critério e serve como análise de sensibilidade:**
  - Condições: K-means, 10.000 amostras, 3 sementes, `n_init=3`, sem split.
  - "Sem colunas raras" é uma alternativa testada e descartada: remover as colunas em que o valor difere da mediana em menos de q das amostras.

  | Pré-processamento | MNIST: colunas | MNIST: ACC | MNIST: ARI | Fashion: ACC | Fashion: ARI |
  |---|---|---|---|---|---|
  | pixels/255 (literatura: ACC ≈ 53% no MNIST) | 784 | 0,545 | 0,360 | 0,508 | 0,345 |
  | padronizado | 682 | 0,489 | 0,313 | 0,497 | 0,337 |
  | padronizado, sem colunas raras q=1% | 488 | 0,503 | 0,315 | 0,516 | 0,356 |
  | padronizado, sem colunas raras q=5% | 394 | 0,510 | 0,325 | 0,509 | 0,338 |
  | padronizado, sem colunas raras q=10% | 342 | 0,526 | 0,333 | 0,508 | 0,343 |

- **O que a tabela mostra:**
  - **MNIST:** padronizar dá o mesmo peso a todos os pixels, inclusive aos periféricos de pouca variância, e isso custa ~0,05 de ACC.
  - **Remover colunas raras** recupera só parte da perda, mesmo descartando mais da metade dos pixels, e não compensa o parâmetro a mais.
  - **Fashion-MNIST:** as diferenças ficam dentro da variação entre sementes (±0,01 a 0,03).
  - Centralizar sem dividir pelo desvio dá o mesmo resultado que pixels/255: o K-means não muda com translação nem com escala uniforme.

**Consequência para o PCA:** o algoritmo é o mesmo, mas a entrada muda.
- **Imagens:** dados só centralizados, ou seja, PCA da matriz de **covariância**. Os pixels que mais variam (os centrais, que carregam a forma) dominam os primeiros componentes.
- **`tumors`:** dados padronizados, ou seja, PCA da matriz de **correlação**. Todos os genes pesam igual, e os componentes capturam grupos de genes que variam juntos.
- As conclusões valem para o PCA de covariância nas imagens e para o de correlação nos genes.

**Consequência para as regras de *d* (#4):** elas trabalham com os autovalores da covariância dos dados pré-processados. Com dados padronizados, isso é a matriz de correlação, e as regras se reduzem às definições usuais.
- **Kaiser:** "autovalor > média dos autovalores". Com dados padronizados a média é 1, então é a regra usual.
- **Marchenko-Pastur com σ² estimado:** funciona igual, porque a estimativa de σ² absorve a escala.
- **Marchenko-Pastur com σ² = 1:** só faz sentido no `tumors` ou no sintético com `noise_std=1`. No sintético o σ² verdadeiro é conhecido, o que ajuda a validar a regra.
- **Análise paralela, 90% da variância e cotovelo:** funcionam igual.

**Limitação a declarar (#8):** se uma regra funcionar no `tumors` e falhar no MNIST, não dá para separar o efeito do dataset do efeito do pré-processamento.

### Métricas

Todas são externas: comparam com os rótulos, que os algoritmos nunca veem.

| Métrica | O que mede | Por que usar | Cuidado |
|---|---|---|---|
| ARI | Concordância entre partições, contando pares de pontos | Régua principal: ajustada ao acaso (1 = idênticas, ~0 = acaso) e funciona com K estimado ≠ K | — |
| NMI | Informação mútua normalizada entre clusters e classes | Complemento usual na literatura | Não é ajustada ao acaso e cresce com o número de clusters. Com K estimado, prefira AMI. |
| ACC | Fração de pontos no grupo certo após a melhor correspondência um-para-um entre clusters e classes (algoritmo húngaro) | Fácil de ler, comparável à literatura (K-means no MNIST ≈ 53%). Os percentuais do quadro do professor têm cara de acurácia. | Não é ajustada ao acaso (~10% com 10 classes balanceadas). Clusters sem classe contam como erro. |
| Erro de reconstrução | Distância quadrática média entre X e a reconstrução com *d* componentes | Mede quanto o PCA perde, sem rótulos | Sempre cai com *d*; sozinho não aponta o melhor *d*. |

O KNN como sonda de informação preservada foi **removido**: o escopo ficou só em clustering (resposta do professor na #11). Com isso saiu também a coluna `knn_acc_test` do `run_sweep`.

### Sementes

`pcacluster.SEEDS` = 0 a 9, uma semente por repetição. A semente `s` vale para tudo que é aleatório na repetição `s`: geração do sintético ou subamostragem, split e inicialização do clustering.
- **Motivo:** evitar que cada pessoa use sementes diferentes para dado, split e K-means, e permitir refazer qualquer repetição isoladamente.
- **Por que 10:** é o mesmo número de repetições do piloto. Se o MNIST completo ficar lento, a quantidade muda num lugar só.

### Testes

- **O que os testes automáticos cobrem:**
  - Ausência de vazamento: o `Preprocessor` usa só estatísticas do treino e remove colunas constantes no treino.
  - Com `scale=False`, o `Preprocessor` só centraliza.
  - ARI dá 1,0 com rótulos trocados e 0,0 com tudo num cluster.
  - ACC dá 1,0 com rótulos trocados e 5/6 num caso parcial.
  - Erro de reconstrução e formato do `digits` subamostrado e do sintético.
- MNIST, Fashion-MNIST e 11_Tumors exigem download e não estão nos testes automáticos. Foram verificados à mão, carregando, dividindo e pré-processando cada um.

### Pendências

- Quais configurações do sintético testar (p, ruído, separação) fica em aberto para quem trabalhar na P3 e na P5.

## Issue #4: Regras para escolher d (P3)

Todas as regras ficam em `rules.py`, recebem o `X_train` já pré-processado e devolvem um `int`. Nenhuma usa rótulos.

### Autovalores (base de todas as regras)

- **O que se calcula:** os autovalores de `XᵀX/n`, depois de recentralizar X (inofensivo, pois o `Preprocessor` já centraliza). Só os `min(n−1, p)` possivelmente não nulos entram.
- **Como:** pela menor das duas matrizes de Gram (`XᵀX` ou `XXᵀ`). Com p > n (`tumors`, p/n ≈ 100) nunca se monta uma matriz p×p.
- **Consequência:** com p > n, nenhuma regra devolve d > n−1, que é o limite do PCA.

### Decisões por regra

| Regra | Decisão | Motivo |
|---|---|---|
| Marchenko-Pastur | borda σ²·(1+√(p/n))², com γ = p/n; d é o número de autovalores acima dela | definição da seção 3.2 |
| MP, σ² estimado | σ² = mediana(autovalores) / mediana esperada de ruído N(0,1) n×p, média de 30 simulações (`n_sim`, `seed`) | a estimativa absorve a escala dos dados; o resultado de cada (n, p) fica em cache, então o sweep não repete a simulação |
| MP, σ² = 1 | `sigma2=1` | só faz sentido com dados padronizados ou sintético com `noise_std=1` |
| Análise paralela | `rng.permuted` embaralha cada coluna de forma independente; percentil 95 por índice; **conta até o primeiro autovalor que não passa** | ver abaixo |
| d = K ou K−1 | `d_from_K` | já estava pronta |
| 90% da variância | menor d com variância acumulada ≥ 0,90 (com tolerância de 1e-12 para o caso exato) | — |
| Kaiser | autovalores acima de `soma/p`, ou seja, da média sobre as **p** dimensões | com p > n−1 existem menos de p autovalores; tirar a média só dos existentes inflaria a média e subestimaria d. Com dados padronizados é o usual "> 1". |
| Cotovelo | ponto da curva de autovalores mais distante da reta entre o primeiro e o último, com os dois eixos reescalados para [0, 1]. d é o **número de pontos antes do cotovelo** (mínimo 1) | escolha nossa, a issue não fixava um método. Segue o teste de Cattell: o próprio ponto do cotovelo já é cauda. |

### Desvio em relação à issue: análise paralela para no primeiro fracasso

A issue diz "contar autovalores acima do percentil 95". Contar todas as ultrapassagens superestima d: cada índice tem ~5% de chance de passar, então em ruído puro o resultado esperado é 5% dos índices.

Em n=500 e p=50, com 40 sementes de ruído puro, a versão que soma tudo devolveu em média **2,5 componentes** (até 9). A regra de parada de Horn conta só a sequência inicial de autovalores acima do limiar e devolve 0 ou 1 em ruído puro. Em dados com sinal forte as duas coincidem.

### Testes

- **Autovalores conhecidos:** matriz com autovalores exatos para 90% da variância, Kaiser e cotovelo.
- **Ruído puro (n=500, p=50), 5 sementes:** MP (σ² = 1 e estimado) e análise paralela devolvem no máximo 1. Escala 7× maior: MP estimado continua no máximo 1; σ² = 1 devolve 50 (a regra usada fora do domínio).
- **5 spikes plantados:** MP e análise paralela devolvem 5.
- **Sintético (n=1000, p=50, K=10):** MP (σ² = 1 e estimado) e análise paralela devolvem 9 = K−1.
- **p > n:** nenhuma regra passa de n−1.

### Resultados (treino de 70%, dados pré-processados como no sweep)

| Dataset (n treino × p) | MP σ²=1 | MP σ² est. | Análise paralela | 90% var. | Kaiser | Cotovelo |
|---|---|---|---|---|---|---|
| `digits` (1257 × 60), sementes 0, 1, 2 | 44 | 25 a 26 | 9 | 21 | 14 | 12 |
| `synthetic` K=10 (700 × 50) | 9 | 9 | 9 | 40 | 10 | 9 |
| `synthetic` K=10 (700 × 500) | 9 | 9 | 14 | 293 | 197 | 103 |
| `synthetic` K=10 (70 × 500) | 1 | 1 | 0 | 56 | 69 | 33 |
| `mnist` n=5000 (3500 × 652) | 601 | 225 | 42 | 85 | 78 | 42 |
| `mnist` n=1000 (700 × 592) | 507 | 152 | 31 | 75 | 73 | 42 |
| `mnist` n=500 (350 × 555) | 349 | 94 | 26 | 65 | 71 | 26 |
| `tumors` (121 × 12533) | 19 | 49 | 21 | 78 | 120 | 8 |

O verdadeiro d no sintético é K−1 = 9.

**O que se lê:**
- **Sintético:** MP e análise paralela acham 9 com p/n baixo (0,07); o 90% da variância e o Kaiser erram. Com p = 500 (p/n ≈ 0,7), só o MP continua em 9; a análise paralela sobe para 14. Com n = 70 (p/n ≈ 7) todas as regras de ruído falham, porque o sinal some dentro do ruído (primeiro indício para a H5).
- **`digits`:** MP σ²=1 devolve 44, não os 12 do piloto. A diferença vem do pré-processamento: o piloto padronizou, e aqui os pixels só são centralizados (ver "Padronização" na #2), então σ²=1 não é a variância do ruído. Ver "Limitação" abaixo.
- **MNIST:** as regras de ruído divergem muito (26 a 601). MP σ²=1 é inútil sem padronização, e o MP estimado cai de 225 para 94 quando p/n sobe, o que combina com a H2.

### Limitação a declarar

- **MP σ² = 1 nas imagens:** com dados só centralizados, σ² = 1 não é o nível de ruído, e a regra devolve centenas de componentes. Use essa variante apenas no `tumors` e no sintético com `noise_std=1`.
- **Custo:** a análise paralela e a simulação do MP custam ~20 s com 3500 × 652. Se o sweep com MNIST completo ficar lento, reduzir `n_perm` ou `n_sim`.
