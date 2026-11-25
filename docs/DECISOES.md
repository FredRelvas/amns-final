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

## Issue #3: Clustering e estimativa de K (P2)

Tudo fica em `clustering.py`. As classes seguem a interface `ClusterAlgo` (`fit(Z, K)` e `predict(Z)`) e ganham um método `bic(Z)`; o dicionário `ALGOS` liga o nome (`"kmeans"`, `"gmm"`) à classe.

### Algoritmos e parâmetros

| Peça | Decisão | Motivo |
|---|---|---|
| `KMeansAlgo` | `n_init=5` por padrão, `random_state=seed` | K-means é instável (seção 4.5 do contexto). O 5 é o padrão recomendado, não um mínimo obrigatório. |
| `GMMAlgo` | `covariance_type="diag"`, uma execução do EM, `random_state=seed` | covariância diagonal, como combinado na interface |
| `estimate_K` | ajusta cada K de `K_grid` e escolhe o de **menor BIC** ou **maior silhueta**; padrão `criterion="bic"` | os dois critérios pedidos na issue; ver a pendência abaixo |
| Silhueta | com mais de `sample_size` linhas (padrão 5000), é calculada numa subamostra sorteada com a semente | custa O(n²): com n = 20 000 eram ~2 s por K; com a subamostra, 0,2 s, e o mesmo K |
| Clustering original | `fit_original_clustering` ajusta no espaço completo; `save_clustering` grava os rótulos por split, com algoritmo, K e semente, num `.npz` (em `results/`, fora do git); `load_clustering` lê de volta | é a referência do `agreement_full` no sweep |

### BIC do K-means

O K-means não tem verossimilhança. Usamos a do X-means (Pelleg e Moore, 2000): K gaussianas esféricas com **uma variância compartilhada** σ², pesos nₖ/n e centros nos centroides.

- σ² = inércia / ((n − K)·d)
- log L = Σₖ nₖ log(nₖ/n) − (n·d/2)·log(2πσ²) − (n − K)·d/2
- m = (K − 1) + K·d + 1 parâmetros livres (pesos, centros e σ²)
- BIC = −2 log L + m·log n

O BIC do GMM é o do scikit-learn (`GaussianMixture.bic`), com variâncias diagonais por componente.

### Validação no sintético com K conhecido, por d

Sintético padrão (n = 1000, p = 50, K = 10, sinal em K − 1 = 9 dimensões), só centralizado, PCA com d componentes, `K_grid = range(2, 21)`, sementes 0 a 4. Número de sementes, entre 5, em que `estimate_K` devolveu exatamente 10:

| d | K-means, BIC | K-means, silhueta | GMM, BIC | GMM, silhueta |
|---|---|---|---|---|
| 1, 2, 3, 5 | 0 | 0 | 0 | 0 |
| 7 | 4 | 0 | 3 | 0 |
| 8 | 5 | 2 | 4 | 2 |
| 9 (= K − 1) | 5 | 5 | 4 | 4 |
| 10 | 5 | 5 | 3 | 3 |
| 15 | 5 | 5 | 2 | 2 |
| 20 | 5 | 5 | 3 | 3 |
| 30 | 5 | 5 | 5 | 5 |
| 50 (= p) | 4 | 4 | 1 | 2 |

**O que se lê:**
- Com d < K − 1 o K é **subestimado** (com d = 5 o K estimado fica entre 6 e 9): faltam dimensões para separar todos os grupos, como prevê Ding e He.
- A partir de d = K − 1 o K-means acerta quase sempre, com os dois critérios.
- O GMM erra em vários d mesmo com o sinal completo, em geral por 1 a 4 clusters; com d = p chegou a escolher 3. Ele roda o EM uma vez só, enquanto o K-means faz 5 inicializações (causa provável, não testada).

O teste `test_estimate_k_on_synthetic_by_d` confere a mesma coisa numa versão menor (K = 5, n = 500, p = 30, 3 sementes).

### Em dados reais, o BIC acompanha o teto da grade

Resultado da revisão da PR #14 (K verdadeiro = 10, PCA em vários d, `K_grid = range(2, 21)`, sementes 0 e 1):

| Dados | BIC (K-means e GMM) | Silhueta |
|---|---|---|
| Sintético (n = 1000, p = 50) | 10 a partir de d = 9 (= K − 1); com d < 9 subestima | igual ao BIC |
| `digits` | 15 a 20 (K-means: sempre 20) | K-means: 9 em todo d |
| `mnist` (n = 3000) | 20 a partir de d = 9 (GMM com d = 200: 4) | instável, de 2 a 20 |

- No `digits` com d = 20, subindo o teto da grade para 30 e 40, o BIC do K-means escolhe 30 e 40; o do GMM para em 28.
- Não é erro do código, é o comportamento do critério. Explicação provável: os dados reais não são gaussianos esféricos (nem diagonais), e com milhares de pontos o ganho de verossimilhança de partir um grupo supera a penalidade m·log n.
- **Consequência:** com o padrão `criterion="bic"`, o K estimado em dados reais depende de onde a grade termina.

### Pendências

- **O grupo precisa decidir qual critério o sweep usa para estimar K** (`estimate_K_flag=True`): BIC, silhueta ou outro. Em aberto até essa decisão, o padrão continua `"bic"`, como na interface combinada.
- **GMM no espaço completo:** no MNIST (n = 3000), o GMM diagonal dá ARI 0,15 contra 0,34 do K-means (revisão da PR #14). O `agreement_full` com GMM compara com uma referência fraca, o que deve ser lembrado na análise.

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

## Issue #5: Sweep e comparação (P4)

O código fica em `sweep.py` (sweep e análise), `plots.py` (figuras) e `scripts/run_sweep.py` (um comando por dataset). A regra de Gavish e Donoho entrou em `rules.py`.

### O que o `run_sweep` faz

Segue o protocolo da seção "Treino e teste" acima, com estas decisões:

| Peça | Decisão | Motivo |
|---|---|---|
| PCA | ajustado uma vez por semente, com o maior *d* da grade; o PCA com *d* componentes são as *d* primeiras colunas | os componentes são aninhados, então o resultado é o mesmo de ajustar um PCA por *d*, e o custo cai de um ajuste por *d* para um por semente. O teste `test_run_sweep_matches_an_independent_pca_per_d` compara com um PCA ajustado do zero para cada *d* e obtém o mesmo ARI e o mesmo erro de reconstrução |
| `svd_solver="full"` | SVD exata | com o padrão (`"auto"`), o scikit-learn pode escolher o solver aleatorizado, cujo resultado depende de semente |
| Linha "sem PCA" | cada semente ganha uma linha com `space="full"` e `d = p_eff`: o clustering dos dados pré-processados, sem PCA | é a referência de melhora/piora. Não dá para usar o PCA com *d* máximo no lugar: o PCA completo só gira os dados, o que não muda o K-means mas muda o GMM diagonal (ver "Efeito do PCA" em "Resultados") |
| Grade de *d* | valores acima de min(n_treino − 1, p_eff) são pulados naquela semente, com aviso | o p_eff muda com a semente: no `digits` o treino fica com 60 ou 61 colunas, conforme o split |
| Erro de reconstrução | calculado para todos os *d* de uma vez: ‖x − média‖² menos a soma dos *d* primeiros escores ao quadrado | os componentes são ortonormais (Pitágoras). Dá o mesmo valor de `metrics.reconstruction_error` (conferido no teste citado acima) |
| `agreement_baseline` | ARI entre dois clusterings sem PCA que só diferem na inicialização (semente s e s + 10 000) | a seção 4.4 do contexto pede essa linha de base: o `agreement_full` só tem sentido comparado com a concordância do algoritmo consigo mesmo |
| `estimate_K_flag=True` | K é estimado com `estimate_K` em cada espaço que se agrupa (sem PCA e cada *d*), e vai para a coluna `K_fit`; `K` continua sendo o número de classes | o K estimado muda com *d* (ver #3) |

Colunas além das combinadas: `space`, `n_train`, `p_eff`, `K_fit`, `nmi_train`, `acc_train`, `acc_test`, `recon_err_train` e `agreement_baseline`. O `n`, o `p` e o `p_over_n` são os do dataset que entrou (`p_over_n` = 784/1000 = 0,78 no MNIST com n = 1000, como no contexto).

### Comparação com as regras (`rule_ds` e `compare_rules`)

- **Mesmo treino:** `rule_ds` chama a mesma função `prepare` do sweep, então cada regra vê exatamente o `X_train` que o sweep agrupa, semente por semente.
- **Regras comparadas:** MP com σ² estimado e com σ² = 1, análise paralela, Gavish e Donoho, d = K, d = K − 1, 90% da variância, Kaiser e cotovelo. Com `--k-est`, também d = K estimado e d = K estimado − 1, com K estimado no espaço completo, que é o procedimento do quadro do professor.
- **d = 0:** uma regra de ruído pode não achar nenhum componente (sinal abaixo do ruído). O clustering precisa de pelo menos uma dimensão, então o sweep usa `d_used = max(d, 1)`. A coluna `d` guarda o valor original.
- **Os *d* das regras entram na grade:** o script acrescenta à grade todos os `d_used` de todas as sementes, para que o desempenho no *d* de cada regra seja medido, nunca interpolado.
- **Perda da regra:** para cada semente, ARI no *d* da regra menos ARI no melhor *d* (o máximo da curva média), na mesma semente. Pareado, porque split e inicialização são os mesmos.

### Melhor *d*: máximo, platô e `oracle_cv`

- **Máximo da curva (`d_best`):** o *d* de maior ARI médio entre as sementes. A média só usa os *d* presentes em todas as sementes.
- **Platô (`d_plateau`):** o menor *d* cuja média atinge 95% do máximo. O 95% vem da seção 4.5 do contexto e foi escolhido pelo grupo, sem fonte.
- **`oracle_cv` (validação cruzada entre sementes):** para cada semente s, escolhe o *d* pelo máximo da curva média das outras sementes e mede o ARI na semente s. O máximo da curva é otimista: entre dezenas de *d* com ARI parecido, o maior tende a ser o que teve sorte nas sementes usadas. O `oracle_cv` mede o que escolher *d* com os rótulos rende numa repetição nova, e por isso é a referência justa para as regras. A coluna `loss_cv` do `compare.csv` é a perda contra ele: negativa quer dizer que a regra, sem rótulos, rende mais do que escolher *d* olhando os rótulos.

### Melhora, piora ou indiferente

Para cada *d* e cada regra, a diferença ARI(PCA) − ARI(sem PCA) é tomada por semente (pareada) e resumida pela média e por um intervalo de confiança t de 95%.

| Classe | Condição |
|---|---|
| melhora | diferença média maior que a margem **e** intervalo de 95% acima de zero |
| piora | diferença média menor que −margem **e** intervalo de 95% abaixo de zero |
| indiferente | qualquer outro caso |

A margem é 1 desvio-padrão do ARI sem PCA entre as sementes, como propõe a seção 4.3 do contexto (o "1" foi escolhido pelo grupo, sem fonte).

**Desvio em relação à proposta do contexto:** a proposta usava só a margem. Exigir também o intervalo evita chamar de melhora uma diferença que o ruído entre sementes explica. Quantas classificações mudam está em "Classificação: efeito de exigir o intervalo", nos resultados.

**Efeito com o melhor *d*:** para dizer se o PCA com o melhor *d* melhora, a diferença é tomada no *d* que o `oracle_cv` escolhe para cada semente (`effect_at_oracle_cv`), e não no máximo da curva. O *d* do máximo é escolhido nas mesmas sementes em que a diferença é medida, então a diferença nele é otimista pelo mesmo motivo que o máximo é.

### Regra nova: limiar ótimo de Gavish e Donoho (`d_gavish_donoho`)

A leitura da P4 (Gavish e Donoho, 2014) dá um limiar para valores singulares com nível de ruído desconhecido: manter os valores singulares acima de ω(β)·y_med, em que y_med é a mediana dos valores singulares e β = min(n, p)/max(n, p).

- **ω(β) = λ\*(β)/√μ_β**, com λ\*(β) da eq. (11) do artigo e μ_β a mediana da distribuição de Marchenko-Pastur de razão β, calculada por integração numérica (`scipy.integrate.quad` e `brentq`).
- **Conferência com o artigo:** a conta reproduz a Tabela IV do artigo (ω(β) de β = 0,05 a 1) com diferença máxima de 0,0007; ω(1) = 2,858, como no resumo; e a aproximação cúbica da eq. (5), 0,56β³ − 0,95β² + 1,82β + 1,43, fica a menos de 0,02 da conta exata em β de 0,05 a 1 (teste `test_gavish_donoho_omega_matches_the_paper`). A mediana μ_β bate com uma simulação de Monte Carlo (0,830 contra 0,832, com β = 0,5).
- **Diferença para o MP:** o MP conta o que está acima da borda do ruído (2 vezes a escala do ruído numa matriz quadrada). O Gavish e Donoho é mais exigente (2,31 vezes): só mantém o componente cuja estimativa ajuda a recuperar o sinal. Testes: ruído puro dá no máximo 1, 5 spikes plantados dão 5, e o sintético dá 9 = K − 1.

### Saídas do script

O `scripts/run_sweep.py` grava em `results/<tag>/`:

| Arquivo | Conteúdo |
|---|---|
| `sweep.csv` | uma linha por algoritmo, semente e *d* (saída do `run_sweep`) |
| `rules.csv` | *d* de cada regra em cada semente |
| `compare.csv` | perda de cada regra (`loss_mean`, `loss_cv`), fração das sementes no platô (`in_plateau`) e efeito contra o sem PCA |
| `effect.csv` | melhora/piora/indiferente para cada *d* |
| `best.csv` | máximo, platô, `oracle_cv` e o efeito no *d* dele (`effect_cv`, `diff_cv`, `ci_cv_low`, `ci_cv_high`), margem e `agreement_baseline` médio |
| `curves.png` | ARI por *d* (média ± 1 DP entre sementes), sem PCA tracejado e, embaixo, o *d* de cada regra (mediana e faixa entre sementes) |
| `effect.png` | diferença para o sem PCA com o intervalo de 95% e a faixa da margem |

### Resultados

Todos com K = número de classes, 10 sementes (0 a 9) e o ARI como métrica: o de teste quando há split e o de todos os dados no `tumors`. Comandos:

```bash
uv run python scripts/run_sweep.py --dataset synthetic
uv run python scripts/run_sweep.py --dataset digits
uv run python scripts/run_sweep.py --dataset mnist --n 10000 --grid 1-30,35-100:5,120-600:40
uv run python scripts/run_sweep.py --dataset fashion_mnist --n 10000 --grid 1-30,35-100:5,120-600:40
uv run python scripts/run_sweep.py --dataset tumors --K 11 --test-size 0
```

| Dataset | n de treino | p_eff | p/n do dataset | Grade de d | Divisão |
|---|---|---|---|---|---|
| sintético | 700 | 50 | 0,050 | todos de 1 a 50 | 70% treino, 30% teste |
| digits | 1257 | 60 a 61 | 0,036 | todos de 1 a 61 | 70% treino, 30% teste |
| MNIST | 7000 | 660 a 676 | 0,078 | 87 valores de 1 a 621 | 70% treino, 30% teste |
| Fashion | 7000 | 783 a 784 | 0,078 | 75 valores de 1 a 773 | 70% treino, 30% teste |
| tumors | 174 | 12533 | 72,029 | todos de 1 a 173 | sem split |

O sintético é o padrão (K = 10, sinal em K − 1 = 9 dimensões). No MNIST e no Fashion-MNIST, cada semente sorteia 10 000 imagens. Nesses dois a grade é esparsa (de 1 a 30, depois de 5 em 5 até 100 e de 40 em 40 até 600) mais os *d* escolhidos pelas regras. O sweep completo levou 45 min no MNIST e 26 min no Fashion-MNIST.

**Conferência com números de fora do sweep.** K-means sem PCA no MNIST: ARI 0,365 e ACC 0,536. A #2 mediu ARI 0,360 e ACC 0,545 com 10 000 amostras sem split, e a literatura cita ACC ≈ 53%. No `KBENCH`, reduzir o MNIST para 30 dimensões deu ARI 0,36; aqui o máximo é 0,368, em d = 27.

#### Efeito do PCA

| Dataset | Algoritmo | Sem PCA | Máximo da curva (d) | Platô de 95% (d) | `oracle_cv` | Efeito no *d* do `oracle_cv` (diferença) |
|---|---|---|---|---|---|---|
| sintético | kmeans | 0,997 ± 0,005 | 0,998 (9) | 0,979 (8) | 0,998 | indiferente (+0,001) |
| sintético | gmm | 0,898 ± 0,079 | 0,972 (46) | 0,945 (8) | 0,851 | indiferente (−0,047) |
| digits | kmeans | 0,653 ± 0,016 | 0,669 (48) | 0,647 (9) | 0,669 | indiferente (+0,016) |
| digits | gmm | 0,465 ± 0,057 | 0,631 (15) | 0,604 (9) | 0,607 | melhora (+0,142) |
| MNIST | kmeans | 0,365 ± 0,015 | 0,368 (27) | 0,352 (8) | 0,360 | indiferente (−0,005) |
| MNIST | gmm | 0,166 ± 0,027 | 0,327 (8) | 0,325 (7) | 0,318 | melhora (+0,152) |
| Fashion | kmeans | 0,344 ± 0,015 | 0,366 (58) | 0,349 (6) | 0,349 | indiferente (+0,005) |
| Fashion | gmm | 0,317 ± 0,021 | 0,371 (8) | 0,353 (5) | 0,371 | melhora (+0,054) |
| tumors | kmeans | 0,340 ± 0,041 | 0,461 (17) | 0,454 (16) | 0,442 | melhora (+0,102) |
| tumors | gmm | 0,317 ± 0,064 | 0,472 (19) | 0,460 (13) | 0,405 | melhora (+0,087) |

A diferença e o efeito são contra o sem PCA, no *d* que o `oracle_cv` escolhe para cada semente, pela regra de "Melhora, piora ou indiferente". No *d* do máximo, a classe só muda no Fashion-MNIST com K-means: seria melhora (+0,022 em d = 58), por viés de seleção.

- **K-means:** o PCA é indiferente no sintético, no `digits`, no MNIST e no Fashion-MNIST (diferença de −0,005 a +0,016) e melhora só no `tumors` (+0,102, com p/n = 72). A partir do platô (d entre 6 e 16) a curva fica plana: o máximo cai em d = 27 a 58 nas imagens, mas com ganho de no máximo 0,022 sobre o platô.
- **GMM:** o PCA melhora nos quatro datasets reais (+0,054 a +0,152), com o máximo em d pequeno (8 a 19). Com muitos componentes o ARI cai abaixo do sem PCA: a partir de d = 54 no `digits`, 83 no Fashion-MNIST, 85 no `tumors` e 210 no MNIST.
- **PCA completo não é "sem PCA" no GMM:** no K-means, o PCA com todos os componentes dá exatamente o mesmo ARI que o sem PCA (diferença 0,000 em todas as sementes do sintético, do `digits` e do `tumors`), porque só gira os dados. No GMM diagonal a diferença chega a 0,375 (`digits`). Por isso o sem PCA tem linha própria.

#### Perda de cada regra

Perda = ARI no *d* da regra menos ARI no máximo da curva, pareada por semente e com média nas 10 sementes. Entre parênteses, a mediana do *d* escolhido. M e P marcam as regras cujo *d* melhora ou piora o resultado em relação ao sem PCA (as demais são indiferentes).

K-means:

| Regra | sintético | digits | MNIST | Fashion | tumors |
|---|---|---|---|---|---|
| MP, σ² estimado | 0,000 (9) | 0,016 (26) | 0,008 (257) | 0,019 (277,5) | 0,134 (67) |
| MP, σ² = 1 | 0,000 (9) | 0,013 (44) | 0,003 (618) | 0,022 (771,5) | 0,055 (24) M |
| análise paralela | 0,000 (9) | 0,022 (9) | 0,010 (43) | 0,017 (32) | 0,058 (28) M |
| Gavish e Donoho | 0,000 (9) | 0,018 (20) | 0,006 (208,5) | 0,014 (219) | 0,036 (32) M |
| d = K | 0,000 (10) | 0,038 (10) | 0,020 (10) P | 0,020 (10) | 0,074 (11) M |
| d = K − 1 | 0,000 (9) | 0,022 (9) | 0,019 (9) P | 0,012 (9) | 0,078 (10) |
| 90% da variância | 0,011 (40) | 0,014 (21) | 0,003 (86) | 0,012 (81) | 0,135 (110) |
| Kaiser | 0,000 (10) | 0,013 (14) | 0,006 (78) | 0,010 (59) | 0,121 (172) |
| cotovelo | 0,000 (9) | 0,020 (12) | 0,007 (43,5) | 0,020 (14) | 0,074 (11) M |
| sem PCA | 0,001 (50) | 0,016 (61) | 0,003 (667) | 0,022 (784) | 0,121 (12533) |

GMM:

| Regra | sintético | digits | MNIST | Fashion | tumors |
|---|---|---|---|---|---|
| MP, σ² estimado | 0,002 (9) | 0,092 (26) M | 0,177 (257) | 0,172 (277,5) P | 0,204 (67) |
| MP, σ² = 1 | 0,002 (9) | 0,137 (44) | 0,214 (618) P | 0,226 (771,5) P | 0,016 (24) M |
| análise paralela | 0,002 (9) | 0,027 (9) M | 0,125 (43) | 0,036 (32) | 0,007 (28) M |
| Gavish e Donoho | 0,002 (9) | 0,062 (20) M | 0,181 (208,5) | 0,161 (219) P | 0,012 (32) M |
| d = K | 0,018 (10) | 0,051 (10) M | 0,011 (10) M | 0,017 (10) M | 0,043 (11) M |
| d = K − 1 | 0,002 (9) | 0,027 (9) M | 0,041 (9) M | 0,007 (9) M | 0,049 (10) M |
| 90% da variância | 0,024 (40) | 0,019 (21) M | 0,128 (86) M | 0,070 (81) | 0,263 (110) P |
| Kaiser | 0,005 (10) | 0,012 (14) M | 0,136 (78) | 0,061 (59) | 0,257 (172) P |
| cotovelo | −0,009 (9) M | 0,030 (12) M | 0,120 (43,5) M | 0,022 (14) M | 0,043 (11) M |
| sem PCA | 0,073 (50) | 0,166 (61) | 0,161 (667) | 0,054 (784) | 0,155 (12533) |

- **Sintético:** as regras de ruído, d = K − 1, Kaiser e cotovelo escolhem 9 ou 10 e perdem no máximo 0,005 (o d = K, no GMM, perde 0,018). O 90% da variância escolhe 40.
- **K-means em imagens:** todas as regras ficam a no máximo 0,038 do máximo, porque a curva é plana depois do platô. A menor perda foi do Kaiser no `digits` e no Fashion-MNIST e do MP com σ² = 1 no MNIST.
- **GMM em imagens:** MP (os dois) e Gavish e Donoho escolhem de 20 a 772 componentes, em geral muito além do máximo, e perdem de 0,062 a 0,226. d = K e d = K − 1 perdem no máximo 0,051 e têm a menor perda no MNIST (d = K) e no Fashion-MNIST (d = K − 1). No MNIST e no Fashion-MNIST, 90% da variância e Kaiser perdem menos que o MP e o Gavish e Donoho.
- **`tumors` (p/n = 72):** o padrão se inverte. Análise paralela, Gavish e Donoho e MP com σ² = 1 têm as menores perdas no GMM (0,007 a 0,016), e o Gavish e Donoho tem a menor no K-means (0,036). 90% da variância (110 componentes) e Kaiser (172 de 173) perdem até 0,263 e pioram o GMM em relação ao sem PCA.
- **MP com σ² estimado:** em nenhum dataset real teve a menor perda; no GMM perdeu de 0,092 a 0,204 em todos eles. No piloto, com o KNN, ela era a melhor regra.
- **d = K estimado** (`digits`, K estimado sem PCA, `--k-est`): com o BIC, K = 20 em todas as sementes do K-means e de 18 a 20 no GMM, o teto da grade, como já se viu na #3. Com a silhueta, mediana 9 (9 a 17) no K-means e 12 (9 a 20) no GMM. A perda (0,018 a 0,055) fica na faixa do d = K com o K verdadeiro (0,038 e 0,051).

#### Máximo da curva contra `oracle_cv`

A diferença entre o máximo e o `oracle_cv` ficou entre 0,000 e 0,024 em 8 dos 10 casos. As exceções são as curvas mais ruidosas: GMM no sintético (0,121) e no `tumors` (0,067). No sintético com GMM, todas as regras renderam mais que o `oracle_cv` (`loss_cv` ≤ 0): escolher *d* olhando os rótulos de outras sementes rendeu menos que qualquer regra.

#### Concordância com o clustering sem PCA

| Dataset | Algoritmo | `agreement_full` no máximo | `agreement_baseline` |
|---|---|---|---|
| sintético | kmeans | 1,00 | 1,00 |
| sintético | gmm | 0,87 | 0,83 |
| digits | kmeans | 0,94 | 0,92 |
| digits | gmm | 0,49 | 0,54 |
| MNIST | kmeans | 0,82 | 0,82 |
| MNIST | gmm | 0,14 | 0,39 |
| Fashion | kmeans | 0,75 | 0,81 |
| Fashion | gmm | 0,34 | 0,73 |
| tumors | kmeans | 0,50 | 0,45 |
| tumors | gmm | 0,37 | 0,46 |

No K-means, a concordância no máximo fica perto da linha de base (de −0,06 a +0,05). No GMM do MNIST ela cai para 0,14 enquanto o ARI dobra (0,166 para 0,327): o clustering com PCA é outro, e melhor. É o caso da seção 4.5 do contexto: concordância com o clustering original não mede qualidade.

#### Classificação: efeito de exigir o intervalo

Com só a margem, 50 das 890 combinações de dataset, algoritmo e *d* teriam outra classe (melhora ou piora em vez de indiferente). O maior grupo é o K-means no `tumors` (17), em que a margem é 0,041 e a variação entre sementes vem só da inicialização.

### Limitações

- **MNIST e Fashion-MNIST com 10 000 amostras e grade esparsa:** o máximo é o da grade. Os *d* das regras foram medidos exatamente.
- **`tumors` sem split:** os dados são os mesmos em todas as sementes, então o desvio-padrão mede só a inicialização do algoritmo, e as regras dão o mesmo *d* em todas as sementes.
- **GMM com uma inicialização (#3):** as curvas do GMM oscilam mais que as do K-means, e o máximo delas é menos confiável. Por isso o `oracle_cv` e o platô acompanham o máximo nas tabelas.
- **K fixo:** todos os sweeps usam o número de classes. A versão com K estimado em cada *d* (`estimate_K_flag=True`) está implementada e testada, mas não foi rodada nos datasets.

### Para a #6

- O `run_sweep` com `test_size=0` serve para o p/n alto, e o script sorteia uma subamostra por semente com `--n`. Exemplo: `--dataset mnist --n 500 --test-size 0`.
- A diferença treino–teste da H4 sai das colunas `ari_train` e `ari_test` do `sweep.csv`.
- A perda de cada regra em função de p/n (H5) sai do `compare.csv` de cada n.
