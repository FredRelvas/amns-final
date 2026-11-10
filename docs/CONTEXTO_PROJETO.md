# Contexto do projeto: PCA antes do clustering

Documento de contexto para o agente de código. Reúne tudo o que o grupo discutiu até 09/11/2026. O PDF `trabalho_final_pca_clustering.pdf` (enviado junto) é a versão resumida para humanos; este arquivo é a versão completa para execução.

## 0. O que o agente deve fazer

1. Verificar `gh auth status` e `uv --version`. Se algo faltar, avisar o usuário antes de continuar.
2. Perguntar ao usuário (se não estiver claro): nome do repositório (sugestão: `pca-before-clustering`), visibilidade (sugestão: privado) e se deve ficar numa organização ou na conta pessoal.
3. Criar o repositório com **uv** (o usuário prefere uv a pip em qualquer instalação), seguindo a estrutura da seção 5.
4. Criar os **stubs** das interfaces da seção 6 (assinaturas, docstrings, `NotImplementedError`) e testes só para o que for trivial. **Não implementar a parte científica**: isso é o trabalho do grupo, dividido nas issues.
5. Colocar o PDF em `docs/` e este arquivo em `docs/CONTEXTO_PROJETO.md`.
6. Criar labels, milestones e as **issues da seção 8** com `gh`. Uma issue por tarefa, com critérios de aceitação em checkboxes.
7. Fazer commit e push. Escrever o `README.md` em português, curto, com como instalar (`uv sync`), como rodar os testes e como o trabalho está organizado.

Convenções: código e nomes de funções em inglês; commits, README, docs e issues em português do Brasil.

## 1. Contexto acadêmico

- Trabalho final da disciplina de **Aprendizado Não Supervisionado** (curso de Inteligência Artificial, UFG). Grupo de 5 pessoas.
- O professor pediu escolher um tema dos livros (Murphy, *Probabilistic Machine Learning: An Introduction*; Bishop, *Deep Learning: Foundations and Concepts*) ou uma pergunta da área não supervisionada.
- Estrutura exigida do trabalho: **Pergunta de Pesquisa, Motivação, Metodologia, Resultados, Conclusões**.
- Referência de como conduzir a pesquisa, indicada pelo professor: os 10 passos de modelagem da Neuromatch Academy (https://deeplearning.neuromatch.io/projects/modelingsteps/Example_Deep_Learning_Project.html).
- O grupo já apresentou KNN e GMM no começo do semestre.
- Em conversa com o professor (09/11/2026), ele desenhou no quadro o esquema da seção 3 e confirmou que usar o MNIST para as métricas não tem problema.

## 2. Pergunta de pesquisa e hipóteses

> Reduzir a dimensão com PCA antes de um algoritmo de clustering melhora, piora ou é indiferente para a qualidade dos clusters? Uma regra sem rótulos para escolher o número de componentes *d* (usar *d* = *K*, o número de clusters estimado pelo algoritmo, ou critérios de ruído como a borda de Marchenko-Pastur e a análise paralela) acerta o *d* de melhor desempenho de um sweep completo? E como isso muda com p/n, e a partir de qual p/n o método quebra?

| | Hipótese | Como testar |
|---|---|---|
| H1 | O *d* indicado pela regra (d = K, Marchenko-Pastur com σ² estimado, análise paralela) fica próximo do melhor *d* do sweep, e mais perto dele que os baselines simples (90% da variância, Kaiser). | Comparar o desempenho no *d* de cada regra com o máximo da curva do sweep, no conjunto de teste, com repetições. |
| H2 | O *d* ótimo e o *d* das regras diminuem quando p/n aumenta. | Subamostrar n (p/n de 0,01 a acima de 1) e repetir o sweep. |
| H3 | O efeito do PCA depende do algoritmo (K-means, GMM; KNN como sonda da informação preservada). | Uma curva por algoritmo, mesmos dados. |
| H4 | O PCA reduz o overfitting com muitas dimensões: a diferença treino–teste cai ao reduzir *d*. | PCA e padronizador ajustados só no treino; medir a diferença treino–teste por *d*. |
| H5 | Existe um p/n a partir do qual a regra deixa de acertar o *d* ótimo (o método "quebra"). | Medir a perda da regra em relação ao melhor *d* em função de p/n; definir o ponto de quebra. |

Extensões (só se sobrar tempo): versão recursiva do procedimento; outras reduções de dimensão além do PCA (projeção aleatória, autoencoder, UMAP).

## 3. Esquema do quadro do professor

```
        A = algoritmo de clustering (K é dado!)

X (200×1) ──A──▶ Z ∈ {1,…,K}            ⇒ ~80%   (número em vermelho, pouco legível)
   │
  PCA
   ▼
X~ (K×1) ──A──▶ Z~ ∈ {1,…,K~}           ⇒ ~95–98% (número em vermelho, pouco legível)
```

Leitura: o algoritmo A (com K dado) agrupa X em alta dimensão. O PCA reduz X para X~, e o mesmo A agrupa o resultado, com acurácia maior. Os percentuais são só ordem de grandeza.

### Notação adotada (no quadro a letra K tem dois sentidos; aqui ficam separados)

| Símbolo | Significado |
|---|---|
| n, p, p/n | nº de amostras, nº de variáveis (dimensão original) e a razão entre eles |
| d | nº de componentes do PCA (dimensão reduzida; no quadro: K×1) |
| K | nº de clusters dado (ex.: 10 no MNIST e no digits) |
| K_est | nº de clusters estimado pelo algoritmo (no quadro: K~ no espaço reduzido) |

## 4. Desenho experimental

### 4.1 Procedimento (nota do usuário)

1. Criar um algoritmo de clustering que calcula um K e define clusters.
2. Pegar um dataset com certo p/n, fazer os clusters e guardar (o clustering "original", em alta dimensão).
3. Passar um PCA com K componentes (o mesmo K calculado pelo algoritmo) e rodar o mesmo algoritmo no espaço reduzido; avaliar como a performance mudou em relação ao clustering original.
4. Em paralelo, fazer um **sweep** de todos os *d* possíveis (1 até min(n−1, p)), rodar o clustering em cada um, achar o *d* com melhor performance e avaliar se a regra (d = K, ou critérios de ruído) acerta esse melhor *d*.

### 4.2 Regras candidatas para escolher *d* (todas sem rótulos, exceto a referência)

| Regra | Definição |
|---|---|
| d = K (ou K−1) | usa o nº de clusters (Ding e He indicam K−1 componentes para K clusters) |
| Marchenko-Pastur | padronizar os dados; autovalores `ev` da matriz de correlação; γ = p/n; σ² estimado pela mediana: `σ²_est = mediana(ev) / E[mediana dos autovalores da correlação de ruído N(0,1) n×p]` (simular ~30 matrizes); borda = σ²_est·(1+√γ)²; **d = nº de autovalores > borda** |
| Análise paralela | embaralhar cada coluna independentemente (60 a 100 repetições), recalcular os autovalores, tomar o percentil 95 por índice; **d = nº de autovalores reais acima desse percentil** |
| 90% da variância | menor d tal que a variância explicada acumulada ≥ 0,90 |
| Kaiser | autovalores > 1 (dados padronizados) |
| Sweep completo | melhor d medido contra os rótulos: **referência**, não regra |

Variância explicada de d componentes = soma dos d maiores autovalores dividida pela soma de todos. O erro de reconstrução (média da distância quadrática entre X e X reconstruído) é a soma dos autovalores descartados.

### 4.3 Decisões já tomadas

- **Rótulos só como régua**: os algoritmos nunca os veem (validação externa). O MNIST pode ser usado para métricas (confirmado pelo professor).
- **K fixo** como desenho principal (K = nº de classes). Limitação a declarar: o K "ideal" depende de d (no digits: BIC escolheu 5 a 16 e silhouette 8 a 15, contra os 10 reais). Testar a regra d = K com K verdadeiro **e** com K estimado.
- **Treino × teste**: ajustar padronizador, remoção de colunas constantes e PCA **só no treino**; avaliar no teste (clustering: `fit` no treino, `predict` no teste).
- **Classificar o efeito como melhora, piora ou indiferente**, com uma margem baseada na variação entre repetições (proposta: diferença menor que 1 desvio-padrão entre sementes = indiferente).
- **Variar p/n** subamostrando n (de ~0,01 até acima de 1, com MNIST: n=1000 dá p/n≈0,78; n=500 dá ≈1,6).
- **Baselines simples** sempre incluídos: sem PCA, 90% da variância, Kaiser, cotovelo. Justificativa do projeto: as regras de ruído não precisam de rótulos nem de testar todos os d.
- O sweep vai até min(n−1, p): para p > n o PCA tem no máximo n−1 componentes não nulos.

### 4.4 Métricas

| Métrica | Uso |
|---|---|
| ARI e NMI contra os rótulos | qualidade do clustering (régua principal) |
| Acurácia do KNN (k=5, treino→teste) | sonda da informação útil preservada pela redução |
| Erro de reconstrução | quanto o PCA perde, sem clustering e sem rótulos |
| Concordância (ARI) entre o clustering reduzido e o original | quanto os clusters se mantêm; **exige linha de base de estabilidade** (ARI entre duas sementes no espaço completo), porque o K-means muda de partição até com d=40 |

ARI: 1 = partições idênticas, ~0 = acaso, invariante à troca de nomes dos clusters.

### 4.5 Pontos de atenção

- O K-means é instável: sempre várias sementes, `n_init` ≥ 5, barras de erro.
- "Em relação ao cluster original" mede concordância, não qualidade: o ARI contra os rótulos continua necessário.
- O desempenho do K-means e do KNN costuma formar **platô** a partir de d≈8 a 15; o argmax do sweep é ruidoso. Reportar também o menor d que atinge ≥95% do máximo.
- A silhouette **não é comparável** entre espaços de dimensões diferentes.

## 5. Estrutura sugerida do repositório

```
pca-before-clustering/
├── README.md
├── pyproject.toml            # gerenciado por uv (uv init / uv add)
├── uv.lock
├── .python-version           # 3.12 (ou 3.11)
├── .gitignore                # data/, results/ (exceto .gitkeep), .venv, __pycache__
├── docs/
│   ├── CONTEXTO_PROJETO.md   # este arquivo
│   └── trabalho_final_pca_clustering.pdf
├── src/pcacluster/
│   ├── __init__.py
│   ├── data.py               # carga, split, pré-processamento, subamostragem
│   ├── metrics.py
│   ├── clustering.py         # K-means, GMM, estimate_K
│   ├── rules.py              # regras para escolher d
│   ├── sweep.py
│   └── plots.py
├── scripts/                  # run_sweep.py, run_pn.py (CLI com argparse)
├── notebooks/                # exploração (limpar saídas antes de commitar)
├── configs/                  # YAML/TOML com datasets, grades de d, sementes
├── tests/
├── data/                     # cache local (ignorado pelo git)
└── results/                  # saídas (CSV/parquet e figuras), ignorado pelo git
```

Dependências (via `uv add`): `numpy`, `scipy`, `scikit-learn`, `pandas`, `pyarrow`, `matplotlib`, `joblib`, `tqdm`. Dev (`uv add --dev`): `pytest`, `ruff`. Opcional (grupo `extras`): `torch` (autoencoder), `umap-learn`. MNIST e Fashion-MNIST via `sklearn.datasets.fetch_openml` (`mnist_784`, `Fashion-MNIST`), com cache em `data/`. O `digits` vem do scikit-learn.

Configurar também: `ruff` (lint + format) e um workflow mínimo de GitHub Actions que roda `uv sync` e `uv run pytest`.

## 6. Interfaces compartilhadas (stubs; o grupo implementa)

Definir cedo para as 5 pessoas trabalharem em paralelo. Assinaturas sugeridas:

```python
# data.py
def load_dataset(name: str, *, n: int | None = None, seed: int = 0, **kw) -> tuple[np.ndarray, np.ndarray]:
    """name em {'digits','mnist','fashion_mnist','synthetic'}. 'synthetic' aceita n_clusters, p, noise_std, separation."""
def subsample(X, y, n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]: ...
def split(X, y, test_size: float = 0.3, seed: int = 0): ...   # estratificado
class Preprocessor:
    def fit(self, X_train): ...        # remove colunas constantes + padroniza, tudo no treino
    def transform(self, X): ...

# metrics.py
def ari(y_true, y_pred) -> float: ...
def nmi(y_true, y_pred) -> float: ...
def knn_accuracy(Z_train, y_train, Z_test, y_test, k: int = 5) -> float: ...
def reconstruction_error(X, X_hat) -> float: ...

# clustering.py
class ClusterAlgo(Protocol):
    name: str
    def fit(self, Z, K: int) -> "ClusterAlgo": ...
    def predict(self, Z) -> np.ndarray: ...
class KMeansAlgo: ...      # n_init>=5
class GMMAlgo: ...         # covariance_type='diag'
def estimate_K(Z, algo_name: str, K_grid: range, criterion: str = "bic") -> int: ...  # 'bic' | 'silhouette'

# rules.py  (todas recebem X_train já pré-processado e devolvem int)
def d_marchenko_pastur(X_train, sigma2: float | str = "median") -> int: ...
def d_parallel_analysis(X_train, n_perm: int = 100, q: float = 95, seed: int = 0) -> int: ...
def d_from_K(K: int, minus_one: bool = False) -> int: ...
def d_variance_threshold(X_train, frac: float = 0.90) -> int: ...
def d_kaiser(X_train) -> int: ...

# sweep.py
def run_sweep(X, y, algo_name, d_grid, K, *, seeds, test_size=0.3, estimate_K_flag=False) -> pd.DataFrame:
    """Colunas: dataset, n, p, p_over_n, seed, algo, d, K, ari_train, ari_test, nmi_test,
       knn_acc_test, recon_err_test, agreement_full (ARI vs clustering em dimensão completa)."""
```

## 7. Resultados do piloto (para orientar; não são conclusões)

Piloto exploratório no `digits` (n=1.797, p=61 após remover pixels constantes; 10 classes). **Limitações**: um dataset, 10 repetições na subamostragem, sem barras de erro, grade grosseira de d, PCA ajustado com todos os dados (sem split treino/teste). Os scripts do piloto não estão no repositório; reimplementar a partir das definições da seção 4.

| Observação | Números |
|---|---|
| d pelas regras no digits completo | MP com σ²=1: 12; análise paralela: 16; Kaiser: 17; MP com σ² estimado (0,45): **24**; 90% da variância: 31 |
| K-means (sem PCA → melhor d) | ARI 0,47 → 0,53 a 0,55 (ganho ≤ 0,04, dentro do ruído entre repetições); curva praticamente plana a partir de d≈8 a 12 |
| GMM (diag) | ARI 0,36 sem PCA → 0,49 no melhor d (4 a 12); o d das regras fica acima do ótimo |
| KNN (k=5) | acurácia ~0,944 sem PCA; platô a partir de d≈10 a 15; com o d de MP (σ² estimado) fica a ≤0,006 do máximo nos 5 tamanhos de n (subamostragem n = 1797, 900, 450, 225, 120; p/n de 0,03 a 0,45) |
| Perda da regra vs melhor d (KNN) | MP (σ² estimado): ≤0,006; análise paralela: 0,009 a 0,045; MP (σ²=1): 0,042 a 0,064 |
| d = K = 10 | perde 0,013 a 0,048 de ARI no K-means, 0,026 a 0,056 no GMM, 0,016 a 0,047 de acurácia no KNN |
| K estimado vs d | BIC (GMM diag): 5 a 16; silhouette (K-means): 8 a 15 (K real = 10); não recupera K=10 de forma confiável |
| Concordância com o clustering completo | só passa de 0,9 perto da dimensão total (d ≥ 54): K-means é instável |
| Recursão da regra de ruído | 61 → 24 → 8 → 4 → 1 (colapsa; interpretação de "recursivo" ainda a confirmar com o professor) |
| Ruído puro (n=500, p=50) | 2 primeiros componentes explicam 6,8% da variância (não 4%); trustworthiness 0,61 |

Teoria de apoio (Johnstone e Paul, 2018): modelo spiked, ruído σ²I; autovalores de ruído puro ocupam [σ²(1−√γ)², σ²(1+√γ)²]; um sinal só é detectável se ℓ₁ > σ²(1+√γ) (transição BBP); acima do limiar, o cos² entre o autovetor estimado e o verdadeiro é (1−γ/(ℓ₁−1)²)/(1+γ/(ℓ₁−1)) (com σ²=1), ou seja, a direção estimada nunca é perfeita.

## 8. Issues a criar

Labels sugeridas: `setup`, `dados`, `clustering`, `regras-d`, `sweep`, `p-sobre-n`, `leitura`, `escrita`, `extensao`, `pergunta-professor`, `P1` a `P5`. Milestones: `M1 Infraestrutura` (issues 1 a 3), `M2 Experimentos` (4 a 6), `M3 Escrita e entrega` (7 a 9). As pessoas (P1 a P5) serão atribuídas pelo usuário depois; usar as labels P1 a P5 por enquanto. Estimativas são só de código (+ ~3 h de leitura e ~3 h de escrita por pessoa).

### Issue 1: Estrutura do repositório e interfaces compartilhadas (`setup`) · agente
- [ ] Repositório com uv, estrutura da seção 5, README, CI mínimo
- [ ] Stubs das interfaces da seção 6 com docstrings e `NotImplementedError`
- [ ] PDF e este contexto em `docs/`
- [ ] `uv sync && uv run pytest` passa em máquina limpa

### Issue 2: Dados e infraestrutura (`dados`, `P1`) · ~7 h · leitura: Yeung e Ruzzo
- [ ] `load_dataset` para `digits`, `mnist`, `fashion_mnist` e `synthetic` (clusters e ruído conhecidos; parâmetros de separação e ruído)
- [ ] `split` estratificado e `Preprocessor` ajustado só no treino (remove colunas constantes, padroniza)
- [ ] `subsample` para variar p/n
- [ ] Funções de `metrics.py` (ARI, NMI, KNN, reconstrução) com testes
- Critério de aceitação: testes cobrindo ausência de vazamento (o `Preprocessor` não vê o teste) e valores conhecidos de ARI em exemplos pequenos (ex.: rótulos trocados dão 1,0; tudo num cluster dá 0,0).

### Issue 3: Algoritmo de clustering que estima K (`clustering`, `P2`) · ~8 h · leitura: Ding e He
- [ ] `KMeansAlgo` e `GMMAlgo` (diag) com `fit`/`predict`
- [ ] `estimate_K` por BIC e por silhouette
- [ ] Validação em dados sintéticos com K conhecido (quanto acerta, por dimensão d)
- [ ] Gera e guarda o clustering "original" em alta dimensão
- Depende de: Issue 2 (pode começar com `digits` do scikit-learn).

### Issue 4: Regras para escolher d (`regras-d`, `P3`) · ~7 h · leitura: Johnstone e Paul; Horn
- [ ] Marchenko-Pastur com σ² = 1 e com σ² estimado pela mediana
- [ ] Análise paralela (percentil 95)
- [ ] d = K, d = K−1, 90% da variância, Kaiser, cotovelo
- [ ] Teste: em ruído puro (n=500, p=50) a regra MP devolve ~0; em dados com spike forte devolve o número de spikes
- Depende de: Issue 2 (pré-processamento).

### Issue 5: Sweep e comparação (`sweep`, `P4`) · ~9 h · leitura: Gavish e Donoho
- [ ] `run_sweep` para todo d, com várias sementes e split treino/teste, salvando o DataFrame (colunas na seção 6)
- [ ] Curvas de desempenho por d com barras de erro
- [ ] Classificação em melhora/piora/indiferente com margem baseada na variância entre sementes
- [ ] Comparação do d de cada regra com o melhor d (perda de desempenho)
- Depende de: Issues 2, 3 e 4.

### Issue 6: p/n, overfitting e ponto de quebra (`p-sobre-n`, `P5`) · ~8 h · leitura: Murphy cap. 20.1; estudo sistemático recente (arXiv 2604.22099)
- [ ] Sweep para p/n de 0,01 a acima de 1 (MNIST subamostrado)
- [ ] Diferença treino–teste por d (H4)
- [ ] Definição operacional de "quebra" e estimativa do ponto de quebra (H5)
- Depende de: Issue 5.

### Issue 7: Leitura e resumos (`leitura`) · todos
- [ ] Johnstone e Paul (2018), PCA in High Dimensions: An Orientation — https://pmc.ncbi.nlm.nih.gov/articles/PMC6167023/ (foco: modelo spiked, Marchenko-Pastur, BBP; pode pular universalidade e testes de esfericidade)
- [ ] Horn (1965), análise paralela — Springer (provavelmente pago; tentar Portal CAPES com login da UFG); alternativas gratuitas: slides https://psychometrika.org/sites/main/files/file-attachments/horn_1965_slides.pdf e pacote hornpa
- [ ] Ding e He (2004), K-means clustering via PCA — https://icml.cc/Conferences/2004/proceedings/papers/262.pdf
- [ ] Yeung e Ruzzo (2001), PCA for clustering gene expression data — https://homes.cs.washington.edu/~ruzzo/papers/pca-bioinf.html
- [ ] Gavish e Donoho (2014), optimal hard threshold 4/√3 — https://arxiv.org/abs/1305.5870
- [ ] Murphy cap. 20.1 e 21; arXiv 2604.22099 (só o título foi visto; ainda não lido)
- Entregável: resumo de meia página por pessoa em `docs/leituras/`. Antes de dizer que o trabalho é inédito, fazer busca de literatura mais completa (há trabalhos próximos).

### Issue 8: Escrita do relatório (`escrita`) · todos
- [ ] Pergunta e Motivação (P5) · Metodologia (P1 a P3) · Resultados (P4 e P5) · Conclusões (todos)
- [ ] Declarar limitações: K fixo, rótulos como gabarito (não necessariamente clusters naturais), poucos datasets, instabilidade do K-means

### Issue 9: Extensão: recursão (`extensao`)
- [ ] Reaplicar a regra de ruído sobre dados já reduzidos (no piloto colapsa para 1 dimensão) **ou** reestimar K no espaço reduzido (K~). Depende da resposta da Issue 11.

### Issue 10: Extensão: outras reduções de dimensão (`extensao`)
- [ ] Comparar com projeção aleatória e, se der tempo, autoencoder e UMAP, usando as mesmas interfaces e o mesmo sweep.

### Issue 11: Perguntas para o professor (`pergunta-professor`)
- [ ] O que exatamente significa "recursivo" na anotação ("convergir em um cluster por amostra ou um cluster pra todos ou o ótimo")? Reaplicar a redução ou reestimar K~?
- [ ] O escopo é só PCA, ou o professor espera autoencoders?
- [ ] O KNN como sonda de informação preservada cabe no escopo, ou ficamos só em clustering?
- [ ] Quanto de teoria de matrizes aleatórias é esperado na Metodologia?
- [ ] Os percentuais do quadro (~80% e ~95–98%) são ilustrativos ou há uma métrica específica em mente?

## 9. Ordem de trabalho

Issue 1 primeiro (agente). Depois, Issues 2, 3 e 4 em paralelo (as pessoas 2 e 3 já podem usar o `digits` do scikit-learn enquanto a infraestrutura de dados termina). Issue 5 depende de 2 a 4, e Issue 6 depende de 2 e 5. Leitura e escrita acontecem em paralelo.

## 10. Reprodutibilidade (definição de pronto de qualquer experimento)

- Todas as sementes explícitas e registradas; configs em `configs/`.
- Resultados em `results/` como CSV/parquet com as colunas da seção 6; figuras geradas por scripts (nunca à mão).
- Rodar com um comando (`uv run python scripts/run_sweep.py --config configs/digits.yaml`).
- Notebooks só para exploração; a lógica vive em `src/`.
