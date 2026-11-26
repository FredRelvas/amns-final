# amns-final

Trabalho final de Aprendizado Não Supervisionado (IA/UFG). Pergunta: reduzir a dimensão com PCA antes do clustering melhora, piora ou é indiferente? E regras sem rótulos para escolher o número de componentes *d* acertam o melhor *d* de um sweep completo?

Contexto completo em [`docs/CONTEXTO_PROJETO.md`](docs/CONTEXTO_PROJETO.md) e resumo em [`docs/trabalho_final_pca_clustering.pdf`](docs/trabalho_final_pca_clustering.pdf). As decisões tomadas em cada parte, com os motivos, ficam em [`docs/DECISOES.md`](docs/DECISOES.md).

## Instalação

Requer [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Para adicionar uma dependência: `uv add <pacote>`.

## Testes

```bash
uv run pytest
```

## Organização

| Módulo | Conteúdo | Issue |
|---|---|---|
| `src/pcacluster/data.py` | carga dos datasets, subamostragem, split e pré-processamento | 2 |
| `src/pcacluster/metrics.py` | ARI, NMI, ACC e erro de reconstrução | 2 |
| `src/pcacluster/clustering.py` | K-means, GMM e estimativa de K | 3 |
| `src/pcacluster/rules.py` | regras para escolher *d* | 4 |
| `src/pcacluster/sweep.py` | sweep de *d* com sementes e treino/teste | 5 |
| `src/pcacluster/pn.py` | ao longo de p/n: acerto das regras, ponto de quebra e diferença treino–teste | 6 |
| `src/pcacluster/plots.py` | figuras do sweep e ao longo de p/n | 5 e 6 |

As tarefas estão nas [issues](https://github.com/FredRelvas/amns-final/issues). Dados (`data/`) e saídas (`results/`) ficam fora do git.

## Experimentos

`scripts/run_sweep.py` roda o sweep de um dataset com K-means e GMM, calcula o *d* de cada regra em cada semente e grava tabelas e figuras em `results/<tag>/`, cujo nome junta o dataset e as opções (ex.: `mnist_n10000`, `tumors_sem-split`):

```bash
uv run python scripts/run_sweep.py --dataset digits
uv run python scripts/run_sweep.py --dataset tumors --K 11 --test-size 0
uv run python scripts/run_sweep.py --dataset mnist --n 10000 --grid 1-30,35-100:5,120-600:40
```

`scripts/run_pn.py` repete o sweep para vários n, uma chamada ao `run_sweep.py` por n, e junta os resultados em `results/pn_<nome>/`. Cada n fica numa pasta própria (`n<n>` e `n<n>_sem-split`). As rodadas que já terminaram são puladas, então dá para interromper e continuar depois:

```bash
uv run python scripts/run_pn.py --name synthetic_s5 --dataset synthetic \
    --synthetic 1000,1,5 --ns 4000,2000,1000,500,250,125 --no-split-max 4000
uv run python scripts/run_pn.py --name mnist --dataset mnist \
    --ns 70000,35000,10000,3500,1400,700,500,350,250,150,100 --no-split-max 3500
```

Com `--analyze-only`, o script só refaz a análise e as figuras.

As opções estão em `--help`. O que cada arquivo de saída contém e como ler os resultados está nas seções das issues #5 e #6 em [`docs/DECISOES.md`](docs/DECISOES.md).

## Sementes

Cada repetição de um experimento usa uma semente `s` de `pcacluster.SEEDS` (0 a 9), passada a tudo que é aleatório nela: geração do sintético ou subamostragem, split treino/teste e inicialização do clustering. Assim qualquer repetição pode ser refeita isoladamente.
