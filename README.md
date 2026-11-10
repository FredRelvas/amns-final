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

As funções são stubs com as assinaturas combinadas e levantam `NotImplementedError`; cada pessoa implementa a sua parte na issue correspondente. As tarefas estão nas [issues](https://github.com/FredRelvas/amns-final/issues). Dados (`data/`) e saídas (`results/`) ficam fora do git.

## Sementes

Cada repetição de um experimento usa uma semente `s` de `pcacluster.SEEDS` (0 a 9), passada a tudo que é aleatório nela: geração do sintético ou subamostragem, split treino/teste e inicialização do clustering. Assim qualquer repetição pode ser refeita isoladamente.
