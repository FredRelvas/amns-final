# Experimento de p/n (issue #6, P5)

Como a #6 roda e é analisada. Tudo segue o que a #5 já definiu (`sweep.py`, `scripts/run_sweep.py` e a seção da #5 em `DECISOES.md`): mesma grade, mesma métrica, mesma referência de melhor *d* e mesmo critério de melhora/piora/indiferente. Os resultados e as decisões estão na seção da #6 em `DECISOES.md`.

## Como rodar

O `scripts/run_pn.py` chama o `scripts/run_sweep.py` para cada n e junta os resultados em `results/pn_<nome>/`. Cada n fica numa pasta própria (`n<n>` e `n<n>_sem-split`). Se uma rodada já terminou, ela é pulada, então dá para interromper e continuar depois.

```bash
# Sintético
uv run python scripts/run_pn.py --name synthetic_s5 --dataset synthetic \
    --synthetic 1000,1,5 --ns 4000,2000,1000,500,250,125 --no-split-max 4000

# MNIST
uv run python scripts/run_pn.py --name mnist --dataset mnist \
    --ns 70000,35000,10000,3500,1400,700,500,350,250,150,100 --no-split-max 3500

# Só refazer a análise e as figuras
uv run python scripts/run_pn.py --name mnist --dataset mnist \
    --ns 70000,35000,10000,3500,1400,700,500,350,250,150,100 --no-split-max 3500 --analyze-only
```

## Escolhas, todas herdadas da #5

| Item | Como fica | De onde vem |
|---|---|---|
| p/n | coluna `p_over_n` do `run_sweep`: p do dataset / n que entra (784/n no MNIST) | #5 |
| Grade de *d* | `1-30,35-100:5,120-600:40` mais os *d* das regras | grade do MNIST na #5 |
| Sementes | 0 a 9; a semente também sorteia a subamostra | #2 e #5 |
| Métrica | ARI de teste com split, ARI de todos os dados sem split (`default_score`) | #5 |
| Melhor *d* | `oracle_cv` (o *d* é escolhido nas outras sementes) | #5 |
| Margem e classificação | `_classify`: margem de 1 DP do ARI sem PCA **e** intervalo de 95% pareado | #5 |
| Split | com split em todos os n; também sem split nos n pequenos | "Treino e teste" e "Para a #6" em `DECISOES.md` |
| Regras | as da #5, sem d = K estimado | o critério de K ainda está em aberto (#3) |

## Valores de n

| Dataset | n | p/n |
|---|---|---|
| MNIST (p = 784) | 70 000, 35 000, 10 000, 3500, 1400, 700, 500, 350, 250, 150, 100 | 0,011 a 7,8 |
| Sintético (p = 1000, K = 10, ruído 1, separação 5) | 4000, 2000, 1000, 500, 250, 125 | 0,25 a 8 |

- O n = 10 000 é o mesmo da #5, o que permite conferir os números.
- No sintético a verdade é conhecida (*d* = K − 1 = 9). A teoria (transição BBP, Johnstone e Paul) prevê que o sinal se perca no ruído perto de p/n ≈ (separação²/K)² = 6,25.

## Análise (`src/pcacluster/pn.py`)

- **H5, a regra acerta o melhor *d*?** (`rule_vs_oracle`)
  - Para cada semente, calcula a diferença entre o ARI no *d* da regra e o ARI do `oracle_cv` nessa semente.
  - As diferenças passam pelo `_classify` da #5. Se a classe é "piora", a regra **erra**. Caso contrário, **acerta**.
  - A linha "sem PCA" entra como referência.
- **Ponto de quebra** (`breaking_point`): é o menor p/n a partir do qual a regra erra em todos os p/n maiores. Ele é reportado como intervalo, entre o último acerto e o primeiro erro.
- **Colapso do problema** (`has_signal`): com p/n alto, nem o melhor *d* agrupa (ARI ≈ 0). Aí todas as regras empatam e "acertam" sem mérito.
  - Um p/n só entra no cálculo da quebra enquanto o `oracle_cv` mantém pelo menos 50% do valor que tem no menor p/n. O 50% é escolha nossa, sem fonte.
  - O primeiro p/n sem sinal vai para a coluna `collapse` do `breaks.csv`. Nas figuras, a região sem sinal aparece sombreada.
- **H4, diferença treino–teste** (`train_test_gap`, só com split):
  - O gap é `ari_train − ari_test`, calculado por *d*.
  - O gap sem PCA é comparado, semente a semente, com o gap no *d* do `oracle_cv`, também pelo `_classify`. "Melhora" quer dizer que o PCA reduz o overfitting.

### Saídas em `results/pn_<nome>/`

| Arquivo | Conteúdo |
|---|---|
| `rules.csv` | por n, split, algoritmo e regra: *d* mediano, diferença para o `oracle_cv`, intervalo e acerto |
| `breaks.csv` | ponto de quebra de cada regra |
| `gap.csv`, `gap_by_d.csv` | H4: gap sem PCA e no *d* do `oracle_cv`, e gap por *d* |
| `rules.png`, `gap.png` | as duas coisas ao longo de p/n |
