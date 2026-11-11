# Trabalho externo: escolher K por autoconsistência (kbench)

Um colega de fora do grupo testou, por conta própria, uma ideia muito próxima da nossa recursão (#9) e mandou os resultados em 11/11/2026. O material completo (notebook Colab, resultados por job, tabelas e figuras, ~32 MB) não está no repositório: está com o Fred. Quem quiser usar ou testar alguma parte, peça a ele.

Nada disso é resultado nosso. Se algo daqui entrar no relatório, é preciso dar o crédito.

## O que foi testado

A "pirâmide": estimar K, reduzir os dados a K dimensões, estimar K de novo nesse espaço e repetir até K parar de mudar (ponto fixo). Se K volta a um valor anterior sem parar, a rodada termina num ciclo. A partida é K = 2 ou o K escolhido no espaço completo (K₀).

Comparações:

| Método | O que faz |
|---|---|
| grid | escolhe K no espaço completo (o usual) |
| fixed m | reduz uma vez para 30 dimensões e escolhe K uma vez |
| eigengap | K tirado do maior salto do espectro do PCA |
| pirâmide | a recursão acima |

- **Dados:** 23 datasets. Incluem MNIST e Fashion-MNIST (só os 10 000 de teste), digits, olivetti (p/n ≈ 10), alguns tabulares de UCI/OpenML e sintéticos. Entre os sintéticos há os blobs com p de 64 a 4096 para n = 5000.
- **Combinações:** K-means e GMM diagonal. K escolhido por silhueta, Davies–Bouldin ou gap statistic; BIC não foi testado. Reduções: PCA, CTE (embedding de grafo) e projeção nos centroides.
- **Repetições:** a versão principal (K-means + silhueta, PCA e CTE) tem 3 sementes. A versão ampla tem 1 semente.

## Resultados com PCA (o nosso escopo)

ARI médio nos 23 datasets, com K-means e silhueta, 3 sementes:

| grid | fixed m | eigengap | pirâmide (de 2) | pirâmide (de K₀) |
|---|---|---|---|---|
| 0,48 | 0,54 | 0,36 | 0,53 | 0,54 |

- **A recursão não ganha de reduzir uma vez:** pirâmide de K₀ contra fixed m dá 0,539 contra 0,536, sem diferença em 18 dos 23 datasets (Wilcoxon p = 0,80). Contra o grid ela ganha (0,54 contra 0,48, p = 0,02). O ganho vem de reduzir, não de repetir.
- **O ganho sobre o grid é, sobretudo, escolher K melhor.** Em dados com muitas colunas, a silhueta no espaço completo escolhe K = 2 (MNIST, Fashion-MNIST, USPS, isolet, HAR). No MNIST o grid fica com ARI 0,08 e a pirâmide chega a 0,31, com K = 5.
- **Com PCA, o eigengap é ruim:** 0,36. Nos blobs ele aponta K = 21 (ver o sintético abaixo).

## O que interessa para cada issue

### #9 Recursão

- **Para onde converge depende do critério que escolhe K:**
  - Com a silhueta, quase sempre chega a um ponto fixo: 58 de 69 rodadas partindo de K₀, mais 8 ciclos e 3 no teto. Nunca chega a K = 1, porque a silhueta só avalia K ≥ 2.
  - Com a gap statistic (que aceita K = 1), partindo de 2 dimensões, a recursão colapsa em um cluster: 13 de 23 datasets com K-means e 15 de 23 com GMM. É o mesmo colapso do nosso piloto (61 → 24 → 8 → 4 → 1).
  - "Um cluster por amostra" nunca aparece, porque K tem teto.
- **K aumenta em algum nível em 31 de 69 rodadas** (MNIST: 2 → 3 → 4 → 5 → 5). Ele sempre pega os K primeiros componentes do PCA do X original. A definição do professor é "tirar outro PCA a partir da projeção". As duas coincidem enquanto K não aumenta, porque os componentes da projeção são os primeiros componentes do X. Quando K aumenta, a versão do professor não tem como continuar: não há 5 componentes numa projeção de 4 dimensões. Isso precisa de uma decisão nossa.
- **Lógica de parada:** a função dele para no ponto fixo, detecta ciclos comparando com os estados já visitados e tem um limite de níveis por segurança. Dá para reaproveitar a ideia.

### #3 Clustering que estima K

- **A silhueta no espaço completo** cai para K = 2 em dados com muitas colunas (com K-means: MNIST, Fashion-MNIST, USPS, isolet, HAR). O Davies–Bouldin cai para 2 ou 3 em parte deles (Fashion-MNIST, HAR, isolet). No espaço reduzido os dois funcionam melhor.
- **A gap statistic no espaço completo** acerta mais (ARI 0,52 com K-means e 0,54 com GMM) e quase nunca devolve K = 1 (1 a 2 de 23). Com K-means, porém, escolhe K de 26 a 30 em MNIST, Fashion-MNIST, USPS e HAR, perto do teto de 30.
- **BIC com GMM, o nosso padrão, não foi testado.**
- **O teto de K usou os rótulos:** é o maior entre 30 e 2 × K verdadeiro. Nós precisamos de um teto que não dependa deles.

### #6 p/n e o sintético

- **Gerador diferente do nosso.** Cada cluster se espalha num subespaço latente de 20 dimensões. O espectro do PCA fica com 9 autovalores grandes (entre clusters, K−1) e mais 11 médios (espalhamento dentro dos clusters), e só depois vem o ruído. Uma regra de ruído (MP, análise paralela) contaria cerca de 20, não 9. O nosso sintético tem sinal exatamente em K−1 dimensões, então "d verdadeiro = K−1" só vale para o nosso gerador. Testar as regras também num gerador assim é uma boa checagem de robustez.
- **Blobs com n = 5000 e p de 64 a 4096 (p/n de 0,013 a 0,82):**
  - Com K escolhido no espaço completo, a escolha de K piora com p: o ARI cai de 1,00 para 0,81 e K de 10 para 8.
  - Reduzindo antes para 30 dimensões, o resultado fica em ARI 1,00 e K = 10 em todos os p.

### #5 Sweep

- **Números de referência** para conferir o nosso sweep, todos com K-means e silhueta, sem separação treino/teste:
  - digits: ARI ≈ 0,60 com K = 9.
  - MNIST: ARI 0,36 reduzindo para 30 dimensões.
- **Comparação estatística:** ele comparou os métodos pareados por dataset (Wilcoxon) e por ranking (Friedman com diferença crítica de Nemenyi). Serve de modelo para comparar as regras na H1.

## Ressalvas

- **Sem separação treino/teste:** tudo é ajustado e avaliado nos mesmos dados. Não serve para a H4.
- **Teto de K e dimensão do fixed m:** ambos usam o K verdadeiro (ver #3). Na prática, isso só muda o teto em letter, isolet e olivetti.
- **Variância e literatura:** a versão ampla tem uma semente só, sem estimativa de variância. Ele não verificou se o método já existe na literatura.
- **CTE:** é a melhor configuração dele (pirâmide 0,64 contra grid 0,48), mas é uma redução por grafo, fora do nosso escopo atual, que é só PCA.
