# Painel OCR do teste

Execute na raiz do MambaLP, após ajustar os caminhos do YAML:

```bash
python3 basicsr/test.py -opt basicsr/test_benchmark.yml
```

Configuração (dentro de `test`, ao lado de `warmup_iterations`):

```yaml
test:
  warmup_iterations: 10
  ocr_dashboard:
    enabled: true
    update_interval: 0.5
    top_k: 10
    block_on_finish: true
```

`update_interval` é o intervalo mínimo entre redesenhos, em segundos. Todas as
amostras contribuem para os contadores, mesmo quando a tela não é redesenhada.
A última atualização sempre usa o resultado completo. `block_on_finish` mantém
a janela aberta até você fechá-la. Fechá-la antes não interrompe a avaliação.

Os dados são acumulados por dataset. Com `temporal_fusion.enabled: true`, cada
trilha entra uma vez, após majority vote das suas predições. Caso contrário,
cada imagem fornecida pelo dataset entra uma vez.

- Char precision: acertos na posição correta divididos pelas ocorrências
  preditas daquele caractere (TP / predições). Ao lado, a barra de acerto usa
  ocorrências do GT como denominador (TP / GT). Taxas sem suporte são zero no
  gráfico; os denominadores estão no JSON.
- Position precision: acertos / caracteres GT avaliados naquela posição.
- Most common error: caracteres GT com maior quantidade de erros posicionais.
- Most common confusion: pares GT → OCR incorretos mais frequentes.
- Matriz: linhas representam GT, colunas representam a predição, incluindo a
  diagonal de acertos. `∅` (`<PAD>` no CSV) representa caracteres faltantes ou
  extras. O botão alterna contagens e porcentagens normalizadas por linha.
  Passe o mouse sobre uma célula para consultar sua contagem e porcentagem.

A métrica de caracteres sem considerar ordem continua no cabeçalho. Ela é
separada das métricas posicionais usadas nos gráficos e na matriz.

Os gráficos são atualizados fora do intervalo cronometrado da inferência SR.
O número de parâmetros, o tempo médio e o desvio padrão continuam disponíveis.

Ao final, cada dataset recebe uma pasta em `results/<experimento>/ocr_dashboard/`
com `ocr_dashboard.png`, `ocr_metrics.json` e `character_confusion_matrix.csv`.
O `test_summary.json` também contém todas as estatísticas OCR.

A janela requer um backend gráfico Matplotlib. Em um desktop Linux, se o
backend estiver definido como Agg, execute com:

```bash
MPLBACKEND=TkAgg python3 basicsr/test.py -opt basicsr/test_benchmark.yml
```

Sem ambiente gráfico, o teste informa isso no log e ainda exporta os gráficos.
