# Revisão das imagens SR por acerto OCR

No YAML de teste, dentro de `test`:

```yaml
test:
  warmup_iterations: 0
  save_ocr_images:
    enabled: true
```

Execute o teste na raiz do MambaLP:

```bash
python3 basicsr/test.py -opt basicsr/test_benchmark.yml
```

O conjunto é escolhido por `datasets.test.split` (padrão `testing`). Aceita
`test`/`testing`, `val`/`validation` e `train`/`training`, ignorando caixa e espaços.
Para avaliar intencionalmente as imagens de treino, use `split: training` mantendo
a chave `datasets.test`. Isso mede desempenho no treino, não generalização em teste.
Mudar somente `phase` não seleciona o split, pois o leitor de opções o sobrescreve.
Os caminhos relativos no CSV são resolvidos contra `dataroot`. No teste, cada pasta
precisa apenas das LR e da anotação da placa no CSV; HR é opcional. Treino e validação
continuam exigindo HR. Os prefixos são `lr-`/`lr_` e `hr-`/`hr_`, nas extensões
PNG/JPG/JPEG (sem distinção de caixa). O log informa contagens por split, trilhas
sem HR e motivos de descarte.

Sem HR, continuam sendo calculados OCR, tempo de inferência e número de parâmetros;
as SRs e LRs continuam sendo salvas. As métricas de imagem com referência (PSNR,
SSIM etc.) são calculadas somente nos itens com HR. Se nenhum tiver HR, ficam
indisponíveis (`null` no relatório), sem valor artificial de zero. O relatório
informa `reference_metric_count`: número de itens usados nas métricas de referência
(LRs no teste individual). O visualizador indica quando não há HR registrada.

Cada SR avaliada será salva em:

```text
results/<experimento>/ocr_images/<dataset>/N_acertos/<identificador>/sr.png
```

`N` é o número de caracteres corretos **na posição esperada**. Um caractere
faltante ou extra conta como erro. O visualizador mostra o total de posições
comparadas; por exemplo, uma predição com os sete caracteres certos e um extra
fica em `7_acertos`, mas aparece como `7/8`, com um erro e placa incorreta.

O teste processa **somente uma LR por trilha**, sem majority vote. Seleciona a
primeira LR em ordem de nome (ordem lexicográfica) para uma escolha reproduzível.
Essa LR gera uma SR e uma predição OCR, usada nas métricas, nos gráficos e nos
arquivos salvos. Assim, 3.000 trilhas geram 3.000 avaliações, mesmo que cada trilha
tenha cinco LR. Todas as HRs disponíveis continuam acessíveis no visualizador.

`test.py` força `single_lr_per_track: true`, desativa `temporal_fusion` e zera
`warmup_iterations`, inclusive com YAML antigo. Assim não há passes extras de
aquecimento; os tempos incluem as primeiras inferências, que podem ser mais lentas.
O modo de validação durante o treinamento continua seguindo sua configuração.

Cada registro também contém:

- `lr.png`: a LR redimensionada efetivamente fornecida ao modelo.
- GT, OCR, caracteres corretos/incorretos e caminhos originais em `result.json`.
- Cópias de todas as HRs de referência disponíveis na trilha na pasta `references`.
- Uma entrada no `manifest.jsonl` do dataset, escrita após os arquivos completos.

As HRs são copiadas uma única vez e compartilhadas pelas SRs correspondentes.
Todos os caminhos usados pelo visualizador são relativos ao arquivo de índice:
você pode mover a pasta de resultados completa. O dataset original não é alterado.
A cópia e gravação são feitas fora da medição do tempo de inferência SR.

## Abrir o visualizador

```bash
python3 basicsr/view_ocr_results.py --results results/MambaIR_RealSR_TestBenchmark/ocr_images
```

Também pode executar sem `--results` e selecionar a pasta pela interface.
O programa usa somente Pillow e Tkinter; não precisa de PyTorch ou GPU.

Controles:

- Filtre pelo número de acertos ou dataset e marque `Apenas com erro`.
- Busque por GT, OCR ou caminho da LR; pressione Enter ou `Filtrar`.
- Selecione uma linha ou use `Anterior`/`Próxima` e as setas do teclado.
- O OCR fica verde nos acertos e vermelho nos erros. `∅` marca ausência.
- Veja LR, SR e todas as HRs; role a área de imagens se necessário.
- Dê duplo clique em uma imagem para ampliar em 4×.
- Clique em `Atualizar lista` para ler novas SRs enquanto o teste ainda roda.

Ao atualizar a instalação, substitua os arquivos do ZIP na pasta `basicsr`.
