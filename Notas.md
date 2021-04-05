# Problema de classificação
:::info
Versão mais crua: apenas teste e treino
:::

----

# Modelos de classificação a utilizar:
- VGG-16
- VGG-19
- ResNet (sugestão) (em keras)
- Inception (sugestão) (em keras)

VGG-16 -19 em keras é uma linha

----

Aqui vamos treinar os nossos modelos
(Learning from scratch [aprender do zero])

----

:::info
keras vgg-16 learning from scratch
:::

não é para introduzirmos os pesos

inicializamos os pesos através de uma função gausiana

---

Tipicamente treinamos por lotes (batchs)

input_shape: 16 (batch_size)
16, 128, 128, 3 (imagem 128x128)
no shape dizemos a organização

cada vez que tivermos a treinar a rede vamos ler 16 imagens
for onde
preenchemos uma extrutura 16, 128, 128, 3
outro vetor com labels

keras: train_on_batch

loss: diferença entre o previsto e o resultado

---

tabela (ficheiro csv)
path_imagem 1, classe 1

treinar com batch
a cada iteraçao escolhe aleatoriamente 16 elementos

train on batch

media loss por epoca
cada epoca: qnd precorre todas as imagens (16 img x 1000)


## Pt1:

Divisão do dataset em 3 partes
treino
teste
validação

## Pt2: Data aumentation

Pequenas rotações

## Pt3:


