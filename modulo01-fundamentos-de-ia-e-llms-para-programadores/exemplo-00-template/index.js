import * as tf from '@tensorflow/tfjs';

async function trainModel(inputXs, outputYs) {
    const model = tf.sequential();

    // Primeira camada de rede:
    // Entrada de 7 posições (idade normalizada, 3 cores, 3 localizações)

    // 80 neurônios = aqui coloquei tudo isso pois tem pouca base de treino
    // Quanto mais neurônios, mais complexidade a rede pode aprender
    // e consequentemente mais processamento usará

    // A ReLU age como um filtro:
    // É como se deixasse somendo dados interessantes seguirem viagem na rede
    // Se a informação chegou nesse neurônio é positiva, passe para frente!
    // Se for negativa, descaste.
    model.add(tf.layers.dense({ inputShape: [7], units: 80, activation: 'relu' }));

    // Saída: 3 neurônios
    // Um para cada categoria (premium, medium, basic)

    // Activation: softmax - Normaliza a saída em probabilidades
    model.add(tf.layers.dense({ units: 3, activation: 'softmax' }));

    // Compilando o modelo
    // optimizer: 'adam' - Algoritmo de otimização para ajustar os pesos da rede
    // Aprende com histórico de erros e acertos durante o treino

    // loss: 'categoricalCrossentropy'
    // Compara o que o modelo "acha" (os scores de cada categoria)
    // com o que é esperado (labels) e calcula o erro

    // metrics: ['accuracy'] - Métrica para avaliar o desempenho do modelo
    // Exemplo: classificação de imagem, recomendação, categorização de usuário

    model.compile({
        optimizer: 'adam',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    // Treinando do modelo
    await model.fit(
        inputXs,
        outputYs,
        {
            verbose: 0, // 0 para não mostrar o progresso do treino
            epochs: 100, // Número de vezes que o modelo verá todo o dataset
            shuffle: true, // Embaralha os dados a cada época para evitar padrões de ordem
            callbacks: {
                // onEpochEnd: (epoch, logs) => {
                //     console.log(`epoch ${epoch}: loss = ${logs.loss}`);
                // }
            }
        }
    );

    return model;
}

async function predict(model, pessoa) {
    // Transforma o array js para o tensor (tfjs)
    const tInput = tf.tensor2d(pessoa);

    // Fazendo a previsão (output será um vetor de 3 pprobabilidades)
    const pred = model.predict(tInput);
    const predArray = await pred.array();

    return predArray[0].map((prob, index) => ({ prob, index }));
}

// Exemplo de pessoas para treino (cada pessoa com idade, cor e localização)
// const pessoas = [
//     { nome: "Erick", idade: 30, cor: "azul", localizacao: "São Paulo" },
//     { nome: "Ana", idade: 25, cor: "vermelho", localizacao: "Rio" },
//     { nome: "Carlos", idade: 40, cor: "verde", localizacao: "Curitiba" }
// ];

// Vetores de entrada com valores já normalizados e one-hot encoded
// Ordem: [idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]
// const tensorPessoas = [
//     [0.33, 1, 0, 0, 1, 0, 0], // Erick
//     [0, 0, 1, 0, 0, 1, 0],    // Ana
//     [1, 0, 0, 1, 0, 0, 1]     // Carlos
// ]

// Usamos apenas os dados numéricos, como a rede neural só entende números.
// tensorPessoasNormalizado corresponde ao dataset de entrada do modelo.
const tensorPessoasNormalizado = [
    [0.33, 1, 0, 0, 1, 0, 0], // Erick
    [0, 0, 1, 0, 0, 1, 0],    // Ana
    [1, 0, 0, 1, 0, 0, 1]     // Carlos
]

// Labels das categorias a serem previstas (one-hot encoded)
// [premium, medium, basic]
const labelsNomes = ["premium", "medium", "basic"]; // Ordem dos labels
const tensorLabels = [
    [1, 0, 0], // premium - Erick
    [0, 1, 0], // medium - Ana
    [0, 0, 1]  // basic - Carlos
];

// Criamos tensores de entrada (xs) e saída (ys) para treinar o modelo
const inputXs = tf.tensor2d(tensorPessoasNormalizado)
const outputYs = tf.tensor2d(tensorLabels)

// inputXs.print();
// outputYs.print();

// Quanto mais dados melhor!
// Assim o algoritmo entende melhor os padrões e consegue generalizar melhor para novos casos
const model = await trainModel(inputXs, outputYs);

const pessoa = { nome: 'Zé', cor: 'verde', idade: 28, localizacao: 'Curitiba' };

// Normalizando a idade da nova pessoa usando o mesmo padrão do treino
// Exemplo: idade_min = 25, idade_max = 40, então (28 - 25) / (40 - 25) = 0.2
const pessoaTensorNormalizado = [
    [
        0.2, // idade normalizada
        1,   // azul
        0,   // vermelho
        0,   // verde
        1,   // São Paulo
        0,   // Rio
        0    // Curitiba

    ]
];

const predictions = await predict(model, pessoaTensorNormalizado);
const results = predictions
    .sort((a, b) => b.prob - a.prob)
    .map(p => `${labelsNomes[p.index]}: ${(p.prob * 100).toFixed(2)}%`)
    .join('\n');

console.log(results);