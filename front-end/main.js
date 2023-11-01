const fs = require('fs');

const nomeDoArquivo = 'trainedNetwork.json';

fs.readFile(nomeDoArquivo, 'utf8', (err, data) => {
  if (err) {
    console.error(`Erro ao ler o arquivo ${nomeDoArquivo}: ${err}`);
    return;
  }

  const objetoJSON = JSON.parse(data);

  // Acesse a matriz "weights" no objeto JSON
  const weights = objetoJSON.weights;

  // Agora você pode realizar cálculos com a matriz "weights"
  // Exemplo: Calcule a soma de todos os valores na matriz
  let soma = 0;
  for (let i = 0; i < weights.length; i++) {
    for (let j = 0; j < weights[i].length; j++) {
      soma += weights[i][j];
    }
  }

  console.log('Soma de todos os valores em "weights":', soma);
});
