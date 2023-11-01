fetch('data.json') // Substitua pela URL correta do seu arquivo JSON
.then(response => {
  if (!response.ok) {
    throw new Error('Não foi possível ler o arquivo JSON.');
  }
  return response.json();
})
.then(data => {
  // Armazene os dados em uma variável JavaScript
  const jsonData = data;

  // Exiba os dados na página
  const resultado = document.getElementById('resultado');
  resultado.textContent = JSON.stringify(jsonData, null, 2);

  // Calcule a soma dos elementos em "weights"
  const weights = jsonData.weights;
  let soma = 0;
  for (let i = 0; i < weights.length; i++) {
    for (let j = 0; j < weights[i].length; j++) {
      soma += weights[i][j];
    }
  }

  // Exiba a soma no HTML
  const somaElement = document.getElementById('soma');
  somaElement.textContent = soma;
})
.catch(error => {
  console.error('Erro: ' + error.message);
});