const jsonFileName = "trainedNetwork.json";

        // Função para carregar e exibir o conteúdo JSON
        function loadAndDisplayJSON(fileName) {
            const jsonDataDisplay = document.getElementById("jsonDataDisplay");

            // Requisição para carregar o arquivo JSON
            const xhr = new XMLHttpRequest();
            xhr.open("GET", fileName, true);
            xhr.onreadystatechange = function () {
                if (xhr.readyState === 4 && xhr.status === 200) {
                    const jsonData = JSON.parse(xhr.responseText);
                    jsonDataDisplay.innerText = JSON.stringify(jsonData, null, 2);
                }
            };
            xhr.send();
        }

        // Carregar e exibir o JSON
        loadAndDisplayJSON(jsonFileName);