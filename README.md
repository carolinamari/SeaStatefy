# SeaStatefy
Ferramenta para estimação do estado do mar a partir do processamento de imagens e utilizando visão computacional. Desenvolvida como projeto de TCC do curso de Engenharia Elétrica com ênfase em Computação da Escola Politécnica da Universidade de São Paulo.

## Arquivos
Os principais arquivos relacionados ao projeto estão divididos em 4 pastas principais:

* **Backend:** contém os arquivos referentes ao back-end da ferramenta SeaStatefy, hospedado em [https://sea-state-classifier-api.herokuapp.com](https://sea-state-classifier-api.herokuapp.com) ([Swagger](https://sea-state-classifier-api.herokuapp.com/docs)).
* **Frontend:** contém os arquivos relacionados ao front-end da ferramenta SeaStatefy, hospedado em [https://sea-state-classifier.herokuapp.com](https://sea-state-classifier.herokuapp.com/).
* **Notebooks:** contém os Jupyter notebooks desenvolvidos ao longo do projeto. O propósito de cada um é descrito a seguir.
    * `batch_script.ipynb`: separação das imagens em batches;
    * `beaufort_images_test.ipynb`: teste dos modelos usando as 13 imagens representantes da escala beaufort;
    * `resize_script_drive.ipynb`: resize e crop das imagens e geração dos arquivos `.pkl`;
    * `split_full_script.ipynb`: separação das imagens em conjuntos de treino e teste;
    * `test_script.ipynb`: teste dos modelos usando as imagens dos datasets MU-SSiD e Stereo;
    * `training_script.ipynb`: treinamento dos modelos no Google Colab.
* **Scripts:** contém os scripts desenvolvidos ao longo do projeto. 
    * `Dockerfile`: Dockerfile para geração do container utilizado no treinamento dos modelos;
    * `dockercomands.txt`: comandos básicos do Docker;
    * `plot_metrics.py`: plot dos grafico das métricas;
    * `training_script.py`: treinamento dos modelos.

## Modelos de CNN
Todos os modelos de CNN treinados e testados podem ser encontrados [aqui](https://drive.google.com/drive/folders/1kae5sJRFqhnBFeruGfWU_QBhJ-MUgER7?usp=sharing).

## Demais arquivos
Os demais arquivos - as imagens dos datasets empregados, outros scripts e notebooks criados mas não utilizados no final, etc - estão arquivados no [drive do projeto](https://drive.google.com/drive/folders/1L5Q8jnDvpb7OohzvCsOn6G91G52sFcm1?usp=sharing).
