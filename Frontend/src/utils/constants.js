// Header buttons name
export const HOME = 'Início'
export const TOOL = 'Ferramenta'
export const ABOUT = 'Saiba mais'

// Home page
export const TITLE = 'SeaStatefy'
export const DESCRIPTION = 'Propõe-se uma ferramenta de classificação de imagens capaz de estimar o estado do mar segundo a Escala Beaufort utilizando técnicas de visão computacional.'
export const START_BUTTON_NAME = 'Começar'
export const TUTORIAL_TITLE = 'COMO USAR'
export const TUTORIAL_STEP_1 = 'Clique em “Começar” ou acesse a ferramenta na aba “Ferramenta”.'
export const TUTORIAL_STEP_2 = 'Arraste uma imagem para a área indicada ou carregue uma imagem armazenada em seu dispositivo local.'
export const TUTORIAL_STEP_3 = 'Clique no botão ”Classificar”.'
export const TUTORIAL_STEP_4 = 'Observe o intervalo da Escala Beaufort predito pelo classificador e suas informações correspondentes associadas à imagem carregada.'

// Tool page - upload image
export const UPLOAD_IMAGE_TITLE = 'Carregue uma imagem'
export const SUPPORTED_FORMATS = 'Formatos suportados: PNG, JPEG e TIFF'
export const DRAG_AND_DROP_TEXT = 'Arraste e solte uma imagem aqui ou'
export const UPLOAD_IMAGE_BUTTON = 'Selecione uma imagem'
export const MAX_FILE_NAME_LENGTH = 35
export const FILE_UPLOAD_ERROR_TITLE = 'Erro ao importar'
export const FILE_UPLOAD_ERROR_MESSAGE = 'O formato do arquivo não é suportado. Por favor, certifique-se de que a imagem possui extensão .png, .jpeg ou .tif e tente novamente.'
export const API_RESPONSE_ERROR_TITLE = 'Atenção!'
export const API_RESPONSE_ERROR_MESSAGE = 'Houve um erro em sua requisição, tente novamente mais tarde.'

// Tool page - results
export const NEW_IMAGE_BUTTON = 'Carregue uma nova imagem'
export const PREDICTED_BS_INTERVAL_TITLE = 'Intervalo estimado da Escala Beaufort'
export const CLASS_LABEL = 'Classe'
export const DESCRIPTION_LABEL = 'Descrição'
export const WIND_SPEED_LABEL = 'Velocidade do vento'
export const WAVE_HEIGHT_LABEL = 'Altura média das ondas'
export const SEA_APPEARANCE_LABEL = 'Aspecto do mar'
export const PREDICTED_PROBABILITIES_TITLE = 'Probabilidades'
export const SIMILAR_IMAGES_TITLE = 'Imagens similares'
export const SIMILAR_IMAGES_INFO = 'Exemplos de imagens catalogadas pelo modelo como pertencentes a mesma classe da imagem carregada.'

// About page
export const PROJECT_NAME = 'SeaStatefy'
export const PROJECT_SUMMARY = 'Uma ferramenta para estimação do estado do mar a partir do processamento de imagens e utilizando visão computacional. Apresentam-se aqui os motivadores, os objetivos e o processo de desenvolvimento que culminaram na criação da plataforma SeaStatefy.'
export const MOTIVATION_SECTION_NAME = 'MOTIVAÇÃO'
export const MOTIVATION_TEXT = 'O estado do mar (sea state) é uma medida da agitação da superfície do mar. Suas variáveis, em especial a altura das ondas, têm impactado, por exemplo, as estruturas offshore de petróleo e gás [1] e até mesmo a costa, sendo seu conhecimento de interesse da guarda costeira para a prevenção de catástrofes. Atualmente, os métodos para classificar os estados do mar são baseados na representação estatística de parâmetros de onda ou por modelagem numérica. Esses métodos são caros, propensos a mau funcionamento do equipamento e exigem alto poder de computação e tempo.'
export const MOTIVATION_TEXT_REFS = '[1] VANNAK, D.; LIEW, M.S.; YEW, G.Z. Time Domain and Frequency Domain Analyses of Measured Metocean Data for Malaysian Waters.; Int. J. Geol. Environ. Eng. 2013, 7, 549–554.'
export const OBJECTIVE_SECTION_NAME = 'OBJETIVOS'
export const OBJECTIVE_TEXT = 'Uma vez que modelos de aprendizado de máquina têm sido empregados como uma solução alternativa para prever e classificar as condições das ondas, o objetivo estabelecido para o projeto foi a criação de uma ferramenta de classificação do estado do mar utilizando Deep Learning. Denominada SeaStatefy, a plataforma deveria ser capaz de associar uma imagem da superfície do mar ao intervalo da Escala Beaufort mais semelhante:'
export const OBJECTIVE_ITEMS = 'Graus 0 a 3 - Mar calmo/leve;\n Graus 4 a 7 - Mar moderado/agitado;\nGraus 8 a 12 - Mar forte/extremo.'
export const METODOLOGY_SECTION_NAME = 'METODOLOGIA'
export const METODOLOGY_TEXT = 'O processo de desenvolvimento consistiu em sete etapas principais. Primeiramente, buscou-se datasets de imagens da superfície do mar rotuladas segundo o grau da escala Beaufort correspondente ou anotadas com informações referentes a altura média das ondas para aquele conjunto de dados. Em seguida, estudou-se algumas arquiteturas de CNNs, partindo-se de modelos utilizados em artigos relacionados a este problema ou a problemas similares, e selecionou-se, então, os candidatos para realizar a extração de features e a classificação das imagens.\nDe posse dos modelos, partiu-se para o pré-processamento das imagens selecionadas, manipulando suas dimensões e aplicando técnicas de data augumentation para obter maior diversidade de dados. Treinou-se os modelos selecionados utilizando técnicas para datasets desbalanceados e comparou-se a performance de cada um com base em testes de desempenho e métricas de avaliação, como a acurácia.\nPor fim, implementou-se e integrou-se a plataforma com o modelo classificador, desenvolvendo uma interface gráfica para a ferramenta de classificação a partir do mapeamento do seu usuário principal.'
export const RESULTS_SECTION_NAME = 'RESULTADOS'
export const RESULTS_TEXT = 'O modelo com o melhor desempenho foi a ResNet-34 pré-treinada, na qual adotou-se o oversampling como técnica de dataset desbalanceado, isto é, igualou-se a quantidade de dados em cada classe mediante duplicação das imagens das classes minoritárias de forma aleatória. Ele obteve o melhor equilíbrio de acertos nos cenários de teste propostos, uma média superior a 90%, conseguindo classificar relativamente bem imagens de um mar forte/extremo (graus 8 - 12 da Escala Beaufort), apesar da baixa quantidade de representantes. Sua acurácia em cada dataset é registrada abaixo.\nObservou-se, entretanto, limitações na capacidade de generalização do modelo para novas imagens que sejam muito diferentes daquelas utilizadas no treinamento. Tal circunstância decorre da baixa diversidade dos conjuntos de imagens utilizados para o treinamento do modelo, tanto para imagens pertencentes a uma mesma classe quanto em representantes de classes diferentes, havendo majoritariamente amostras dos graus 0 a 3 da Escala Beaufort.\nEm relação à plataforma SeaStatefy, foi possível desenvolver uma interface simples, fácil e rápida de ser utilizada, capaz de associar uma imagem ao intervalo da Escala Beaufort mais provável em poucos segundos.'
export const TCC_INFO = 'Esta ferramenta foi desenvolvida pelos alunos Carolina Mari Miyashiro e Pedro Gabriel Mascarenhas Maronezi como projeto de TCC do curso de Engenharia Elétrica com ênfase em Computação da Escola Politécnica da Universidade de São Paulo.\nLinks relevantes:'