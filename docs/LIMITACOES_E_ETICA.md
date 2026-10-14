# Limitações e uso responsável

Este é um protótipo acadêmico de visão computacional, não um sistema de segurança
homologado.

Principais limitações:
- apenas 80 imagens;
- somente duas classes;
- desempenho sensível a iluminação, oclusão, distância, ângulo e domínio das imagens;
- possibilidade de falso positivo e falso negativo;
- uma pessoa detectada não implica ameaça;
- uma classificação ou detecção não deve ser usada isoladamente para decisões de acesso.

O dataset deve respeitar licenças e privacidade. O Open Images informa que suas
anotações são CC BY 4.0 e as imagens são listadas como CC BY 2.0, recomendando verificar
a licença de cada imagem antes de redistribuição. Por isso o repositório não precisa
redistribuir as imagens: ele preserva o manifesto e o procedimento de aquisição.

Referência: https://storage.googleapis.com/openimages/web/factsfigures_v7.html
