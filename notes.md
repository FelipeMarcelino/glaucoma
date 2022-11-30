* Usar collate\_fn e em seguida enviar para o device e aplicar transforms (Batch)
    - Verificar se tem como fazer toPil já em tensor
* Usar Kornia e fazer as transforms em GPU (Provável uso maior de GPU nesse caso) no entanto pode
trazer um benefício de ser rápido e diminuir o bottleneck do CPU

