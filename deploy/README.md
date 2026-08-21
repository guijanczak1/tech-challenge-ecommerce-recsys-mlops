# Deploy em nuvem (bônus) — AWS Lambda (Function URL)

Serve a API de recomendação (`recsys.serving.api:app`) como uma função Lambda
com imagem de container, acessível por **URL pública** via Function URL.
Requer credenciais AWS (modo híbrido — fornecidas pelo usuário).

## Por que Lambda (e não App Runner/EC2)?

- **App Runner não tem free tier** — cobra desde o primeiro minuto.
- **EC2 free tier** só vale nos primeiros 12 meses da conta AWS.
- **Lambda tem free tier perpétuo**: 1.000.000 de requisições/mês + 400.000
  GB-segundos de execução, **para sempre**, independente da idade da conta.
  Com 1536 MB de memória, isso dá ~74h de execução gratuita por mês — muito
  acima do necessário para uma demo.
- Trade-off aceito: **cold start** (alguns segundos na primeira chamada após
  ociosidade, enquanto importa `torch` e carrega o modelo).

## Pré-requisitos
- AWS CLI autenticado, Docker e Terraform instalados.
- Artefatos do modelo gerados (`poetry run dvc repro`): a imagem **embute**
  `models/mlp.pt` + `data/processed/dims.json` (serving self-contained, sem
  depender de servidor MLflow na nuvem).

## Passos

```bash
# 0. Garanta os artefatos do modelo (embutidos na imagem)
poetry run dvc repro          # gera models/mlp.pt e data/processed/dims.json

# 1. Build da imagem (usa Dockerfile.serve.dockerignore); precisa do
#    Docker com suporte a --platform (a imagem deve ser linux/amd64 ou
#    linux/arm64 — escolha uma arquitetura e mantenha consistente com o passo 4)
docker build --platform linux/amd64 -f Dockerfile.serve -t recsys-api:latest .

# 2. Criar o repositório ECR (via Terraform) e obter a URI
cd deploy
terraform init
terraform apply -target=aws_ecr_repository.recsys_api \
  -var="aws_region=<REGIAO>" -var="image_uri=placeholder"

# 3. Login, tag e push da imagem para o ECR
aws ecr get-login-password --region <REGIAO> | docker login --username AWS --password-stdin <ACCOUNT>.dkr.ecr.<REGIAO>.amazonaws.com
docker tag recsys-api:latest <ACCOUNT>.dkr.ecr.<REGIAO>.amazonaws.com/recsys-api:latest
docker push <ACCOUNT>.dkr.ecr.<REGIAO>.amazonaws.com/recsys-api:latest

# 4. Aplicar a função Lambda + Function URL apontando para a imagem
terraform apply \
  -var="aws_region=<REGIAO>" \
  -var="image_uri=<ACCOUNT>.dkr.ecr.<REGIAO>.amazonaws.com/recsys-api:latest"

# 5. A URL pública sai em `terraform output function_url`
curl "$(terraform output -raw function_url)health"
curl "$(terraform output -raw function_url)recommend?user=1&k=5"
```

## O que é necessário de você (modo híbrido)
- Região AWS, `account_id` e credenciais (via `aws configure` / variáveis de
  ambiente — **nunca** coladas em texto/commitadas).
- O modelo vai **embutido** na imagem, então não é preciso um backend MLflow na
  nuvem nem RDS/S3 extra.
- Segredos só via ambiente / AWS Secrets Manager.

## Limpeza (evitar custo residual)
```bash
terraform destroy -var="aws_region=<REGIAO>" -var="image_uri=<IMAGE_URI>"
```
