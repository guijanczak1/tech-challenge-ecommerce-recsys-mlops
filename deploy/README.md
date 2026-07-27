# Deploy em nuvem (bônus) — AWS App Runner

Serve a API de recomendação (`recsys.serving.api:app`) num container acessível
por URL pública. Requer credenciais AWS (modo híbrido — fornecidas pelo usuário).

## Pré-requisitos
- AWS CLI autenticado, Docker e Terraform instalados.
- Um backend MLflow acessível pelo container (para carregar
  `models:/recsys-mlp@production`).

## Passos

```bash
# 1. Build da imagem de serving
docker build -f Dockerfile.serve -t recsys-api:latest .

# 2. Criar o repositório ECR (via Terraform) e obter a URI
cd deploy
terraform init
terraform apply -target=aws_ecr_repository.recsys_api \
  -var="aws_region=<REGIAO>" -var="image_uri=placeholder" -var="mlflow_tracking_uri=<URI>"

# 3. Login, tag e push da imagem para o ECR
aws ecr get-login-password --region <REGIAO> | docker login --username AWS --password-stdin <ACCOUNT>.dkr.ecr.<REGIAO>.amazonaws.com
docker tag recsys-api:latest <ACCOUNT>.dkr.ecr.<REGIAO>.amazonaws.com/recsys-api:latest
docker push <ACCOUNT>.dkr.ecr.<REGIAO>.amazonaws.com/recsys-api:latest

# 4. Aplicar o App Runner apontando para a imagem
terraform apply \
  -var="aws_region=<REGIAO>" \
  -var="image_uri=<ACCOUNT>.dkr.ecr.<REGIAO>.amazonaws.com/recsys-api:latest" \
  -var="mlflow_tracking_uri=<URI>"

# 5. A URL pública sai em `terraform output service_url`
```

## O que o harness precisa de você (modo híbrido)
- Região AWS, `account_id` e credenciais (via ambiente/`aws configure`).
- Definição do backend MLflow acessível pelo container (RDS+S3, ou imagem com
  o modelo embutido — a decidir).
- Segredos **nunca** são versionados; use variáveis de ambiente / AWS Secrets.
