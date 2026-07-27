# Deploy em nuvem (bônus) — AWS App Runner

Serve a API de recomendação (`recsys.serving.api:app`) num container acessível
por URL pública. Requer credenciais AWS (modo híbrido — fornecidas pelo usuário).

## Pré-requisitos
- AWS CLI autenticado, Docker e Terraform instalados.
- Artefatos do modelo gerados (`poetry run dvc repro`): a imagem **embute**
  `models/mlp.pt` + `data/processed/dims.json` (serving self-contained, sem
  depender de servidor MLflow na nuvem).

## Passos

```bash
# 0. Garanta os artefatos do modelo (embutidos na imagem)
poetry run dvc repro          # gera models/mlp.pt e data/processed/dims.json

# 1. Build da imagem de serving (usa Dockerfile.serve.dockerignore)
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

## O que é necessário de você (modo híbrido)
- Região AWS, `account_id` e credenciais (via `aws configure` / variáveis de
  ambiente — **nunca** coladas em texto/commitadas).
- O modelo vai **embutido** na imagem, então não é preciso um backend MLflow na
  nuvem. `var.mlflow_tracking_uri` pode ser um valor placeholder.
- Segredos só via ambiente / AWS Secrets Manager.
