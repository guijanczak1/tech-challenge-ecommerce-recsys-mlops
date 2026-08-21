# IaC do deploy em nuvem — AWS Lambda (imagem de container) + Function URL.
# Free tier perpétuo (1M requisições/mês + 400.000 GB-s de execução, sem
# expirar em 12 meses). Requer credenciais AWS. Fluxo: build & push da imagem
# para o ECR, depois `terraform init && terraform apply`. A Function URL fica
# acessível publicamente por HTTPS.

terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

variable "aws_region" {
  type        = string
  description = "Região AWS do deploy."
}

variable "image_uri" {
  type        = string
  description = "URI da imagem no ECR (ex.: <acct>.dkr.ecr.<region>.amazonaws.com/recsys-api:latest)."
}

provider "aws" {
  region = var.aws_region
}

resource "aws_ecr_repository" "recsys_api" {
  name                 = "recsys-api"
  image_tag_mutability = "MUTABLE"
  force_delete         = true
}

# Permite o serviço Lambda puxar a imagem deste repositório ECR privado.
resource "aws_ecr_repository_policy" "lambda_pull" {
  repository = aws_ecr_repository.recsys_api.name
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid       = "AllowLambdaPull"
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
      Action = [
        "ecr:BatchGetImage",
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchCheckLayerAvailability",
      ]
    }]
  })
}

# Role de execução da função (permissões mínimas: logs no CloudWatch).
resource "aws_iam_role" "lambda_exec" {
  name = "recsys-lambda-exec"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_logs" {
  role       = aws_iam_role.lambda_exec.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_lambda_function" "recsys_api" {
  function_name = "recsys-api"
  role          = aws_iam_role.lambda_exec.arn
  package_type  = "Image"
  image_uri     = var.image_uri
  timeout       = 30    # cobre o cold start (import do torch + carga do modelo)
  memory_size   = 1536  # folga para o torch; ainda bem dentro do free tier

  depends_on = [aws_ecr_repository_policy.lambda_pull]
}

resource "aws_lambda_function_url" "recsys_api" {
  function_name      = aws_lambda_function.recsys_api.function_name
  authorization_type = "NONE" # pública, sem IAM — atende "container acessível via URL pública"
}

# authorization_type = "NONE" exige esta permissão explícita para invoke público.
resource "aws_lambda_permission" "public_invoke" {
  statement_id           = "AllowPublicInvokeFunctionUrl"
  action                 = "lambda:InvokeFunctionUrl"
  function_name          = aws_lambda_function.recsys_api.function_name
  principal              = "*"
  function_url_auth_type = "NONE"
}

output "function_url" {
  description = "URL pública da API (Lambda Function URL)."
  value       = aws_lambda_function_url.recsys_api.function_url
}
