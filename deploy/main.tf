# IaC (dry-run) do deploy em nuvem — AWS App Runner servindo a imagem da API.
# Requer credenciais AWS. Fluxo: build & push da imagem para o ECR, depois
# `terraform init && terraform apply`. Container fica acessível por URL pública.

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

variable "mlflow_tracking_uri" {
  type        = string
  description = "Backend MLflow acessível pelo container (registry do modelo)."
}

provider "aws" {
  region = var.aws_region
}

resource "aws_ecr_repository" "recsys_api" {
  name                 = "recsys-api"
  image_tag_mutability = "MUTABLE"
  force_delete         = true
}

# Role que autoriza o App Runner a puxar a imagem do ECR privado.
resource "aws_iam_role" "apprunner_ecr" {
  name = "recsys-apprunner-ecr"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "build.apprunner.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "apprunner_ecr" {
  role       = aws_iam_role.apprunner_ecr.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess"
}

resource "aws_apprunner_service" "recsys_api" {
  service_name = "recsys-api"

  source_configuration {
    authentication_configuration {
      access_role_arn = aws_iam_role.apprunner_ecr.arn
    }
    image_repository {
      image_identifier      = var.image_uri
      image_repository_type = "ECR"
      image_configuration {
        port = "8080"
        runtime_environment_variables = {
          MLFLOW_TRACKING_URI = var.mlflow_tracking_uri
        }
      }
    }
    auto_deployments_enabled = false
  }

  instance_configuration {
    cpu    = "1024"
    memory = "2048"
  }
}

output "service_url" {
  description = "URL pública do serviço."
  value       = "https://${aws_apprunner_service.recsys_api.service_url}"
}
