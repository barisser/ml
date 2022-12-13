# Configure the AWS provider
provider "aws" {
  region = "us-east-2"
}

resource "aws_sagemaker_code_repository" "example" {
  code_repository_name = "my-notebook-instance-code-repo"

  git_config {
    repository_url = "https://github.com/barisser/ml.git"
  }
}

resource "aws_sagemaker_notebook_instance" "ni" {
  name                    = "my-notebook-instance"
  role_arn                = aws_iam_role.sagemaker.arn
  instance_type           = "ml.g4dn.xlarge"
  default_code_repository = aws_sagemaker_code_repository.example.code_repository_name

}

# Create the IAM role for Amazon SageMaker
resource "aws_iam_role" "sagemaker" {
  name               = "SageMakerRole"
  assume_role_policy = <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Action": "sts:AssumeRole",
      "Principal": {
        "Service": "sagemaker.amazonaws.com"
      },
      "Effect": "Allow",
      "Sid": ""
    }
  ]
}
EOF
}
