# Configure the AWS provider
provider "aws" {
  region = "us-east-2"
}

# Create a new VPC
resource "aws_vpc" "my_vpc" {
  cidr_block = "10.0.0.0/16"
}

# Create a new subnet in the VPC
resource "aws_subnet" "my_subnet" {
  vpc_id     = aws_vpc.my_vpc.id
  cidr_block = "10.0.0.0/24"
}

# Create an EC2 instance
resource "aws_instance" "my_instance" {
  ami           = "ami-0176478f493e6143b"
  instance_type = "p2.xlarge"

  vpc_security_group_ids = [aws_security_group.my_security_group.id]
  subnet_id              = aws_subnet.my_subnet.id

  key_name = "mlkeypair"

  tags = {
    Name = "my-ec2-instance"
  }
}

resource "aws_eip" "my_eip" {
  instance = "${aws_instance.my_instance.id}"
}

# Create a security group for the EC2 instance
resource "aws_security_group" "my_security_group" {
  name        = "my-ec2-security-group"
  description = "Security group for my EC2 instance"
  vpc_id      = aws_vpc.my_vpc.id

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}