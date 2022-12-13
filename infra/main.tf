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
  map_public_ip_on_launch = true
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


  # jupyter uses 8888 by default
  ingress {
    from_port   = 8888
    to_port     = 8888
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # tensorboard uses 6006
  ingress {
    from_port   = 6006
    to_port     = 6006
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

# allow our VPC to talk to the public internet
resource "aws_internet_gateway" "tf" {
  vpc_id = "${aws_vpc.my_vpc.id}"
}


# this is for the default route table that was created with our VPC
resource "aws_default_route_table" "rt" {
  default_route_table_id = "${aws_vpc.my_vpc.default_route_table_id}"

  # make sure all outbound traffic goes through the internet gateway
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = "${aws_internet_gateway.tf.id}"
  }

}


# attach route table to the sbnet
resource "aws_route_table_association" "rt_assoc" {
  subnet_id      = "${aws_subnet.my_subnet.id}"
  route_table_id = "${aws_vpc.my_vpc.default_route_table_id}"
}



output "instance_public_ip" {
  value = "${aws_instance.my_instance.public_ip}"
}

output "instance_elastic_ip" {
  value = "${aws_eip.my_eip.public_ip}"
}
/*
resource "aws_ebs_volume" "ebs" {
  availability_zone = "${aws_instance.my_instance.availability_zone}"
  size              = 20
  type              = "gp2"

}


resource "aws_ebs_volume_attachment" "example" {
  device_name = "/dev/sda1"
  volume_id   = aws_ebs_volume.ebs.id
  instance_id = aws_instance.my_instance.id
}
*/

