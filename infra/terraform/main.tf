terraform {
  required_version = ">= 1.14"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# ── Variables ────────────────────────────────────────────────────

variable "aws_region" {
  default = "ap-northeast-1"
}

variable "instance_type" {
  default = "t3.small"
}

variable "key_name" {
  default = "cta-forge"
}

variable "volume_size" {
  default = 20
}

# ── Data sources ─────────────────────────────────────────────────

# Latest Debian 13 AMI (official, HVM, amd64)
data "aws_ami" "debian" {
  most_recent = true
  owners      = ["136693071363"] # Debian official

  filter {
    name   = "name"
    values = ["debian-13-amd64-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

data "aws_vpc" "default" {
  default = true
}

# ── Security Group ───────────────────────────────────────────────

resource "aws_security_group" "cta_forge" {
  name        = "cta-forge-sg"
  description = "SSH access for cta-forge EC2"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description = "SSH"
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

  tags = {
    Name = "cta-forge-sg"
  }
}

# ── EC2 Instance ─────────────────────────────────────────────────

resource "aws_instance" "cta_forge" {
  ami                    = data.aws_ami.debian.id
  instance_type          = var.instance_type
  key_name               = var.key_name
  vpc_security_group_ids = [aws_security_group.cta_forge.id]

  root_block_device {
    volume_size = var.volume_size
    volume_type = "gp3"
    iops        = 3000
    throughput  = 125
  }

  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"
    http_put_response_hop_limit = 2
  }

  credit_specification {
    cpu_credits = "unlimited"
  }

  tags = {
    Name = "cta-forge"
  }
}

# ── Elastic IP ───────────────────────────────────────────────────

resource "aws_eip" "cta_forge" {
  instance = aws_instance.cta_forge.id
  domain   = "vpc"

  tags = {
    Name = "cta-forge-eip"
  }
}

# ── Outputs ──────────────────────────────────────────────────────

output "public_ip" {
  value = aws_eip.cta_forge.public_ip
}

output "instance_id" {
  value = aws_instance.cta_forge.id
}

output "ami_id" {
  value = data.aws_ami.debian.id
}
