# EC2 launch commands (Console-to-Code export, 2026-04-15)
# Region: ap-northeast-1 (Tokyo)
# AMI: Debian 13 (ami-0072ec893f617e897)
# Instance: t3.small, 20GB gp3
# Key pair: cta-forge
# VPC: vpc-0f6f5556d67817d3f

aws ec2 create-security-group --group-name 'launch-wizard-1' --description 'launch-wizard-1 created 2026-04-15T15:40:40.206Z' --vpc-id 'vpc-0f6f5556d67817d3f'
aws ec2 authorize-security-group-ingress --group-id 'sg-preview-1' --ip-permissions '{"IpProtocol":"tcp","FromPort":22,"ToPort":22,"IpRanges":[{"CidrIp":"119.74.29.247/32"}]}'
aws ec2 run-instances --image-id 'ami-0072ec893f617e897' --instance-type 't3.small' --key-name 'cta-forge' --block-device-mappings '{"DeviceName":"/dev/xvda","Ebs":{"Encrypted":false,"DeleteOnTermination":true,"Iops":3000,"SnapshotId":"snap-0609f14dfe015d7c8","VolumeSize":20,"VolumeType":"gp3","Throughput":125}}' --network-interfaces '{"AssociatePublicIpAddress":true,"DeviceIndex":0,"Groups":["sg-preview-1"]}' --credit-specification '{"CpuCredits":"unlimited"}' --metadata-options '{"HttpEndpoint":"enabled","HttpPutResponseHopLimit":2,"HttpTokens":"required"}' --private-dns-name-options '{"HostnameType":"ip-name","EnableResourceNameDnsARecord":true,"EnableResourceNameDnsAAAARecord":false}' --count '1'
