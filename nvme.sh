#!/usr/bin/env bash
sudo apt-get update && sudo apt-get install mdadm --no-install-recommends
sudo mdadm --create /dev/md0 --level=0 --raid-devices=4 /dev/nvme0n1 /dev/nvme0n2 /dev/nvme0n3 /dev/nvme0n4
sudo mkfs.ext4 -F /dev/md0
sudo mkdir -p /mnt/disks/nvme
sudo mount /dev/md0 /mnt/disks/nvme
sudo chmod a+w /mnt/disks/nvme
echo UUID=`sudo blkid -s UUID -o value /dev/md0` /mnt/disks/nvme ext4 discard,defaults,nofail 0 2 | sudo tee -a /etc/fstab
cat /etc/fstab

gcloud init
cd /mnt/disks/nvme/
gsutil cp -r gs://despi-0-1/DeSPI/* .

ln -s /mnt/disks/nvme/dump /home/minjoon/dump
ln -s /mnt/disks/nvme/data /home/minjoon/data
ln -s /mnt/disks/nvme/metadata /home/minjoon/metadata
ln -s /mnt/disks/nvme/models /home/minjoon/models