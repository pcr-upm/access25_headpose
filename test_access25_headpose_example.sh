#!/bin/bash
echo 'Using Docker to start the container and run tests ...'
sudo docker build --force-rm --build-arg SSH_PRIVATE_KEY="$(cat ~/.ssh/id_rsa)" -t access25_headpose_image .
sudo docker volume create --name access25_headpose_volume
sudo docker run --name access25_headpose_container -v access25_headpose_volume:/home/username/access25_headpose --rm --gpus all -it -d access25_headpose_image bash
sudo docker exec -w /home/username/access25_headpose access25_headpose_container python test/access25_headpose_test.py --input-data test/example.tif --database aflw --gpu 0 --backbone resnet --save-image
sudo docker stop access25_headpose_container
echo 'Transferring data from docker container to your local machine ...'
mkdir -p output
sudo chown -R "${USER}":"${USER}" /var/lib/docker/
rsync --delete -azvv /var/lib/docker/volumes/access25_headpose_volume/_data/access25_headpose/output/images/ output
sudo docker system prune --all --force --volumes
sudo docker volume rm $(sudo docker volume ls -qf dangling=true)
