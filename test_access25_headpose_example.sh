#!/bin/bash
echo 'Using Docker to start the container and run tests ...'
sudo docker build --force-rm --ssh default=$HOME/.ssh/id_rsa -t access25_headpose_image .
sudo docker run --name access25_headpose_container --rm --gpus all -it -d access25_headpose_image bash
sudo docker exec -w /home/username/access25_headpose access25_headpose_container python test/access25_headpose_test.py --input-data test/example.tif --database aflw --gpu 0 --backbone resnet --save-image
echo 'Transferring data from docker container to your local machine ...'
mkdir -p output
sudo docker cp access25_headpose_container:/home/username/conda/envs/access25/lib/python3.10/site-packages/images_framework/output/images/. output/
sudo chown -R "${USER}":"${USER}" output
sudo docker rm -f access25_headpose_container
sudo docker image rm access25_headpose_image
sudo docker builder prune -a -f