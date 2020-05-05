#!/bin/bash

directory_flag="$(pwd)"
network_name='carla_lab_network'
cu_version='101'
margs=1

function example () {
    echo -e "example: $0 -d /your/directory/ -n network_name -c 100"
}


function help () {
  echo -e "  -d -  absolute path to working directory (make sure it's writable)"
  echo -e "  -n -  name for the network (default carla_lab_network)"
  echo -e "  -c -  cuda version, default is 101, alternative is 100"
  echo -e "  -r -  rebuild docker images for carla and lab"
}


function margs_check () {
	if [ $# -lt $margs ]; then
	        example
	    exit 1 # error
	fi
}

function rebuild () {
  docker kill carla lab
  docker rm carla lab
  docker network rm carla_lab_network
  echo "Images killed"

  docker image rm -f carla
  echo "carla removed"
  docker image rm -f lab
  echo "lab removed"
}

while getopts 'hd:n:c:r' flag; do
  case "$flag" in
    h) help;exit;;
    d) directory_flag="$OPTARG" ;;
    n) network_name="$OPTARG" ;;
    c) cu_version="$OPTARG" ;;
    r) rebuild ;;
  esac
done

# margs_check $directory_flag

############ MAIN

echo "Working directory for lab: $directory_flag"
echo "Network name: $network_name"


if [ -e LinuxNoEditor ]
then
  echo "Skipping download as Carla Binaries are already downloaded"
else
  echo "Downloading carla binaries"
  wget -ct 5 -O carla.tar.xz  "https://onedrive.live.com/download?cid=83F0E720C8D5226F&resid=83F0E720C8D5226F%21105&authkey=ADrHG2yXG6tWu7U"
  echo "Unpacking carla binaries"
  tar -xf carla.tar.xz
  cp docker/Dockerfile_carla LinuxNoEditor/Dockerfile
  cp docker/Dockerfile_cu$cu_version LinuxNoEditor/lab/Dockerfile
  rm carla.tar.xz
fi

cd LinuxNoEditor

docker build -t carla .
echo 'built carla image'

docker network create --driver=bridge $network_name
echo "network: $network_name"

docker run --gpus all --name="carla" --network=$network_name -d carla:latest
echo 'carla container is up'

carla_ip=$(docker inspect --format='{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' carla)
echo "{\"carla_ip\": \"$carla_ip\"}" > "$directory_flag/config.json"
chmod 777 $directory_flag/config.json
echo 'carla container ip acquired and saved'
echo "{\"carla_ip\": \"$carla_ip\"}"

cd lab/
#docker build -t lab --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) . # <- in case of using user setup
docker build -t lab .
echo 'lab build'

if [ $cu_version == '101' ]
then
  docker run --gpus all --name="lab" --network=$network_name -p 8888:8888 -p 6006:6006 -v $directory_flag:/home/user/workspace/ -d lab
else
    docker run --gpus all --name="lab" --network=$network_name -p 8888:8888 -p 6006:6006 -v $directory_flag:/workspace/ -d lab
fi

#Run as a root
[ "$EUID" != 0 ] || exec sudo bash "$0" "$@"
cd ../../.. && sudo chmod -R 777 carla_racetrack_BA
cd carla_racetrack_BA && rm -rf LinuxNoEditor
echo "Alles gute mein Freund!"
