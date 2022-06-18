<p align="center">
  <img height="60" src="https://carla.org//img/logo/carla-black-m.png" />
  <img height="60" src="https://devblogs.nvidia.com/wp-content/uploads/2017/04/pytorch-logo-dark.png" />
  <img height="60" src="https://avatars1.githubusercontent.com/u/22800682?v=4" />
</p>

# Carla Racetrack BS

> Contenerized developement enviroment created for purpose of BS Thesis.

**Architecture**

![image](https://user-images.githubusercontent.com/31616749/174434601-66903153-4ecc-4a63-a3b7-b80844565e15.png)


**Demo**

![Carla simulator](https://s6.gifyu.com/images/carla_gif_down2.gif)

![Carla client](https://s6.gifyu.com/images/carla_gif_up.gif)


## Installation
Requires Docker (>=19.03), NVIDIA driver and nvidia-docker (see how to install it [here](https://github.com/NVIDIA/nvidia-docker)). Tested on Ubuntu 18.04 and Debian 10.

* default password is: 'ridin dirty', but you can change it in dockerfiles by insertting your own sha password.

### Setup

> For setting up the environment you'll need docker.
> All you need to do to make things work after cloning the repo is to run setup file with

```shell
$ bash setup.sh
```

> Avialable options during the installation
```shell
-d /path/to/working/dir - absolute path to jupyter lab working directory (make sure it's writable)
-n network_name -  name for the network (default carla_lab_network) while setting carla and environment containers
-c 10* - cuda version, default is 101, alternative is 100
-r - rebuild docker images for carla and lab 
```

---
## How to setup workspace
In order to use scripts from source you'll need to generate spawnpoints in the first place.
In order to do so use notebook notebooks/14042020_setting_points.ipynb and save the spawn points for each track in data/spawn_points.
Voila, you can use experiments:
* src/runner.py - utilizes MPC for the purpose of data generation
* src/offline_training.py - for the purpose of offline Actor and Critic networks offline training
* src/runner_NN.py - for the DDPG online training
 

## License

[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)

- **[MIT license](http://opensource.org/licenses/mit-license.php)**
