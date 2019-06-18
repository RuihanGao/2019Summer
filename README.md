# 2019 Summer
## 5.23 Install Carla
1. [Getting Started with CARLA] (https://carla.readthedocs.io/en/stable/getting_started/) -> [build with Linux](https://carla.readthedocs.io/en/latest/how_to_build_on_linux/) <br/>
Remember to [build Unreal Engine](https://wiki.unrealengine.com/Building_On_Linux) <br/>
[install clang 3.9 on ubuntu 16](https://askubuntu.com/questions/787383/how-to-install-llvm-3-9) (the clang default on Ubuntu is 16.04 is 3.8)
```
Add the archive signature:

wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add -
Add the PPA:

sudo apt-add-repository "deb https://apt.llvm.org/xenial/ llvm-toolchain-xenial-3.9 main"
sudo apt-get update
Note: There may be some authentication warnings for the llvm key which I have safely overridden on my own system. To bypass authentication, replace sudo apt-get update with

sudo apt -o Acquire::AllowInsecureRepositories=true update
Allow the Repository to reload and then run the following command:

sudo apt-get install clang-3.9 lldb-3.9
Test your installation as follows, as shown on my own Xenial system:

$ clang-3.9 --version
clang version 3.9.0-svn275716-1~exp1 (trunk)
Target: x86_64-pc-linux-gnu
Thread model: posix
InstalledDir: /usr/bin
```

[install clang 6.0 on ubuntu16](https://blog.kowalczyk.info/article/k/how-to-install-latest-clang-6.0-on-ubuntu-16.04-xenial-wsl.html) <br/>
After installation make "alternative link" following [this](https://github.com/carla-simulator/carla/issues/53)
To debugh `update-alternatives warning: broken link group, need to force reinstallation`, try [this](https://dev1galaxy.org/viewtopic.php?id=649), type `yes '' | sudo update-alternatives --force --all` in terminal, press `Enter` and redo the alternative link <br/>
After installing [Clang](https://clang.llvm.org/), `./Setup.sh` can run properly].

*Misc*
* Learn C++, [notes for `reference`](https://en.wikipedia.org/wiki/Reference_(C%2B%2B))

## 5.30
1. Change to Clang v3.8 in order to run ` ./GenerateProjectFiles.sh ` as normal. Follow [Build UE on Lunux](https://wiki.unrealengine.com/Building_On_Linux), do `make` and test with `./Engine/Binaries/Linux/UE4Editor` successfully.

## 6.10
1. First meeting with Prof. Li <br/>
Papers to read: <br/>
  * https://arxiv.org/pdf/1812.03079.pdf
  * http://www.cds.caltech.edu/~murray/preprints/comb19-aiaa.pdf
  * https://arxiv.org/pdf/1710.02410.pdf
  * https://arxiv.org/pdf/1611.01779.pdf
  * https://arxiv.org/pdf/1905.10691.pdf
2. To set up CARLA, although "build on linux" still get error of "No rules to make 'launch'", can download from `releases` from [CARLA github](https://github.com/carla-simulator/carla/releases) \[Linux] CARLA_0.9.5.tar.gz <br/>
Then directly run the program, see more at `README`

*Notes*
* [One-hot encoded data](https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/)
  
## 6.11
1. J-1 orientation
2. seminars @ MIT
3. Read paper`Learning to Act by Predicting the Future`

## 6.12
1. Read paper `MPSC`

*Notes*
* dynamical system: In mathematics, a dynamical system is a system in which a function describes the time dependence of a point in a geometrical space.
* [tube-based MPC](https://web.stanford.edu/~pavone/papers/Singh.Pavone.Slotine.CDC16EV.pdf): In tube MPC, an ancillary feedback controller is designed to keep the actual state within an invariant “tube” around a nominal trajectory computed neglecting disturbances.
* [constrained RL](https://arxiv.org/pdf/1801.08099.pdf) **To Do**
* [Lyapunov equation](https://stanford.edu/class/ee363/lectures/lq-lyap.pdf) AXA^{{H}}-X+Q=0 **To Do** [coursera videos](https://www.coursera.org/lecture/nonlinear-spacecraft-attitude-control/2-lyapunov-function-definition-no5pF)
* [automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation)

## 6.13
1. Install Carla 0.8.4 release and copy the `carla` folder inside to the git-cloned `conditional imitation learning` repo <br/>
install [tensorflow](https://www.tensorflow.org/install/pip?lang=python2) for python2.7 in virtualenv callled (venv) <br/>
encounter `RuntimeWarning: numpy.dtype size changed, may indicate binary incompatibility.` and solved by upgrading to numy 1.15.2 `pip install numpy==1.15.2` <br/>
correct way to add path: `export PATH=$PATH:/usr/local/cuda`. If use  `export PATH=/usr/local/cuda`, it will replace the path and cannnot execute normal command like `ls` `sudo`. Soln: restore to some commonly added path like `/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin` <br/>
use python3 for carla
2. Try to install CUDA 8.0, 10.0, nvidia driver. libpng library can be manuallly installed. <br/>
Final resolution: cuda 8.0 + nvidia 384 + manully install libpng library. <br/>
[CUDA 10 install](https://devtalk.nvidia.com/default/topic/1045400/cuda-setup-and-installation/cuda-10-installation-problems-on-ubuntu-18-04/)
* [How to Fix an Ubuntu System When It Won’t Boot](https://www.howtogeek.com/196740/how-to-fix-an-ubuntu-system-when-it-wont-boot/)
* [boot to a black screen](https://askubuntu.com/questions/162075/my-computer-boots-to-a-black-screen-what-options-do-i-have-to-fix-it/162078#162078)
* [Graphics issues after/while installing Ubuntu 16.04/16.10 with NVIDIA graphics](https://askubuntu.com/questions/760934/graphics-issues-after-while-installing-ubuntu-16-04-16-10-with-nvidia-graphics)
* [tensorflow and compatible version](https://www.tensorflow.org/install/source#tested_source_configurations)

## 6.14
1. group meeting: try data augmentationm <br/>
[Behavioral cloning w D.A. (from Udacity)](https://medium.com/@ksakmann/behavioral-cloning-make-a-car-drive-like-yourself-dc6021152713)
2. set up `MPSC_Guided_Imitation_Learning` so that both lab comp and my mac `can run pendulum.ipynb` <br/>
mac run `car2.ipynb` till before MPSC guided learning

*Notes*
* add path to jupyter notebook 
```
import os
import sys
nb_dir = os.path.split(os.getcwd())[0]
if nb_dir not in sys.path:
    sys.path.append(nb_dir)
```
* Since version 0.8, Pip supports pip-{version}. You can use it the same as easy_install-{version}:
```
$ pip-2.5 install myfoopackage
$ pip-2.6 install otherpackage
$ pip-2.7 install mybarpackage
```
same for python3
if pip3 doesn't help, use sth like `pip3.6`

*Others*
* ubuntu change font size `gsettings set org.gnome.desktop.interface text-scaling-factor 1.5`, change `1.5` to any factor you want.
* check numpy version: in python, `import numpy     numpy.version.version`

## 6.17
* Weichao's github link for [MPSC_Guided_Imitation_Learning](https://github.com/zwc662/MPSC_Guided_Imitation_Learning)
* [cvxopt cone programming](https://cvxopt.org/userguide/coneprog.html)

*Note* 
* Github In order to synchronize a fork: [config an upstream (source repo)](https://help.github.com/en/articles/configuring-a-remote-for-a-fork) and then [synchronize with it](https://help.github.com/en/articles/syncing-a-fork)
* [Theano documentation/tutorial](http://deeplearning.net/software/theano/) Read till [Seeding Streams](http://deeplearning.net/software/theano/tutorial/examples.html#seeding-streams)
* [Python memory](https://www.evanjones.ca/memoryallocator/)
