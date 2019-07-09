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

## 6.18
1. lecture about conve optimiaion (Introl only
2. Figure out the naming and way of storage of the checkpoints files.
3. Save your own mpc_safe datapoints, try to save mpc safe traj and re-train

*Note*
* Pickle streams are entirely self-contained, and so unpickling will unpickle one object at a time. <br/>
Therefore, to unpickle multiple streams, you should repeatedly unpickle the file until you get an EOFError:
```
>>> f=open('a.p', 'wb')
>>> pickle.dump({1:2}, f)
>>> pickle.dump({3:4}, f)
>>> f.close()
>>> 
>>> f=open('a.p', 'rb')
>>> pickle.load(f)
{1: 2}
>>> pickle.load(f)
{3: 4}
>>> pickle.load(f)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module> EOFError
```
so your unpickle code might look like
```
import pickle
objs = []
while 1:
    try:
        objs.append(pickle.load(f))
    except EOFError:
        break
```

## 6.19
Meeting with Wenchao
1. Read paper [Berkeley: counter-example guided data augmentation](https://arxiv.org/pdf/1805.06962.pdf)
2. Continue to implement fine-tuning model. <br/>

*Note*
* add 
```
%load_ext autoreload
%autoreload 2
```
in jupyter notebook (iPython for python3.5) to autoreload

## 6.20 
1. Solve problem `can't convert np.ndarray of type numpy.object_.` <br/>
* change `x0_` to `x0`, less one layer of `[]` <br/>
* change `for j in range(0, n+1)` to `for j in range(0, n)`
After changing some format, finally able to train a new model with MPC output. However, the output runs in a circle at the starting point. **TODO** check the x0_ilqr_path, may be modified unintentionally
2. Read about online learning [basic idea](https://medium.com/value-stream-design/online-machine-learning-515556ff72c5) [Thesis: Online Learning: Theory, Algorithms, and Applications](https://ttic.uchicago.edu/~shai/papers/ShalevThesis07.pdf)  [MIT notes](http://www.mit.edu/~rakhlin/6.883/lectures/lecture01.pdf)
3. Modify the code and put `create_expert_traj_boundary/ref` into funcitons. Collect new expert points `[x0, upper_boundary, lower_boundary]`. **TODO** use the new points to train a new model and compare with that with MPC safe control.

*Python*
* [parameter passing for mutable & immutable objects](https://medium.com/@tyastropheus/tricky-python-ii-parameter-passing-for-mutable-immutable-objects-10e968cbda35)

## 6.21
1. Group meeting [EWRL18 tutorial on safe RL](https://las.inf.ethz.ch/files/ewrl18_SafeRL_tutorial.pdf)
2. Set up new desktop following [lambda stack](https://deeptalk.lambdalabs.com/t/install-ubuntu-18-04-with-lambda-stack-from-scratch/490).  Boot program [Legacy VS UEFI](https://phoenixts.com/blog/uefi-vs-legacy-bios/) <br/>
* Ubuntu setup [Top 10 things to do](https://www.linuxtechi.com/top10-things-after-installing-ubuntu-18-04/) 
* [install terminator](https://blog.arturofm.com/install-terminator-terminal-emulator-in-ubuntu/) 
* "switching between screens" is called `workspace`; in 16.04, `menu bar -> Appearances -> Behavior -> Enable workspace` allows you to switch between four workspaces; in 18.04 just move the applicaiton by `ctrl + alt + arrow` and it will automatically create new workspace
* the `super` key is the key with windows logo
3. Big group meeting at Robotics Lab
4. Finish collecting 50000 data points of \[current position, upper&lower boundary], train a model and compare with MPC safe controller. have 5 (fewer) points violating safety constraint, and the "switching" controller seems to work much smoothier than CNN learnt from ref traj. However, when I use the five MPC controller o\p to retrain the model, it flies to sky.

## 6.22
Running, hotpot, chatting, go to PHO and study Cvx Opt Lec2

## 6.23
Chatting, meet Wanzheng, study Cvx Opt Lec3, read part of paper reduction to IL, 选课

## 6.24
0. Get the account ready
1. Collect points for non-symmetric boundaries, train the model and draw the plots 
2. Finish the paper `A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning` <br/>
**TODO** find code & try to implement on Carla <br/>
Found [one implemented on car racing game with Gym](https://github.com/havefun28/imitation-dagger) <br/>
able to run dagger.py, **TODO** Look into the dagger code
3. Install Carla (download Carl 0.9.5 release package, as building on Linux encounter the same problem of "No rule to make target 'launch'") <br/>
Run `sudo apt-get install python-pygame` and `sudo apt-get install python3-pygame` in terminal to install pygame <br/>
Try carla/CIL again and notice that `run_CIL.py` only works with python 2.7 & 3.5 <br/>
[intall python3.5](https://askubuntu.com/questions/682869/how-do-i-install-a-different-python-version-using-apt-get ), since originally only have 3.6, 3.7. <br/>
However, since packages like numpy were only installed on PYthon 3.6 & 3.7, choose to run CIL with python 2.7 and Carla 0.8.4 (Carla 0.9 encounter error `no 'carla' found` or `no 'carla.benchmark' found`) <br/>
Encounter error `ERROR: (localhost:2000) failed to connect: [Errno 111] Connection refused`, solved by adding `carla-server` tag to `./CarlaUE4.sh` server command <br/>
Encounter error `module 'scipy.misc' has no attribute 'imresize`, only solved in PYthon 2.7 by `sudo pip install Pillow` and reopen the terminal
4. Current solution to run carla/CIL: <br/>
* in one terminal of `~/CARLA_0.8.4` run ` ./CarlaUE4.sh -carla-server` (which will [by default connect to port 200, 2001, 2002](https://carla.readthedocs.io/en/stable/connecting_the_client/) can be check by running `./client_example.py --autopilot` under `~/CARLA_0.8.4/PythonClient`)<br/>
* in another terminal of `~/imitation-learning` run `python run_CIL.py` (works fine with 2.7, but not with `python3` because of error `module 'scipy.misc' has no attribute 'imresize` >> check Pillow has been installed, scipy is updated, so the problem is maybe due to the fact that [`imread` and `imresize` have been removed](https://github.com/lengstrom/fast-style-transfer/issues/106#issuecomment-501967909))
5. Go through run_CIL code and it only loads a pre-trained model (an agent) and benchmark it. To train the model or modify and retrain it with dagger, you need to look at their training code. see below
6. On old desktop, install [sublime merge](https://www.sublimemerge.com/docs/linux_repositories#apt)

*TODO*
[argparse tutorial](https://docs.python.org/2/howto/argparse.html)
* posts to look at for Carla CIL training <br/>
[Questions on model & training](https://github.com/carla-simulator/imitation-learning/issues/1)
[merantix](https://github.com/merantix/imitation-learning)
[felip](https://github.com/felipecode/coiltraine/)
*Note*
* use [lsof command](https://www.cyberciti.biz/faq/unix-linux-check-if-port-is-in-use-command/) to check the ports been used by programs
* CARLA tip: <br/>
 * You can launch the simulator in windowed mode by using the argument -windowed, and control the window size with -ResX=N and -ResY=N.
 * [Github version control](https://ourcodingclub.github.io/2017/02/27/git.html)
 
## 6.25
1. Install [Miniconda 2](https://www.osetc.com/en/how-to-install-miniconda-on-ubuntu-18-04-16-04-linux.html) on new desktop.
2. Try training carla using [felip repo coiltraine](https://github.com/havefun28/coiltraine) <br/>
To run coiltraine, need to [run Carla in docker](https://carla.readthedocs.io/en/latest/carla_docker/)
3. Create a Docker account. Username: ruihan pw: ILstudy28! email: gaor0007@entu.edu.sg
4. Install docker  
 * for Docker only can follow  [How to Install Docker On Ubuntu 18.04 Bionic Beaver](https://linuxconfig.org/how-to-install-docker-on-ubuntu-18-04-bionic-beaver), change `bionic` to [`artful`](https://askubuntu.com/questions/1030179/package-docker-ce-has-no-installation-candidate-in-18-04) and finally get `(coiltraine) ruihan@depend-XPS-8930:~$ docker --version Docker version 18.06.3-ce, build d7080c1`
 * in coiltraine case where nvidia-docker2 is required, follow [Get Docker CE for Ubuntu](https://docs.docker.com/install/linux/docker-ce/ubuntu/#extra-steps-for-aufs). Try with repo, but encounter the problem cuz "docker-ce package is missing for Ubuntu "Bionic" 18.04 LTS x86_64" Soln: follow [QuickStart](https://github.com/NVIDIA/nvidia-docker#quickstart) and after adding the repos, do the following to pin all three specific versions
```
sudo apt-get install docker-ce=18.06.0~ce~3-0~ubuntu
sudo apt-get install nvidia-container-runtime=2.0.0+docker18.06.0-1
sudo apt-get install nvidia-docker2=2.0.3+docker18.06.0-1
```
* Note:when create a new file in `/etc/`, permission denied; soln: create in home directory and copy it to `/etc`
* [os.environ](https://kite.com/python/docs/os.environ)

* After a day's debugging (should use `suppress_output=0` at the first beginning), find the error and many others [encounter the same thing](https://github.com/felipecode/coiltraine/issues/12). <br/>
the error is   
```
File "/home/ruihan/miniconda2/envs/coiltraine/lib/python3.5/site-packages/numpy/core/fromnumeric.py", line 41, in _wrapit
    result = getattr(asarray(obj), method)(*args, **kwds)
numpy.core._internal.AxisError: axis1: axis 0 is out of bounds for array of dimension 0
```
when running `for data in data_loader` in `train.execute`

## 6.26
1. Continue to debug CoIL training.  <br/>
`python3 coiltraine.py --folder sample --gpus 0 -de TestT1_Town01 -vd CoILVal1 --docker carlasim/carla:0.8.4`

*Debug*
* `STATUS: Error - cuda runtime error (38) : no CUDA-capable device is detected at /pytorch/aten/src/THC/THCGeneral.cpp:74` <br/>
Soln: set `gpus` = 0 (Restart the program with os.environ\["CUDA_VISIBLE_DEVICES"] = '0'can solve the problem)
* `AttributeError: 'NoneType' object has no attribute 'swapaxes'`<br/>
Find that the error is due to inconsistency of img paths when reading the files. <br/>
check `def _pre_load_image_folders(self, path):` in `coil_dataset.py` <br/>
"THCudaCheck FAIL file=/pytorch/aten/src/THC/THCGeneral.cpp line=663 error=11 : invalid argument" Soln: add `torch.backends.cudnn.benchmark = False` in `train.py` before using torch <br/>
the program stops/hangs at `loss.backward()` and cannot proceed to `optimizer.step()` after one iteration. Soln: upgrade pytorch from 0.4 to 1.0 by `pip install --upgrade torch torchvision` <br/>
after upgrading, encounter "Error - cuDNN error: CUDNN_STATUS_EXECUTION_FAILED", need to upgrade corresponding packages by `conda install pytorch torchvision cudatoolkit=10.0 -c pytorch` (my cuda is actually 10.1 but this works) <br/>
Till now it can load data properly and proceed from `Loading` status to `Iterating` status. Hopefully it goes well tomorrow.

*Python*
* check meaning and [differences between `def __init__` and  ` def __call__`](https://stackoverflow.com/questions/9663562/what-is-the-difference-between-init-and-call), [`def __repr__`](https://stackoverflow.com/questions/1984162/purpose-of-pythons-repr) , and [`def __getitem__`](https://stackoverflow.com/questions/43627405/understanding-getitem-method)

* [git ignore](https://git-scm.com/docs/gitignore)

## 6.27
1. **TODO** [exploration](http://www.cs.cmu.edu/~rsalakhu/10703/Lecture_Exploration.pdf)
2. To avoid "CUDA out of memory" error: change GPU value from 3.5 to 2 so that only one process is going on.
3. Debug: `driving` is in `iterating` for too long, then check the o`outputl_log` find the error `ERROR:root:(127.0.0.1:49971) failed to connect: [Errno 111] Connection refused`
4. After training and validation, can run the following to execute one single process `python3 coiltraine.py --folder sample --single-process drive --exp coil_icra --gpus 0 -de TestT1_Town01 -vd CoILVal1 --docker carlasim/carla:0.8.4` (`coil_icra` is found by searching the subfolder of `_logs/sample/`)
5. When run the single process of `drive`, <br/>
encounter "ERROR:root:(127.0.0.1:49971) failed to connect: \[Errno 111] Connection refused" <br/>
Soln: set `suppress_output` to False and find the internal error is `permission denied` (docker: Got permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock: Post http://%2Fvar%2Frun%2Fdocker.sock/v1.38/containers/create: dial unix /var/run/docker.sock: connect: permission denied. See 'docker run --help') from `docker`. Should [add user to docker group](https://techoverflow.net/2018/12/15/how-to-fix-docker-got-permission-denied-while-trying-to-connect-to-the-docker-daemon-socket/) by running command [`sudo usermod -a -G docker $USER`](https://techoverflow.net/2019/04/30/what-does-sudo-usermod-a-g-docker-user-do-on-linux/), where `$USER` should be either `$USER` or explicitly `ruihan`
To run `docker` without `sudo`, log out and login again to the desktop (not only docker account) <br/>
encouter error "ERROR:root:(127.0.0.1:54729) connection closed" <br/>
Encounter "connection closed when running carla", waiting for a few seconds helps since [the client may be waiting](https://github.com/carla-simulator/carla/issues/263#issuecomment-383113144)
6. Try Penny's [Get started](https://carlachallenge.org/get-started/) <br/>
When running `bash srunner/challenge/run_evaluator.sh`, encounter `ImportError: No module named carla`. Soln: in `~/.bashrc`, add `export PYTHONPATH="${PYTHONPATH}:/home/ruihan/CARLA_0.9.5/PythonAPI/"` <br/>
if follow [tutorial](https://github.com/carla-simulator/scenario_runner/blob/development/Docs/getting_started.md/#install_prerequisites), encounter `https://github.com/carla-simulator/scenario_runner/blob/development/Docs/getting_started.md/#install_prerequisites` when running `ython scenario_runner.py --scenario FollowLeadingVehicle` [possible soln](https://github.com/carla-simulator/scenario_runner/issues/172)
7. Note: when running srunner, remember to do `conda deactivte`, otherwise need to install packages separtely, e.g. `cv2`


Commands
python $/home/ruihan/scenario_runner/srunner/challenge/challenge_evaluator_routes.py \
--scenarios=$/home/ruihan/scenario_runner/srunner/challenge/all_towns_traffic_scenarios1_3_4.json \
--routes=$/home/ruihan/scenario_runner/srunner/challenge/routes_training.xml \
--repetitions=3 \
--debug=0 \
--agent=autonomous_agent \
--config=FollowLeadingVehicle.xml

*Notes*
* Docker 
 * check process and port `docker ps -a`
 * [remove a container](https://linuxize.com/post/how-to-remove-docker-images-containers-volumes-and-networks/)
 * Error response from daemon::Cannot kill container  signaling init process caused "permission denied" 
 ```
 For anyone that does not wish to completely purge AppArmor.

 Check status: sudo aa-status

 Shutdown and prevent it from restarting: sudo systemctl disable apparmor.service --now

 Unload AppArmor profiles: sudo service apparmor teardown

 Check status: sudo aa-status

 You should now be able to stop/kill containers.
 ```
 * [dlib](http://dlib.net/)
 
## 6.28
1. Group meeting "App-Aware Scheduling" shard by Kacper, "enumerating the other labelling is already #P complete, i.e. too complex, so only evaluate optimal solution for one labelling.
2. Meeting with Wenchao, use the Carla Challenge one and make your own agent. Can refer to `coiltraine` if necessary but no need to spend too much time debugging it. <br/>
He stresses 
 1. for different scenarios, like the routes, turns, and notice that we can create some json files like `srunner/challenge/all_towns_traffic_scenarios1_3_4.json` and add more interesting scenarios.
 2.Add more controllable agents at the same time: cooperative imitation learning
3. Reading group meeting at Robotics Lab. Reinforcement learning. <br/>
Parameters tuning, [neural architecture search](https://www.automl.org/automl/literature-on-neural-architecture-search/), [knowledge distillation](https://arxiv.org/pdf/1503.02531.pdf)
4. Experiment with other agents built in scenario_runner and try to create your own imitation learning agent. <br/>
**Total installation:**
 0. `conda deactivate` to deactivate any running virtual environment and in `~/.bashrc`, comment out miniconda. Check by `which python` >> `usr/bin/python` <br/>
Clone [scenario_runner repo](https://github.com/carla-simulator/scenario_runner) to my account, then follow [tutorial](https://carlachallenge.org/get-started/) and [installation](https://github.com/carla-simulator/scenario_runner/blob/development/Docs/getting_started.md/#installing-prerequisites)
 1. Follow Step 1 to clone `carla_challenge` branch
 2. Follow Step 2 and set `bash setup_environment.sh --carla-root /home/ruihan/CARLA_0.9.5` followed by `source ~/.bashrc`
Routine procedure: <br/>
 3. Follow Step 3 exactly. When firstly start the server `./CarlaUE4.sh`, the town appears. But after running `bash srunner/challenge/run_evaluator.sh`, it will automatically change to racing challenge setting, so don't worry.
**After first time setting, just run Step 3 to start experiment**
5. First glance into the agents: 
 * Dummy agent does nothing but stays there. 
 * HumanAgent receives keyboard-mouse control.
 * NPCAgent follows pre-defined path. (Normal players as others)
 * ros_agent needs package `rospy` and not in interest now
 * Track4SampleAgent "THis is a human controlled agent with track 4 access for testing", did not get it yet
6. [Fov: field of view](https://en.wikipedia.org/wiki/Field_of_view_in_video_games)
7. Try the `coiiltrane` imitation learning agent with scenario_runner.py
when running [visualization](https://github.com/felipecode/coiltraine/blob/master/docs/carla_challenge_coil_baseline.md#visualize-the-agent-results)
```
$ python3 view_model.py -f baselines -e resnet34imnet -cp 180000 -cv 0.9
pygame 1.9.4
Hello from the pygame community. https://www.pygame.org/contribute.html
Traceback (most recent call last):
  File "view_model.py", line 89, in <module>
    import model_view.carla09interface as carla09interface
  File "/home/ruihan/coiltraine/model_view/carla09interface.py", line 72, in <module>
    import carla
  File "/home/ruihan/CARLA_0.9.5/PythonAPI/carla/__init__.py", line 8, in <module>
    from .libcarla import *
ImportError: /home/ruihan/CARLA_0.9.5/PythonAPI/carla/libcarla.so: undefined symbol: PyString_Type
```

when running [this](https://github.com/felipecode/coiltraine/blob/master/docs/carla_challenge_coil_baseline.md#get-the-agent-performance-on-the-carla-challenge)
```
CHALLENGE_PHASE_CODENAME=dev_track_2 python3 ${ROOT_SCENARIO_RUNNER}/srunner/challenge/challenge_evaluator_routes.py --scenarios=${ROOT_SCENARIO_RUNNER}/srunner/challenge/all_towns_traffic_scenarios1_3_4.json --routes=${ROOT_SCENARIO_RUNNER}/srunner/challenge/routes_training.xml --debug=0 --agent=../coiltraine/drive/CoILBaseline.py --config=../coiltraine/drive/sample_agent.json
Traceback (most recent call last):
  File "/home/ruihan/scenario_runner/srunner/challenge/challenge_evaluator_routes.py", line 31, in <module>
    import py_trees
ImportError: No module named 'py_trees'
```
## 6.29
* successful install `sudo python -m easy_install ${CARLA_ROOT}/PythonAPI/carla/dist/*-py2.7-linux-x86_64.egg`
* `sudo ln -sfn /usr/bin/python3.6 /usr/bin/python3`
```
CHALLENGE_PHASE_CODENAME=dev_track_2 python ${ROOT_SCENARIO_RUNNER}/srunner/challenge/challenge_evaluator_routes.py\
--scenarios=${ROOT_SCENARIO_RUNNER}/srunner/challenge/all_towns_traffic_scenarios1_3_4.json \
--routes=${ROOT_SCENARIO_RUNNER}/srunner/challenge/routes_training.xml \
--debug=0 \
--agent=../coiltraine/drive/CoILBaseline.py \
--config=../coiltraine/drive/sample_agent.json
```
"error: command 'x86_64-linux-gnu-gcc' failed with exit status 1 ---------------------------------------- Failed building wheel for psutil" Soln: `$ sudo apt install python3.5-dev`

* use python3.5 for coiltraine and scenario_run. Install some packages for 3.5 (system default only has 3.6 and 3.7)
* switch default python3 version `sudo ln -sfn /usr/bin/python3.5 /usr/bin/python3`
 * Debug: "Unable to find `imresize`". If got this error even with pillow installed on scipy==1.3.0, then fixed it by installing an earlier version:  `pip3 install scipy==1.1.0 --user`
 * Debug "ModuleNotFoundError: No module named `torch._C` ", `sudo pip3 install --upgrade torch` 
 * [easy_install](https://setuptools.readthedocs.io/en/latest/easy_install.html)

* when run `python3 view_model.py  -f baselines -e resnet34imnet -cp 180000 -cv 0.9`, pygame blank black screen, but does not report any error
* successfully run  
```
CHALLENGE_PHASE_CODENAME=dev_track_2 python3 ${ROOT_SCENARIO_RUNNER}/srunner/challenge/challenge_evaluator_routes.py \
--scenarios=${ROOT_SCENARIO_RUNNER}/srunner/challenge/all_towns_traffic_scenarios1_3_4.json \
--routes=${ROOT_SCENARIO_RUNNER}/srunner/challenge/routes_training.xml \
--debug=0 \
--agent=../coiltraine/drive/CoILBaseline.py \
--config=../coiltraine/drive/sample_agent.json
```
under `~/scenario_run` without any conda environment

## 7.1
1. Fix the bug of "cannot open terminator/terminal". It is the problem of python version. <br/>
When running CARLA, I changed the default `python3` from 3.6 to 3.5 globally, while in `/usr/bin` the `_gi` file required for lauching `gnome-terminal` is for 3.6 only. (Launch `xTerm` and type `gnome-terminal` returns the error "Import Error cannot import name '\_gi'") <br/>
Soln: 
```
cd /usr/lib/python3/dist-packages/gi/
ls
sudo cp _gi.cpython-36m-x86_64-linux-gnu.so _gi.cpython-35m-x86_64-linux-gnu.so
sudo cp _gi_cairo.cpython-36m-x86_64-linux-gnu.so _gi_cairo.cpython-35m-x86_64-linux-gnu.so
```
so that `gnome-terminal` works fine with python3.5. <br/>
For terminator, which needs to work with python2 (check by running`terminator` in terminal and get error "except(KeyError, ValueError), ex") <br/>
Soln: edit `/usr/bin/terminator` and changing the python version to python2 (the default python was changed to python3). In my case, the file `/usr/bin/terminator` is read-only, so copy it to `~/` directory (`sudo cp /usr/bin/terminator /home/ruihan/terminator`), change the first line from `#!/usr/bin/python` to `#!/usr/bin/python2`. Finally copy it back `sudo cp /home/ruihan/terminator /usr/bin/terminator` and you're ready to go!
2. **SMOOTH flow**
Start the CARLA server on one terminal without conda env:
```
cd ~/CARLA_0.9.5
./CarlaUE4.sh -benchmark -fps=20 -quality-level=Epic
```
Execute the challenge with the conditional imitation learning baseline in another terminal
```
cd ~/scenario_runner
conda activate coiltraine
CHALLENGE_PHASE_CODENAME=dev_track_2 python3 ${ROOT_SCENARIO_RUNNER}/srunner/challenge/challenge_evaluator_routes.py \
--scenarios=${ROOT_SCENARIO_RUNNER}/srunner/challenge/all_towns_traffic_scenarios1_3_4.json \
--routes=${ROOT_SCENARIO_RUNNER}/srunner/challenge/routes_training.xml \
--debug=0 \
--agent=../coiltraine/drive/CoILBaseline.py \
--config=../coiltraine/drive/sample_agent.json
```
Watch the results.

*Note*
* `sudo apt-get install` option: `sudo apt-get install --reinstall terminator`
* change the `~/.bashrc` to default setting 
```
cp ~/.bashrc ~/Documents/.bashrc.bak (save a copy of current file)
cp /etc/skel/.bashrc ~/ (recreate a fresh bashrc file)
```
* set the locale
```
sudo locale-gen
sudo localectl set-locale LANG="en_US.UTF-8"
```
* correct way to add a PYTHONPATH `export PYTHONPATH=/home/ruihan/coiltraine:$PYTHONPATH`; `:` stands for "adding" and no space in between!
* kill process in terminal, e.g. for pygame that does not close window `kill -9 xxx`(xxx is the process ID) or `pkill -9 python`

*Python*
* [getattr](https://effbot.org/zone/python-getattr.html)

## 7.2
*QuickLinks*
* [Policy aggregation in SMPyBandit](https://smpybandits.github.io/Aggregation.html)
* [CARLA API ref](https://github.com/carla-simulator/carla/blob/master/Docs/python_api.md) <br/>
* [PYTHON API methods ref](https://carla.readthedocs.io/en/latest/python_api/) <br/>
* [Git commands](https://github.com/joshnh/Git-Commands)

1. Individual meeting with Wenchao. Suggest to implement safe IL & BC on Scenario 1, ControlLoss of Carla AD Challenge.
2. Try to run scenario_runner with ControlLoss.
```
export ROOT_SCENARIO_RUNNER=/home/ruihan/scenario_runner
python ${ROOT_SCENARIO_RUNNER}/srunner/challenge/challenge_evaluator_routes.py \
--scenarios=${ROOT_SCENARIO_RUNNER}/srunner/challenge/all_towns_traffic_scenarios1_3_4.json \
--routes=${ROOT_SCENARIO_RUNNER}/srunner/challenge/routes_training.xml \
--repetitions=3 \
--debug=0 \
--agent=${TEAM_AGENT} \
--config=${TEAM_CONFIG}
```
```
 CHALLENGE_PHASE_CODENAME=dev_track_2 python3 ${ROOT_SCENARIO_RUNNER}/srunner/challenge/challenge_evaluator_routes.py \
--scenarios=${ROOT_SCENARIO_RUNNER}/srunner/challenge/all_towns_traffic_scenarios1_3_4.json \
--routes=${ROOT_SCENARIO_RUNNER}/srunner/challenge/routes_training.xml \
--debug=0 \
--agent=../coiltraine/drive/CoILBaseline.py \
--config=../coiltraine/drive/sample_agent.json
```
3. Look into scenario files and want to train the ciltriane agent.

* [Environment variables](https://medium.com/chingu/an-introduction-to-environment-variables-and-how-to-use-them-f602f66d15fa) e.g. `CHALLENGE_PHASE_CODENAME` to run `challenge_evaluater_routes.py`
* [behaviour tree](https://py-trees.readthedocs.io/en/devel/behaviours.html)
* [xml py-tree](https://www.datacamp.com/community/tutorials/python-xml-elementtree)

*Debug*
When running [training](https://github.com/havefun28/coiltraine/blob/master/docs/carla_challenge_coil_baseline.md#training)
* "Object arrays cannot be loaded when allow_pickle=False' for `np.load` function" <br/>
Soln: change `with np.load(path) as f` to `with np.load(path, allow_pickle=True) as f`
* "cannot import name '\_validate_lengths' " <br/>
`conda install -c conda-forge scikit-image` and/or `pip install -U scikit-image`
* "THCudaCheck FAIL file=/pytorch/aten/src/THC/THCGeneral.cpp line=383 error=11 : invalid argument" <br/>
Soln: [Incompatible torch version and RTX](https://github.com/pytorch/pytorch/issues/15797)
```
$ wget https://download.pytorch.org/whl/cu100/torch-1.0.0-cp35-cp35m-linux_x86_64.whl
$ python3.5 -m pip install torch-1.0.0-cp35-cp35m-linux_x86_64.whl
```
Leave with: successfully run --scenario FollowLeadingVehicle with scenario_runner of carla_challenge branch
## 7.3
1. Switch to master branch of scenario_runner, which allows multiple ego_vehicle and fix the 'group' option for running one scenario <br/>
Experiment: 1. run CARLA_0.95 as usual; 2. `python scenario_runner.py --scenario group:FollowLeadingVehicle` in `~/scenario_runner_master`; 3. Before it times out, run ` python manual_control.py` also in  `~/scenario_runner_master` <br/>
Results: 
Currently the smoothiest one. Can see around three (fixed) scenarios, but still need debugging.
```
(coiltraine) ruihan@depend-XPS-8930:~/scenario_runner_master$ python scenario_runner.py --scenario group:FollowLeadingVehicle
Preparing scenario: FollowLeadingVehicle_1
ScenarioManager: Running scenario FollowVehicle
Resetting ego-vehicle!
Resetting ego-vehicle!
Failure!
Resetting ego-vehicle!
Preparing scenario: FollowLeadingVehicleWithObstacle_1
ScenarioManager: Running scenario FollowLeadingVehicleWithObstacle
Resetting ego-vehicle!
Resetting ego-vehicle!
Resetting ego-vehicle!
Failure!
Resetting ego-vehicle!
Resetting ego-vehicle!
Preparing scenario: FollowLeadingVehicle_2
ScenarioManager: Running scenario FollowVehicle
Resetting ego-vehicle!
Failure!
Resetting ego-vehicle!
Preparing scenario: FollowLeadingVehicleWithObstacle_2
ScenarioManager: Running scenario FollowLeadingVehicleWithObstacle
eeResetting ego-vehicle!
Resetting ego-vehicle!
Resetting ego-vehicle!
Failure!
Resetting ego-vehicle!
Resetting ego-vehicle!
Preparing scenario: FollowLeadingVehicle_3
The CARLA server uses the wrong map!
This scenario requires to use map Town02
The scenario cannot be loaded
The CARLA server uses the wrong map!
Preparing scenario: FollowLeadingVehicleWithObstacle_3
The CARLA server uses the wrong map!
This scenario requires to use map Town02
The scenario cannot be loaded
The CARLA server uses the wrong map!
Preparing scenario: FollowLeadingVehicle_4
The CARLA server uses the wrong map!
This scenario requires to use map Town03
The scenario cannot be loaded
The CARLA server uses the wrong map!
Preparing scenario: FollowLeadingVehicleWithObstacle_4
The CARLA server uses the wrong map!
This scenario requires to use map Town03
The scenario cannot be loaded
The CARLA server uses the wrong map!
Preparing scenario: FollowLeadingVehicle_5
The CARLA server uses the wrong map!
This scenario requires to use map Town04
The scenario cannot be loaded
The CARLA server uses the wrong map!
Preparing scenario: FollowLeadingVehicleWithObstacle_5
The CARLA server uses the wrong map!
This scenario requires to use map Town04
The scenario cannot be loaded
The CARLA server uses the wrong map!
Preparing scenario: FollowLeadingVehicle_6
The CARLA server uses the wrong map!
This scenario requires to use map Town04
The scenario cannot be loaded
The CARLA server uses the wrong map!
Preparing scenario: FollowLeadingVehicleWithObstacle_6
The CARLA server uses the wrong map!
This scenario requires to use map Town04
The scenario cannot be loaded
The CARLA server uses the wrong map!
Preparing scenario: FollowLeadingVehicle_7
The CARLA server uses the wrong map!
This scenario requires to use map Town04
The scenario cannot be loaded
The CARLA server uses the wrong map!
Preparing scenario: FollowLeadingVehicleWithObstacle_7
The CARLA server uses the wrong map!
This scenario requires to use map Town04
The scenario cannot be loaded
The CARLA server uses the wrong map!
Preparing scenario: FollowLeadingVehicle_8
The CARLA server uses the wrong map!
This scenario requires to use map Town05
The scenario cannot be loaded
The CARLA server uses the wrong map!
Preparing scenario: FollowLeadingVehicleWithObstacle_8
The CARLA server uses the wrong map!
This scenario requires to use map Town05
The scenario cannot be loaded
The CARLA server uses the wrong map!
Preparing scenario: FollowLeadingVehicle_9
The CARLA server uses the wrong map!
This scenario requires to use map Town05
The scenario cannot be loaded
The CARLA server uses the wrong map!
Preparing scenario: FollowLeadingVehicleWithObstacle_9
The CARLA server uses the wrong map!
This scenario requires to use map Town05
The scenario cannot be loaded
The CARLA server uses the wrong map!
Preparing scenario: FollowLeadingVehicle_10
The CARLA server uses the wrong map!
This scenario requires to use map Town05
The scenario cannot be loaded
The CARLA server uses the wrong map!
Preparing scenario: FollowLeadingVehicleWithObstacle_10
The CARLA server uses the wrong map!
This scenario requires to use map Town05
The scenario cannot be loaded
The CARLA server uses the wrong map!
Preparing scenario: FollowLeadingVehicle_11
The CARLA server uses the wrong map!
This scenario requires to use map Town02
The scenario cannot be loaded
The CARLA server uses the wrong map!
Preparing scenario: FollowLeadingVehicleWithObstacle_11
The CARLA server uses the wrong map!
This scenario requires to use map Town02
The scenario cannot be loaded
The CARLA server uses the wrong map!
No more scenarios .... Exiting
```
If for Step 2, use ` python scenario_runner.py --reloadWorld --scenario group:FollowLeadingVehicle` instead, will encounter error
```
ERROR: trying to access an expired episode; a new episode was started in the simulation but an object tried accessing the old one.
Traceback (most recent call last):
  File "manual_control.py", line 622, in game_loop
    if not world.tick(clock):
  File "manual_control.py", line 158, in tick
    if len(self.world.get_actors().filter(self.vehicle_name)) < 1:
RuntimeError: trying to access an expired episode; a new episode was started in the simulation but an object tried accessing the old one.

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "manual_control.py", line 681, in main
    game_loop(args)
  File "manual_control.py", line 630, in game_loop
    world.destroy()
  File "manual_control.py", line 177, in destroy
    actor.destroy()
RuntimeError: trying to access an expired episode; a new episode was started in the simulation but an object tried accessing the old one.
```
*Debug*
* Fix [error in loading scenarios](https://github.com/carla-simulator/scenario_runner/issues/172)
* Fix Git error "remote: error: File is 268.01 MB; this exceeds GitHub's file size limit of 100.00 MB" <br/>
1. [Remove a big file wrongly committed](https://thomas-cokelaer.info/blog/2018/02/git-how-to-remove-a-big-file-wrongly-committed/)
```
git filter-branch --tree-filter 'rm -rf path/to/your/file' HEAD
git push
```
2. if get error "\[rejected] master -> master (non-fast-forward) error: failed to push some refs to 'https://github.com/havefun28/coiltraine.git' hint: Updates were rejected because the tip of your current branch is behind hint: its remote counterpart. Integrate the remote changes" <br/>
Soln: [deal with fast-forward error](https://help.github.com/en/articles/dealing-with-non-fast-forward-errors)
3. error "fatal: refusing to merge unrelated histories"
```
git pull origin master --allow-unrelated-histories
git merge origin origin/master
```
4. error "Auto-merging input/coil_dataset.py CONFLICT (add/add): Merge conflict in input/coil_dataset.py" <br/>
use [`git diff`](https://git-scm.com/docs/git-diff) to check
```
git diff branch1 branch2 -- path/to/file
git checkout name_of_branch path/to/file
```
Sometimes the right content is a mix of the two files. In that case you have to resolve manually, possibly by studying the differences using the git diff command and manully modify in text editor or use Git desktop GUI.

* Running CARLA
 * "The CARLA server uses the wrong map!" <BR/>
 Soln: when running [`scenario_runner`](https://github.com/carla-simulator/scenario_runner/blob/master/Docs/getting_started.md), use `./CarlaUE4.sh /Game/Carla/Maps/Town01 -benchmark -fps=20 -windowed` for server; <br/>
 when running [challenge](https://carlachallenge.org/get-started/), use ` ./CarlaUE4.sh -benchmark -fps=20 -quality-level=Epic` for server
 *  two different systems, `scenario_runner` and `challenge_routes.py`
 
* try to convert xml to json file. `ControlLoss.xml` contains all scenarios (Town 01-05) of control loss and I try to convert it to `.json` file so that can run in `challenge`, which currently use `Scenario1_3_4` to contain different types of scenarios in different towns. <br/>
Can use library [xmltodict](https://pypi.org/project/xmltodict/) but need to pay close attention to the hierachy

## 7.4
* Starting point (what works) <br/>
```
./CarlaUE4.sh /Game/Carla/Maps/Town01 -benchmark -fps=20 -windowed 
python scenario_runner.py --scenario group:FollowLeadingVehicle
python manual_control.py
```
Changing the second line to `python scenario_runner.py --scenario FollowLeadingVehicle_1` will encounter error "Configuration for scenario FollowLeadingVehicle_1 cannot be found!"
* Debug: "ImportError: No module named 'tools.download_tools'" <br/>
Soln: need to add to [PYTHONPATH](https://stackoverflow.com/questions/51288512/difference-between-path-sys-path-and-os-environ/51290910), not only `sys.path` 
```
cd ~/coitraine
export PYTHONPATH=`pwd`:$PYTHONPATH
```
* Error: "json.decoder.JSONDecodeError: Expecting value: line 68 column 21 (char 2463)" <br/>
Soln: use json formatter to check; there shouldn't be `,` after the last element in an array
* To create a json file only for ControlLoss, previously convert `ControlLoss.xml` into json format. However, when put in `run_evaluator.sh`, nothing shows on the road, probably due to wrong coordinates. <br/>
what works now, modify `all_towns_traffic_scenarios1_3_4.json` and only keep `Scenario1` and get the file `all_towns_traffic_scenarios1.json`, which successfully show on the road. <br/>
Performance: not good. On the highway, the car just crashes into it and disappear, on the road sometimes run above it.
* Git. add a local copy of a cloned repo to my own repos and set the `origin` as `upstream` <br/>
```
 git push
remote: Permission to carla-simulator/scenario_runner.git denied to havefun28.
fatal: unable to access 'https://github.com/carla-simulator/scenario_runner.git/': The requested URL returned error: 403

 git remote -v
origin	https://github.com/carla-simulator/scenario_runner.git (fetch)
origin	https://github.com/carla-simulator/scenario_runner.git (push)

 git remote add upstream https://github.com/carla-simulator/scenario_runner.git
 git remote set-url origin https://github.com/havefun28/scenario_runner_master.git

 git remote -v
origin	https://github.com/havefun28/scenario_runner_master.git (fetch)
origin	https://github.com/havefun28/scenario_runner_master.git (push)
upstream	https://github.com/carla-simulator/scenario_runner.git (fetch)
upstream	https://github.com/carla-simulator/scenario_runner.git (push)

 git push
```

## 7.8
**Terminal commmands**
` conda deactivate` `cd CARLA_0.9.5` `./CarlaUE4.sh -benchmark -fps=20 -quality-level=Epic` <br/>
`conda activate coiltraine` `cd scenario_runner_master` `bash srunner/challenge/run_evaluator_co.sh`
*ToDo*
Design an agent for MPSC <br/>
 * "inner-loop" to simulate the vehicle model with `apply_control`
 * get bondary info that is consistent with GPS/localizaiton obtained from `transform_to_location`

*Carla*
* blueprint library for UE Carla => folder: /home/ruihan/CARLA_0.9.5/CarlaUE4/Content/Carla/Static

*Python*
* [`import logging`](https://docs.python.org/2/howto/logging.html)

*Others*
* git clone weichao's [MPSC repo](https://github.com/zwc662/MPSC_Guided_Imitation_Learning) but encounter the error of "ImportError: cannot import name 'constants'" when lauching `jupyter notebook`. 
Soln:
```
Anaconda
pip uninstall jupyter
pip install jupyter
pip install --force-reinstall --upgrade pyzmq
```
* Git: to ignore an untracked file, a simple way to add it to .gitignore is: Change to the root of the git tree and do `git ls-files --others --exclude-standard >> .gitignore`
* [RPC(remote procedure call) model](https://www.ibm.com/support/knowledgecenter/ssw_aix_72/com.ibm.aix.progcomc/rpc_mod.htm)

## 7.9 
1. Meeting with Wenchao
Parameter fitting
online/offline approximate model
pomdp strategy improvement
Slack penalty
Monotonic
Solve mdp
Termination
Maximize  fur
Overleaf sharing document
