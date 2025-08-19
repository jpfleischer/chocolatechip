# 🍪 chocolatechip

```
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⡴⠚⣉⡙⠲⠦⠤⠤⣤⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⢀⣴⠛⠉⠉⠀⣾⣷⣿⡆⠀⠀⠀⠐⠛⠿⢟⡲⢦⡀⠀⠀⠀⠀
⠀⠀⠀⠀⣠⢞⣭⠎⠀⠀⠀⠀⠘⠛⠛⠀⠀⢀⡀⠀⠀⠀⠀⠈⠓⠿⣄⠀⠀⠀
⠀⠀⠀⡜⣱⠋⠀⠀⣠⣤⢄⠀⠀⠀⠀⠀⠀⣿⡟⣆⠀⠀⠀⠀⠀⠀⠻⢷⡄⠀
⠀⢀⣜⠜⠁⠀⠀⠀⢿⣿⣷⣵⠀⠀⠀⠀⠀⠿⠿⠿⠀⠀⣴⣶⣦⡀⠀⠰⣹⡆
⢀⡞⠆⠀⣀⡀⠀⠀⠘⠛⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢿⣿⣶⠇⠀⢠⢻⡇
⢸⠃⠘⣾⣏⡇⠀⠀⠀⠀⠀⠀⠀⡀⠀⠀⠀⠀⠀⠀⣠⣤⣤⡉⠁⠀⠀⠈⠫⣧
⡸⡄⠀⠘⠟⠀⠀⠀⠀⠀⠀⣰⣿⣟⢧⠀⠀⠀⠀⠰⡿⣿⣿⢿⠀⠀⣰⣷⢡⢸
⣿⡇⠀⠀⠀⣰⣿⡻⡆⠀⠀⠻⣿⣿⣟⠀⠀⠀⠀⠀⠉⠉⠉⠀⠀⠘⢿⡿⣸⡞
⠹⣽⣤⣤⣤⣹⣿⡿⠇⠀⠀⠀⠀⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡔⣽⠀
⠀⠙⢻⡙⠟⣹⠟⢷⣶⣄⢀⣴⣶⣄⠀⠀⠀⠀⠀⢀⣤⡦⣄⠀⠀⢠⣾⢸⠏⠀
⠀⠀⠘⠀⠀⠀⠀⠀⠈⢷⢼⣿⡿⡽⠀⠀⠀⠀⠀⠸⣿⣿⣾⠀⣼⡿⣣⠟⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⢠⡾⣆⠑⠋⠀⢀⣀⠀⠀⠀⠀⠈⠈⢁⣴⢫⡿⠁⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠈⠙⣧⣄⡄⠴⣿⣶⣿⢀⣤⠶⣞⣋⣩⣵⠏⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⢺⣿⢯⣭⣭⣯⣯⣥⡵⠿⠟⠛⠉⠉⠀⠀⠀⠀⠀⠀⠀
```

This python package makes computing at the MALTLab easier. its delicious

## Installation

Please ensure your ssh key is set up with GitHub.
if not, do `ssh-keygen`, and then once done, do 

```bash
cat ~/.ssh/id_rsa.pub
```

and take that key and put it into your github settings new SSH key.


```bash
git clone git@github.com:jpfleischer/chocolatechip.git
cd chocolatechip
make pip
```

The `make pip` makes sure that you are in a Python virtual environment.
It tells you how to make one if you aren't in one. In any case, `make` is
required-- if you are on Windows, and you don't have `make`, follow
https://github.com/cybertraining-dsc/reu2022/blob/main/project/windows-configuration.md#install-chocolatey
then `choco install make -y`

For some of the functionalities, chocolatechip needs to connect to a
MySQL database specifically configured for the near miss pipeline.
You need to create a `login.env`.

```bash
nano src/chocolatechip/login.env
# 
# it looks like this
#
host=FillMeOut
user=FillMeOut
passwd=FillMeOut
db=FillMeOut
testdb=FillMeOut
port=FillMeOut
SSH_USER=FillMeOut
#
# you have to ask someone in the lab for the actual credentials.
#
```

## Use


You can use `chip` now. Try it now!


Generally this is meant for use in a data pipeline that
analyses videos from signalized intersections. If you would
like to add a new intersection to analyse, you need footage
from that intersection.

Take one video that you have saved from that intersection, and
take a snapshot (if you can, get one snapshot with no vehicles
in sight). Then, you get a top-down Google Maps view of that same
intersection. This way, you can rectify the fisheye distortion
of the videos using thin-plate spline. This can be done in the
src/chocolatechip/lanes folder.

`chip fastmot` will benchmark fastmot for you

```bash
Memory Usage - NVIDIA TITAN RTX #1
 1076.00  ┼
  923.14  ┤           ╭──────────────────
  770.29  ┤           │
  617.43  ┤           │
  464.57  ┤          ╭╯
  311.71  ┤      ╭───╯
  158.86  ┤   ╭──╯
    6.00  ┼───╯
Wattage - NVIDIA TITAN RTX #1
   71.59  ┤                            ╭╮
   63.57  ┤   ╭────────────────────────╯╰
   55.54  ┤   │
   47.52  ┤   │
   39.49  ┤   │
   31.47  ┤   │
   23.44  ┼───╯
   15.42  ┤
Temperature - NVIDIA TITAN RTX #1
   36.00  ┤                       ╭──────
   35.33  ┤                       │
   34.67  ┤              ╭────────╯
   34.00  ┤     ╭────────╯
   33.33  ┤   ╭─╯
   32.67  ┤   │
   32.00  ┼───╯
Fan Speed - NVIDIA TITAN RTX #1
   41.00  ┼╮╭╮╭─╮╭──╮╭────╮╭─────────────
   40.83  ┤││││ ││  ││    ││
   40.67  ┤││││ ││  ││    ││
   40.50  ┤││││ ││  ││    ││
   40.33  ┤││││ ││  ││    ││
   40.17  ┤││││ ││  ││    ││
   40.00  ┤╰╯╰╯ ╰╯  ╰╯    ╰╯
1280x960
```


stream.py does mem usage over time for num streams, not resolution related.
jsut change the parameter to gpu plotter.

## Long paths

This may be necessary

```bash
git config --global core.longpaths true
```
