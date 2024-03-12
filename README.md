# chocolatechip

```
      _____________,-.___     _
     |____        { {]_]_]   [_]
     |___ `-----.__\ \_]_]_    . `
     |   `-----.____} }]_]_]_   ,
     |_____________/ {_]_]_]_] , `
                   `-'
```

This python package makes computing at the MALTLab easier

## Installation

sign on to any lab machine and do the following:

```
# ls ~/ENV3
# does it not exist? then do:
# python3.12 -m venv ~/ENV3
# if not already activated,
source ~/ENV3/bin/activate
pip install chocolatechip
```

## Use

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

ASCII ART BY Hayley Jane Wakenshaw
