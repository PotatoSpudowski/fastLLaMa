# fastLLaMa

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

`fastLLaMa` is an experimental high-performance framework designed to tackle the challenges associated with deploying large language models (LLMs) in production environments.


It offers a user-friendly Python interface to a C++ library, [llama.cpp](https://github.com/ggerganov/llama.cpp), enabling developers to create custom workflows, implement adaptable logging, and seamlessly switch contexts between sessions. This framework is geared towards enhancing the efficiency of operating LLMs at scale, with ongoing development focused on introducing features such as optimized cold boot times, Int4 support for NVIDIA GPUs, model artifact management, and multiple programming language support.

```
                ___            __    _    _         __ __
                | | '___  ___ _| |_ | |  | |   ___ |  \  \ ___
                | |-<_> |<_-<  | |  | |_ | |_ <_> ||     |<_> |
                |_| <___|/__/  |_|  |___||___|<___||_|_|_|<___|



                                                       .+*+-.
                                                      -%#--
                                                    :=***%*++=.
                                                   :+=+**####%+
                                                   ++=+*%#
                                                  .*+++==-
                  ::--:.                           .**++=::
                 #%##*++=......                    =*+==-::
                .@@@*@%*==-==-==---:::::------::==*+==--::
                 %@@@@+--====+===---=---==+=======+++----:
                 .%@@*++*##***+===-=====++++++*++*+====++.
                 :@@%*##%@@%#*%#+==++++++=++***==-=+==+=-
                  %@%%%%%@%#+=*%*##%%%@###**++++==--==++
                  #@%%@%@@##**%@@@%#%%%%**++*++=====-=*-
                  -@@@@@@@%*#%@@@@@@@%%%%#+*%#++++++=*+.
                   +@@@@@%%*-#@@@@@@@@@@@%%@%**#*#+=-.
                    #%%###%:  ..+#%@@@@%%@@@@%#+-
                    :***#*-         ...  *@@@%*+:
                     =***=               -@%##**.
                    :#*++                -@#-:*=.
                     =##-                .%*..##
                      +*-                 *:  +-
                      :+-                :+   =.
                       =-.               *+   =-
                        :-:-              =--  :::


```
---
### Prerequisites
1. [Node](https://nodejs.org/en)
2. [pnpm](https://pnpm.io) 

### How to Run

1. Clone this branch
2. Go inside fastLLaMa
3. Install dependencies 
```
pnpm install
```
4. Run dev server
```
pnpm run dev
```

### How to Build
```
pnpm run build
```

### Contributing
* Contributors can open PRs
* Collaborators can push to branches to the repo and merge PRs into the main branch
* Collaborators will be invited based on contributions
* Any help with managing issues and PRs is very appreciated!
* Make sure to read about our [vision](https://github.com/PotatoSpudowski/fastLLaMa/discussions/46)

### Notes

* Tested on
    * Hardware: Apple silicon, Intel, Arm (Pending)
    * OS: MacOs, Linux, Windows (Pending), Android (Pending)
