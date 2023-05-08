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

### How to run

#### Running Completely Locally 

1. Clone the webUI from the [webui branch](https://github.com/PotatoSpudowski/fastLLaMa/tree/webui) and follow the steps given in the branch's readme.
2. Clone the repo from the [websocket-server](https://github.com/PotatoSpudowski/fastLLaMa/tree/websocket-server)
3. Go inside the folder and install requirements using pip
4. run the server 
```
uvicorn src.main:app 
```

#### Running through hosted UI

Incase you want to use fastLLaMa using the hosted webUI, You need to satisfy browser security requirements (SSL certificates etc)

A simple way to bypass this would be using [NGROK](https://ngrok.com).

1. Follow the same steps to get the fastLLaMa server running as given previously.  
2. Create a reverse proxy
```
ngrok http 8000 
```
3. Click the connect button on the webUI and provide the URL to connect to the backend server 

Example
```
wss://1c89-49-207-197-248.in.ngrok.io/ws
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