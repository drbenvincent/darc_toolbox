# dev stuff

The `/dev` folder is just development code and is not intended for experimenter's eyes ;) 

## UML diagram

I can get a _really_ nice UML class diagram. From the command line, based in the root of the repo, the following command creates exactly what I want. 

    pyreverse -o pdf -A -p bad darc

I did have to do `conda install pygraphviz` once to installe