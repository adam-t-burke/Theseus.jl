[![Docs](https://img.shields.io/badge/docs-in_progress-red)](https://adam-t-burke.github.io/Theseus.jl/docs/build)
# Theseus

Theseus is a [Julia](https://julialang.org/) package for form-finding and optimization of tensile structures, compression structures, and mixed tension/compression structures. It is a companion package to [Ariadne](https://github.com/fibrous-tendencies/Ariadne). Theseus starts a local server and can communicate with Ariadne via WebSockets.

## Installation

Installation of Theseus requires the Julia REPL which can be downloaded from [here](https://julialang.org/downloads/). Theseus was developed using Julia 1.9.3. Theseus is not registered in the Julia package registry so you will need to install it using the directions below. Even if you do not plan on using the Julia REPL for anything else it is generally recommended that you create a new environment. This is similar to environments when using conda for Python development. On Windows you may need to run the Julia REPL.

To create a new environment in Julia, open the REPL that you just installed. Enter package mode by typing:

```julia
]
```

This should display the current environment name. By default this will say something like @v1.9 if you have downloaded Julia v1.9.XX. Once you are in this mode, enter the following:

```julia
activate Theseus
```

Press enter to execute this command. This will create a new environment with the name Theseus. You will need to navigate back into this environment every time you start up Theseus. It doesn't need to be named Theseus, it could be named anything you like, but here I name is with the same name as the package for clarity.

To actually install the package type:

```julia
add https://github.com/adam-t-burke/Theseus.jl
```

Press enter to execute this command. Once all of the dependencies have been installed you can press backspace to exit pkg mode. The word julia should reappear.

## Usage

To use Theseus you always need to activate the environment where it has been installed. If you are just following the steps for installation up to here then you should already be in the correct environment. If you are re-starting Theseus you will need to re-activate the environment every time you want to use it. To activate the environment again just follow the same instructions listed in Installation.

There is only one functiona available to call with Theseus. To use this function first type:

```julia
using Theseus
```

Then press enter.

> [!NOTE]
> Capitalization is important here.

Finally, to start the server type:

```julia
start!()
```

Then press enter. A message that the server has opened should appear. Theseus is now running, you can go back into Grasshopper and set up your networks for form-finding and optimization with Ariadne.
