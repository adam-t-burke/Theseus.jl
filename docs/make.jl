push!(LOAD_PATH, "../src/")  # Include the source directory
include("copy_deps.jl")

using Pkg

# create or update docs/Project.toml (copy_deps.jl writes docs/Project.toml)
include(joinpath(@__DIR__, "copy_deps.jl"))

# activate the docs environment (docs/Project.toml in this dir)
Pkg.activate(@__DIR__)

# ensure Documenter is present in the docs env
try
    Pkg.add("Documenter")
catch
    # proceed; instantiate() below will surface issues
end

# pick concrete versions for deps (populate Manifest)
Pkg.resolve()

# make the local package available in the docs env (adds Theseus as a path dep)
try
    Pkg.develop(path="..")
catch
    # ignore if already present
end

# install / instantiate the environment
Pkg.instantiate()

# build docs
using Documenter
using Theseus

makedocs(
    sitename = "Theseus",
    modules = [Theseus],
    repo = "https://github.com/adam-t-burke/Theseus.jl",
    pages = [
        "Home" => "index.md",
        "API"  => "src/api.md",
    ],
    # optional: set the output directory (default is "build")
    # builddir = joinpath(@__DIR__, "build"),
)
