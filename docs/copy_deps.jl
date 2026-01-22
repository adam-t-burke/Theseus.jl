using TOML, UUIDs

main_project_path = joinpath(dirname(@__DIR__), "Project.toml")
docs_project_path = joinpath(@__DIR__, "Project.toml")

main = TOML.parsefile(main_project_path)

# preserve existing docs uuid if present
existing = isfile(docs_project_path) ? TOML.parsefile(docs_project_path) : Dict()
uuid_str = get(existing, "uuid", string(UUIDs.uuid4()))

docs_toml = Dict(
    "deps" => get(main, "deps", Dict()),
    "compat" => get(main, "compat", Dict())
)

buf = IOBuffer()
TOML.print(buf, docs_toml)
toml_text = String(take!(buf))

open(docs_project_path, "w") do io
    write(io, toml_text)
end

println("Wrote docs Project.toml -> ", docs_project_path)