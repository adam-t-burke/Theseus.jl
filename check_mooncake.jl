using Mooncake
println("Names in Mooncake:")
for n in names(Mooncake)
    if contains(lowercase(string(n)), "rule") || contains(lowercase(string(n)), "grad")
        println(n)
    end
end
