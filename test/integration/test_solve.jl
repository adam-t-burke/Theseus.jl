using Theseus
using JSON3
using SparseArrays
using Optim

# Mimicking the expected format from Ariadne
problem_json = """
{
    "Network": {
        "Graph": {"Ne": 2, "Nn": 3},
        "FreeNodes": [0],
        "FixedNodes": [1, 2]
    },
    "I": [0, 0, 1, 1],
    "J": [0, 1, 0, 2],
    "V": [1, -1, 1, -1],
    "XYZf": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    "P": [[0.0, 0.0, -1.0]],
    "LoadNodes": [0],
    "Q": [1.0, 1.0],
    "Parameters": {
        "Objectives": [
            {
                "Type": "TargetXYZ",
                "OBJID": 1,
                "Weight": 10.0,
                "Indices": [0],
                "Points": [[0.5, 0.5, -0.5]]
            }
        ],
        "Bounds": {
            "Min": [0.1, 0.1],
            "Max": [10.0, 10.0]
        },
        "Solver": {
            "MaxIterations": 100,
            "ShowProgress": true
        }
    }
}
"""

println("Parsing JSON message...")
msg = JSON3.read(problem_json)

println("Building problem...")
problem, state = Theseus.build_problem(msg)

println("Initial free node position:")
println(problem.context.xyz_free)

println("Running optimization...")
result, snapshot = Theseus.optimize_problem!(problem, state)

println("\n--- Results ---")
println("Final Q: ", state.force_densities)
println("Final Free Node Pos: ", snapshot.xyz_free)
println("Final Loss: ", Optim.minimum(result))
println("Iterations: ", Optim.iterations(result))
println("Converged: ", Optim.converged(result))
