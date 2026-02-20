using System;
using Theseus.Interop;

/// <summary>
/// Smoke test that replicates the Rust integration test "optimize_target_xyz"
/// using the 7-node / 8-edge arch network.  Validates end-to-end P/Invoke
/// interop with theseus.dll.
/// </summary>
class Program
{
    static int Main()
    {
        int passed = 0;
        int failed = 0;

        Run("ForwardSolve",        TestForwardSolve,       ref passed, ref failed);
        Run("OptimizeTargetXyz",   TestOptimizeTargetXyz,  ref passed, ref failed);
        Run("CombinedObjectives",  TestCombinedObjectives, ref passed, ref failed);
        Run("ConstrainedOptimize", TestConstrainedOptimize,ref passed, ref failed);

        Console.WriteLine($"\n{passed + failed} tests: {passed} passed, {failed} failed");
        return failed == 0 ? 0 : 1;
    }

    static void Run(string name, Action test, ref int passed, ref int failed)
    {
        try
        {
            test();
            Console.WriteLine($"  PASS  {name}");
            passed++;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"  FAIL  {name}: {ex.Message}");
            failed++;
        }
    }

    // ── Arch network builder (mirrors Rust integration tests) ──

    static TheseusSolver CreateArch(double lowerBound = 0.1, double upperBound = 100.0)
    {
        int numNodes = 7;
        int numEdges = 8;
        int numFree = 5;

        int[][] edges = [
            [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6],
            [1, 5], [2, 4],
        ];

        var cooRows = new int[numEdges * 2];
        var cooCols = new int[numEdges * 2];
        var cooVals = new double[numEdges * 2];
        for (int e = 0; e < numEdges; e++)
        {
            cooRows[e * 2] = e; cooCols[e * 2] = edges[e][0]; cooVals[e * 2] = -1.0;
            cooRows[e * 2 + 1] = e; cooCols[e * 2 + 1] = edges[e][1]; cooVals[e * 2 + 1] = 1.0;
        }

        int[] freeIdx = [1, 2, 3, 4, 5];
        int[] fixedIdx = [0, 6];

        double[] loads = [
            0, 0, -1,
            0, 0, -1,
            0, 0, -2,
            0, 0, -1,
            0, 0, -1,
        ];

        double[] fixedPos = [0, 0, 0, 6, 0, 0];

        var qInit = new double[numEdges];
        var lower = new double[numEdges];
        var upper = new double[numEdges];
        for (int i = 0; i < numEdges; i++)
        {
            qInit[i] = 1.0;
            lower[i] = lowerBound;
            upper[i] = upperBound;
        }

        return TheseusSolver.Create(
            numEdges, numNodes, numFree,
            cooRows, cooCols, cooVals,
            freeIdx, fixedIdx,
            loads, fixedPos,
            qInit, lower, upper);
    }

    // ── Tests ────────────────────────────────────────────────

    static void TestForwardSolve()
    {
        using var solver = CreateArch();
        var result = solver.SolveForward();

        Assert(result.Xyz.Length == 7 * 3, "xyz length");

        // Anchors preserved
        Assert(Math.Abs(result.Xyz[0 * 3 + 0] - 0.0) < 1e-10, "anchor 0 x");
        Assert(Math.Abs(result.Xyz[6 * 3 + 0] - 6.0) < 1e-10, "anchor 6 x");

        // All positions finite
        foreach (var v in result.Xyz)
            Assert(double.IsFinite(v), $"non-finite xyz: {v}");

        // All lengths positive
        foreach (var l in result.MemberLengths)
            Assert(l > 0 && double.IsFinite(l), $"bad length: {l}");
    }

    static void TestOptimizeTargetXyz()
    {
        using var solver = CreateArch();

        int[] targetNodes = [1, 2, 3, 4, 5];
        double[] targetXyz = [
            1, 0, 1,
            2, 0, 2,
            3, 0, 2.5,
            4, 0, 2,
            5, 0, 1,
        ];

        solver.AddTargetXyz(1.0, targetNodes, targetXyz);
        solver.SetSolverOptions(maxIterations: 200);

        var result = solver.Optimize();

        Assert(result.Iterations > 0, "should run at least 1 iteration");

        // Positions should be close to target
        double totalError = 0;
        for (int i = 0; i < targetNodes.Length; i++)
        {
            int node = targetNodes[i];
            for (int d = 0; d < 3; d++)
            {
                double diff = result.Xyz[node * 3 + d] - targetXyz[i * 3 + d];
                totalError += diff * diff;
            }
        }
        Assert(totalError < 20.0, $"total squared error = {totalError:F4}, expected < 20.0");

        // All geometry finite
        foreach (var l in result.MemberLengths)
            Assert(l > 0 && double.IsFinite(l), $"bad length: {l}");
        foreach (var q in result.ForceDensities)
            Assert(q > 0 && double.IsFinite(q), $"bad q: {q}");
    }

    static void TestCombinedObjectives()
    {
        using var solver = CreateArch();

        int[] targetNodes = [1, 2, 3, 4, 5];
        double[] targetXyz = [
            1, 0, 0.8,
            2, 0, 1.5,
            3, 0, 2.0,
            4, 0, 1.5,
            5, 0, 0.8,
        ];

        solver.AddTargetXyz(1.0, targetNodes, targetXyz);

        int[] allEdges = [0, 1, 2, 3, 4, 5, 6, 7];
        solver.AddLengthVariation(0.5, allEdges, 20.0);
        solver.AddSumForceLength(0.01, allEdges);
        solver.SetSolverOptions(maxIterations: 200);

        var result = solver.Optimize();

        Assert(result.Iterations > 0, "should run at least 1 iteration");
        foreach (var l in result.MemberLengths)
            Assert(l > 0 && double.IsFinite(l), $"bad length: {l}");
    }

    static void TestConstrainedOptimize()
    {
        using var solver = CreateArch();

        int[] targetNodes = [1, 2, 3, 4, 5];
        double[] targetXyz = [
            1, 0, 1,
            2, 0, 2,
            3, 0, 2.5,
            4, 0, 2,
            5, 0, 1,
        ];

        solver.AddTargetXyz(1.0, targetNodes, targetXyz);

        int[] allEdges = [0, 1, 2, 3, 4, 5, 6, 7];
        double[] maxLens = [2, 2, 2, 2, 2, 2, 2, 2];
        solver.AddConstraintMaxLength(allEdges, maxLens);
        solver.SetSolverOptions(maxIterations: 200);

        var result = solver.OptimizeConstrained();

        Assert(result.Iterations > 0, "should run iterations");
        for (int k = 0; k < result.MemberLengths.Length; k++)
            Assert(result.MemberLengths[k] <= 2.0 + 0.01,
                $"edge {k}: length={result.MemberLengths[k]:F4} > 2.01");
        Assert(result.ConstraintMaxViolation < 0.1,
            $"violation={result.ConstraintMaxViolation}");
    }

    static void Assert(bool condition, string message)
    {
        if (!condition) throw new Exception($"Assertion failed: {message}");
    }
}
