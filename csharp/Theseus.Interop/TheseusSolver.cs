using System.Text;

namespace Theseus.Interop;

/// <summary>
/// Result of an optimisation or forward solve.
/// All arrays use row-major layout: xyz[node * 3 + dim].
/// </summary>
public sealed class SolverResult
{
    public double[] Xyz { get; init; } = [];
    public double[] MemberLengths { get; init; } = [];
    public double[] MemberForces { get; init; } = [];
    public double[] ForceDensities { get; init; } = [];
    public double[] Reactions { get; init; } = [];
    public int Iterations { get; init; }
    public bool Converged { get; init; }
    public double ConstraintMaxViolation { get; init; }
}

/// <summary>
/// Managed wrapper around the native Theseus solver (theseus.dll).
///
/// Implements <see cref="IDisposable"/> to ensure the native handle is freed.
/// Typical usage:
/// <code>
/// using var solver = TheseusSolver.Create(numEdges, numNodes, numFree, ...);
/// solver.AddTargetXyz(1.0, nodeIndices, targetXyz);
/// var result = solver.Optimize();
/// </code>
/// </summary>
public sealed class TheseusSolver : IDisposable
{
    private IntPtr _handle;
    private readonly int _numNodes;
    private readonly int _numEdges;
    private bool _disposed;

    private TheseusSolver(IntPtr handle, int numNodes, int numEdges)
    {
        _handle = handle;
        _numNodes = numNodes;
        _numEdges = numEdges;
    }

    /// <summary>
    /// Retrieve the last error message from the native library.
    /// </summary>
    public static string GetLastError()
    {
        var buf = new byte[2048];
        int n = TheseusInterop.theseus_last_error(buf, (nuint)buf.Length);
        if (n <= 0) return string.Empty;
        return Encoding.UTF8.GetString(buf, 0, n);
    }

    private static void Check(int rc)
    {
        if (rc != 0)
            throw new TheseusException(GetLastError(), rc);
    }

    // ── Construction ─────────────────────────────────────────

    /// <summary>
    /// Create a new solver instance from network topology and initial data.
    /// </summary>
    /// <param name="numEdges">Number of edges (members) in the network.</param>
    /// <param name="numNodes">Total number of nodes.</param>
    /// <param name="numFree">Number of free (non-anchor) nodes.</param>
    /// <param name="cooRows">COO row indices of the incidence matrix.</param>
    /// <param name="cooCols">COO column indices of the incidence matrix.</param>
    /// <param name="cooVals">COO values (+1 / -1) of the incidence matrix.</param>
    /// <param name="freeNodeIndices">Global indices of free nodes.</param>
    /// <param name="fixedNodeIndices">Global indices of fixed (anchor) nodes.</param>
    /// <param name="loads">Flat row-major array of free-node loads (numFree x 3).</param>
    /// <param name="fixedPositions">Flat row-major array of anchor positions (numFixed x 3).</param>
    /// <param name="qInit">Initial force densities (numEdges).</param>
    /// <param name="lowerBounds">Lower bounds on q (numEdges).</param>
    /// <param name="upperBounds">Upper bounds on q (numEdges).</param>
    public static TheseusSolver Create(
        int numEdges, int numNodes, int numFree,
        int[] cooRows, int[] cooCols, double[] cooVals,
        int[] freeNodeIndices, int[] fixedNodeIndices,
        double[] loads, double[] fixedPositions,
        double[] qInit, double[] lowerBounds, double[] upperBounds)
    {
        int numFixed = fixedNodeIndices.Length;

        var handle = TheseusInterop.theseus_create(
            (nuint)numEdges, (nuint)numNodes, (nuint)numFree,
            ToNuint(cooRows), ToNuint(cooCols), cooVals, (nuint)cooRows.Length,
            ToNuint(freeNodeIndices), ToNuint(fixedNodeIndices), (nuint)numFixed,
            loads, fixedPositions,
            qInit, lowerBounds, upperBounds);

        if (handle == IntPtr.Zero)
            throw new TheseusException(GetLastError(), -1);

        return new TheseusSolver(handle, numNodes, numEdges);
    }

    // ── Objectives ───────────────────────────────────────────

    public void AddTargetXyz(double weight, int[] nodeIndices, double[] targetXyz)
    {
        Check(TheseusInterop.theseus_add_target_xyz(
            _handle, weight, ToNuint(nodeIndices), (nuint)nodeIndices.Length, targetXyz));
    }

    public void AddTargetXy(double weight, int[] nodeIndices, double[] targetXy)
    {
        Check(TheseusInterop.theseus_add_target_xy(
            _handle, weight, ToNuint(nodeIndices), (nuint)nodeIndices.Length, targetXy));
    }

    public void AddTargetLength(double weight, int[] edgeIndices, double[] targets)
    {
        Check(TheseusInterop.theseus_add_target_length(
            _handle, weight, ToNuint(edgeIndices), (nuint)edgeIndices.Length, targets));
    }

    public void AddLengthVariation(double weight, int[] edgeIndices, double sharpness)
    {
        Check(TheseusInterop.theseus_add_length_variation(
            _handle, weight, ToNuint(edgeIndices), (nuint)edgeIndices.Length, sharpness));
    }

    public void AddForceVariation(double weight, int[] edgeIndices, double sharpness)
    {
        Check(TheseusInterop.theseus_add_force_variation(
            _handle, weight, ToNuint(edgeIndices), (nuint)edgeIndices.Length, sharpness));
    }

    public void AddSumForceLength(double weight, int[] edgeIndices)
    {
        Check(TheseusInterop.theseus_add_sum_force_length(
            _handle, weight, ToNuint(edgeIndices), (nuint)edgeIndices.Length));
    }

    public void AddMinLength(double weight, int[] edgeIndices, double[] thresholds, double sharpness)
    {
        Check(TheseusInterop.theseus_add_min_length(
            _handle, weight, ToNuint(edgeIndices), (nuint)edgeIndices.Length, thresholds, sharpness));
    }

    public void AddMaxLength(double weight, int[] edgeIndices, double[] thresholds, double sharpness)
    {
        Check(TheseusInterop.theseus_add_max_length(
            _handle, weight, ToNuint(edgeIndices), (nuint)edgeIndices.Length, thresholds, sharpness));
    }

    public void AddMinForce(double weight, int[] edgeIndices, double[] thresholds, double sharpness)
    {
        Check(TheseusInterop.theseus_add_min_force(
            _handle, weight, ToNuint(edgeIndices), (nuint)edgeIndices.Length, thresholds, sharpness));
    }

    public void AddMaxForce(double weight, int[] edgeIndices, double[] thresholds, double sharpness)
    {
        Check(TheseusInterop.theseus_add_max_force(
            _handle, weight, ToNuint(edgeIndices), (nuint)edgeIndices.Length, thresholds, sharpness));
    }

    public void AddRigidSetCompare(double weight, int[] nodeIndices, double[] targetXyz)
    {
        Check(TheseusInterop.theseus_add_rigid_set_compare(
            _handle, weight, ToNuint(nodeIndices), (nuint)nodeIndices.Length, targetXyz));
    }

    public void AddReactionDirection(double weight, int[] anchorIndices, double[] targetDirs)
    {
        Check(TheseusInterop.theseus_add_reaction_direction(
            _handle, weight, ToNuint(anchorIndices), (nuint)anchorIndices.Length, targetDirs));
    }

    public void AddReactionDirectionMagnitude(double weight, int[] anchorIndices, double[] targetDirs, double[] targetMags)
    {
        Check(TheseusInterop.theseus_add_reaction_direction_magnitude(
            _handle, weight, ToNuint(anchorIndices), (nuint)anchorIndices.Length, targetDirs, targetMags));
    }

    // ── Constraints ──────────────────────────────────────────

    public void AddConstraintMaxLength(int[] edgeIndices, double[] maxLengths)
    {
        Check(TheseusInterop.theseus_add_constraint_max_length(
            _handle, ToNuint(edgeIndices), (nuint)edgeIndices.Length, maxLengths));
    }

    // ── Solver options ───────────────────────────────────────

    public void SetSolverOptions(
        int maxIterations = 500,
        double absTol = 1e-6,
        double relTol = 1e-6,
        double barrierWeight = 1000.0,
        double barrierSharpness = 10.0)
    {
        Check(TheseusInterop.theseus_set_solver_options(
            _handle, (nuint)maxIterations, absTol, relTol, barrierWeight, barrierSharpness));
    }

    // ── Solve ────────────────────────────────────────────────

    /// <summary>
    /// Run unconstrained L-BFGS optimisation.
    /// </summary>
    public SolverResult Optimize()
    {
        var xyz = new double[_numNodes * 3];
        var lengths = new double[_numEdges];
        var forces = new double[_numEdges];
        var q = new double[_numEdges];
        var reactions = new double[_numNodes * 3];
        nuint iterations = 0;
        byte converged = 0;

        Check(TheseusInterop.theseus_optimize(
            _handle, xyz, lengths, forces, q, reactions,
            ref iterations, ref converged));

        return new SolverResult
        {
            Xyz = xyz,
            MemberLengths = lengths,
            MemberForces = forces,
            ForceDensities = q,
            Reactions = reactions,
            Iterations = (int)iterations,
            Converged = converged != 0,
        };
    }

    /// <summary>
    /// Run constrained optimisation via augmented Lagrangian.
    /// </summary>
    public SolverResult OptimizeConstrained(
        double muInit = 10.0,
        double muFactor = 5.0,
        double muMax = 1e6,
        int maxOuterIters = 15,
        double constraintTol = 1e-3)
    {
        var xyz = new double[_numNodes * 3];
        var lengths = new double[_numEdges];
        var forces = new double[_numEdges];
        var q = new double[_numEdges];
        var reactions = new double[_numNodes * 3];
        nuint iterations = 0;
        byte converged = 0;
        double maxViolation = 0.0;

        Check(TheseusInterop.theseus_optimize_constrained(
            _handle,
            muInit, muFactor, muMax, (nuint)maxOuterIters, constraintTol,
            xyz, lengths, forces, q, reactions,
            ref iterations, ref converged, ref maxViolation));

        return new SolverResult
        {
            Xyz = xyz,
            MemberLengths = lengths,
            MemberForces = forces,
            ForceDensities = q,
            Reactions = reactions,
            Iterations = (int)iterations,
            Converged = converged != 0,
            ConstraintMaxViolation = maxViolation,
        };
    }

    /// <summary>
    /// Single forward FDM solve (no optimisation).
    /// </summary>
    public SolverResult SolveForward()
    {
        var xyz = new double[_numNodes * 3];
        var lengths = new double[_numEdges];
        var forces = new double[_numEdges];

        Check(TheseusInterop.theseus_solve_forward(_handle, xyz, lengths, forces));

        return new SolverResult
        {
            Xyz = xyz,
            MemberLengths = lengths,
            MemberForces = forces,
        };
    }

    // ── IDisposable ──────────────────────────────────────────

    public void Dispose()
    {
        if (!_disposed && _handle != IntPtr.Zero)
        {
            TheseusInterop.theseus_free(_handle);
            _handle = IntPtr.Zero;
            _disposed = true;
        }
    }

    // ── Helpers ──────────────────────────────────────────────

    private static nuint[] ToNuint(int[] arr)
    {
        var result = new nuint[arr.Length];
        for (int i = 0; i < arr.Length; i++)
            result[i] = (nuint)arr[i];
        return result;
    }
}

/// <summary>
/// Exception thrown when the native Theseus library returns an error.
/// </summary>
public class TheseusException : Exception
{
    public int NativeCode { get; }

    public TheseusException(string message, int nativeCode)
        : base(message)
    {
        NativeCode = nativeCode;
    }
}
