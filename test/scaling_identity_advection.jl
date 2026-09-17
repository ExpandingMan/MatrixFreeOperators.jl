@testset "ScalingOp, IdentityOp, Advection" begin
    @testset "scaling by a Number" begin
        g = CartesianGrid(((0.0, 1.0),), (6,))
        u = set!(x -> x[1], scalar_field(g))
        S = scaling(2.5)
        @test collect(interior(S * u)) ≈ 2.5 .* collect(interior(u))
        @test islinear(S) && isconstant(S) && isdiagonal(S) && isselfadjoint(S)
        @test adjoint(S) === S
        @test MatrixFreeOperators.operator_grid(S) === nothing
    end

    @testset "scaling by a coefficient field" begin
        g = CartesianGrid(((0.0, 1.0),), (6,))
        κ = set!(x -> 1 + x[1]^2, scalar_field(g))
        u = set!(x -> sin(x[1]), scalar_field(g))
        S = scaling(κ)
        @test collect(interior(S * u)) ≈ collect(interior(κ)) .* collect(interior(u))
        @test isselfadjoint(S) && adjoint(S) === S
        @test MatrixFreeOperators.operator_grid(S) === g

        v = set!(x -> SVector(x[1]), vector_field(g))
        Sv = S * v
        @test getindex.(collect(interior(Sv)), 1) ≈
            collect(interior(κ)) .* getindex.(collect(interior(v)), 1)

        @test_throws ArgumentError scaling(vector_field(g))
    end

    @testset "identity_op" begin
        g = CartesianGrid(((0.0, 1.0),), (6,))
        u = set!(x -> x[1]^3, scalar_field(g))
        I = identity_op()
        @test collect(interior(I * u)) == collect(interior(u))
        @test islinear(I) && isselfadjoint(I) && isdiagonal(I)
        @test adjoint(I) === I
        z = set!(x -> 1.0, scalar_field(g))
        MatrixFreeOperators.apply!(z, I, u, g, 2.0, -1.0)
        @test collect(interior(z)) ≈ 2 .* collect(interior(u)) .- 1
    end

    @testset "prescribed advection: analytic action and convergence" begin
        function adv_error(n)
            g = CartesianGrid(
                ((0.0, 2π), (0.0, 2π)), (n, n);
                bc=((Periodic(), Periodic()), (Periodic(), Periodic())),
            )
            v = set!(x -> SVector(sin(x[2]), cos(x[1])), vector_field(g))
            u = set!(x -> sin(x[1]) * sin(x[2]), scalar_field(g))
            A = advection(g, v)
            @test islinear(A) && isconstant(A)
            y = A * u
            ref = set!(
                x ->
                    sin(x[2]) * cos(x[1]) * sin(x[2]) + cos(x[1]) * sin(x[1]) * cos(x[2]),
                scalar_field(g),
            )
            return maximum(abs, collect(interior(y)) .- collect(interior(ref)))
        end
        e32 = adv_error(32)
        e64 = adv_error(64)
        @test log2(e32 / e64) ≥ 1.9
    end

    @testset "advection validation" begin
        g = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (4, 4))
        @test_throws ArgumentError advection(g, scalar_field(g))
        g1 = CartesianGrid(((0.0, 1.0),), (4,))
        @test_throws ArgumentError advection(g, vector_field(g1))
    end

    @testset "self-advection is nonlinear and computes u·∇u" begin
        g = CartesianGrid(((0.0, 2π),), (64,); bc=((Periodic(), Periodic()),))
        A = advection(g, SelfAdvection())
        @test !islinear(A)
        @test_throws ArgumentError adjoint(A)

        u = set!(x -> SVector(sin(x[1])), vector_field(g))
        y = A * u
        ref = set!(x -> SVector(sin(x[1]) * cos(x[1])), vector_field(g))
        @test maximum(norm.(collect(interior(y)) .- collect(interior(ref)))) < 0.01

        @test_throws ArgumentError apply(A, scalar_field(g))
    end
end
