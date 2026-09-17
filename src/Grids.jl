#--------------------------------------------------------------------------------# Grids

"""
    AbstractGrid{N}

Supertype for all `N`-dimensional grids. Operators are authored once against the
grid interface (`spacing`, `interior`, [`halo_update!`](@ref), …); single-device,
distributed, and adaptive grids differ only in what the grid object is and what
`halo_update!` does.
"""
abstract type AbstractGrid{N} end

"""
    CartesianGrid(extent, ncells; bc, halo, device)

Uniform cell-centered Cartesian grid.

# Arguments
- `extent::NTuple{N,Tuple{T,T}}`: physical `(min, max)` per dimension
- `ncells::NTuple{N,Int}`: interior cell counts per dimension

# Keyword Arguments
- `bc`: per-dimension `(low, high)` boundary-condition pairs (default: homogeneous
  [`Dirichlet`](@ref) on every face)
- `halo`: ghost-layer width per dimension (default: 1 per dimension)
- `device`: KernelAbstractions backend used for field allocation (default: `CPU()`)

Cell spacing is derived from `extent` and `ncells`. Cell centers sit at
`min + (i - 1/2)Δ` for interior cell `i`.

### Examples

```julia
g = CartesianGrid(((0.0, 1.0),), (64,))                       # 1-D, Dirichlet
g = CartesianGrid(((0.0, 1.0), (0.0, 2.0)), (32, 64);
                  bc = ((Periodic(), Periodic()),
                        (Neumann(), Neumann())))              # 2-D, mixed BCs
```

See also: [`spacing`](@ref), [`interior`](@ref), [`boundary_conditions`](@ref).
"""
struct CartesianGrid{N,T,BC<:Tuple,Dev,Topo} <: AbstractGrid{N}
    extent::NTuple{N,Tuple{T,T}}
    spacing::NTuple{N,T}
    size::NTuple{N,Int}
    halo::NTuple{N,Int}
    bc::BC
    device::Dev
    local_range::NTuple{N,UnitRange{Int}}
    topology::Topo
end

function CartesianGrid(
    extent::NTuple{N,Tuple{T,T}},
    ncells::NTuple{N,Int};
    bc::Tuple=ntuple(_ -> (Dirichlet(), Dirichlet()), Val(N)),
    halo::NTuple{N,Int}=ntuple(_ -> 1, Val(N)),
    device=KernelAbstractions.CPU(),
) where {N,T<:Real}
    all(>=(1), ncells) || throw(ArgumentError("ncells must be ≥ 1 per dimension, got $ncells"))
    all(>=(1), halo) || throw(ArgumentError("halo width must be ≥ 1 per dimension, got $halo"))
    for d in 1:N
        lo, hi = extent[d]
        lo < hi || throw(ArgumentError("extent must satisfy min < max in dimension $d, got ($lo, $hi)"))
    end
    _validate_bc(bc, Val(N))
    grid_spacing = ntuple(d -> (extent[d][2] - extent[d][1]) / ncells[d], Val(N))
    local_range = ntuple(d -> 1:ncells[d], Val(N))
    return CartesianGrid{N,T,typeof(bc),typeof(device),Nothing}(
        extent, grid_spacing, ncells, halo, bc, device, local_range, nothing
    )
end

function _validate_bc(bc::Tuple, ::Val{N}) where {N}
    length(bc) == N || throw(ArgumentError("bc must provide one (low, high) pair per dimension"))
    for d in 1:N
        pair = bc[d]
        pair isa Tuple{AbstractBC,AbstractBC} ||
            throw(ArgumentError("bc[$d] must be a (low, high) pair of AbstractBC, got $(typeof(pair))"))
        if (pair[1] isa Periodic) != (pair[2] isa Periodic)
            throw(ArgumentError("Periodic boundary conditions must be paired on both faces of dimension $d"))
        end
    end
    return nothing
end

AbstractGrid(g::AbstractGrid) = g

#--------------------------------------------------------------------------------# Grid interface

"""
    dimension(g::AbstractGrid) -> Int

Spatial dimension of the grid.
"""
dimension(::AbstractGrid{N}) where {N} = N

"""
    spacing(g::AbstractGrid) -> NTuple{N}

Cell spacing per dimension.
"""
spacing(g::CartesianGrid) = g.spacing

"""
    local_size(g::AbstractGrid) -> NTuple{N,Int}

Interior (owned, non-halo) cell counts per dimension.
"""
local_size(g::CartesianGrid) = g.size

"""
    halo_width(g::AbstractGrid) -> NTuple{N,Int}

Ghost-layer width per dimension.
"""
halo_width(g::CartesianGrid) = g.halo

"""
    boundary_conditions(g::AbstractGrid) -> NTuple{N,Tuple}

Per-dimension `(low, high)` boundary-condition pairs.
"""
boundary_conditions(g::CartesianGrid) = g.bc

"""
    interior(g::AbstractGrid) -> CartesianIndices

Indices of the interior (owned, non-halo) cells in halo-padded index space.
"""
function interior(g::AbstractGrid{N}) where {N}
    return CartesianIndices(
        ntuple(d -> (halo_width(g)[d] + 1):(halo_width(g)[d] + local_size(g)[d]), Val(N))
    )
end

"""
    padded_size(g::AbstractGrid) -> NTuple{N,Int}

Array size per dimension including ghost layers on both faces.
"""
padded_size(g::AbstractGrid{N}) where {N} =
    ntuple(d -> local_size(g)[d] + 2 * halo_width(g)[d], Val(N))

"""
    cell_center(g::AbstractGrid, I::CartesianIndex) -> SVector

Physical coordinates of the center of cell `I` (in halo-padded index space).

Evaluated at the *global* cell index `local_range` reports, against the global
`extent` origin — so a [`partition_grid`](@ref) slab and the grid it was cut from
return **bitwise identical** coordinates for the same cell. Recomputing from a
slab-local origin instead would round twice and drift by an ulp, which is enough
to make a coordinate-assembled right-hand side depend on the partition count.
For every undistributed grid and every [`BlockForest`](@ref) leaf grid
`first(local_range[d]) == 1`, so this is bit-for-bit the plain formula
`min + (i - 1/2)Δ`.
"""
function cell_center(g::CartesianGrid{N,T}, I::CartesianIndex{N}) where {N,T}
    return SVector(
        ntuple(Val(N)) do d
            z = first(g.local_range[d]) - 1 + I[d] - g.halo[d]   # global cell index
            g.extent[d][1] + (T(z) - T(0.5)) * g.spacing[d]
        end
    )
end

"""
    coarsen(g::CartesianGrid) -> CartesianGrid

The next-coarser grid in a 2:1 multigrid hierarchy: half the cells per
dimension over the same extent, so the spacing doubles. Boundary conditions,
halo width, and device carry over verbatim. Requires an even cell count in
every dimension.

### Examples

```julia
g = CartesianGrid(((0.0, 1.0), (0.0, 1.0)), (64, 64))
gc = coarsen(g)          # 32×32, spacing doubled
```
"""
function coarsen(g::CartesianGrid{N}) where {N}
    all(iseven, local_size(g)) || throw(
        ArgumentError(
            "coarsen requires an even cell count in every dimension, got $(local_size(g))"
        ),
    )
    g.topology === nothing ||
        throw(ArgumentError("coarsen does not support distributed grids yet"))
    return CartesianGrid(
        g.extent, ntuple(d -> g.size[d] >> 1, Val(N)); bc=g.bc, halo=g.halo, device=g.device
    )
end

KernelAbstractions.get_backend(g::CartesianGrid) = g.device

_inv_spacing(g::AbstractGrid{N}) where {N} = ntuple(d -> inv(spacing(g)[d]), Val(N))
_inv_spacing2(g::AbstractGrid{N}) where {N} = ntuple(d -> inv(spacing(g)[d]^2), Val(N))

"""
    halo_update!(x, g::AbstractGrid) -> x

Fill ghost layers with neighbor data. The single distributed seam: a no-op on
single-device grids; distributed grids overload it to exchange halos. Operators
call this before any stencil that reads neighbor cells.
"""
halo_update!(x, ::AbstractGrid) = x

nleaves(::CartesianGrid) = 1
function leaf_grid(g::CartesianGrid, j::Integer)
    j == 1 || throw(ArgumentError("CartesianGrid only has a single leaf"))
    g
end

function Adapt.adapt_structure(to, g::CartesianGrid{N}) where {N}
    device = KernelAbstractions.get_backend(Adapt.adapt(to, similar(Vector{Bool}, 0)))
    return CartesianGrid{N,eltype(g.spacing),typeof(g.bc),typeof(device),typeof(g.topology)}(
        g.extent, g.spacing, g.size, g.halo, g.bc, device, g.local_range, g.topology
    )
end


@inline function _grid_mismatch(a::CartesianGrid{N}, b::CartesianGrid{N}) where {N}
    a === b && return nothing
    a.size == b.size || return :size
    a.halo == b.halo || return :halo
    a.spacing == b.spacing || return :spacing
    a.extent == b.extent || return :extent
    a.bc === b.bc || return :bc
    a.local_range == b.local_range || return :local_range
    a.topology === b.topology || return :topology
    return nothing
end

# Different grid types, or the same type in different dimensions: never compatible.
_grid_mismatch(::AbstractGrid, ::AbstractGrid) = :type

@inline function _layout_mismatch(a::CartesianGrid{N}, b::CartesianGrid{N}) where {N}
    a === b && return nothing
    a.size == b.size || return :size
    a.halo == b.halo || return :halo
    return nothing
end

_layout_mismatch(::AbstractGrid, ::AbstractGrid) = :type

"""
    a::AbstractGrid == b::AbstractGrid -> Bool

Whether `a` and `b` describe the same discretization: equal cell counts, halo,
spacing, extent, boundary conditions, ownership (`local_range`) and distributed
topology for [`CartesianGrid`](@ref)s; equal block size, halo, root spacing,
extent, physical boundary conditions and forest topology (root tiling,
periodicity, `maxlevel`, leaf set) for [`BlockForest`](@ref)s. Grids of
different types or dimensions are never equal. The `device` is deliberately
**not** compared: a host grid and its `Adapt`-ed twin are the same
discretization.

`a === b` short-circuits, so the common case — two fields allocated from one
grid object — costs a single comparison. `hash` is not specialized to match, so
grids that are `==` but distinct objects are not interchangeable `Dict` keys.

See also: [`same_layout`](@ref), [`check_compatible`](@ref).
"""
Base.:(==)(a::AbstractGrid, b::AbstractGrid) = isnothing(_grid_mismatch(a, b))

@inline function _check_pair(what, a, b, level::String)
    what === nothing || _throw_incompatible(what, a, b, level)
    return nothing
end

# Grid-level twins of the field checks; documented with them in Fields.jl.
# Tuple recursion, not `foreach` + closure — see the note there.
@inline check_compatible(::AbstractGrid) = nothing
@inline function check_compatible(a::AbstractGrid, b::AbstractGrid, rest::AbstractGrid...)
    _check_pair(_grid_mismatch(a, b), a, b, "grid")
    return check_compatible(a, rest...)
end
@inline check_layout(::AbstractGrid) = nothing
@inline function check_layout(a::AbstractGrid, b::AbstractGrid, rest::AbstractGrid...)
    _check_pair(_layout_mismatch(a, b), a, b, "layout")
    return check_layout(a, rest...)
end

"""
    same_layout(a::AbstractGrid, b::AbstractGrid) -> Bool

Whether fields on `a` and `b` have the same padded storage shape, so that a
pointwise broadcast between them (block by block for forests) is legal. Weaker
than grid `==`: says nothing about geometry or boundary conditions.
The right check for kernels that only ever combine values cell by cell.

See also: [`check_layout`](@ref), [`check_compatible`](@ref).
"""
same_layout(a::AbstractGrid, b::AbstractGrid) = isnothing(_layout_mismatch(a, b))


# Why each property matters, for an error that names the consequence rather than
# just the field name.
const _MISMATCH_REASONS = Dict{Symbol,String}(
    :type => "they are different grid types or dimensions",
    :size => "their interior cell counts differ, so the padded arrays cannot be broadcast together",
    :halo => "their ghost-layer widths differ, so the same padded index means different cells",
    :spacing => "their cell spacings differ",
    :extent => "they cover different physical domains",
    :bc => "their boundary conditions differ, so ghost cells would mean different things",
    :local_range => "they own different global index ranges (a slab vs. the grid it was cut from?)",
    :topology => "they belong to different distributed topologies",
    :blocksize => "their block sizes differ",
    :spacing0 => "their root-level spacings differ",
    :nroot => "their root tilings differ",
    :periodic => "their periodicity differs",
    :maxlevel => "their maximum refinement levels differ",
    :leaves => "their leaf sets differ (same domain, different refinement)",
    :nleaves => "they have different numbers of leaf blocks",
    :stale => "a forest field was allocated before its forest was regridded; allocate a fresh field on the current forest",
)

@noinline function _throw_incompatible(what::Symbol, a, b, level::String)
    reason = get(_MISMATCH_REASONS, what, "they differ in $what")
    throw(
        ArgumentError(
            "incompatible $level: $reason. Got " *
            "$(summary(AbstractGrid(a))) vs $(summary(AbstractGrid(b))) [$what]",
        ),
    )
end


