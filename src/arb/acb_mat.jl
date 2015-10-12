###############################################################################
#
#   acb_mat.jl : acb_mat matrices
#
#   Copyright (C) 2015 Tommy Hofmann
#
###############################################################################

export parent, elem_type, prec, base_ring, cols, rows, deepcopy, getindex!,
       setindex!, one, zero, show, strongequal, overlaps, contains, issquare,
       transpose, bound_inf_norm, -, +, *, //,  swap_rows!, lufact!, lufact,
       solve, solve!, solve_lu_precomp, solve_lu_precomp!, inv, det, charpoly, 
       exp, add!, mul!, sub!, call, MatrixSpace

###############################################################################
#
#   Basic manipulation
#
###############################################################################

parent(a::acb_mat) = a.parent

elem_type(a::AcbMatSpace) = acb_mat

prec(a::AcbMatSpace) = prec(a.base_ring)

base_ring(a::AcbMatSpace) = a.base_ring

base_ring(a::acb_mat) = base_ring(parent(a))

cols(x::acb_mat) = x.c

rows(x::acb_mat) = x.r

function deepcopy(x::acb_mat)
  z = parent(x)()
  ccall((:acb_mat_set, :libarb), Void, (Ptr{acb_mat}, Ptr{acb_mat}), &z, &x)
  return z
end

function getindex!(z::acb, x::acb_mat, r::Int, c::Int)
  v = ccall((:acb_mat_entry_ptr, :libarb), Ptr{acb},
              (Ptr{acb_mat}, Int, Int), &x, r - 1, c - 1)
  ccall((:acb_set, :libarb), Void, (Ptr{acb}, Ptr{acb}), &z, v)
  return z
end

function getindex(x::acb_mat, r::Int, c::Int)
  _checkbounds(rows(x), r) || throw(BoundsError())
  _checkbounds(cols(x), c) || throw(BoundsError())

  z = base_ring(x)()
  v = ccall((:acb_mat_entry_ptr, :libarb), Ptr{acb},
              (Ptr{acb_mat}, Int, Int), &x, r - 1, c - 1)
  ccall((:acb_set, :libarb), Void, (Ptr{acb}, Ptr{acb}), &z, v)
  return z
end

function setindex!(x::acb_mat,
                   y::Union{Int, UInt, Float64, fmpz, fmpq, arb, BigFloat,
                            acb, AbstractString},
                   r::Int, c::Int)
  _checkbounds(rows(x), r) || throw(BoundsError())
  _checkbounds(cols(x), c) || throw(BoundsError())

  z = ccall((:acb_mat_entry_ptr, :libarb), Ptr{acb},
              (Ptr{acb_mat}, Int, Int), &x, r - 1, c - 1)
  _acb_set(z, y, prec(base_ring(x)))
end

function setindex!{T <: Union{Int, UInt, Float64, fmpz, fmpq, arb, BigFloat,
                              AbstractString}}(x::acb_mat, y::Tuple{T, T},
                              r::Int, c::Int)
  _checkbounds(rows(x), r) || throw(BoundsError())
  _checkbounds(cols(x), c) || throw(BoundsError())

  z = ccall((:acb_mat_entry_ptr, :libarb), Ptr{acb},
              (Ptr{acb_mat}, Int, Int), &x, r - 1, c - 1)
  _acb_set(z, y[1], y[2], prec(base_ring(x)))
end

function one(x::AcbMatSpace)
  z = x()
  ccall((:acb_mat_one, :libarb), Void, (Ptr{acb_mat}, ), &z)
  return z
end

function zero(x::AcbMatSpace)
  z = x()
  ccall((:acb_mat_zero, :libarb), Void, (Ptr{acb_mat}, ), &z)
  return z
end

################################################################################
#
#  String I/O
#
################################################################################

function show(io::IO, a::AcbMatSpace)
   print(io, "Matrix Space of ")
   print(io, a.rows, " rows and ", a.cols, " columns over ")
   print(io, base_ring(a))
end

function show(io::IO, a::acb_mat)
   rows = a.parent.rows
   cols = a.parent.cols
   for i = 1:rows
      print(io, "[")
      for j = 1:cols
         print(io, a[i, j])
         if j != cols
            print(io, " ")
         end
      end
      print(io, "]")
      if i != rows
         println(io, "")
      end
   end
end

################################################################################
#
#  Comparison
#
################################################################################

function ==(x::acb_mat, y::acb_mat)
  check_parent(x, y)
  r = ccall((:acb_mat_eq, :libarb), Cint, (Ptr{acb_mat}, Ptr{acb_mat}), &x, &y)
  return Bool(r)
end

function !=(x::acb_mat, y::acb_mat)
  r = ccall((:acb_mat_ne, :libarb), Cint, (Ptr{acb_mat}, Ptr{acb_mat}), &x, &y)
  return Bool(r)
end

function strongequal(x::acb_mat, y::acb_mat)
  r = ccall((:acb_mat_equal, :libarb), Cint,
              (Ptr{acb_mat}, Ptr{acb_mat}), &x, &y)
  return Bool(r)
end

function overlaps(x::acb_mat, y::acb_mat)
  r = ccall((:acb_mat_overlaps, :libarb), Cint,
              (Ptr{acb_mat}, Ptr{acb_mat}), &x, &y)
  return Bool(r)
end

function contains(x::acb_mat, y::acb_mat)
  r = ccall((:acb_mat_contains, :libarb), Cint,
              (Ptr{acb_mat}, Ptr{acb_mat}), &x, &y)
  return Bool(r)
end

function contains(x::acb_mat, y::fmpz_mat)
  r = ccall((:acb_mat_contains_fmpz_mat, :libarb), Cint,
              (Ptr{acb_mat}, Ptr{fmpz_mat}), &x, &y)
  return Bool(r)
end

#function contains(x::acb_mat, y::fmpq_mat)
#  r = ccall((:acb_mat_contains_fmpq_mat, :libarb), Cint,
#              (Ptr{acb_mat}, Ptr{acb_mat}))
#  return Bool(r)
#end

==(x::acb_mat, y::fmpz_mat) = x == parent(x)(y)

==(x::fmpz_mat, y::acb_mat) = y == x

==(x::acb_mat, y::arb_mat) = x == parent(x)(y)

==(x::arb_mat, y::acb_mat) = y == x

################################################################################
#
#  Predicates
#
################################################################################

issquare(x::acb_mat) = cols(x) == rows(x)

isreal(x::acb_mat) =
            Bool(ccall((:acb_mat_is_real, :libarb), Cint, (Ptr{acb_mat}, ), &x))

################################################################################
#
#  Transpose
#
################################################################################

function transpose(x::acb_mat)
  z = MatrixSpace(base_ring(x), cols(x), rows(x))()
  ccall((:acb_mat_transpose, :libarb), Void,
              (Ptr{acb_mat}, Ptr{acb_mat}), &z, &x)
  return z
end

################################################################################
#
#  Norm
#
################################################################################

function bound_inf_norm(x::acb_mat)
  z = arb()
  t = ccall((:arb_rad_ptr, :libarb), Ptr{mag_struct}, (Ptr{arb}, ), &z)
  ccall((:acb_mat_bound_inf_norm, :libarb), Void,
              (Ptr{mag_struct}, Ptr{acb_mat}), t, &x)
  s = ccall((:arb_mid_ptr, :libarb), Ptr{arf_struct}, (Ptr{arb}, ), &z)
  ccall((:arf_set_mag, :libarb), Void,
              (Ptr{arf_struct}, Ptr{mag_struct}), s, t)
  ccall((:mag_zero, :libarb), Void,
              (Ptr{mag_struct},), t)
  return ArbField(prec(parent(x)))(z)
end

################################################################################
#
#  Unary operations
#
################################################################################

function -(x::acb_mat)
  z = parent(x)()
  ccall((:acb_mat_neg, :libarb), Void, (Ptr{acb_mat}, Ptr{acb_mat}), &z, &x)
  return z
end

################################################################################
#
#  Binary operations
#
################################################################################

function +(x::acb_mat, y::acb_mat)
  check_parent(x, y)
  z = parent(x)()
  ccall((:acb_mat_add, :libarb), Void,
              (Ptr{acb_mat}, Ptr{acb_mat}, Ptr{acb_mat}, Int),
              &z, &x, &y, prec(parent(x)))
  return z
end

function -(x::acb_mat, y::acb_mat)
  check_parent(x, y)
  z = parent(x)()
  ccall((:acb_mat_sub, :libarb), Void,
              (Ptr{acb_mat}, Ptr{acb_mat}, Ptr{acb_mat}, Int),
              &z, &x, &y, prec(parent(x)))
  return z
end

function *(x::acb_mat, y::acb_mat)
  check_parent(x, y)
  cols(x) != rows(y) && error("Matrices have wrong  dimensions")
  z = MatrixSpace(base_ring(x), rows(x), cols(y))()
  ccall((:acb_mat_mul, :libarb), Void,
              (Ptr{acb_mat}, Ptr{acb_mat}, Ptr{acb_mat}, Int),
              &z, &x, &y, prec(parent(x)))
  return z
end

function ^(x::acb_mat, y::UInt)
  rows(x) != cols(x) && error("Matrix must be square")
  z = parent(x)()
  ccall((:acb_mat_pow_ui, :libarb), Void,
              (Ptr{acb_mat}, Ptr{acb_mat}, UInt, Int),
              &z, &x, y, prec(parent(x)))
  return z
end

function *(x::acb_mat, y::Int)
  z = parent(x)()
  ccall((:acb_mat_scalar_mul_si, :libarb), Void,
              (Ptr{acb_mat}, Ptr{acb_mat}, Int, Int),
              &z, &x, y, prec(parent(x)))
  return z
end

*(x::Int, y::acb_mat) = y*x

function *(x::acb_mat, y::fmpz)
  z = parent(x)()
  ccall((:acb_mat_scalar_mul_fmpz, :libarb), Void,
              (Ptr{acb_mat}, Ptr{acb_mat}, Ptr{fmpz}, Int),
              &z, &x, &y, prec(parent(x)))
  return z
end

*(x::fmpz, y::acb_mat) = y*x

function *(x::acb_mat, y::arb)
  z = parent(x)()
  ccall((:acb_mat_scalar_mul_arb, :libarb), Void,
              (Ptr{acb_mat}, Ptr{acb_mat}, Ptr{arb}, Int),
              &z, &x, &y, prec(parent(x)))
  return z
end

*(x::arb, y::acb_mat) = y*x

function *(x::acb_mat, y::acb)
  z = parent(x)()
  ccall((:acb_mat_scalar_mul_acb, :libarb), Void,
              (Ptr{acb_mat}, Ptr{acb_mat}, Ptr{acb}, Int),
              &z, &x, &y, prec(parent(x)))
  return z
end

*(x::acb, y::acb_mat) = y*x

function //(x::acb_mat, y::Int)
  y == 0 && throw(DivideError())
  z = parent(x)()
  ccall((:acb_mat_scalar_div_si, :libarb), Void,
              (Ptr{acb_mat}, Ptr{acb_mat}, Int, Int),
              &z, &x, y, prec(parent(x)))
  return z
end

function //(x::acb_mat, y::fmpz)
  z = parent(x)()
  ccall((:acb_mat_scalar_div_fmpz, :libarb), Void,
              (Ptr{acb_mat}, Ptr{acb_mat}, Ptr{fmpz}, Int),
              &z, &x, &y, prec(parent(x)))
  return z
end

function //(x::acb_mat, y::arb)
  z = parent(x)()
  ccall((:acb_mat_scalar_div_arb, :libarb), Void,
              (Ptr{acb_mat}, Ptr{acb_mat}, Ptr{arb}, Int),
              &z, &x, &y, prec(parent(x)))
  return z
end

function //(x::acb_mat, y::acb)
  z = parent(x)()
  ccall((:acb_mat_scalar_div_acb, :libarb), Void,
              (Ptr{acb_mat}, Ptr{acb_mat}, Ptr{acb}, Int),
              &z, &x, &y, prec(parent(x)))
  return z
end

################################################################################
#
#  Precision, shifting and other operations
#
################################################################################

function ldexp(x::acb_mat, y::Int)
  z = parent(x)()
  ccall((:acb_mat_scalar_mul_2exp_si, :libarb), Void,
              (Ptr{acb_mat}, Ptr{acb_mat}, Int), &z, &x, y)
  return z
end

################################################################################
#
#  Permutation
#
################################################################################

# this can be done faster
function *(P::perm, x::acb_mat)
   z = parent(x)()
   m = rows(x)
   n = cols(x)
   for i = 1:m
      for j = 1:n
         z[P[i], j] = x[i, j]
      end
   end
   return z
end

################################################################################
#
#  Solving
#
################################################################################

function swap_rows!(x::acb_mat, i::Int, j::Int)
  _checkbounds(rows(x), i) || throw(BoundsError())
  _checkbounds(rows(x), j) || throw(BoundsError())
  ccall((:acb_mat_swap_rows, :libarb), Void,
              (Ptr{acb_mat}, Ptr{Void}, Int, Int),
              &x, C_NULL, i - 1, j - 1)
end

function lufact!(P::perm, x::acb_mat)
  r = ccall((:acb_mat_lu, :libarb), Cint,
              (Ptr{Int}, Ptr{acb_mat}, Ptr{acb_mat}, Int),
              P.d, &x, &x, prec(parent(x)))
  r == 0 && error("Could not find $(rows(x)) invertible pivot elements")
  return r
end

function lufact(P::perm, x::acb_mat)
  cols(x) != rows(x) && error("Matrix must be square")
  parent(P).n != rows(x) && error("Permutation does not match matrix")
  R = base_ring(x)
  L = parent(x)()
  U = deepcopy(x)
  n = cols(x)
  lufact!(P, U)
  for i = 1:n
    for j = 1:n
      if i > j
        L[i, j] = U[i, j]
        U[i, j] = R()
      elseif i == j
        L[i, j] = one(R)
      else
        L[i, j] = R()
      end
    end
  end
  return L, U
end

function solve!(z::acb_mat, x::acb_mat, y::acb_mat)
  r = ccall((:acb_mat_solve, :libarb), Cint,
              (Ptr{acb_mat}, Ptr{acb_mat}, Ptr{acb_mat}, Int),
              &z, &x, &y, prec(parent(x)))
  r == 0 && error("Matrix cannot be inverted numerically")
  nothing
end

function solve(x::acb_mat, y::acb_mat)
  cols(x) != rows(x) && error("First argument must be square")
  cols(x) != rows(y) && error("Matrix dimensions are wrong")
  z = parent(y)()
  solve!(z, x, y)
  return z
end

function solve_lu_precomp!(z::acb_mat, P::perm, LU::acb_mat, y::acb_mat)
  ccall((:acb_mat_solve_lu_precomp, :libarb), Void,
              (Ptr{acb_mat}, Ptr{Int}, Ptr{acb_mat}, Ptr{acb_mat}, Int),
              &z, P.d, &LU, &y, prec(parent(LU)))
  nothing
end

function solve_lu_precomp(P::perm, LU::acb_mat, y::acb_mat)
  cols(LU) != rows(y) && error("Matrix dimensions are wrong")
  z = parent(y)()
  solve_lu_precomp!(z, P, LU, y)
  return z
end

function inv(x::acb_mat)
  cols(x) != rows(x) && error("Matrix must be square")
  z = parent(x)()
  r = ccall((:acb_mat_inv, :libarb), Cint,
              (Ptr{acb_mat}, Ptr{acb_mat}, Int), &z, &x, prec(parent(x)))
  Bool(r) ? (return z) : error("Matrix cannot be inverted numerically")
end

################################################################################
#
#  Determinant
#
################################################################################

function det(x::acb_mat)
  cols(x) != rows(x) && error("Matrix must be square")
  z = base_ring(x)()
  ccall((:acb_mat_det, :libarb), Void,
              (Ptr{acb}, Ptr{acb_mat}, Int), &z, &x, prec(parent(x)))
  return z
end

################################################################################
#
#  Characteristic polynomial
#
################################################################################

function charpoly(x::acb_mat, y::AcbPolyRing)
  base_ring(x) != base_ring(y) && error("Base rings must coincide")
  z = y()
  ccall((:acb_mat_charpoly, :libarb), Void,
              (Ptr{acb_poly}, Ptr{acb_mat}, Int), &z, &x, prec(parent(x)))
  return z
end

################################################################################
#
#  Special functions
#
################################################################################

function exp(x::acb_mat)
  cols(x) != rows(x) && error("Matrix must be square")
  z = parent(x)()
  ccall((:acb_mat_exp, :libarb), Void,
              (Ptr{acb_mat}, Ptr{acb_mat}, Int), &z, &x, prec(parent(x)))
  return z
end

################################################################################
#
#  Unsafe operations
#
################################################################################

for (s,f) in (("add!","acb_mat_add"), ("mul!","acb_mat_mul"),
              ("sub!","acb_mat_sub"))
  @eval begin
    function ($(symbol(s)))(z::acb_mat, x::acb_mat, y::acb_mat)
      ccall(($f, :libarb), Void,
                  (Ptr{acb_mat}, Ptr{acb_mat}, Ptr{acb_mat}, Int),
                  &z, &x, &y, prec(parent(x)))
    end
  end
end

################################################################################
#
#  Parent object overloading
#
################################################################################

function call(x::AcbMatSpace)
  z = acb_mat(x.rows, x.cols)
  z.parent = x
  return z
end

function call(x::AcbMatSpace, y::fmpz_mat)
  (x.cols != cols(y) || x.rows != rows(y)) &&
      error("Dimensions are wrong")
  z = acb_mat(y, prec(x))
  z.parent = x
  return z
end

function call(x::AcbMatSpace, y::arb_mat)
  (x.cols != cols(y) || x.rows != rows(y)) &&
      error("Dimensions are wrong")
  z = acb_mat(y, prec(x))
  z.parent = x
  return z
end


function call{T <: Union{Int, UInt, Float64, fmpz, fmpq, BigFloat, arb, acb,
                         AbstractString}}(x::AcbMatSpace, y::Array{T, 2})
  (x.rows, x.cols) != size(y) && error("Dimensions are wrong")
  z = acb_mat(x.rows, x.cols, y, prec(x))
  z.parent = x
  return z
end

call{T <: Union{Int, UInt, Float64, fmpz, fmpq, BigFloat, arb, acb,
                AbstractString}}(x::AcbMatSpace, y::Array{T, 1}) = x(y'')


function call{T <: Union{Int, UInt, Float64, fmpz, fmpq, BigFloat, arb,
                         AbstractString}}(x::AcbMatSpace,
                                          y::Array{Tuple{T, T}, 2})
  (x.rows, x.cols) != size(y) && error("Dimensions are wrong")
  z = acb_mat(x.rows, x.cols, y, prec(x))
  z.parent = x
  return z
end

call{T <: Union{Int, UInt, Float64, fmpz, fmpq, BigFloat, AbstractString,
                arb}}(x::AcbMatSpace, y::Array{Tuple{T, T}, 1}) = x(y'')

################################################################################
#
#  Matrix space constructor
#
################################################################################

function MatrixSpace(R::AcbField, r::Int, c::Int)
  return AcbMatSpace(R, r, c)
end
