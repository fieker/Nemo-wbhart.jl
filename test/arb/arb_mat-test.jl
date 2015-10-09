if !on_windows64

RR = ArbField(64)

M = MatrixSpace(RR, 2, 2)

function test_arb_mat_constructors()
   print("arb_mat.constructors()...")

   @test isa(M, ArbMatSpace)
   @test isa(M(), MatElem)

   @test elem_type(M) == arb_mat
   @test base_ring(M) == RR

   @test prec(M) == 64

   println("PASS")
end

function test_arb_mat_basic_ops()
   print("arb.basic_ops()...")

   zz = M(MatrixSpace(ZZ, 2, 2)(2))

   @test parent(zz) == M

   @test M([2 0; 0 2]) == zz
   @test M([UInt(2) UInt(0); UInt(0) UInt(2)]) == zz
   @test M([fmpz(2) fmpz(0); fmpz(0) fmpz(2)]) == zz
   @test M([2.0 0.0; 0.0 2.0]) == zz
   @test M(["2" "0"; "0" "2"]) == zz
   @test contains(M([BigFloat(2) BigFloat(0); BigFloat(0) BigFloat(2)]), zz)

   a = one(M)

   @test a[1, 1] == 1
   @test a[1, 2] == 0
   @test a[2, 2] == 1
   @test a[2, 1] == 0

   z = RR()

   getindex!(z, a, 1, 1)

   @test parent(z) == base_ring(a)
   @test z == 1

   b = M()

   @test b[1, 1] == 0
   @test b[1, 2] == 0
   @test b[2, 1] == 0
   @test b[2, 2] == 0

   b[1, 1] = 1
   b[2, 2] = 1

   @test a == b

   b = M()

   b[1, 1] = UInt(1)
   b[2, 2] = UInt(1)

   @test a == b

   b = M()

   b[1, 1] = fmpz(1)
   b[2, 2] = fmpz(1)

   @test a == b

   b = M()

   b[1, 1] = fmpq(1)
   b[2, 2] = fmpq(1)

   @test a == b

   b = M()

   b[1, 1] = 1.0
   b[2, 2] = 1.0

   @test a == b

   b = M()

   b[1, 1] = "1.0 +/- 0"
   b[2, 2] = "1.0 +/- 0"

   @test a == b

   println("PASS")
end

function test_arb_mat_comparison()
   print("arb_mat.comparison()...")

   exact = M()
   exact[1, 1] = 2
   exact[2, 2] = 2

   exact2 = M()
   exact2[1, 1] = 3
   exact2[2, 2] = 3

   approx = M()

   approx[1, 1] = "2 +/- 0.1"
   approx[2, 2] = "2 +/- 0.1"

   approx2 = M()

   approx2[1, 1] = "3 +/- 0.1"
   approx2[2, 2] = "3 +/- 0.1"

   @test exact == exact
   @test exact != exact2
   @test exact != approx2

   @test strongequal(exact, deepcopy(exact))
   @test !strongequal(exact, approx)

   @test overlaps(approx, exact)
   @test overlaps(exact, approx)
   @test overlaps(approx, approx)
   @test !overlaps(approx, approx2)

   @test contains(approx, exact)
   @test contains(approx, approx)
   @test !contains(exact, approx)

   @test contains(approx, MatrixSpace(ZZ, 2, 2)(2))
   @test !contains(approx2, MatrixSpace(ZZ, 2, 2)(2))

   println("PASS")
end

function test_arb_mat_predicates()
   print("arb_mat.predicates()...")

   a = M()

   @test issquare(a)

   b = MatrixSpace(RR, 1, 2)()

   @test !issquare(b)

   println("PASS")
end

function test_arb_mat_transpose()
   print("arb_mat.transpose()...")

   a = M()

   a[1, 2] = 2
   a[2, 1] = 3

   b = transpose(a)

   @test b[1, 2] == 3
   @test b[2, 1] == 2
   @test b[1, 1] == 0
   @test b[2, 2] == 0

   println("PASS")
end

function test_arb_mat_norm_bound()
   print("arb_mat.norm_bound()...")
 
   a = M()
   a[1, 1] = 2
   a[1, 2] = 3
   a[2, 1] = -4
   a[2, 2] = 10
 
   z = bound_inf_norm(a)
 
   @test parent(z) == RR
   @test 14 <= z
 
   println("PASS")
end

function test_arb_mat_unary_ops()
   print("arb_mat.unary_ops()...")

   a = one(M)

   b = -a

   @test b[1, 1] == -1
   @test b[2, 2] == -1
   @test b[1, 2] == 0
   @test b[2, 1] == 0

   println("PASS")
end

function test_arb_mat_binary_ops()
   print("arb_mat.binary_ops()...")

   a = one(M)
   b = M()
   c = M()
   d = M()
   f = RR(1)
   g = RR(2)

   b[1, 1] = 2
   b[2, 2] = 2

   c[1, 1] = 3
   c[2, 2] = 3

   d[1, 1] = 4
   d[2, 2] = 4

   @test a + a == b
   @test a + b == c
   @test a * a == a
   @test b * b == d
   @test d - c == a

   @test b^2 == d
   @test b^UInt(2) == d

   @test b * 2 == d
   @test b * fmpz(2) == d
   @test b * RR(2) == d
   @test 2 * b == d
   @test fmpz(2) * b == d
   @test RR(2) * b == d

   @test d//2 == b
   @test d//fmpz(2) == b
   @test d//RR(2) == b

   println("PASS")
end

function test_arb_mat_permutation()
   print("arb_mat.permuation()...")

   z = one(M)
   p = PermutationGroup(2)([2, 1])

   z = p*z

   @test z[1, 1] == 0
   @test z[1, 2] == 1
   @test z[2, 1] == 1
   @test z[2, 2] == 0

   println("PASS")
end

function test_arb_mat_solving()
   print("arb_mat.solving()...")

   z = one(M)

   swap_rows!(z, 1, 2) 

   @test z[1, 1] == 0
   @test z[1, 2] == 1
   @test z[2, 1] == 1
   @test z[2, 2] == 0

   z = M()

   z[1, 1] = 3
   z[1, 2] = 6
   z[2, 1] = 14
   z[2, 2] = 654

   p = PermutationGroup(2)()

   l, u = lufact(p, z)

   @test overlaps(p*l*u, z)

   lufact!(p, z)

   @test overlaps(l + u - one(M), z)

   z = 2*one(M)
   w = MatrixSpace(RR, 2, 1)()
   w[1, 1] = 4
   w[2, 1] = 4
   x = deepcopy(w)
   solve!(x, z, w)

   @test overlaps(x, w//2)
   @test overlaps(solve(z, w), w//2)
   
   lufact!(p, z)

   @test overlaps(solve_lu_precomp(p, z, w), w//2)

   x = deepcopy(w)
   solve_lu_precomp!(x, p, z, w)

   @test overlaps(x, w//2)

   zz = inv(z)

   @test overlaps(zz, one(M)//2)

   println("PASS")
end

function test_arb_mat_determinant()
   print("arb_mat.determinant()...")

   z = M()

   z[1, 1] = "3 +/- 0.1"
   z[1, 2] = "2 +/- 0.1"
   z[2, 1] = "5 +/- 0.1"
   z[2, 2] = "4 +/- 0.1"
   
   d = det(z)

   @test parent(d) == base_ring(z)

   @test contains(d, 2)

   println("PASS")
end

function test_arb_mat_special_functions()
   print("arb_mat.special_functions()...")

   z = M()

   z[1, 1] = 2
   z[1, 2] = 1
   z[2, 2] = 2

   zz = M()

   zz[1, 1] = RR(e)^2
   zz[1, 2] = RR(e)^2
   zz[2, 2] = RR(e)^2

   @test overlaps(exp(z), zz)

   println("PASS")
end

function test_arb_mat_unsafe_ops()
   print("arb_mat.unsafe_ops()...")

   a = one(M)
   b = M()
   c = M()
   d = M()

   b[1, 1] = 2
   b[2, 2] = 2

   c[1, 1] = 3
   c[2, 2] = 3

   d[1, 1] = 4
   d[2, 2] = 4

   z = M()

   add!(z, a, b)

   @test z == c

   mul!(z, b, b)

   @test z == d

   sub!(z, d, c)

   @test z == a

   println("PASS")
end

function test_arb_mat_charpoly()
   print("arb_mat.charpoly()...")

   z = M([1 0; 0 2])

   Rx, x = PolynomialRing(RR, "x")

   @test charpoly(z, Rx) == x^2 - 3x + 2

   println("PASS")
end

function test_arb_mat()
   test_arb_mat_constructors()
   test_arb_mat_basic_ops()
   test_arb_mat_comparison()
   test_arb_mat_predicates()
   test_arb_mat_transpose()
   test_arb_mat_norm_bound()
   test_arb_mat_unary_ops()
   test_arb_mat_binary_ops()
   test_arb_mat_permutation()
   test_arb_mat_solving()
   test_arb_mat_determinant()
   test_arb_mat_charpoly()
   test_arb_mat_special_functions()
   test_arb_mat_unsafe_ops()

   println("")
end

end # on_windows64
