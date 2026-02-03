// The original paper is overly complicated in it's notations, so here is a description.
// We will use capital letter for univariate polys, and small one for multivatiate, and same letter
// of different capitalization is just reinterpretation of one for another
// - Prover starts with oracle of evaluations F0 of the original poly F(X) at some smooth domain L0
// - also we assume that we have an original claim that F(Y) = Z, that can also we rewritten as sumcheck claim
// F(Y) = Z = f(y^0, y^1, y^2, ...) = \sum_{x} eq(x, y^0, y^1, y^2, ...) f(x) - our original sumcheck claim.
// If we sum over all the {x} in the right-hand side, but one, we can view it as a univariate f0(Y), and f0(0) + f0(1) == Z - 
// all the standard sumcheck staff
// - Note that in the same manner we can express in-domain value F(omega^k) = \sum_{x} eq(x, omega^k decomposition over powers) f(x)
// - Prover and verifier can engage in more than 1 sumcheck steps (here the tradeoff is less steps later, but more accesses to F0 oracle)
// ---- Steps below are recursive, but we only use indexes 0/1 for clarity
// - At this moment we would have something like
// claim_0 = \sum_{x/folded coordiantes} eq(r1, r2, r3, x4, x5, ... y^0, y^1, y^2, y^4, ...) f(r1, r2, r3, x4, x5, ...)
// - Now prover sends an oracle F1 to f1(x4, x5, ...) = f(r1, r2, r3, x4, x5, ...) at domain L1. Note that "degree" of f1(x4, x5, ...)
// is smaller that of original f(x), but prover can decrease the rate for further iterations of the protocol
// - As in STIR, we want to perform out of domain sampling. So, we draw OOD point y1 and prover sends evaluation of f1(y1^0, y1^1, ...) = z1
// - Now prover also samples NUM_QUERIES indexes in the 3 (in our example) times folded image of L0. Those indexes trivially map 
// into the |L0|/2^3 roots of unity. We will use notations Q_i for such indexes and corresponding roots of unity interchangeably
// - As in FRI, verifier has oracle access to f1(Q_i) by accessing 2^3 corresponding elements in F0 (at L0) and folding them.
// - We denote those values as G_i and in the original paper we do not need those values from prover YET, and instead they update our sumcheck claim formally at first,
// but it doesn't affect the protocol, and we will show that verification can be performed right away
// - start with the old one (prefactors aside)
// claim_0 = \sum_{x} eq(x, y^4, y^8, ...) f1(x)
// - add a contribution about f1(y1) = z1
// claim_0 + gamma^1 * z1 = \sum_{x} eq(x, y^4, y^8, ...) f1(x) + gamma^1 * \sum_{x} eq(x, y1^0, y1^1, ...) f1(x)
// - add NUM_QUERIES contribution about Q_i
// claim_0 + gamma^1 * z1 + \sum_{i = 0..NUM_QUERIES} gamma^{i + 1} G_i = 
// = \sum_{x} eq(x, y^4, y^8, ...) f1(x) +
// + gamma^1 * \sum_{x} eq(x, y1^0, y1^1, ...) f1(x) +
// + \sum_{i = 0..NUM_QUERIES} gamma^{1+i} * \sum_{x} eq(x, Q_i) f1(x)
// - Those terms re-arrange nicely over f1(x)
// - To continue the sumcheck prover would send some univariate poly f1(Y), but as usual
// f1(0) + f1(1) == claim_0 + gamma^1 * z1 + \sum_{i = 0..NUM_QUERIES} gamma^{i + 1} G_i
// and verifier already has all the values to perform this check and forget about anything that happened before:
// - claim_0 comes from the previous work
// - z1 was sent by the prover
// - G_i are available via oracle access to F0 at L0 (in our example verifier needs 8 elements to fold 3 times and get those values)
// ---- Steps above are recursive until f_i(x) becomes "small", and so prover can send it explicitly
// - Verifier explicitly performs the last sumcheck
// - that would formally involve evaluating eq(x, small roots of unity), that ARE just selecting corresponding value
// - and evaluating eq(x, high powers of ys) - but it's not too large number of them