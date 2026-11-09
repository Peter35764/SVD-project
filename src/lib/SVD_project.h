#pragma once

// Testing generator
#include "testing/generate_svd.h"

// Standard algorithms
#include "dqds/dqds.h"
#include "golub_kahan/golub_kahan.h"
#include "iterative_refinement/iterative_refinement.h"
#include "jacobi/jacobi.h"

// Experimental ideas
#include "idea1_givref/givens_refinement.h"
#include "idea2_revjac/pseudo_reverse_jacobi.h"
#include "idea2_revjac/reverse_jacobi.h"
#include "idea3_naive/naive_bidiag_svd/naive_bidiag_svd.h"
#include "idea3_naive/naive_mrrr/naive_mrrr.h"

// Legacy
#include "idea1_givref/legacy/v0_givens_refinement.h"
#include "idea1_givref/legacy/v1_givens_refinement.h"
#include "idea2_revjac/legacy/v0_reverse_jacobi.h"

// Testing framework (depends on all algorithms and generator)
#include "testing/SVD_Test.h"
