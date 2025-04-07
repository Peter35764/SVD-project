#include <Eigen/Jacobi>
#include <iostream> // cout
#include <iomanip> // setprecision
#include <cmath> // abs, sqrt
//Проверка кода

// ROTResult :: { cs :: Float, sn :: Float, r :: Float }
struct ROTdeps {
    float cs, sn, r;
};

// after test should be optimized or replaced with Eigen if it would prove itself sufficient
// naiveImplementation :: Float -> Float -> ROTdeps, based on lawn03.pdf, JUST for comparison with Eigen's impl
ROTdeps naiveImplementation(float f, float g) {
    ROTdeps result;
    
    if (f == 0.0f) {
        result.cs = 0.0f;
        result.sn = 1.0f;
        result.r = g;
    }
    else if (std::abs(f) >= std::abs(g)) {
        float t = g/f;
        float u = std::sqrt(1.0f + t*t);
        result.cs = 1.0f/u;
        result.sn = t*result.cs;
        result.r = f*u;
    }
    else {
        float t = f/g;
        float u = std::sqrt(1.0f + t*t);
        result.sn = 1.0f/u;
        result.cs = t*result.sn;
        result.r = g*u;
    }
    
    return result;
}

// testROTImplementations :: Float -> Float -> Float -> Bool, epsilon to compare floats
bool testROTImplementations(float f, float g, float epsilon = 1e-6f) {
    using namespace std;
    using namespace Eigen;
    
    ROTdeps paper = naiveImplementation(f, g);
    
    JacobiRotation<float> eigen;
    float r;
    eigen.makeGivens(f, g, &r);
    
    bool cs_match = std::abs(paper.cs - eigen.c()) <= epsilon;
    bool sn_match = std::abs(std::abs(paper.sn) - std::abs(eigen.s())) <= epsilon;
    bool r_match = std::abs(paper.r - r) <= epsilon;
    
    float result1 = eigen.c() * f - eigen.s() * g;
    float result2 = eigen.s() * f + eigen.c() * g;
    
    cout << "Eigen: {cs=" << eigen.c() << ", sn=" << eigen.s() << ", r=" << r << "}\n"; // In the end we shall prefer Eigen implementations, so comparing from it
    cout <<  "Matched with naive implementation? "<< (cs_match && sn_match && r_match ? "YES" : "NO") << "\n"; // comparing up to epsilon beacuse of float
    cout << "Verify ROT: [" << result1 << "; " << result2 << "] should be [r, ~0]\n"; //  [cs -sn; sn cs][f; g] = [r; 0]
    
    return cs_match && sn_match && r_match;
}

int main()
{
    using namespace std;
    using namespace Eigen;
    Eigen::Matrix<float, Dynamic, Dynamic> A(10,9);
    A << 1, 2, 3, 4, 5, 6, 7, 8, 9,
        10, 11, 12, 13, 14, 15, 16, 17, 18,
        19, 20, 21, 22, 23, 24, 25, 26, 27,
        28, 29, 30, 31, 32, 33, 34, 35, 36,
        37, 38, 39, 40, 41, 42, 43, 44, 45,
        46, 47, 48, 49, 50, 51, 52, 53, 54,
        55, 56, 57, 58, 59, 60, 61, 62, 63,
        64, 65, 66, 67, 68, 68, 70, 71, 72,
        73, 74, 75, 76, 77, 78, 79, 80, 81,
        3, 9, (float)4.98942, (float)0.324235,  443534, 345, (float)56.543853, (float)450.435234, (float)43.34353221;

    cout << A << endl << endl << endl << endl << endl << endl << endl << endl << endl;

	//
	//
	// Testing ROT
	//
	//
	
    cout << fixed << setprecision(10);
    
    cout << "Testcase 1: f = 0\n";
    testROTImplementations(0.0f, 1.0f);
    cout << "\n";
    cout << "Testcase 2: |f| > |g|\n";
    testROTImplementations(2.0f, 1.0f);
    cout << "\n";
    cout << "Testcase 3: |f| < |g|\n";
    testROTImplementations(1.0f, 2.0f);
    cout << "\n";
    cout << "Testcase 4: Large numbers\n";
    testROTImplementations(1e5f, 2e5f);
    cout << "\n";
    cout << "Testcase 5: Small numbers\n";
    testROTImplementations(1e-5f, 2e-5f);
    cout << "\n";
    cout << "Testcase 6: Matrix values\n";
    testROTImplementations(A(0,0), A(1,0));

    return 0;
}
