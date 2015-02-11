#include <cstdlib>
#include <cmath>
#include <iostream>

using namespace std;

double GaussRand()
{
    static double v1, v2, s;
    static int phase  = 0;
    double x;

    if (0 == phase)
    {
        do 
        {
            double u1 = (double)rand()/RAND_MAX;
            double u2 = (double)rand()/RAND_MAX;

            v1 = 2 * u1 - 1;
            v2 = 2 * u2 - 1;
            s = v1 * v1 + v2 * v2;
        } while ( 1 <= s || 0 == s);
        x = v1 * sqrt(-2 * log(s) / s);
    }
    else
    {
        x = v2 * sqrt(-2 * log(s) / s);
    }
    phase = 1 - phase;

    return x; 
}

int main()
{
    srand(time(NULL));
    for (size_t i = 0; i < 10; ++i)
    {
        cout << GaussRand() << "    ";
    }
    cout << endl;
}
