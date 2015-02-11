/*************************************************************************
    > File Name: utils.h
    > Author: chenrudan
    > Mail: chenrudan123@gmail.com
    > Created Time: 2014年07月15日 星期二 09时23分18秒
 ************************************************************************/
#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <cstdlib>
#include <cmath>

using namespace std;


inline float RandomWeight(float low, float upper){
    return (rand() * 1.0 / RAND_MAX) * (upper - low) + low;
}

inline float RandomNormal(){
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

inline float RandomNumber(){
    return (rand() / (RAND_MAX + 1.0));
}

inline bool CompareFloat(float a, float b){
    return (a > b);
}

inline float Logisitc(float a){
    return 1.0 / (1 + exp(-a));
}



#endif
