#include <iostream>
#include "points.h"

int main(int argc,char** argv)
{
    // Load points
    Points points;
    points.read_points(argv[1]);
    points.write_ply(argv[2]);
    int n = points.info().pt_num();
    std::cout << argv[2] << ": " << n << " points" << std::endl;
    return 0;
}


