/*--------------------------------*- C++ -*----------------------------------*\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  v2312                                 |
|   \\  /    A nd           | Website:  www.openfoam.com                      |
|    \\/     M anipulation  |                                                 |
\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

scale   1;

vertices
(
    (0 0 0)
    (0.1 0 0)
    (0.1 0.1 0)
    (0 0.1 0)
    (0 0 0.01)
    (0.1 0 0.01)
    (0.1 0.1 0.01)
    (0 0.1 0.01)
);

blocks
(
    hex (0 1 2 3 4 5 6 7) (100 50 1) simpleGrading (1 1 1)
);

edges
(
);

boundary
(
    inlet
    {
        type patch;
        faces
        (
            (0 3 7 4)
        );
    }
    outlet
    {
        type patch;
        faces
        (
            (1 2 6 5)
        );
    }
    walls
    {
        type wall;
        faces
        (
            (0 1 5 4)
            (3 2 6 7)
        );
    }
    frontAndBack
    {
        type empty;
        faces
        (
            (0 1 2 3)
            (4 5 6 7)
        );
    }
);

// ************************************************************************* //