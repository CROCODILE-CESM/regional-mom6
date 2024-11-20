# MOM6 Angle Calculation Steps 

## Process of calculation -> Only works on t-points
1. Calculate pi/4rads / 180 degress  = Gives a 1/4 conversion of degrees to radians. I.E. multiplying an angle in degrees by this gives the conversion to radians at 1/4 the value. 
2. Figure out the longitudunal extent of our domain, or periodic range of longitudes. For global cases it is len_lon = 360, for our regional cases it is given by the hgrid.
3. At each point on our hgrid, we find the point to the left, bottom left diag, bottom, and itself. We adjust each of these longitudes to be in the range of len_lon around the point itself. (module_around_point)
4. We then find the lon_scale, which is the "trigonometric scaling factor converting changes in longitude to equivalent distances in latitudes". Whatever that actually means is we add the latitude of all four of these points from part 3 and basically average it and convert to radians. We then take the cosine of it. As I understand it, it's a conversion of longitude to equivalent latitude distance. 
5. Then we calculate the angle. This is a simple arctan2 so y/x. 
    1. The "y" component is the addition of the difference between the diagonals in longitude of lonB multiplied by the lon_scale, which is our conversion to latitude.
    2. The "x" component is the same addition of differences in latitude.
    3. Thus, given the same units, we can call arctan to get the angle in degrees

## Conversion to Q points
1. (Recommended by Gustavo)
2. We use XGCM to interpolate from the t-points to all other points in the supergrid.

## Implementation

1. Direct implementation of MOM6 grid angle initalization function (and modulo_around_point)
2. Wrap direct implementation combined with XGCM interpolation for grid angles

