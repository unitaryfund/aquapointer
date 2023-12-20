import numpy as np

# If a ray is identified by the vector O of its origin and the vector D of its direction,
# and a triangle is identified by its three vertices A, B, C, then:
# RAY = O + tD, t>=0
# TRIANGLE = A + uE1 + vE2, u,v>=0, (u+v)<=1, E1=B-A, E2=C-A
# the following function gives the parameters u, v, t of the intersection between RAY and TRIANGLE if it exists
# (based on Moeller–Trumbore Algorithm)

# ray is an array np.array([[O],[D]]), triangle is an array np.array([[A],[B],[C]])
def intersection_ray_triangle(ray, triangle):
    O, D = ray
    A, B, C = triangle
    E1 = B-A
    E2 = C-A
    N = np.cross(E1, E2)
    det = -np.dot(D, N)

    if np.abs(det) > 1e-8:
        invdet = 1.0/det
    else:
        return False

    AO  = O-A
    DAO = np.cross(AO, D)
    u = np.dot(E2, DAO) * invdet
    v = -np.dot(E1, DAO) * invdet
    t =  np.dot(AO, N)  * invdet

    if (t < 0) or (u < 0) or (v < 0) or ((u+v) > 1):
        return False
    else:
        return u, v, t

# returns True if triangle intersects the square, False otherwise 
# triangle is an array np.array([[A],[B],[C]]), square in an array np.array([[P1],[P2],[P2],[P4])
def intersection_triangle_square(triangle, square):
    # divide square in two triangles
    tri1 = np.array([square[0], square[1], square[3]])
    tri2 = np.array([square[1], square[2], square[3]])
    tris = [tri1, tri2]
    
    # define the six rays (triangle side extensions)
    ray1 = np.array([triangle[0], triangle[1]-triangle[0]])
    ray2 = np.array([triangle[1], triangle[2]-triangle[1]])
    ray3 = np.array([triangle[2], triangle[0]-triangle[2]])
    rays = [ray1, ray2, ray3]
    
    # check if any segment intersects any triangle
    for ray in rays:
        for tri in tris:
            try:
                u, v, t = intersection_ray_triangle(ray, tri)
                if t <= 1:
                    return True
            except:
                pass
    return False

# return the barycenter of the triangle
def barycenter(triangle):
    A, B, C = triangle
    return (A+B+C)/3.


# Returns True if the vector going from point P to the barycenter of triangle
# is aligned with the triangle normal N, False otherwise.
# Aligned means that it is at an angle of less than pi/2 with the normal
# to the triangle.
def is_aligned(P, triangle, N):
    M = barycenter(triangle)
    PM = M-P

    theta = np.arccos(np.dot(PM,N)/(np.linalg.norm(PM)*np.linalg.norm(N)))

    if theta<np.pi/2.:
        return True
    else:
        return False

# Calculates the distance between point P and triangle as described in
# Mark W. Jones - 3D Distance from a Point to a Triangle
def distance_point_triangle(P0, triangle):
    P1, P2, P3 = triangle
    
    # Check degeneracy
    if (np.linalg.norm(P0-P1) < 1e-8) or (np.linalg.norm(P2-P1) < 1e-8) or (np.linalg.norm(P0-P2) < 1e-8):
        return (1e10, np.array([1e10, 1e10, 1e10]))

    # Find P0i, the projection of P0 along the normal N
    # onto the plane defined by the triangle
    N = np.cross(P2-P1, P3-P1)
    
    # Check degeneracy
    if np.linalg.norm(N) < 1e-8:
        return (1e10, np.array([1e10, 1e10, 1e10]))
    
    
    cosalpha = np.dot(P0-P1, N)/(np.linalg.norm(P0-P1)*np.linalg.norm(N))

    P0P0i_l = np.linalg.norm(P1-P0)*np.abs(cosalpha)
    P0P0i_1 = - N * P0P0i_l/np.linalg.norm(N)
    P0P0i_2 = + N * P0P0i_l/np.linalg.norm(N)
    P0i_1 = P0 + P0P0i_1
    P0i_2 = P0 + P0P0i_2
    if np.linalg.norm(P1-P0i_1) < np.linalg.norm(P1-P0i_2):
        P0i = P0i_1
        P0P0i = P0P0i_1
    else:
        P0i = P0i_2
        P0P0i = P0P0i_2

    # If P0i lies inside the triangle, you're done
    if intersection_ray_triangle(np.array([P0, P0P0i]), triangle) != False:
        return (P0P0i_l, P0i)

    # Otherwise you need to understand if P0i is closer to a
    # vertex or an edge

    # Project P0i onto the edges.
    # Call Rab the direction of the projection of P0i
    # onto edge PaPb.
    # NOTE: the projection might lie outside the edge itself
    R12 = np.cross(np.cross(P2-P0i, P1-P0i), P2-P1)
    R23 = np.cross(np.cross(P3-P0i, P2-P0i), P3-P2)
    R31 = np.cross(np.cross(P1-P0i, P3-P0i), P1-P3)


    # Find the point P0ii_12, the projection of P0i
    # onto edge P1P2
    if np.linalg.norm(R12) > 1e-8:
        cosgamma12 = np.dot(P1-P0i, R12)/(np.linalg.norm(P1-P0i)*np.linalg.norm(R12))
        P0iP0ii_12_l = np.linalg.norm(P1-P0i)*cosgamma12
        P0iP0ii_12 = R12 * P0iP0ii_12_l/np.linalg.norm(R12)
        P0ii_12 = P0i + P0iP0ii_12
    else:
        P0ii_12 = P0i

    # Find out if P0 is closer to the edge itself or one
    # of the two vertices
    cosbeta12 = np.dot(P0ii_12-P1, P2-P1)/(np.linalg.norm(P0ii_12-P1)*np.linalg.norm(P2-P1))
    if cosbeta12 < 0:
        d_12 = np.linalg.norm(P0-P1)
        d_P12 = P1
    else:
        if np.linalg.norm(P0ii_12-P1)/np.linalg.norm(P2-P1) > 1:
            d_12 = np.linalg.norm(P0-P2)
            d_P12 = P2
        else:
            d_12 = np.linalg.norm(P0-P0ii_12)
            d_P12 = P0ii_12

    # Find the point P0ii_23, the projection of P0i
    # onto edge P2P3
    if np.linalg.norm(R23) > 1e-8:
        cosgamma23 = np.dot(P2-P0i, R23)/(np.linalg.norm(P2-P0i)*np.linalg.norm(R23))
        P0iP0ii_23_l = np.linalg.norm(P2-P0i)*cosgamma23
        P0iP0ii_23 = R23 * P0iP0ii_23_l/np.linalg.norm(R23)
        P0ii_23 = P0i + P0iP0ii_23
    else:
        P0ii_23 = P0i

    # Find out if P0 is closer to the edge itself or one
    # of the two vertices
    cosbeta23 = np.dot(P0ii_23-P2, P3-P2)/(np.linalg.norm(P0ii_23-P2)*np.linalg.norm(P3-P2))
    if cosbeta23 < 0:
        d_23 = np.linalg.norm(P0-P2)
        d_P23 = P2
    else:
        if np.linalg.norm(P0ii_23-P2)/np.linalg.norm(P3-P2) > 1:
            d_23 = np.linalg.norm(P0-P3)
            d_P23 = P3
        else:
            d_23 = np.linalg.norm(P0-P0ii_23)
            d_P23 = P0ii_23

    # Find the point P0ii_31, the projection of P0i
    # onto edge P3P1
    if np.linalg.norm(R31) > 1e-8:
        cosgamma31 = np.dot(P3-P0i, R31)/(np.linalg.norm(P3-P0i)*np.linalg.norm(R31))
        P0iP0ii_31_l = np.linalg.norm(P3-P0i)*cosgamma31
        P0iP0ii_31 = R31 * P0iP0ii_31_l/np.linalg.norm(R31)
        P0ii_31 = P0i + P0iP0ii_31
    else:
        P0ii_31 = P0i

    # Find out if P0 is closer to the edge itself or one
    # of the two vertices
    cosbeta31 = np.dot(P0ii_31-P3, P1-P3)/(np.linalg.norm(P0ii_31-P3)*np.linalg.norm(P1-P3))
    if cosbeta31 < 0:
        d_31 = np.linalg.norm(P0-P3)
        d_P31 = P3
    else:
        if np.linalg.norm(P0ii_31-P3)/np.linalg.norm(P1-P3) > 1:
            d_31 = np.linalg.norm(P0-P1)
            d_P31 = P1
        else:
            d_31 = np.linalg.norm(P0-P0ii_31)
            d_P31 = P0ii_31

    candidates_d = [d_12, d_23, d_31]
    candidates_P = [d_P12, d_P23, d_P31]
    idx = np.argmin(candidates_d)
    return (candidates_d[idx], candidates_P[idx])

# Returns tuple (index, distance, interior) where 'index' is the index of the
# closest triangle of 'triangles' to the point P, 'distance' is its distance,
# and 'interior' is True if P is inside the mesh and False otherwise
def relationship(P, triangles, normals):
    idx = -1
    distance = 1e10
    point = [1e10, 1e10, 1e10]
    for i in range(len(triangles)):
        d, d_P = distance_point_triangle(P, triangles[i])
        if d < distance:
            distance = d
            point = d_P
            idx = i

    # If T is the closest triangle to P (which was found just above this
    # comment), send a ray in the direction P->M_T, where M_T is the barycenter
    # of T. Call T' the closest triangle that intersects the ray P->M_T.
    # If the intersection happens on the boundary of T', repeat the procedure
    # with the barycenter M_T' instead of M_T, until the intersection happens
    # in the interior of a triangle. That triangle is a good one with which the
    # alignment can be calculated.
    M = barycenter(triangles[idx])
    idx_ray = -1
    prev_idx_ray = idx
    distance_ray = 1e10
    intersect_on_boundary = True
    while intersect_on_boundary:
        for i in range(len(triangles)):
            try:
                u, v, t = intersection_ray_triangle(np.array([P, M-P]), triangles[i])
            except TypeError:
                continue
            d = np.linalg.norm(t*(M-P))
            if d < distance_ray:
                distance_ray = d
                idx_ray = i
                candidate_u = u
                candidate_v = v
        if (
            (np.abs(candidate_u + candidate_v - 1) < 1e-8) or
            (candidate_u < 1e-8) or
            (candidate_v < 1e-8)
        ):
            # if condition is true, intersection is on the border
            # and the procedure needs to be repeated, unless we are hitting
            # the same border over and over again
            if idx_ray == prev_idx_ray:
                # if condition is true, we are on an infinite loop and need to
                # exit
                return None
            M = barycenter(triangles[idx_ray])
            prev_idx_ray = idx_ray
            idx_ray = -1
            distance_ray = 1e10
        else:
            intersect_on_boundary = False


    return (idx, distance, point, is_aligned(P, triangles[idx_ray], normals[idx_ray]))

# DEBUG VERSION
# Calculates the distance between point P and triangle as described in
# Mark W. Jones - 3D Distance from a Point to a Triangle
def distance_point_triangle_debug(P0, triangle):
    P1, P2, P3 = triangle

    # Find P0i, the projection of P0 along the normal N
    # onto the plane defined by the triangle
    N = np.cross(P2-P1, P3-P1)
    print(f"N: {N}")
    cosalpha = np.dot(P0-P1, N)/(np.linalg.norm(P0-P1)*np.linalg.norm(N))
    print(f"cosalpha: {cosalpha}")

    P0P0i_l = np.linalg.norm(P1-P0)*np.abs(cosalpha)
    print(f"P0P0i_l: {P0P0i_l}")
    P0P0i = - N * P0P0i_l/np.linalg.norm(N)
    print(f"P0P0i: {P0P0i}")
    P0i = P0 + P0P0i
    print(f"P0i: {P0i}")

    # If P0i lies inside the triangle, you're done
    if intersection_ray_triangle(np.array([P0, P0P0i]), triangle) != False:
        print("P0i intersects triangle")
        return P0P0i_l

    # Otherwise you need to understand if P0i is closer to a
    # vertex or an edge

    # Project P0i onto the edges.
    # Call Rab the direction of the projection of P0i
    # onto edge PaPb.
    # NOTE: the projection might lie outside the edge itself
    R12 = np.cross(np.cross(P2-P0i, P1-P0i), P2-P1)
    print(f"R12: {R12}")
    R23 = np.cross(np.cross(P3-P0i, P2-P0i), P3-P2)
    print(f"R23: {R23}")
    R31 = np.cross(np.cross(P1-P0i, P3-P0i), P1-P3)
    print(f"R31: {R31}")


    # Find the point P0ii_12, the projection of P0i
    # onto edge P1P2
    if np.linalg.norm(R12) > 1e-8:
        print("R12 has length != 0")
        cosgamma12 = np.dot(P1-P0i, R12)/(np.linalg.norm(P1-P0i)*np.linalg.norm(R12))
        print(f"cosgamma12: {cosgamma12}")
        P0iP0ii_12_l = np.linalg.norm(P1-P0i)*cosgamma12
        print(f"P0iP0ii_12_l: {P0iP0ii_12_l}")
        P0iP0ii_12 = R12 * P0iP0ii_12_l/np.linalg.norm(R12)
        print(f"P0iP0ii_12: {P0iP0ii_12}")
        P0ii_12 = P0i + P0iP0ii_12
        print(f"P0ii_12: {P0ii_12}")
    else:
        print("R12 has length 0")
        P0ii_12 = P0i
        print(f"P0ii_12: {P0ii_12}")

    # Find out if P0 is closer to the edge itself or one
    # of the two vertices
    cosbeta12 = np.dot(P0ii_12-P1, P2-P1)/(np.linalg.norm(P0ii_12-P1)*np.linalg.norm(P2-P1))
    print(f"cosbeta12: {cosbeta12}")
    if cosbeta12 < 0:
        print("P0 is closer to P1")
        d_12 = np.linalg.norm(P0-P1)
        print(f"d_12: {d_12}")
    else:
        if np.linalg.norm(P0ii_12-P1)/np.linalg.norm(P2-P1) > 1:
            print("P0 is closer to P2")
            d_12 = np.linalg.norm(P0-P2)
            print(f"d_12: {d_12}")
        else:
            print("P0 is closer to the edge P1P2 rather than its vertices")
            d_12 = np.linalg.norm(P0-P0ii_12)
            print(f"d_12: {d_12}")

    # Find the point P0ii_23, the projection of P0i
    # onto edge P2P3
    if np.linalg.norm(R23) > 1e-8:
        print("R23 has length != 0")
        cosgamma23 = np.dot(P2-P0i, R23)/(np.linalg.norm(P2-P0i)*np.linalg.norm(R23))
        print(f"cosgamma23: {cosgamma23}")
        P0iP0ii_23_l = np.linalg.norm(P2-P0i)*cosgamma23
        print(f"P0iP0ii_23_l: {P0iP0ii_23_l}")
        P0iP0ii_23 = R23 * P0iP0ii_23_l/np.linalg.norm(R23)
        print(f"P0iP0ii_23: {P0iP0ii_23}")
        P0ii_23 = P0i + P0iP0ii_23
        print(f"P0ii_23: {P0ii_23}")
    else:
        print("R23 has length 0")
        P0ii_23 = P0i
        print(f"P0ii_23: {P0ii_23}")

    # Find out if P0 is closer to the edge itself or one
    # of the two vertices
    cosbeta23 = np.dot(P0ii_23-P2, P3-P2)/(np.linalg.norm(P0ii_23-P2)*np.linalg.norm(P3-P2))
    print(f"cosbeta23: {cosbeta23}")
    if cosbeta23 < 0:
        print("P0 is closer to P2")
        d_23 = np.linalg.norm(P0-P2)
        print(f"d_23: {d_23}")
    else:
        if np.linalg.norm(P0ii_23-P2)/np.linalg.norm(P3-P2) > 1:
            print("P0 is closer to P3")
            d_23 = np.linalg.norm(P0-P3)
            print(f"d_23: {d_23}")
        else:
            print("P0 is closer to the edge P2P3 rather than its vertices")
            d_23 = np.linalg.norm(P0-P0ii_23)
            print(f"d_23: {d_23}")

    # Find the point P0ii_31, the projection of P0i
    # onto edge P3P1
    if np.linalg.norm(R31) > 1e-8:
        print("R31 has length != 0")
        cosgamma31 = np.dot(P3-P0i, R31)/(np.linalg.norm(P3-P0i)*np.linalg.norm(R31))
        print(f"cosgamma31: {cosgamma31}")
        P0iP0ii_31_l = np.linalg.norm(P3-P0i)*cosgamma31
        print(f"P0iP0ii_31_l: {P0iP0ii_31_l}")
        P0iP0ii_31 = R31 * P0iP0ii_31_l/np.linalg.norm(R31)
        print(f"P0iP0ii_31: {P0iP0ii_31}")
        P0ii_31 = P0i + P0iP0ii_31
        print(f"P0ii_31: {P0ii_31}")
    else:
        print("R31 has length 0")
        P0ii_31 = P0i
        print(f"P0ii_31: {P0ii_31}")

    # Find out if P0 is closer to the edge itself or one
    # of the two vertices
    cosbeta31 = np.dot(P0ii_31-P3, P1-P3)/(np.linalg.norm(P0ii_31-P3)*np.linalg.norm(P1-P3))
    print(f"cosbeta31: {cosbeta31}")
    if cosbeta31 < 0:
        print("P0 is closer to P3")
        d_31 = np.linalg.norm(P0-P3)
        print(f"d_31: {d_31}")
    else:
        if np.linalg.norm(P0ii_31-P3)/np.linalg.norm(P1-P3) > 1:
            print("P0 is closer to P1")
            d_31 = np.linalg.norm(P0-P1)
            print(f"d_23: {d_31}")
        else:
            print("P0 is closer to the edge P3P1 rather than its vertices")
            d_31 = np.linalg.norm(P0-P0ii_31)
            print(f"d_23: {d_31}")


    return min(d_12, d_23, d_31)
