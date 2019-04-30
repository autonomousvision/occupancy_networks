#!/usr/bin/env python
#
# Tests distance between point and triangle in 3D. Aligns and uses 2D technique.
#
# Was originally some code on mathworks
#
# Implemented for pytorch Variable
# Adapted from https://gist.github.com/joshuashaffer/99d58e4ccbd37ca5d96e

import numpy as np
import torch
from torch.autograd import Variable
import time

one = Variable(torch.ones(1).type(torch.FloatTensor), requires_grad=True)
eps = 1e-8

def pointTriangleDistanceFast(TRI, P):
    # function [dist,PP0] = pointTriangleDistanceFast(TRI,P)
    # calculate distance between a set of points and a triangle in 3D
    # Approximate method, clamp s and t to enable batch calculation
    #
    # Input:
    #       TRI: 3x3 matrix, each column is a vertex
    #       P:   Nx3 matrix, each row is a point
    # Output:
    #       dis: Nx1 matrix, point to triangle distances

    assert(np.isnan(np.sum(TRI.data.cpu().numpy()))==0)
    assert(np.isnan(np.sum(P.data.cpu().numpy()))==0)
    B = TRI[:, 0]
    E0 = TRI[:, 1] - B
    E1 = TRI[:, 2] - B

    D = B.unsqueeze(0).expand_as(P) - P

    d = D.mm(E0.unsqueeze(1))
    e = D.mm(E1.unsqueeze(1))
    f = torch.diag(D.mm(torch.t(D))).unsqueeze(1)

    a = torch.dot(E0, E0).expand_as(d)
    b = torch.dot(E0, E1).expand_as(d)
    c = torch.dot(E1, E1).expand_as(d)


    #print "{0} {1} {2} ".format(B,E1,E0)
    det = a * c - b * b
    s = (b * e - c * d)/(det + eps)
    t = (b * d - a * e)/(det + eps)
    # clamp s and t to be larger than 0
    s = s.clamp(min=0)
    t = t.clamp(min=0)
    # clamp the sum of s and t to be smaller than 1
    norm = (s+t).clamp(min=1)
    s = s/norm
    t = t/norm
    sqrdistance = s * (a * s + b * t + 2.0 * d) + t * (b * s + c * t + 2.0 * e) + f
    # directly return sqrdistance
    #return torch.sqrt(sqrdistance.clamp(min=0) + eps)
    return sqrdistance.clamp(min=0)


def pointTriangleDistance(TRI, P):
    # function [dist,PP0] = pointTriangleDistance(TRI,P)
    # calculate distance between a point and a triangle in 3D
    # SYNTAX
    #   dist = pointTriangleDistance(TRI,P)
    #   [dist,PP0] = pointTriangleDistance(TRI,P)
    #
    # DESCRIPTION
    #   Calculate the distance of a given point P from a triangle TRI.
    #   Point P is a row vector of the form 1x3. The triangle is a matrix
    #   formed by three rows of points TRI = [P1;P2;P3] each of size 1x3.
    #   dist = pointTriangleDistance(TRI,P) returns the distance of the point P
    #   to the triangle TRI.
    #   [dist,PP0] = pointTriangleDistance(TRI,P) additionally returns the
    #   closest point PP0 to P on the triangle TRI.
    #
    # Author: Gwolyn Fischer
    # Release: 1.0
    # Release date: 09/02/02
    # Release: 1.1 Fixed Bug because of normalization
    # Release: 1.2 Fixed Bug because of typo in region 5 20101013
    # Release: 1.3 Fixed Bug because of typo in region 2 20101014

    # Possible extention could be a version tailored not to return the distance
    # and additionally the closest point, but instead return only the closest
    # point. Could lead to a small speed gain.

    # Example:
    # %% The Problem
    # P0 = [0.5 -0.3 0.5]
    #
    # P1 = [0 -1 0]
    # P2 = [1  0 0]
    # P3 = [0  0 0]
    #
    # vertices = [P1; P2; P3]
    # faces = [1 2 3]
    #
    # %% The Engine
    # [dist,PP0] = pointTriangleDistance([P1;P2;P3],P0)
    #
    # %% Visualization
    # [x,y,z] = sphere(20)
    # x = dist*x+P0(1)
    # y = dist*y+P0(2)
    # z = dist*z+P0(3)
    #
    # figure
    # hold all
    # patch('Vertices',vertices,'Faces',faces,'FaceColor','r','FaceAlpha',0.8)
    # plot3(P0(1),P0(2),P0(3),'b*')
    # plot3(PP0(1),PP0(2),PP0(3),'*g')
    # surf(x,y,z,'FaceColor','b','FaceAlpha',0.3)
    # view(3)

    # The algorithm is based on
    # "David Eberly, 'Distance Between Point and Triangle in 3D',
    # Geometric Tools, LLC, (1999)"
    # http:\\www.geometrictools.com/Documentation/DistancePoint3Triangle3.pdf
    #
    #        ^t
    #  \     |
    #   \reg2|
    #    \   |
    #     \  |
    #      \ |
    #       \|
    #        *P2
    #        |\
    #        | \
    #  reg3  |  \ reg1
    #        |   \
    #        |reg0\
    #        |     \
    #        |      \ P1
    # -------*-------*------->s
    #        |P0      \
    #  reg4  | reg5    \ reg6
    # rewrite triangle in normal form


    #
    reg = -1

    assert(np.isnan(np.sum(TRI.data.numpy()))==0)
    assert(np.isnan(np.sum(P.data.numpy()))==0)
    B = TRI[:, 0]
    E0 = TRI[:, 1] - B
    # E0 = E0/sqrt(sum(E0.^2)); %normalize vector
    E1 = TRI[:, 2] - B
    # E1 = E1/sqrt(sum(E1.^2)); %normalize vector
    D = B - P
    a = torch.dot(E0, E0)
    b = torch.dot(E0, E1)
    c = torch.dot(E1, E1)
    d = torch.dot(E0, D)
    e = torch.dot(E1, D)
    f = torch.dot(D, D)

    #print "{0} {1} {2} ".format(B,E1,E0)
    det = a * c - b * b
    s = b * e - c * d
    t = b * d - a * e



    # Terible tree of conditionals to determine in which region of the diagram
    # shown above the projection of the point into the triangle-plane lies.
    if (s.data[0] + t.data[0]) <= det.data[0]:
        if s.data[0] < 0.0:
            if t.data[0] < 0.0:
                # region4
                reg = 4
                if d.data[0] < 0:
                    t = 0.0 
                    if -d.data[0] >= a.data[0]:
                        s = 1.0 
                        sqrdistance = a + 2.0 * d + f
                    else:
                        s = -d / (a + eps)
                        sqrdistance = d * s + f
                else:
                    s.data[0] = 0.0
                    if e.data[0] >= 0.0:
                        t = 0.0 
                        sqrdistance = f
                    else:
                        if -e.data[0] >= c.data[0]:
                            t = 1.0 
                            sqrdistance = c + 2.0 * e + f
                        else:
                            t = -e / (c + eps)
                            sqrdistance = e * t + f

                            # of region 4
            else:
                reg = 3
                # region 3
                s.data[0] = 0
                if e.data[0] >= 0:
                    t = 0 
                    sqrdistance = f
                else:
                    if -e.data[0] >= c.data[0]:
                        t = 1  
                        sqrdistance = c + 2.0 * e + f
                    else:
                        t = -e / (c + eps)
                        sqrdistance = e * t + f
                        # of region 3
        else:
            if t.data[0] < 0:
                reg = 5
                # region 5
                t = 0  
                if d.data[0] >= 0:
                    s = 0  
                    sqrdistance = f
                else:
                    if -d.data[0] >= a.data[0]:
                        s = 1.0  
                        sqrdistance = a + 2.0 * d + f;  # GF 20101013 fixed typo d*s ->2*d
                    else:
                        s = -d / (a + eps)
                        sqrdistance = d * s + f
            else:
                reg = 0
                # region 0
                invDet = 1.0 / (det + eps)
                s = s * invDet
                t = t * invDet
                if s.data[0] == 0:
                    sqrdistance = d
                else:
                    sqrdistance = s * (a * s + b * t + 2.0 * d) + t * (b * s + c * t + 2.0 * e) + f
    else:
        if s.data[0] < 0.0:
            reg = 2
            # region 2
            tmp0 = b + d
            tmp1 = c + e
            if tmp1.data[0] > tmp0.data[0]:  # minimum on edge s+t=1
                numer = tmp1 - tmp0
                denom = a - 2.0 * b + c
                if numer.data[0] >= denom.data[0]:
                    s = 1.0  
                    t = 0.0  
                    sqrdistance = a + 2.0 * d + f;  # GF 20101014 fixed typo 2*b -> 2*d
                else:
                    s = numer / (denom + eps)
                    t = 1 - s
                    sqrdistance = s * (a * s + b * t + 2 * d) + t * (b * s + c * t + 2 * e) + f

            else:  # minimum on edge s=0
                s = 0.0  
                if tmp1.data[0] <= 0.0:
                    t = 1  
                    sqrdistance = c + 2.0 * e + f
                else:
                    if e.data[0] >= 0.0:
                        t = 0.0  
                        sqrdistance = f
                    else:
                        t = -e / (c + eps)
                        sqrdistance = e * t + f
                        # of region 2
        else:
            if t.data[0] < 0.0:
                reg = 6
                # region6
                tmp0 = b + e
                tmp1 = a + d
                if tmp1.data[0] > tmp0.data[0]:
                    numer = tmp1 - tmp0
                    denom = a - 2.0 * b + c
                    if numer.data[0] >= denom.data[0]:
                        t = 1.0  
                        s = 0  
                        sqrdistance = c + 2.0 * e + f
                    else:
                        t = numer / (denom + eps)
                        s = 1 - t
                        sqrdistance = s * (a * s + b * t + 2.0 * d) + t * (b * s + c * t + 2.0 * e) + f

                else:
                    t = 0.0
                    if tmp1.data[0] <= 0.0:
                        s = 1  
                        sqrdistance = a + 2.0 * d + f
                    else:
                        if d.data[0] >= 0.0:
                            s = 0.0
                            sqrdistance = f
                        else:
                            s = -d / (a + eps)
                            sqrdistance = d * s + f
            else:
                reg = 1
                # region 1
                numer = c + e - b - d
                if numer.data[0] <= 0:
                    s = 0.0  
                    t = 1.0  
                    sqrdistance = c + 2.0 * e + f
                else:
                    denom = a - 2.0 * b + c
                    if numer.data[0] >= denom.data[0]:
                        s = 1.0  
                        t = 0.0  
                        sqrdistance = a + 2.0 * d + f
                    else:
                        s = numer / (denom + eps)
                        t = 1 - s
                        sqrdistance = s * (a * s + b * t + 2.0 * d) + t * (b * s + c * t + 2.0 * e) + f

    # account for numerical round-off error
    #dist = torch.sqrt(torch.max(sqrdistance, 0*one))

    # directly return sqr distance
    dist = torch.max(sqrdistance, 0*one)

    #PP0 = B + s.expand_as(E0) * E0 + t.expand_as(E1) * E1
    assert(np.isnan(dist.data[0])==0)
    return dist, reg

if __name__ == '__main__':

    P = Variable(torch.FloatTensor([[-2.0,1.0,1.5]]).view(-1,3), requires_grad=True)
    TRI = Variable(torch.t(torch.FloatTensor([[0.0, 0.0,1.0],[1.0,0.0,0.0],[0.0,1.0,0.0]])), requires_grad=True)

    # approximate batch method
    #P = Variable(torch.randn(100, 3), requires_grad=True) 
    #TRI =  Variable(torch.rand(3, 3), requires_grad=True)
    t0 = time.time()
    dists = pointTriangleDistanceFast(TRI, P)
    t0 = time.time()-t0
    dists.backward()
    print(dists)
    print(TRI.grad.data.numpy())
    print(P.grad.data.numpy())

    ## accurate method
    #t1 = time.time()
    #for i in range(P.size()[0]):
    #    dist, reg = pointTriangleDistance(TRI,P[i, :])
    #    print '%f' % (dist.data[0]-dists[i,0].data[0]), reg
    #t1 = time.time()-t1
    #print "Approximate method time: %f, accurate method time: %f" % (t0, t1)
