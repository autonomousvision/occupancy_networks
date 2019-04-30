#!/usr/bin/env python
"""
Some I/O utilities.
"""

import os
import time
import h5py
import math
import numpy as np

def write_hdf5(file, tensor, key = 'tensor'):
    """
    Write a simple tensor, i.e. numpy array ,to HDF5.

    :param file: path to file to write
    :type file: str
    :param tensor: tensor to write
    :type tensor: numpy.ndarray
    :param key: key to use for tensor
    :type key: str
    """

    assert type(tensor) == np.ndarray, 'expects numpy.ndarray'

    h5f = h5py.File(file, 'w')

    chunks = list(tensor.shape)
    if len(chunks) > 2:
        chunks[2] = 1
        if len(chunks) > 3:
            chunks[3] = 1
            if len(chunks) > 4:
                chunks[4] = 1

    h5f.create_dataset(key, data = tensor, chunks = tuple(chunks), compression = 'gzip')
    h5f.close()

def read_hdf5(file, key = 'tensor'):
    """
    Read a tensor, i.e. numpy array, from HDF5.

    :param file: path to file to read
    :type file: str
    :param key: key to read
    :type key: str
    :return: tensor
    :rtype: numpy.ndarray
    """

    assert os.path.exists(file), 'file %s not found' % file

    h5f = h5py.File(file, 'r')

    assert key in h5f.keys(), 'key %s not found in file %s' % (key, file)
    tensor = h5f[key][()]
    h5f.close()

    return tensor

def write_off(file, vertices, faces):
    """
    Writes the given vertices and faces to OFF.

    :param vertices: vertices as tuples of (x, y, z) coordinates
    :type vertices: [(float)]
    :param faces: faces as tuples of (num_vertices, vertex_id_1, vertex_id_2, ...)
    :type faces: [(int)]
    """

    num_vertices = len(vertices)
    num_faces = len(faces)

    assert num_vertices > 0
    assert num_faces > 0

    with open(file, 'w') as fp:
        fp.write('OFF\n')
        fp.write(str(num_vertices) + ' ' + str(num_faces) + ' 0\n')

        for vertex in vertices:
            assert len(vertex) == 3, 'invalid vertex with %d dimensions found (%s)' % (len(vertex), file)
            fp.write(str(vertex[0]) + ' ' + str(vertex[1]) + ' ' + str(vertex[2]) + '\n')

        for face in faces:
            assert face[0] == 3, 'only triangular faces supported (%s)' % file
            assert len(face) == 4, 'faces need to have 3 vertices, but found %d (%s)' % (len(face), file)

            for i in range(len(face)):
                assert face[i] >= 0 and face[i] < num_vertices, 'invalid vertex index %d (of %d vertices) (%s)' % (face[i], num_vertices, file)

                fp.write(str(face[i]))
                if i < len(face) - 1:
                    fp.write(' ')

            fp.write('\n')

        # add empty line to be sure
        fp.write('\n')

def read_off(file):
    """
    Reads vertices and faces from an off file.

    :param file: path to file to read
    :type file: str
    :return: vertices and faces as lists of tuples
    :rtype: [(float)], [(int)]
    """

    assert os.path.exists(file), 'file %s not found' % file

    with open(file, 'r') as fp:
        lines = fp.readlines()
        lines = [line.strip() for line in lines]

        # Fix for ModelNet bug were 'OFF' and the number of vertices and faces are
        # all in the first line.
        if len(lines[0]) > 3:
            assert lines[0][:3] == 'OFF' or lines[0][:3] == 'off', 'invalid OFF file %s' % file

            parts = lines[0][3:].split(' ')
            assert len(parts) == 3

            num_vertices = int(parts[0])
            assert num_vertices > 0

            num_faces = int(parts[1])
            assert num_faces > 0

            start_index = 1
        # This is the regular case!
        else:
            assert lines[0] == 'OFF' or lines[0] == 'off', 'invalid OFF file %s' % file

            parts = lines[1].split(' ')
            assert len(parts) == 3

            num_vertices = int(parts[0])
            assert num_vertices > 0

            num_faces = int(parts[1])
            assert num_faces > 0

            start_index = 2

        vertices = []
        for i in range(num_vertices):
            vertex = lines[start_index + i].split(' ')
            vertex = [float(point.strip()) for point in vertex if point != '']
            assert len(vertex) == 3

            vertices.append(vertex)

        faces = []
        for i in range(num_faces):
            face = lines[start_index + num_vertices + i].split(' ')
            face = [index.strip() for index in face if index != '']

            # check to be sure
            for index in face:
                assert index != '', 'found empty vertex index: %s (%s)' % (lines[start_index + num_vertices + i], file)

            face = [int(index) for index in face]

            assert face[0] == len(face) - 1, 'face should have %d vertices but as %d (%s)' % (face[0], len(face) - 1, file)
            assert face[0] == 3, 'only triangular meshes supported (%s)' % file
            for index in face:
                assert index >= 0 and index < num_vertices, 'vertex %d (of %d vertices) does not exist (%s)' % (index, num_vertices, file)

            assert len(face) > 1

            faces.append(face)

        return vertices, faces

    assert False, 'could not open %s' % file

def write_obj(file, vertices, faces):
    """
    Writes the given vertices and faces to OBJ.

    :param vertices: vertices as tuples of (x, y, z) coordinates
    :type vertices: [(float)]
    :param faces: faces as tuples of (num_vertices, vertex_id_1, vertex_id_2, ...)
    :type faces: [(int)]
    """

    num_vertices = len(vertices)
    num_faces = len(faces)

    assert num_vertices > 0
    assert num_faces > 0

    with open(file, 'w') as fp:
        for vertex in vertices:
            assert len(vertex) == 3, 'invalid vertex with %d dimensions found (%s)' % (len(vertex), file)
            fp.write('v' + ' ' + str(vertex[0]) + ' ' + str(vertex[1]) + ' ' + str(vertex[2]) + '\n')

        for face in faces:
            assert len(face) == 3, 'only triangular faces supported (%s)' % file
            fp.write('f ')

            for i in range(len(face)):
                assert face[i] >= 0 and face[i] < num_vertices, 'invalid vertex index %d (of %d vertices) (%s)' % (face[i], num_vertices, file)

                # face indices are 1-based
                fp.write(str(face[i] + 1))
                if i < len(face) - 1:
                    fp.write(' ')

            fp.write('\n')

        # add empty line to be sure
        fp.write('\n')

def read_obj(file):
    """
    Reads vertices and faces from an obj file.

    :param file: path to file to read
    :type file: str
    :return: vertices and faces as lists of tuples
    :rtype: [(float)], [(int)]
    """

    assert os.path.exists(file), 'file %s not found' % file

    with open(file, 'r') as fp:
        lines = fp.readlines()
        lines = [line.strip() for line in lines if line.strip()]

        vertices = []
        faces = []
        for line in lines:
            parts = line.split(' ')
            parts = [part.strip() for part in parts if part]

            if parts[0] == 'v':
                assert len(parts) == 4, \
                    'vertex should be of the form v x y z, but found %d parts instead (%s)' % (len(parts), file)
                assert parts[1] != '', 'vertex x coordinate is empty (%s)' % file
                assert parts[2] != '', 'vertex y coordinate is empty (%s)' % file
                assert parts[3] != '', 'vertex z coordinate is empty (%s)' % file

                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == 'f':
                assert len(parts) == 4, \
                    'face should be of the form f v1/vt1/vn1 v2/vt2/vn2 v2/vt2/vn2, but found %d parts (%s) instead (%s)' % (len(parts), line, file)

                components = parts[1].split('/')
                assert len(components) >= 1 and len(components) <= 3, \
                   'face component should have the forms v, v/vt or v/vt/vn, but found %d components instead (%s)' % (len(components), file)
                assert components[0].strip() != '', \
                    'face component is empty (%s)' % file
                v1 = int(components[0])

                components = parts[2].split('/')
                assert len(components) >= 1 and len(components) <= 3, \
                    'face component should have the forms v, v/vt or v/vt/vn, but found %d components instead (%s)' % (len(components), file)
                assert components[0].strip() != '', \
                    'face component is empty (%s)' % file
                v2 = int(components[0])

                components = parts[3].split('/')
                assert len(components) >= 1 and len(components) <= 3, \
                    'face component should have the forms v, v/vt or v/vt/vn, but found %d components instead (%s)' % (len(components), file)
                assert components[0].strip() != '', \
                    'face component is empty (%s)' % file
                v3 = int(components[0])

                #assert v1 != v2 and v2 != v3 and v3 != v2, 'degenerate face detected: %d %d %d (%s)' % (v1, v2, v3, file)
                if v1 == v2 or v2 == v3 or v1 == v3:
                    print('[Info] skipping degenerate face in %s' % file)
                else:
                    faces.append([v1 - 1, v2 - 1, v3 - 1]) # indices are 1-based!
            else:
                assert False, 'expected either vertex or face but got line: %s (%s)' % (line, file)

        return vertices, faces

    assert False, 'could not open %s' % file

def makedir(dir):
    """
    Creates directory if it does not exist.

    :param dir: directory path
    :type dir: str
    """

    if not os.path.exists(dir):
        os.makedirs(dir)

class Mesh:
    """
    Represents a mesh.
    """

    def __init__(self, vertices = [[]], faces = [[]]):
        """
        Construct a mesh from vertices and faces.

        :param vertices: list of vertices, or numpy array
        :type vertices: [[float]] or numpy.ndarray
        :param faces: list of faces or numpy array, i.e. the indices of the corresponding vertices per triangular face
        :type faces: [[int]] fo rnumpy.ndarray
        """

        self.vertices = np.array(vertices, dtype = float)
        """ (numpy.ndarray) Vertices. """

        self.faces = np.array(faces, dtype = int)
        """ (numpy.ndarray) Faces. """

        assert self.vertices.shape[1] == 3
        assert self.faces.shape[1] == 3

    def extents(self):
        """
        Get the extents.

        :return: (min_x, min_y, min_z), (max_x, max_y, max_z)
        :rtype: (float, float, float), (float, float, float)
        """

        min = [0]*3
        max = [0]*3

        for i in range(3):
            min[i] = np.min(self.vertices[:, i])
            max[i] = np.max(self.vertices[:, i])

        return tuple(min), tuple(max)

    def switch_axes(self, axis_1, axis_2):
        """
        Switch the two axes, this is usually useful for switching y and z axes.

        :param axis_1: index of first axis
        :type axis_1: int
        :param axis_2: index of second axis
        :type axis_2: int
        """

        temp = np.copy(self.vertices[:, axis_1])
        self.vertices[:, axis_1] = self.vertices[:, axis_2]
        self.vertices[:, axis_2] = temp

    def mirror(self, axis):
        """
        Mirror given axis.

        :param axis: axis to mirror
        :type axis: int
        """

        self.vertices[:, axis] *= -1

    def scale(self, scales):
        """
        Scale the mesh in all dimensions.

        :param scales: tuple of length 3 with scale for (x, y, z)
        :type scales: (float, float, float)
        """

        assert len(scales) == 3

        for i in range(3):
            self.vertices[:, i] *= scales[i]

    def translate(self, translation):
        """
        Translate the mesh.

        :param translation: translation as (x, y, z)
        :type translation: (float, float, float)
        """

        assert len(translation) == 3

        for i in range(3):
            self.vertices[:, i] += translation[i]

    def _rotate(self, R):

        self.vertices = np.dot(R, self.vertices.T)
        self.vertices = self.vertices.T

    def rotate(self, rotation):
        """
        Rotate the mesh.

        :param rotation: rotation in (angle_x, angle_y, angle_z); angles in radians
        :type rotation: (float, float, float
        :return:
        """

        assert len(rotation) == 3

        x = rotation[0]
        y = rotation[1]
        z = rotation[2]

        # rotation around the x axis
        R = np.array([[1, 0, 0], [0, math.cos(x), -math.sin(x)], [0, math.sin(x), math.cos(x)]])
        self._rotate(R)

        # rotation around the y axis
        R = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
        self._rotate(R)

        # rotation around the z axis
        R = np.array([[math.cos(z), -math.sin(z), 0], [math.sin(z), math.cos(z), 0], [0, 0, 1]])
        self._rotate(R)

    def inv_rotate(self, rotation):
        """
        Rotate the mesh.

        :param rotation: rotation in (angle_x, angle_y, angle_z); angles in radians
        :type rotation: (float, float, float
        :return:
        """

        assert len(rotation) == 3

        x = rotation[0]
        y = rotation[1]
        z = rotation[2]

        # rotation around the x axis
        R = np.array([[1, 0, 0], [0, math.cos(x), -math.sin(x)], [0, math.sin(x), math.cos(x)]])
        R = R.T
        self._rotate(R)

        # rotation around the y axis
        R = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
        R = R.T
        self._rotate(R)

        # rotation around the z axis
        R = np.array([[math.cos(z), -math.sin(z), 0], [math.sin(z), math.cos(z), 0], [0, 0, 1]])
        R = R.T
        self._rotate(R)

    def copy(self):
        """
        Copy the mesh.

        :return: copy of the mesh
        :rtype: Mesh
        """

        mesh = Mesh(self.vertices.copy(), self.faces.copy())

        return mesh

    @staticmethod
    def from_off(filepath):
        """
        Read a mesh from OFF.

        :param filepath: path to OFF file
        :type filepath: str
        :return: mesh
        :rtype: Mesh
        """

        vertices, faces = read_off(filepath)

        real_faces = []
        for face in faces:
            assert len(face) == 4
            real_faces.append([face[1], face[2], face[3]])

        return Mesh(vertices, real_faces)

    def to_off(self, filepath):
        """
        Write mesh to OFF.

        :param filepath: path to write file to
        :type filepath: str
        """

        faces = np.ones((self.faces.shape[0], 4), dtype = int)*3
        faces[:, 1:4] = self.faces[:, :]

        write_off(filepath, self.vertices.tolist(), faces.tolist())

    @staticmethod
    def from_obj(filepath):
        """
        Read a mesh from OBJ.

        :param filepath: path to OFF file
        :type filepath: str
        :return: mesh
        :rtype: Mesh
        """

        vertices, faces = read_obj(filepath)
        return Mesh(vertices, faces)

    def to_obj(self, filepath):
        """
        Write mesh to OBJ file.

        :param filepath: path to OBJ file
        :type filepath: str
        """

        write_obj(filepath, self.vertices.tolist(), self.faces.tolist())

class Timer:
    """
    Simple wrapper for time.clock().
    """

    def __init__(self):
        """
        Initialize and start timer.
        """

        self.start = time.clock()
        """ (float) Seconds. """

    def reset(self):
        """
        Reset timer.
        """

        self.start = time.clock()

    def elapsed(self):
        """
        Get elapsed time in seconds

        :return: elapsed time in seconds
        :rtype: float
        """

        return (time.clock() - self.start)
