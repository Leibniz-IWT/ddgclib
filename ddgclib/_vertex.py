import collections
from abc import ABC, abstractmethod
import logging
import copy
import numpy as np
from functools import partial
import multiprocessing as mp

#from hyperct._field import *

"""Vertex objects"""
class VertexBase(ABC):
    def __init__(self, x, nn=None, index=None):
        self.x = x
        self.hash = hash(self.x)  # Save precomputed hash
        #self.orderv = sum(x)  #TODO: Delete if we can't prove the order triangulation conjecture

        if nn is not None:
            self.nn = set(nn)  # can use .indexupdate to add a new list
        else:
            self.nn = set()

        self.index = index

    def __hash__(self):
        return self.hash
        return self.hash

    def __mul__(self, v2):
        pass
        #s = SimplexOrdered([self, v2])
        #return s

    def __getattr__(self, item):
        if item not in ['x_a']:
            raise AttributeError(f"{type(self)} object has no attribute "
                                 f"'{item}'")
        if item == 'x_a':
            self.x_a = np.array(self.x, dtype=np.longdouble)
            return self.x_a


    @abstractmethod
    def connect(self, v):
        raise NotImplementedError("This method is only implemented with an "
                                  "associated child of the base class.")

    @abstractmethod
    def disconnect(self, v):
        raise NotImplementedError("This method is only implemented with an "
                                  "associated child of the base class.")

    def print_out(self):
        print("Vertex: {}".format(self.x))
        constr = 'Connections: '
        for vc in self.nn:
            constr += '{} '.format(vc.x)

        print(constr)
        #print('Order = {}'.format(self.order))

    def star(self):
        """
        Returns the star domain st(v) of the vertex.

        :param v: The vertex v in st(v)
        :return: st, a set containing all the vertices in st(v)
        """
        self.st = self.nn
        self.st.add(self)
        return self.st

class VertexCube(VertexBase):
    """Vertex class to be used for a pure simplicial complex with no associated
    differential geometry (single level domain that exists in R^n)"""
    def __init__(self, x, nn=None, index=None):
        super().__init__(x, nn=nn, index=index)

    def connect(self, v):
        if v is not self and v not in self.nn:
            self.nn.add(v)  #TODO: Use update for adding multiple vectors?
            v.nn.add(self)

    def disconnect(self, v):
        if v in self.nn:
            self.nn.remove(v)
            v.nn.remove(self)


class VertexScalarField(VertexBase):
    """Add homology properties of a scalar field f: R^n --> R associated with
    the geometry built from the VertexBase class"""

    def __init__(self, x, field=None, nn=None, index=None, field_args=(),
                 g_cons=None, g_cons_args=()):
        """
        :param x: tuple, vector of vertex coordinates
        :param field: function, a scalar field f: R^n --> R associated with
                      the geometry
        :param nn: list, optional, list of nearest neighbours
        :param index: int, optional, index of the vertex
        :param field_args: tuple, additional arguments to be passed to field
        :param g_cons: function, constraints on the vertex
        :param g_cons_args: tuple, additional arguments to be passed to g_cons

        """
        super().__init__(x, nn=nn, index=index)

        # Note Vertex is only initiated once for all x so only
        # evaluated once
        #self.feasible = None

        # self.f is externally defined by the cache to allow parallel
        # processing
        #self.f = None  # None type that will break arithmetic operations unless
                       # defined

        self.check_min = True
        self.check_max = True


    def connect(self, v):
        if v is not self and v not in self.nn:
            self.nn.add(v)
            v.nn.add(self)

            # Flags for checking homology properties:
            self.check_min = True
            self.check_max = True
            v.check_min = True
            v.check_max = True

    def disconnect(self, v):
        if v in self.nn:
            self.nn.remove(v)
            v.nn.remove(self)

            # Flags for checking homology properties:
            self.check_min = True
            self.check_max = True
            v.check_min = True
            v.check_max = True

    def minimiser(self):
        """Check whether this vertex is strictly less than all its neighbours"""
        if self.check_min:
            self._min = all(self.f < v.f for v in self.nn)
            self.check_min = False

        return self._min

    def maximiser(self):
        """Check whether this vertex is strictly greater than all its
        neighbours"""
        if self.check_max:
            self._max = all(self.f > v.f for v in self.nn)
            self.check_max = False

        return self._max

class VertexVectorField(VertexBase):
    """Add homology properties of a scalar field f: R^n --> R^m associated with
    the geometry built from the VertexBase class"""

    def __init__(self, x, sfield=None, vfield=None, field_args=(),
                 vfield_args=(), g_cons=None,
                 g_cons_args=(), nn=None, index=None):
        super(VertexVectorField, self).__init__(x, nn=nn, index=index)

        raise NotImplementedError("This class is still a work in progress")

"""
Cache objects
"""
class VertexCacheBase(object):
    def __init__(self):

        self.cache = collections.OrderedDict()  #TODO: Perhaps unneeded?
        self.nfev = 0  # Feasible points
        self.index = -1  #TODO: Is this needed?

        #TODO: Define a getitem method based on if indexing is on or not so
        # that we do not have to do an if check every call (does the python
        # compiler make this irrelevant or not?) and in addition whether or not
        # we have defined a field function.

    def __iter__(self):
        # We use the following iterator to allow for mutation of self.cache
        iterator = range(len(self.cache))
        items = list(self.cache.items())
        for ind in iterator:
            yield items[ind][1]
            #try:
            #    yield items[ind][1]
            #except KeyError:
            #    continue

        # Alternative:
        #for v in self.cache:
        #    yield self.cache[v]
        return

    def move(self, v, x):
        """
        Move a vertex object v to a new set of coordinates x

        NOTE: You might to recompute f and g pools depending on your application

        :param v: Vertex object to move
        :param x: tuple, new coordinates
        :return:
        """
        #TODO: Instead of using pop try to retain order in list
        # (This turns out to be impractically expensive, requiring iteration
        # of the entire cache)
        self.cache.pop(v.x)

        # Note that we need to remove the object from the nn sets since the hash
        # value is changed although the object stays the same.
        vn = copy.copy(v.nn)
        for vn_i in vn:
            v.disconnect(vn_i)

        v.x = x
        v.hash = hash(x)
        try:
            v.x_a = np.array(x, dtype=np.longdouble)
        except AttributeError:
            pass

        self.cache[x] = v
        # Reconnect new hashes
        for vn_i in vn:
            v.connect(vn_i)

        return self.cache[x]

    def remove(self, v):
        """
        :param v:  Vertex object to remove
        :return:
        """
        #if not (v in self.cache):
        #    return
        ind = v.index

        vn = copy.copy(v.nn)
        for vn in vn:
            v.disconnect(vn)

        self.cache.pop(v.x)

        for v in self:
            if v.index > ind:
                v.index -= 1
        self.index -= 1
        return

    def merge_nn(self, cdist, exclude=set()):
        """
        ICannon 2024 Nov 28
        Merge connected neighbors within the Euclidean norm cdist

        TODO: Use other distants metrics (requires dependencies)

        #NOTE: Appears to work, but re
        :param v:
        :param cdist: float, vertices less than this distance will be merged
        :return:
        """
        # Detect candidate pairs for merging
        merge_pairs_l = []
        merge_vertices_l = []

        for v in self:
            if v in exclude: continue
            if v in merge_vertices_l: continue
            for v2 in v.nn:  
                if v2 in exclude: continue
                if v is v2:
                    continue
                if v2 in merge_vertices_l:
                    continue
                dist = np.linalg.norm(v.x_a - v2.x_a)
                if dist < cdist:
                    merge_pairs_l.append((v, v2))
                    merge_vertices_l.append(v)
                    merge_vertices_l.append(v2)

        n_merged = 0
        for vp in merge_pairs_l:
            try:
                self.merge_pair(vp)
                n_merged += 1 
            except KeyError:
                pass  # pairs have possibly aleady been removed
        return n_merged

    def merge_all(self, cdist):
        """
        Merge all within the Euclidean norm cdist

        TODO: Use other distants metrics (requires dependencies)

        #NOTE: Appears to work, but re
        :param v:
        :param cdist: float, vertices less than this distance will be merged
        :return:
        """
        # Detect candidate pairs for merging
        merge_pairs = set()
        merge_pairs_l = []

        for v in self:
            # for v2 in v.nn:  # NOTE: Can't use nn, because v2 not always in v.nn
            for v2 in self:
                if v is v2:
                    continue
                dist = np.linalg.norm(v.x_a - v2.x_a)
                if dist < cdist:
                    #TODO: Something here is not right
                    if (frozenset({v, v2}) in merge_pairs) or (frozenset({v2, v}) in merge_pairs):
                        continue
                    else:
                        merge_pairs_l.append((v, v2))
                        merge_pairs.add(frozenset({v, v2}))

        #print(f'merge_pairs = {merge_pairs}')
        for vp in merge_pairs_l:
            #self.merge_pair(vp)
            try:
                self.merge_pair(vp)
            except KeyError:
                pass  # pairs have possibly aleady been removed

    def merge_pair(self, vp):
        """
        Merge the vertex pair vp into a single vertex
        :param vp:
        :return:
        """
        #print('merge', vp[0].x_a, vp[1].x_a)
        #print(f'vp[0].nn = {vp[0].nn}')
        #print(f'vp[1].nn = {vp[1].nn}')
        #TODO: Maybe this union operation is the shit part and we should be
        #      doing a loop of .connect operations instead?
        #      NOTE: YES I think this is the issue, try it...
       # nn = vp[0].nn.union(vp[1].nn)  # Union of neighbouring sets
       # nn = nn - set(vp)
        #vp[0].nn = nn
        for vn in vp[1].nn:
            vp[0].connect(vn)

        self.remove(vp[1])

        #TODO: Remove vp[1] from all neighbours? Remove should do this already
        #      But what about the neighbours of vp[0]?
        #vn = copy.copy(v.nn)
        #for vn in vn:
        #    v.disconnect(vn)
        #NO! Bcause

        #TEST"
       # for v2 in vp[0].nn



    def size(self):
        """
        Returns the size of the vertex cache

        :return:
        """
        return self.index + 1

    def print_out(self):
        headlen = len(f"Vertex cache of size: {len(self.cache)}:")
        print('=' * headlen)
        print(f"Vertex cache of size: {len(self.cache)}:")
        print('=' * headlen)
        for v in self.cache:
            self.cache[v].print_out()

class VertexCacheIndex(VertexCacheBase):
    def __init__(self):
        #TODO: Allow for optional constraint arguments for non-linear
        # triangulations
        super().__init__()
        self.Vertex = VertexCube

    def __getitem__(self, x, nn=None):  #TODO: Check if no_index is significant speedup
        try:
            return self.cache[x]
        except KeyError:
            self.index += 1
            xval = self.Vertex(x, index=self.index)
            # logging.info("New generated vertex at x = {}".format(x))
            # NOTE: Surprisingly high performance increase if logging is commented out
            self.cache[x] = xval
            return self.cache[x]

class VertexCacheField(VertexCacheBase):
    def __init__(self, field=None, field_args=(), g_cons=None, g_cons_args=(),
                 workers=None):
        #TODO: Make a non-linear constraint cache with no scalar field
        #TODO: add possible h_cons tolerance check
        super().__init__()
        self.index = -1
        self.Vertex = VertexScalarField
        self.field = field
        self.field_args = field_args
        self.wfield = FieldWraper(field, field_args)  # if workers is not None

        self.g_cons = g_cons
        self.g_cons_args = g_cons_args
        self.wgcons = ConstraintWraper(g_cons, g_cons_args)
        self.gpool = set()  # A set of tuples to process for feasibility

        # Field processing objects
        self.fpool = set()  # A set of tuples to process for scalar function
        self.sfc_lock = False  # True if self.fpool is non-Empty

        if g_cons == None:
            self.proc_fpool = self.proc_fpool_nog
        else:
            self.proc_fpool = self.proc_fpool_g

        if workers == None:
            self.process_gpool = self.proc_gpool
            #self.process_fpool = self.proc_fpool
            if g_cons == None:
                self.process_fpool = self.proc_fpool_nog
            else:
                self.process_fpool = self.proc_fpool_g
        else:
            self.workers = workers
            self.pool = mp.Pool(processes=workers)  #TODO: Move this pool to
                                                    # the complex object
            self.process_gpool = self.pproc_gpool
            if g_cons == None:
                self.process_fpool = self.pproc_fpool_nog
            else:
                self.process_fpool = self.pproc_fpool_g


    def __getitem__(self, x, nn=None): #TODO: Test to add optional nn argument?
        #NOTE: To use nn arg do ex. V.__getitem__((1,2,3), [3,4,7])
        try:
            return self.cache[x]
        except KeyError:
            self.index += 1
            xval = self.Vertex(x, field=self.field, nn=nn, index=self.index,
                               field_args=self.field_args,
                               g_cons=self.g_cons, g_cons_args=self.g_cons_args)

            self.cache[x] = xval  # Define in cache
            self.gpool.add(xval)  # Add to pool for processing feasibility
            self.fpool.add(xval)  # Add to pool for processing field values
            return self.cache[x]

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        try:  #TODO: Reason for this?
            del self_dict['pool']
        except KeyError:
            pass
        
        return self_dict

    def process_pools(self):
        if self.g_cons is not None:
            self.process_gpool()
        self.process_fpool()
        self.proc_minimisers()

    def recompute_pools(self):
        pass
        #TODO: This will recompute pools to include vertices with missing info
        #      and purge vertices that already have info computed. Only to be
        #      run when loading data from hard drive disk
        for v in self:
            # Update function checks
            try:
                v.f
                try:
                    self.fpool.remove(v)
                except KeyError:
                    pass
            except AttributeError:
                self.fpool.add(v)

            # Update feasibility checks
            try:
                v.feasible
                try:
                    self.gpool.remove(v)
                except KeyError:
                    pass
            except AttributeError:
                self.gpool.add(v)

        return self.fpool, self.gpool

    def feasibility_check(self, v):
        v.feasible = True
        for g, args in zip(self.g_cons, self.g_cons_args):
            if g(v.x_a, *args) < 0.0:
                v.f = np.inf
                v.feasible = False
                break

    def compute_sfield(self, v):
        try: #TODO: Remove exception handling?
            v.f = self.field(v.x_a, *self.field_args)
            self.nfev += 1
        except:  #TODO: except only various floating issues
            #logging.warning(f"Field function not found at x = {self.x_a}")
            v.f = np.inf
        if np.isnan(v.f):
            v.f = np.inf

    def proc_gpool(self):
        if self.g_cons is not None:
            for v in self.gpool:
                self.feasibility_check(v)
        # Clean the pool
        self.gpool = set()

    def pproc_gpool(self):
        gpool_l = []
        for v in self.gpool:
            gpool_l.append(v.x_a)

        G = self.pool.map(self.wgcons.gcons, gpool_l)
        for v, g in zip(self.gpool, G):
            v.feasible = g  # set vertex object attribute v.feasible = g (bool)

    def proc_fpool_g(self):
        # TODO: do try check if v.f exists
        for v in self.fpool:
            if v.feasible:
                self.compute_sfield(v)
        # Clean the pool
        self.fpool = set()

    def proc_fpool_nog(self):
        # TODO: do try check if v.f exists
        for v in self.fpool:
            self.compute_sfield(v)
        # Clean the pool
        self.fpool = set()

    #TODO: Make static method to possibly improve pickling speed
    def pproc_fpool_g(self):
        #TODO: Ensure that .f is not already computed? (it shouldn't be addable
        #      to the self.fpool if it is).
        self.wfield.func
        fpool_l = []
        for v in self.fpool:
            if v.feasible:
                fpool_l.append(v.x_a)
            else:
                v.f = np.inf
        F = self.pool.map(self.wfield.func, fpool_l)
        for va, f in zip(fpool_l, F):
            vt = tuple(va)
            self[vt].f = f  # set vertex object attribute v.f = f

    def pproc_fpool_nog(self):
        #TODO: Ensure that .f is not already computed? (it shouldn't be addable
        #      to the self.fpool if it is).
        self.wfield.func
        fpool_l = []
        for v in self.fpool:
            fpool_l.append(v.x_a)
        F = self.pool.map(self.wfield.func, fpool_l)
        for va, f in zip(fpool_l, F):
            vt = tuple(va)
            self[vt].f = f  # set vertex object attribute v.f = f

    def proc_minimisers(self):
        """
        Check for minimisers
        :return:
        """
        for v in self:
            v.minimiser()
            v.maximiser()

        if 0:
            if v.minimiser():
                v2._min = False
                v2.check_min = False
            if v.maximiser():
                v2._max = False
                v2.check_max = False


class ConstraintWraper(object):
    def __init__(self, g_cons, g_cons_args):
        self.g_cons = g_cons
        self.g_cons_args = g_cons_args

    def gcons(self, v_x_a):
        vfeasible = True
        for g, args in zip(self.g_cons, self.g_cons_args):
            if g(v_x_a, *args) < 0.0:  #TODO: Add exception handling?
                vfeasible = False
                break
        return vfeasible

class FieldWraper(object):
    def __init__(self, field, field_args):
        self.field = field
        self.field_args = field_args

    def func(self, v_x_a):
        try:
            v_f = self.field(v_x_a, *self.field_args)
        except:  #TODO: except only various floating issues
            # logging.warning(f"Field function not found at x = {self.x_a}")
            v_f = np.inf
        if np.isnan(v_f):
            v_f = np.inf

        return v_f

if __name__ == '__main__':  # TODO: Convert these to unittests
    v1 = VertexCube((1,2,-3.3))

    print(v1)
    print(v1.x)

    Vertex = VertexCube

    v1 = Vertex((1, 2, 3))
    v1 = Vertex((1, 2, 3))
    print(v1)
    #print(v1.x_a)

    def func(x):
        return np.sum((x - 3) ** 2) + 2.0 * (x[0] + 10)


    def g_cons(x):  # (Requires n > 2)
        # return x[0] - 0.5 * x[2] + 0.5
        return x[0]  # + x[2] #+ 0.5


    v1 = VertexScalarField((1, 2, -3.3), func)
    print(v1)
    print(v1.x)
    print(v1.x_a)

    def func(x):
        return np.sum((x - 3) ** 2) + 2.0 * (x[0] + 10)


    def g_cons(x):  # (Requires n > 2)
        # return x[0] - 0.5 * x[2] + 0.5
        return x[0]  # + x[2] #+ 0.5

    #V = VertexCache()
    V = VertexCacheField(func)
    print(V)
    V[(1,2,3)]
    V[(1,2,3)]
    V.__getitem__((1,2,3), None)
    V.__getitem__((1,2,3), [3,4,7])
    #TODO: ADD THIS TO COMPLEX:
