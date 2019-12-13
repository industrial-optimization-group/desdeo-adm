### Declarations

# Calc
import numpy as np
from scipy.stats import wilcoxon # test for difference


# system
import copy
import pickle

import threeobj as th

# Output
import xlsxwriter
import matplotlib.pyplot as plt
import os
import pandas as pd
# from pylab import *

#import multiprocessing as multiproc

## DESDEO
#from desdeo.method.NIMBUS import NIMBUS
#from desdeo.optimization import SciPyDE
#from desdeo.problem.toy import RiverPollution
#from desdeo.preference import NIMBUSClassification

## Rectangles
from rtree import index as rindex

### Supplementary functions

## Calculate hypervolume of a box given min and max points
def hv_box(mn,mx):
    return np.prod([mxi-mni for mni,mxi in zip(mn,mx)])

## Given a nested list, Returns a list of all lists of size k x 2.
# Used for extracting results of the recursive function divbox_rec
# It's also a recursive function
def flat_boxlist(a,k):
    # leaf of recursion calls
    try: # check if "a" is k x 2 list with non-list elements
        if len(a)==k and any( \
               (
                len(ai)==2 and \
                not( any( isinstance(aii,list) for aii in ai ) )
               ) for ai in a
              ):
            return [a]
    except:
        pass
    # next level recursion call: collect the results at the lower level
    if isinstance(a,list):
        return sum([flat_boxlist(i,k) for i in a],[])
    else:
        return []
            
## Given min and max vectors of a box as lists,
#  Returns the vector representation for rtree: 
#  [min_1,min_2,...,min_k,max_1,...,max_k]
def box2rindex(mn,mx):
    return list(mn)+list(mx)

## Given the rtree representation of a box as a list / numpy array
#       [min_1,min_2,...,min_k,max_1,...,max_k],
## Returns the list [[min vector], [max vector]] of the box 
def rindex2box(v):
    return np.array(v).reshape(2,-1).tolist()

## Recursive function for generating all open boxes partitioning a given box,
#  resulted from subtracting the dual domination cone (represented by its vertex)
# Given: 
#  vrange = [
#           for each i: [min,max] if the part was defined for this i, otherwise
#           [min,mid,max] where mid(=the component of the cone's vertex) 
#           belongs to the open range of the box, 
#           and the selection of higher or lower part was not done for this i
#           ]
#  nlo = nr. of dimensions, for which the box range is determined or selected 
#        to be below the vertex component
#  nhi = nr. of dimensions, for which the range is is determined or selected 
#        to be above the vertex component
#  k = total nr. of dimensions
#  ii = currently considered dimension nr. (for previous ones, 
#       the part of the range was determined or selected)
# Returns:
#  if part of the range is defined for all i (nlo+nhi==k), then 
#      [for each i, [min,max]] (the leaf of the recursion)
#  if some range is not defined (nlo+nhi<k), then the result of branching of 
#      the recursion, i.e. for j=(first index with undefined part), 
#      call divbox_rec for the cases of upper and lower parts
def divbox_rec(vrange,nlo,nhi,k,ii):
    # If in each dimension, range of the considered part is defined,
    # then recursion leaf
    if nlo+nhi==k: 
        # the part of the box is not dominated by / dominating the vertex
        if nlo<k and nhi<k: 
            return [vrange]
        # otherwise, the part will not be included in the potential region
        else:
            return []
    # If for some dimension, the part can be divided into higher/lower parts,
    # create two recursion branches for the considered dimension.
        # initialize two versions of the list of ranges
    vlo=copy.deepcopy(vrange) # initial range list -> ranges with the lower part
    vhi=copy.deepcopy(vrange) # initial range list -> ranges of the upper part
    for i in range(ii,k):
        if len(vrange[i])==3: # i <- first index with undefined part
            vlo[i]=vrange[i][:2] # the list version with the lower part
            vhi[i]=vrange[i][1:] # the list version with the upper range
            # all parts of the box is concatenation of the two branches
            return divbox_rec(vlo,nlo+1,nhi,k,i+1) + \
                   divbox_rec(vhi,nlo,nhi+1,k,i+1)
                           

### Potential region structure for minimization problems based on 
#                                                       rtree package class.
#   Box ID (int) attribute assigned to the boxes in the original rtree class
#                   represents the ordinary number of the act of box creation.
#   Additional attributes of the class object:
#    .ndim = nr. of space dimensions 
#    .nbox = number of boxes in the structure
#    .ncre = number of acts of boxes creation
#    ._hypervol = sum of hypervolume of existing boxes
class potreg(rindex.Index):
    
    def __init__(self,ideal,nadir):
        # setting the space dimension and passing to rtree in a Property object
        ndim=len(ideal)
        p = rindex.Property()
        p.dimension = ndim
        # initializing the object
        rindex.Index.__init__(self,properties=p)
        self.ndim = ndim
        # adding the first rectangle
        self.nbox = 1
        self.ncre = 1
        # the initial box = the Pareto range
        self.insert(1,box2rindex(ideal,nadir))
        self._hypervol=hv_box(ideal,nadir)
    
    # Given a vector v, Returns the list of boxes intersecting with 
    # the dual domination cone at v, presented in rlist format:
    # [id,[rlist mins-maxes vector]]
    def _pintersect(self,v):
        # box ids and coordinates are merged for checking uniqueness
        inter=[# intersection with the positive cone
            [b.id] + b.bbox for b in
            self.intersection(
                    box2rindex(v,[np.inf for i in range(self.ndim)]), 
                    objects=True
                    )
           ]+ \
           [# intersection with the negative cone
            [b.id] + b.bbox for b in
            self.intersection(
                    box2rindex([-np.inf for i in range(self.ndim)],v), 
                    objects=True
                    )
           ]
        # list of unique intersected boxes
        if len(inter)==0:
            h=[]
        else:
            h=np.unique(inter,axis=0).tolist()
        return [[int(i[0]),i[1:]] for i in h]  
    
    ## Given a vector v, transforms the potential region structure by taking
    # set differences between all boxes and the dual domination cone.
    # Returns True if the structure has changed
    def addpoint(self,v):
        # h is the list of boxes [id,[rlist min-max vector]] intersecting with cones
        h=self._pintersect(v)
        if len(h)==0:
            print("### No intersections! Boxes: ", self.nbox," of ",self.ncre)
            return False
        # init the indicator if the potential region changed
        qchange=False
        ## Consider all intersected boxes one-by-one
        for b in h:
            rid=b[0] # box ID
            rv=b[1] # box vector in rtree format
            ## creating the vector of ranges / ranges with a midpoint for the recursive function
            vrange=(np.array(rindex2box(rv)).T).tolist() # init. list of the box ranges 
            # init. nrs. of dimensions with defined lower and higher ranges, respectively
            nlo=0
            nhi=0
            vrange_rec=[] # init. the list for recursive function
            for i in range(self.ndim):
                if v[i]<vrange[i][0]: # the range belongs to the higher part
                    vrange_rec.append(vrange[i])
                    nhi+=1
                elif v[i]>vrange[i][1]: # the range belongs to the lower part
                    vrange_rec.append(vrange[i])
                    nlo+=1
                else: # the vertex point is inside the range => it is undefined
                    vrange_rec.append([vrange[i][0],v[i],vrange[i][1]])
            ## Consider different cases of box-cones intersection
            # Box is a subset of a cone => removed from the potential region
            if nlo==self.ndim or nhi==self.ndim:
                qchange=True
                self.nbox-=1
                self.delete(rid,rv)
                self._hypervol-=hv_box(*rindex2box(rv))
#                print("-  ",np.array(rindex2box(rv)[0]),"\n",
#                      "   ",np.array(rindex2box(rv)[1])) #!#
            # Box does not intersect with either of the cones => do nothing 
            elif nlo>0 and nhi>0:
                break
            # rest of cases: box is intersected => divide into parts
            else:
                qchange=True
                # remove the original box
                self.nbox-=1
                self.delete(rid,rv)
                self._hypervol-=hv_box(*rindex2box(rv))
#                print("-o ",A._box_score(rindex2box(rv)),"\n",
#                      "   ",np.array(rindex2box(rv)[0]),"\n",
#                      "   ",np.array(rindex2box(rv)[1])) #!#
                # insert its remaining parts
                newboxes=flat_boxlist(divbox_rec(vrange_rec,nlo,nhi,self.ndim,0),self.ndim)
                for c in newboxes:
                    self.nbox+=1
                    self.ncre+=1
                    self.insert(self.ncre,box2rindex(*(np.array(c).T.tolist())))
                    self._hypervol+=hv_box(*(np.array(c).T.tolist()))
#                    print("+  ",A._box_score(rindex2box(np.array(c).T)),"\n",
#                          "   ",np.array(c).T[0],"\n",
#                          "   ",np.array(c).T[1]) #!#
        return qchange
    
    # Returns list of al boxes (as [ [[min vect.],[max vect.]],id ]) in the potential region 
    def boxes(self):
        return [[rindex2box(b.bbox),b.id]
                for b in self.intersection(
                    box2rindex(
                            [-np.inf for i in range(self.ndim)],
                            [np.inf for i in range(self.ndim)]
                                ), objects=True
                    )
                ]

### Automatic Decision Maker basic class representing ADM instance
# interacting with a method when solving a minimization problem.
# Input: one or more Pareto optimal objective vectors, 
# Output: two reference points (aspiration,reservation)
## Attributes
#   .k: nr. of objectives
#   .itern: current iteration nr.
#   .c: coefficient of optimism (float)               
#   ._ideal, ._nadir: corresponding points
#   ._potreg: potential region based on potreg class
#   ._paretoset: list of nonuique Pareto objective vectors
#   ._npareto: nr. of unique Pareto objective vectors
#   ._uf: utility function (R^k,Ideal,Nadir -> R)
#   .telemetry: dictionary of lists collecting relevant information in each iteration
## Methods
#   ._box_score: function (box=[min vector,max vector]) -> score (float)
#               which is used when selecting boxes
#   ._ufbox: basic example of _box_score calculating UF at the representative point
#   .box_pref: given a box, returns preference information related to this box
#   ._box_refpoint: basic example of box_pref returning [[min. point],[max. point]]
#           of the box which serve as aspiration and reservation ref. points               
#   ._upd: Given one or list of Pareto optimal objective vectors, 
#          adds new ones to the Pareto optimal set, updates the potential region
#          and returns [True iff potential region was changed, list of new Pareto optima]
#   .potboxes: returns the list of all boxes of the potential region
#   .bestbox: returns the best box [ [[min. point],[max.point]],id ] based on _box_score
#   .nextiter: Given one or set of Pareto optima, updates the potential region
#              and returns new preference information               



class ADM:
    def __init__(self,ideal,nadir,uf,coptimism):
        self.k=len(ideal)
        self._ideal=ideal
        self._nadir=nadir
        self.itern=1
        self._potreg=potreg(ideal,nadir)
        self._paretoset=[]
        self._npareto=0
        self.c=coptimism
        self._uf=uf
        self._box_score=self._ufbox
        self.telemetry={\
                "hypervol": [], # hypervolume of potential region after update
                "maxuf": [], # max. utility of newly obtained solutions
                "Pareto": [], # list of derived Pareto optimal objective in the iter.
                "nboxes": [], # nr. of boxes after update
                "crboxes": [], # nr. of boxes created so far (after the update)
                "npareto": [], # number of Pareto obj. vectors in the pool after update
                "ndifpareto":[], # number of different Pareto optima obtained in each iter.
                "bestbox": [], # the best box selected after update
                "ufbox":[],
                "pref": [] # preference information generated after update
                }
        
## Return hypervolume of boxes
    def hypervol(self):
        return self._potreg._hypervol

## Calculating UF at the representative point of a box (b=[min.v,max.v])
#  used in the basic version as the score function by default       
    def _ufbox(self,b):
        # calculate UF at the point alpha*min + (1-alpha)*max
        return self._uf(
                (np.array(b)*[[self.c],[1-self.c]]).sum(axis=0),
                self._ideal,self._nadir)

## Calculating scalar score of a box, used when selecting the best box,
# the higher score the better
    def _box_score(self,b):
        return self._ufbox(b)

## Returns [aspiration vect., reservation vect] for a given box
    def _box_refpoint(self,b):
        return [
                list((np.array(b)*[[self.c],[1-self.c]]).sum(axis=0)),
                b[1]
                ]
## Returns preference information (wrapper)
    def box_pref(self,b):
        return self._box_refpoint(b)

## Given one or list of objective vectors, 
#     optionally: list of boxes (as bestbox) to remove (e.g. source(s) of given solution(s)),
#  updates Pareto set and potential region;
#  Returns [whether potential region changed, the list of new Pareto objective vectors]
    def _upd(self,pp,remove_boxes=None):
        if len(pp)==0:
            return [False,[]]
        if not(hasattr(pp[0],"__iter__")):
            pp=[pp]
        ## updating Pareto optimal set
        # updating telemetry
        self.telemetry["Pareto"].append(pp)
        ufmax=-np.inf
        for pi in pp:
            ufi=self._uf(pi,self._ideal,self._nadir)
            #x#print("norm: ",normalize(pi,self._ideal,self._nadir))
            #x#print("uf: ",self._uf(pi,self._ideal,self._nadir))
            if ufi>ufmax:
                ufmax=ufi
        self.telemetry["maxuf"].append(ufmax)
        # list of new Pareto solutions which are not in the pool
        pnew=[]
        
        for p in pp:
            qincl=True
            for p1 in self._paretoset:
                if (p==p1).all():
                    qincl=False
                    break
            if qincl:
                pnew.append(p)
        self._paretoset.extend(pnew)
        self._npareto+=len(pnew)
        # updating the potential region and calculating change indicator
        # result (if potreg changed) of adding all Pareto points
        qpoints=any([
                self._potreg.addpoint(point) for point in pnew
                ])
        if remove_boxes is not None:
            for b in remove_boxes:
                # print("***\n",list(self._potreg.intersection(box2rindex(*b[0]))),"\n***")
                # if box is in the potential region
                if b[1] in self._potreg.intersection(box2rindex(*b[0])):
                    self._potreg._hypervol-=hv_box(*b[0])
                    self._potreg.delete(b[1],box2rindex(*b[0]))
                    qpoints=True
                    self._potreg.nbox-=1
        return [qpoints,pnew]

## Returns the potential region as a list of boxes [min vect. , max vect.]
# ADM fatigue, memory etc. are modelled here 
    def potboxes(self):
        return self._potreg.boxes()

## Finds the best box based on _box_score and returns as [[min vect. , max vect.],id=ncre]
    def bestbox(self):
        return max(self.potboxes(),key=lambda b:self._box_score(b[0]))

## Finds a best Pareto optimal vector w.r.t. UF and returns [vect.,UF(vect.)]
    def best_y(self):
        if len(self._paretoset)==0:
            return [None,None]
        y=max(self._paretoset, 
              key = lambda yi: self._uf(yi,self._ideal,self._nadir) 
              )
        return [y,self._uf(y,self._ideal,self._nadir)]

            
## Given one or list of objective vectors, 
#  optionally: list of boxes (as bestbox) to remove (e.g. source(s) of given solution(s)),
#  updates the potential region and
#  Returns {
#           "pref": [asp. vect, reserv. vect], 
#           "boxid": creation nr. of the best box,
#           "nboxes": current nr. of boxes in potreg,
#           "npareto": current nr. of Pareto objective vectors,
#           "changed?" True if potreg changed
#           }
    def nextiter(self,p,remove_boxes=None):
        ## updating the potential region and Pareto set
        upnew=self._upd(p,remove_boxes)
        self.telemetry["hypervol"].append(self.hypervol())
        self.telemetry["nboxes"].append(self._potreg.nbox)
        self.telemetry["crboxes"].append(self._potreg.ncre)
        self.telemetry["npareto"].append(self._npareto)
        bb=self.bestbox()
        #x# print([normalize(x,self._ideal,self._nadir) for x in bb[0]])
        self.telemetry["bestbox"].append(bb)
        self.telemetry["ufbox"].append(self._box_score(bb[0]))
        newpref=self.box_pref(bb[0])
        self.telemetry["pref"].append(newpref)
        self.itern+=1
        return {"pref": newpref,
                "changed?": upnew[0],
                "bestbox": bb # should be also deleted for avoiding cycles
                }

### ADM class for Nimbus method

#? future features:
#    * koef (0,+inf), default=1 for putting temp. ref.point on the half-line
#      between best Pareto and representative point of best box
#    * adjust temp. ref. point components: 
#       o  greater than or close to ideal => "<" class
#       o  less than or close to nadir => ">" class
#       o  close to the best Pareto => "=" class
#
class ADM_Nimbus(ADM):
## Returns Nimbus-specific preference information
    def box_pref(self,b):
        return [("<=",x) for x in self._box_refpoint(b)[0]]


                    ##########
                    ## MAIN ##
                    ##########
                    
## General forms of parametric utility functions 
#  defined for maximization criteria in the region [0,1]^k
# CES based on multiplication
def CES_mult(xx,ww):
    return np.prod([(x+0.01)**w for x,w in zip(xx,ww)])
# CES based on power summation
def CES_sum(xx,ww,p):
    try:
        return sum([w*x**p for x,w in zip(xx,ww)])**(1/p)
    except:
        print("x: ",xx,", w: ",ww)
def UF_TOPSIS(xx,ww):
    d_NIS=sum([(w*(1-x))**2 for x,w in zip(xx,ww)])**(1/2)
    d_PIS=sum([(w*x)**2 for x,w in zip(xx,ww)])**(1/2)
    return d_NIS(d_NIS+d_PIS)

## Linear normalization: ideal -> 1, nadir -> 0
#  which converts minimization objectives to maximization objectives
def normalize(xx,ideal,nadir):
    return np.array([(nad-x)/(nad-idl) for x,idl,nad in zip(xx,ideal,nadir)])


## Interface for MOO methods functions
def get_sol_nimb(pref,w,y,itern=5):
    if y is None:
        return [th.solve_ref(pref,w,itern=itern)["y"]]
    return [r["y"] 
                for r in th.solve_nimb(pref,w,y,itern=itern)]
def get_sol_rpm(pref,w,y,itern=5):
    return [r["y"]
            for r in th.solve_rpm(
                    pref,w,
                    sampl_m='simplicial',itern=5,npoints=100
                    )]
    

############
np.set_printoptions(precision=5)
### Instances of utility functions used in experiments with water treatment problem,
#  defined on [0,1]^k for maximization objectives
## Utility weight examples
ut_ces1=[1,1,1]
ut_ces2=[3,2,1]
ut_mult=[1,1,1]
    
UFs=[
           lambda xx: CES_sum(xx,ut_ces1,0.01),
           lambda xx: CES_sum(xx,ut_ces2,0.8),
           lambda xx: CES_mult(xx,ut_mult)
        ]

UFn=2 # choosing a UF from the list

## testing UF solution
t=[]
#for i in range(1000):
#    w=[1+np.random.rand() for i in range(th.nfun)]
#    sol=th.solve_uf(lambda y: -CES_mult(normalize(y,th.ideal,th.nadir),w),itern=5)
#    t.append(sol[1])
#    print(w)



# multi-experiments

coptimism=0.5 # coefficient of optimism
perturb = 0 # perturbation of UF (+- multiplicative)
fold_name = "out/perturb/"

methods_f=[
        #get_sol_rpm,
        get_sol_nimb]

itertest=10 # nr. of method iterations
iterfail=25 # max iterations number for catching failure
ufmax_frac=0.95 # required fraction of the maximum UF

n_runs=10 # number of experiments
## for collecting results
# maximum UF fraction in itertest iterations
maxuf_l=[[0 for m in methods_f] for i in range(n_runs)] 
# hypervolume at the itertest iteration
hypervol_l=[[0 for m in methods_f] for i in range(n_runs)]
# nr. of boxes at the itertest iteration
nboxes_l=[[0 for m in methods_f] for i in range(n_runs)]
# nr. of solutions in the pool at the itertest iteration
nsols_l=[[0 for m in methods_f] for i in range(n_runs)]
# nr. of iterations before success of achieving ufmax_frac
nsucciter_l=[[iterfail+1 for m in methods_f] for i in range(n_runs)]

nseries=10 # series of experiments for smaller batches
for iex in range(n_runs):
    ut_mult=[1+np.random.rand() for i in range(th.nfun)]
    print("\n*****\nRun ",iex,":\nw = ",ut_mult)
    maxuf_value=-th.solve_uf(
            lambda y: -UFs[UFn](normalize(y,th.ideal,th.nadir)),
            itern=5
            )[2]
    for mi, getsolf in enumerate(methods_f):
        print("Method: ",getsolf.__name__)
        A=ADM(  
                th.ideal,
                th.nadir,
                lambda y,ideal,nadir: UFs[UFn](normalize(y,ideal,nadir))* \
                                      (1-perturb/2+np.random.rand()*perturb),
                coptimism)
        sel_box=None # box based on which the last Pareto optimum was derived    
        p=[] # initial set of current solutions
        iter_fracuf=False # iteration nr. when the fraction of UF has been achieved
        maxyuf=-np.inf
        ## starting iterations
        for i in range(iterfail):
            print("Iteration ",i)
            ## ADM step
            result=A.nextiter(p,[sel_box])
            sel_box=result["bestbox"]
            print("Created: ",A._potreg.ncre, ", left: ",A._potreg.nbox#,", count: ",
                  #A._potreg.count(box2rindex([-np.inf for i in range(th.nfun)],
                  #                           [np.inf for i in range(th.nfun)]))
                  )
            pref=np.array(result["pref"][0])
            #print("Preferences: (",result["bestbox"][1],")\n",pref[0],"\n",pref[1])
            ## METHOD step
            ycurr = A.best_y()[0] # current
            p=getsolf(pref,th.w0,ycurr)
            p=np.unique([ip for ip in p if ip is not None], axis=0)
            A.telemetry["ndifpareto"].append(len(p))
            curr_uf=max([UFs[UFn](normalize(y,th.ideal,th.nadir)) for y in p])
            if curr_uf>maxyuf and i<itertest:
                maxyuf=curr_uf
            if not(iter_fracuf) and curr_uf/maxuf_value>=ufmax_frac:
                iter_fracuf=i+1
            if i>=itertest-1 and iter_fracuf:
                break
        # collect ADM stats at t
        result=A.nextiter(p,[sel_box])
        maxuf_l[iex][mi]=maxyuf/maxuf_value
        hypervol_l[iex][mi]=A.telemetry["hypervol"][itertest-1]
        nboxes_l[iex][mi]=A.telemetry["nboxes"][itertest-1]
        nsols_l[iex][mi]=A.telemetry["npareto"][itertest]
        nsucciter_l[iex][mi]=iter_fracuf
#            
## saving results
#with open(
#        fold_name+"experiments_"+
#        str(perturb).replace(".","")+
#        "_"+str(nseries)+
#        ".pkl",
#        "wb") as fout:
#    pickle.dump(
#            {
#                "maxuf":maxuf_l,
#                "hypervol":hypervol_l,
#                "nboxes":nboxes_l,
#                "nsols":nsols_l,
#                "nsucciter":nsucciter_l
#                    },
#            fout,protocol=pickle.HIGHEST_PROTOCOL)
    
### multi-experiments: collecting results
## maximum UF fraction in itertest iterations
#maxuf_l=[] 
## hypervolume at the itertest iteration
#hypervol_l=[]
## nr. of boxes at the itertest iteration
#nboxes_l=[]
## nr. of solutions in the pool at the itertest iteration
#nsols_l=[]
## nr. of iterations before success of achieving ufmax_frac
#nsucciter_l=[]
#
#methnames=["RPM","Nimbus"]
#for s in os.listdir(fold_name):
#    if s.endswith(".pkl"):
#        with open(fold_name+s,"rb") as f:
#            d=pickle.load(f)
#            maxuf_l.extend(d["maxuf"])
#            hypervol_l.extend(d["hypervol"])
#            nboxes_l.extend(d["nboxes"])
#            nsols_l.extend(d["nsols"])
#            nsucciter_l.extend(d["nsucciter"])        
#for d,dname in zip([maxuf_l,hypervol_l,nboxes_l,nsols_l,nsucciter_l],
#                   ["maxuf_l","hypervol_l","nboxes_l","nsols_l","nsucciter_l"]):
#    print(dname,": ",wilcoxon(*list(map(list, zip(*d))))[1])
    
#
#w=xlsxwriter.Workbook(fold_name+"out100.xlsx")
#sh=w.add_worksheet("Out")
#sh.write(0,0,"Value function")
#for i,s in enumerate(methnames):
#    sh.write(1,i,s)
#for i,l in enumerate(maxuf_l):
#    for j,x in enumerate(l):
#        sh.write(i+2,j,x)
#sh.write(0,2,"Success iteration")        
#for i,s in enumerate(methnames):
#    sh.write(1,i+2,s)
#for i,l in enumerate(nsucciter_l):
#    for j,x in enumerate(l):
#        sh.write(i+2,j+2,x)
#sh.write(0,4,"Hypervolume")        
#for i,s in enumerate(methnames):
#    sh.write(1,i+4,s)
#for i,l in enumerate(hypervol_l):
#    for j,x in enumerate(l):
#        sh.write(i+2,j+4,x)
#sh.write(0,6,"Number of boxes")
#for i,s in enumerate(methnames):
#    sh.write(1,i+6,s)
#for i,l in enumerate(nboxes_l):
#    for j,x in enumerate(l):
#        sh.write(i+2,j+6,x)
#sh.write(0,8,"Number of solutions")
#for i,s in enumerate(methnames):
#    sh.write(1,i+8,s)
#for i,l in enumerate(nsols_l):
#    for j,x in enumerate(l):
#        sh.write(i+2,j+8,x)
#w.close()
#
#
#
#
##
##
#
#
### individual experiments
#coptimism=0.5
#configs={
## considered instance of ADMs -> worksheet
#"ADMs":{ 
#    "names":["Unit","good RPM", "good Nimbus"],
#    "coefs":[
#        [1,1,1],
#        [1.0129412166523248, 1.1838829415870367, 1.5226377382534695],
#        [1.6840234709668256, 1.0281512416317582, 1.6678601370217185]        
#                ]
#    },
## indicator for both methods in each iteration -> table in a worksheet
#"Indicators":{
#        "names":["Max. VF","Nr. sols.","Volume","Nr. boxes"]
#        },
## method -> a column in one table
#"Methods":{
#        "names": ["RPM","Nimbus"],
#        "sol. funct":[get_sol_rpm,get_sol_nimb]
#        }
#        }
## DataFrame for collecting outputs
#Dout=pd.DataFrame(columns=pd.MultiIndex.from_product(
#        [configs[k]["names"] for k in configs],
#        names=[k for k in configs]
#        )
#                  ) 

#
#fig0,ax0=plt.subplots(figsize=(8,6))
#ax0.set_title("Solutions UF")
#fig1,ax1=plt.subplots(figsize=(8,6))
#ax1.set_title("Distance between solutions")
#fig2,ax2=plt.subplots(figsize=(8,6))
#ax2.set_title("Hypervolume")
#fig3,ax3=plt.subplots(figsize=(8,6))
#ax3.set_title("Boxes UF")


#for cname,coefs in zip(configs["ADMs"]["names"],configs["ADMs"]["coefs"]):
#    maxuf_value=-th.solve_uf(
#            lambda y: -CES_mult(normalize(y,th.ideal,th.nadir),coefs),
#            itern=5
#            )[2]
#    print(cname,": ",maxuf_value)
#    for mname, getsolf in zip(
#            configs["Methods"]["names"],
#            configs["Methods"]["sol. funct"]
#            ):
#        A=ADM(
#                th.ideal,
#                th.nadir,
#                lambda x,ideal,nadir: CES_mult(normalize(x,ideal,nadir),coefs)#*(
#                                #1-perturb/2+np.random.rand()*perturb
#                                #)
#                ,coptimism)
#        sel_box=None # box based on which the last Pareto optimum was derived    
#        p=[] # list of P.O. solution to initialize
#        ## starting iterations
#        for i in range(25):
#            print("Iteration ",i)
#            ## ADM step
#            result=A.nextiter(p,[sel_box])
#            sel_box=result["bestbox"]
#            print("Created: ",A._potreg.ncre, ", left: ",A._potreg.nbox#,", count: ",
#                  #A._potreg.count(box2rindex([-np.inf for i in range(th.nfun)],
#                  #                           [np.inf for i in range(th.nfun)]))
#                  )
#            pref=np.array(result["pref"])[0]
#            #print("Preferences: (",result["bestbox"][1],")\n",pref[0],"\n",pref[1])
#            ## METHOD step
#            ycurr=A.best_y()[0] # current
#            p=getsolf(pref,th.w0,ycurr,itern=5)
#            sols.append(p)
#            sols_uf.append([
#                    A._uf(ip,th.ideal,th.nadir) for ip in p if ip is not None
#                    ])
#            p=np.unique([ip for ip in p if ip is not None], axis=0)
#            #print("solution:\n",p,"\n")
#        A.nextiter(p,[sel_box])
#        Dout.loc[:,(cname,"Max. VF",mname)]=[y/maxuf_value for y in A.telemetry["maxuf"]]
#        Dout.loc[:,(cname,"Nr. sols.",mname)]=A.telemetry["npareto"][1:]
#        Dout.loc[:,(cname,"Volume",mname)]=A.telemetry["hypervol"][1:]
#        Dout.loc[:,(cname,"Nr. boxes",mname)]=A.telemetry["nboxes"][1:]
        
## Writing to Excel
#with pd.ExcelWriter("out.xlsx") as f:
#    for cname in configs["ADMs"]["names"]:
#        Dout[cname].to_excel(f,sheet_name=cname)


#        out.append({
#            "method": meth_l[ii],
#            "fname": fname_l[ii],
#            "problem": "f1 <= 1, itern=5, augm. 10E-8",
#            "uf": "mult",
#            "uf_weights": ut_mult,
#            "coptimism":A.c,
#            "solutions": sols,
#            "sol_uf": sols_uf,
#            "hypervol": A.telemetry["hypervol"],
#            "best_boxes": A.telemetry["bestbox"],
#            "box_uf": A.telemetry["ufbox"],
#            "nboxes": A.telemetry["nboxes"]
#                })
#        ax0.plot(A.telemetry["maxuf"])
#        ax1.plot([np.linalg.norm(
#                normalize(p2[0],th.ideal,th.nadir)-
#                normalize(p1[0],th.ideal,th.nadir)
#                ) for p1,p2 in 
#            zip(A.telemetry["Pareto"][1:],A.telemetry["Pareto"][:-1])]
#            )
#        ax2.plot(A.telemetry["hypervol"])
#        ax3.plot(A.telemetry["ufbox"])
#    plt.show()

# saving results
#for d in out:
#    with open(
#            "out/"+d["fname"]+"_"+
#            str(d["coptimism"]).replace(".","")+"_"+
#            d["uf"]+"".join(str(a) for a in d["uf_weights"])+".pkl",
#            "wb") as fout:
#        pickle.dump(d,fout,protocol=pickle.HIGHEST_PROTOCOL)
    

#### comparing results
#met_fnames=["refp","nimb"]
#met_names=["RPM", "Nimbus"]
#met_wvect=[[1.5, 1.3, 1.2]]
#met_copt=[0.5]
#ser_names=[] #names of iteration series
#UF_data=[] # UF(y) iteration series
#HV_data=[] # hypervolume iteration series
#nbox_data=[] # numbers of boxes
#for i,fn in enumerate(met_fnames):
#    for wv in met_wvect:
#        for cop in met_copt:
#            with open(
#                "out/"+fn+"_"+str(cop).replace(".","") + "_mult"+ \
#                "".join(str(a) for a in wv)+".pkl"
#                    , 'rb') as fhandle:
#                dict = pickle.load(fhandle)
#            ser_names.append(met_names[i]+" "+str(wv)+ 
#                             " ("+str(cop)+")")
#            UF_data.append(dict["sol_uf"])
#            HV_data.append(dict["hypervol"])
#            nbox_data.append(dict["nboxes"])
#workbook = xlsxwriter.Workbook('out/compare.xlsx')
#worksheet = workbook.add_worksheet("UF value")
#for i,s in enumerate(ser_names):
#    worksheet.write(0,i,s)
#for i in range(len(UF_data)):
#    for j in range(len(UF_data[1])):
#        worksheet.write(j+1,i,max(UF_data[i][j]))
#worksheet = workbook.add_worksheet("Hypervolume")
#for i,s in enumerate(ser_names):
#    worksheet.write(0,i,s)
#for i in range(len(HV_data)):
#    for j in range(len(HV_data[1])):
#        worksheet.write(j+1,i,HV_data[i][j])
#worksheet = workbook.add_worksheet("NBoxes")
#for i,s in enumerate(ser_names):
#    worksheet.write(0,i,s)
#for i in range(len(nbox_data)):
#    for j in range(len(nbox_data[1])):
#        worksheet.write(j+1,i,nbox_data[i][j])
#        
#workbook.close()
#        
##### Old version using DESDEO
#            
##problem = RiverPollution()
##method = NIMBUS(problem, SciPyDE)
##print("Ideal, nadir",problem.ideal,problem.nadir)
#
### in simpler version, pref.info does not depend on current solution(s)
### results = method.init_iteration()
##A=ADM_Nimbus(
##        problem.ideal,
##        problem.nadir,
##        lambda x,ideal,nadir: water_UFs[UFn](normalize(x,ideal,nadir)),
##        1)
##p=[]
##for i in range(5):
##    print("Iteration ",i)
##    print("Created: ",A._potreg.ncre, ", left: ",A._potreg.nbox,", count: ",
##          A._potreg.count(box2rindex([-np.inf for i in range(4)],
##                                     [np.inf for i in range(4)]))
##          )
##    result=A.nextiter(p)
##    pref=result["pref"]
##    pref1=normalize(pref,A._ideal,A._nadir)
##    print("Preferences:",[format(x,"1.10") for x in pref1])
##    p=[method._factories[0].result(
##                      NIMBUSClassification(method, pref), None
##                      )[1]
##                                                     for i in range(1)]
##    #print("Pareto:",[format(x,"1.10") for x in normalize(p[0],A._ideal,A._nadir)])
##    #x# print("New: ",p)
##
##
##
##
