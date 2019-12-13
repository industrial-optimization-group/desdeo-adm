from scipy.optimize import shgo
import numpy as np
import copy
import sys

## non-dominated sorting
#import pygmo as pg

#outputting
from matplotlib import pyplot as plt


nvar=2 # nr. of variables
nfun=3 # nr. of functions

## objective functions
def psi(x):
    return x[0]*x[0]+x[1]*x[1]
def phi(x):
    return psi(x)-np.exp(-50*psi(x))

def f1(x):
    return phi(x)
def f2(x):
    return phi([x[0],x[1]-1])
def f3(x):
    return phi([x[0]-1,x[1]])

fvect=[f1,f2,f3]
def f(x): # the vector objective function
    return np.array([f1(x),f2(x),f3(x)])

## ideal, nadir
ideal=np.array([-1,-1,-1])
nadir=np.array([1,2,2])
utopia=ideal-10**-5

## basic weights for Chebyshev
w0=1/(nadir-ideal)

## bounds
bnd=[(0,1) for i in range(nvar)]+[(-10,10)]


### PROBLEM FORMULATION
#   decision vector xt = [x1_1,...,x_n,t]

## ASF objective: t + augmentation term 
#  *args = ( ref.point(list), rho(float), weights(list))
def rhosum_f(xt,*args):
    return (
           xt[-1]#+ # t
           #args[1] * sum( args[2]*(f(xt[:-1])-args[0]) ) # augm. term
           )
## l.h.s. fcuntions for ASF constraints: t >= w_i(f_i(x)-ref.point_i), i=1..n
t_constr=[ # *args = (ref.point_i,weight_i)
        lambda xt,*args: 
            -args[1]*fvect[0](xt[:-1])+xt[-1]+args[1]*args[0] - 10**-8*sum(f(xt[:-1])*args[1]),
        lambda xt,*args: 
            -args[1]*fvect[1](xt[:-1])+xt[-1]+args[1]*args[0] - 10**-8*sum(f(xt[:-1])),
        lambda xt,*args: 
            -args[1]*fvect[2](xt[:-1])+xt[-1]+args[1]*args[0] - 10**-8*sum(f(xt[:-1])),
        lambda xt: -fvect[0](xt[:-1]) + 1 # keep artificial upper bound    
        ]

### PROBLEM SOLVING
# solving the scalarized problem
def solve_ref(refpoint,w,sampl_m='simplicial',itern=5,npoints=100,
              subset=None, # subset of objectives to minimize
              upbounds=None # vector of [(upper bound or None) for each objective] 
              ):
    for ishift in [3,5,10]:
        # shifting the ref. point to exceed nadir+(nadir-ideal)
        tadd=max(0,max(w0*(ishift*nadir-(ishift-1)*ideal-refpoint)))
        refp=refpoint+tadd/w
        # list of objectives to minimize
        if subset is None:
            subset=range(len(refpoint))
        #creating constraints
        constr_list=[ 
                     { # ASF linearization constraint
                      "type": "ineq",
                      "fun": t_constr[i],
                      "args": (refp[i],w[i])
                      } for i in subset
                    ]+[{ # problem-specifi constraint
                        "type": "ineq",
                        "fun": t_constr[3]
                            }]
        if upbounds is not None:
            for i,b in enumerate(upbounds):
                if b is not None:
                    constr_list.append({
                      "type": "ineq",
                      "fun": lambda xt, *args: # function nr, constraint
                             -fvect[args[0]](xt[:-1]) + args[1],
                      "args":(i,b)
                      })
        # calling the solver
        sol = shgo(
                rhosum_f, #obj. function
                bounds=bnd, # variable bounds
                args=(np.array(refp),10**-6,w), # parameters for obj. func.
                constraints=constr_list,
                sampling_method=sampl_m,
                iters=itern,
                n=npoints,
                options={"minimize_every_iter":True,"local_iter":False}
               )
        # are ASF constraints tight?
        if sol["x"] is not None:
            constr=[
                    t_constr[i](sol["x"],refp[i],w[i])
                    for i in subset
                    ]
            if min(constr)>10**-6:
                if ishift==10:
                    print("Wrong constraints on shift ",ishift, " for \nRefp: ",
                          refp,"\n Weights: ",w,"\nUpbounds: ",upbounds,
                          "\n Subset: ",subset,"\nWith slack:",constr)
                    sol["message"]="! Solved but ASF constraints are not tight"+ \
                        "x=" + str(sol["x"]) + ", y="+str(f(sol["x"][:-1]))
                    sol["x"]=None
                    sol["y"]=None
            else: # if solved successfuly, enough shifting
                break
        else:
            constr=None
    # return
    return {"message": sol["message"],
            "x": sol["x"],
            "fun": sol["fun"],
            "nfev": sol["nfev"],
            "nlfev":sol["nlfev"],
            "constr":constr,
            "y":f(sol["x"][:-1]) if sol["x"] is not None else None
            }

## Deriving P.O. solutions for the reference point method
def solve_rpm(refpoint,w,
              sampl_m='simplicial',itern=5,npoints=100):
    p=[solve_ref(refpoint,w,
                 sampl_m=sampl_m,itern=itern,npoints=npoints)]

    # Modified ASF solutions
    normdif=np.linalg.norm(refpoint-p[0]["y"]) # perturbation value
    for i in range(nfun):
        pref1=copy.deepcopy(refpoint)
        pref1[i]+=normdif
        p.append(solve_ref(pref1,w,
                 sampl_m=sampl_m,itern=itern,npoints=npoints))
    return p

## Deriving P.P. solutions for the Nimbus method
#   y - current P.opt. solution  
#   tol - tolerance of assignment to classes "<", "=" and ">"    
def solve_nimb(refpoint,w,y,tol=0.01,
               sampl_m='simplicial',itern=5,npoints=100):
    ## assigning objectives to classes from "<" to ">"
    s_l = []
    s_leq = []
    s_eq = []
    s_geq = []
    s_g = []
    for i in range(len(refpoint)):
        # absolute tolerance for this objective
        itol=(nadir[i]-ideal[i])*tol
        #assigning
        if abs(ideal[i]-refpoint[i])<=itol:
            s_l.append(i)
        elif abs(nadir[i]-refpoint[i])<=itol:
            s_g.append(i)
        elif y[i]==refpoint[i]:
            s_eq.append(i)
        elif refpoint[i]<y[i]:
            s_leq.append(i)
        else:
            s_geq.append(i)
    if len(s_l)+len(s_leq)==0 or len(s_g)+len(s_geq)==0:
        print("Nimbus preference error, ref.=",refpoint,", y=",y)
    ## creating the modified ref. point
    ref1=copy.deepcopy(refpoint)
    for i in s_l:
        ref1[i]=ideal[i]
    for i in s_g:
        ref1[i]=nadir[i]
    ## solving scalarized problems
    p=[] # collected solutions
    # original Nimbus (3.1)
    print("3.1", end=" ")
    uporig=[None for i in range(len(refpoint))]
    for i in s_l+s_leq+s_eq:
        uporig[i]=y[i]
    for i in s_geq:
        uporig[i]=refpoint[i]
    p.append(solve_ref(#!! itern changed to 6 here
                ref1,w0,subset=s_l+s_leq,upbounds=uporig,
                sampl_m=sampl_m,itern=6,npoints=npoints)
            )
    # from STOM (3.2)
    print("3.2", end=" ")    
    wstom=1/(np.array(ref1)-utopia)
    p.append(solve_ref(
                utopia, wstom,
                sampl_m=sampl_m,itern=itern,npoints=npoints)
            )
    # simple ASF (3.3)
    print("3.3", end=" ")
    p.append(solve_ref(
                ref1, w0,
                sampl_m=sampl_m,itern=itern,npoints=npoints)
            )
    # from GUESS (3.4)
    print("3.4")
    p.append(solve_ref(
                nadir, 1/(nadir-np.array(ref1)),
                subset = s_l + s_leq + s_eq + s_geq,
                sampl_m=sampl_m,itern=itern,npoints=npoints)
            )
    return p

def solve_uf(uf,sampl_m='simplicial',itern=5,npoints=100):
    sol = shgo(
        lambda x: uf(f(x)), #obj. function
        bounds=bnd[:nvar], # variable bounds
        constraints=[{ # keep artificial upper bound
                    "type": "ineq",
                    "fun": lambda xt: -fvect[0](xt) + 1 
                            }],
        sampling_method=sampl_m,
        iters=itern,
        n=npoints,
        options={"minimize_every_iter":True,"local_iter":False}
       )
    return[ sol["x"],f(sol["x"]), uf(f(sol["x"])),sol]
    
    

if __name__=="__main__":
    rp=np.array([ 7.,12.68296,12.87776])
    for i in [1,2,3]:
        sol=solve_ref(
            rp,
            np.array([0.5,0.33333,0.33333]),
            subset=[0,1],
            upbounds=[-0.871937679931465, 0.9416768454172942, 1.013357076861016],itern=6)
        print(sol["y"])
        rp+=5/np.array([0.5,0.33333,0.33333])
    pass
### generating and saving random reference points
#    ref_l=np.array([
#            [ideal[i]+np.random.random_sample()*(nadir[i]-ideal[i])
#                for i in range(nfun)]
#            for i in range(500)])
#    np.savetxt("reflist.txt",ref_l)
### loading same saved reference points
#    ref_l=np.genfromtxt("reflist.txt")
### collecting solution results    
#    x_l=[]
#    y_l=[]
#    for i in range(500):
#        res=solve_ref(ref_l[i],w0,itern=5)
#        x_l.append(res["x"])
#        y_l.append(f(res["x"]))
###    saving solution results
#    np.savetxt("aug_add_10-6_x.txt",x_l)
#    np.savetxt("aug_add_10-6_y.txt",y_l)


#    ### Testing results of different ASF formulations on random points
#    asfnames=["aug_10-6", # proper augmentation via constraints
#              "noaugm", # no augmentation
#              "aug_add_10-6" # augmented by adding term to the obj. function
#              ]    
#    xx=[np.genfromtxt(s+"_x.txt") for s in asfnames]
#    yy=[np.genfromtxt(s+"_y.txt") for s in asfnames]
#    
#    ### checking non-domination
#    #np.set_printoptions(precision=5)
#    #for i,s in enumerate(asfnames):
#    #    print("\n*******\n"+s)
#    #    for i1,y1 in enumerate(yy[i]):
#    #        for i2,y2 in enumerate(yy[i]):
#    #            if max(y1-y2)<=10**-6 and min(y1-y2)<0:
#    #                print("X: ",xx[i][i1][:-1],xx[i][i2][:-1])
#    #                print("Y: ",y1,y1,min(y1-y2),"\n")
#    
#    ## checking differences
#    for i1,s1 in enumerate(asfnames[:-1]):
#        for i2,s2 in enumerate(asfnames[i1+1:]):
#            print("\n*******************")
#            print(s1+" - "+s2)
#            # difference between y values
#            fig,ax=plt.subplots(figsize=(8,6))
#            ax.set_title("Y")
#            ax.hist([np.linalg.norm(yy[i1][j]-yy[i2][j])
#                        for j in range(len(yy[i1]))
#                    ],bins=20,range=(0,0.000004)
#                    )
#            plt.show()
#            # difference between x values
#            fig,ax=plt.subplots(figsize=(8,6))
#            ax.set_title("X")
#            ax.hist([np.linalg.norm(xx[i1][j][:-1]-xx[i2][j][:-1])
#                        for j in range(len(xx[i1]))
#                    ],bins=20,range=(0,0.000004)
#                    )
#            plt.show()
#            
#            
#            