#!/bin/bash 

DESKTOP=1

if [ $DESKTOP -eq 1 ]
then
  export WORKDIR=$PWD
  # The Eikonal/DREAM physics aborts under MPI ("DREAM/Eikonal physics
  # currently do not support MPI parallelization. Use openMP instead."), so a
  # single rank is mandatory and threads are the only parallelism available.
  #
  # The distributed v20 .deb is built with USE_OPENMP=UTILS, which leaves the
  # physics serial -- it links no libgomp at all.  OPENCARP_OMP below points at
  # a source build configured with -DUSE_OPENMP=ON, whose tenTusscherPanfilov
  # node loop and eikonal solver are threaded.  Results are bit-identical to
  # the .deb build and to each other at every thread count.
  #
  # Measured on 20 ms of this case (48-core host): 1 thread 187 s, 8 threads
  # 45 s, 24 threads 30 s, 48 threads 39 s.  It is memory-bandwidth bound, so
  # 24 is the best setting and using every core is slower than using half.
  NPROCS=1
  export OMP_NUM_THREADS=${OMP_NUM_THREADS:-24}
  GROUPBASE='/usr/local/lib/opencarp'
  OPENCARP_OMP=${OPENCARP_OMP:-$HOME/opencarp-omp/bin}
  CARPDIR=$([ -x "${OPENCARP_OMP}/openCARP" ] && echo ${OPENCARP_OMP} || echo ${GROUPBASE}/bin)
#  EXTIMP=${WORKDIR}'/ionic_model/mMS.so'
#  EXTIMPSTR="-num_external_imp 1 -external_imp ${EXTIMP}"
  CARPBIN="openCARP ${EXTIMPSTR}"
  MPIEX=${GROUPBASE}'/lib/petsc/bin/mpirun'
  echo "running on Desktop; exec is $CARPBIN; NPROCS: $NPROCS "
else
  export CARPFLV=CRAY
  echo "openCARP; compiler is $CARPFLV"
  source /work/e348/e348/shared/opencarp/modules.sh
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
  if [ $OMP_NUM_THREADS -gt 1 ]
  then
          export OMP_PLACES=cores
  fi
  echo " $SLURM_TASKS_PER_NODE MPI process with $SLURM_CPUS_PER_TASK CPUs each"
  module load craype-network-ucx
  module load cray-mpich-ucx
  # as per https://docs.archer2.ac.uk/known-issues/
  export UCX_IB_REG_METHODS=direct
  export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
  export WORKDIR=$SLURM_SUBMIT_DIR

  #export RDS_OUT_DIR=$SLURM_SUBMIT_DIR
  NPROCS=0
  CARPDIR=${EXECDIR}'/bin'
  CARPBIN='openCARP'
  MPIEX='srun --distribution=block:block --hint=nomultithread'
fi
####################################
cd $WORKDIR

MESHFNAME="/home/bzhou6/Data/ventricle/Case_1/Case_1"

LOPTF=${WORKDIR}'/petsc_cg_opts'
# -parab_use_pt / -cg_precond were removed in openCARP v20; the preconditioner
# is selected in ${LOPTF} (-pc_type jacobi).
CGSTR="-cg_maxit_parab 100 -cg_tol_parab 1e-5 -cg_norm_parab 0 -parab_options_file ${LOPTF}"
CARPSTR="${CARPDIR}/${CARPBIN} +F parameters.par -meshname ${MESHFNAME} ${CGSTR}"


if [ $DESKTOP -eq 1 ]
then
    start_time=`date +%s`
    $MPIEX -np $NPROCS $CARPSTR  >out_CARP$1 2>&1
    end_time=`date +%s`
    echo execution time was `expr $end_time - $start_time` s.    
else
    $MPIEX $CARPSTR
fi

