#!/bin/bash 

DESKTOP=1

if [ $DESKTOP -eq 1 ]
then
  export WORKDIR=$PWD
  NPROCS=64
  GROUPBASE='/usr/local/lib/opencarp'
  CARPDIR=${GROUPBASE}'/bin'
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
CGSTR="-parab_use_pt 1 -cg_precond 1 -cg_maxit_parab 100 -cg_tol_parab 1e-5 -cg_norm_parab 0 -parab_options_file ${LOPTF}"
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

