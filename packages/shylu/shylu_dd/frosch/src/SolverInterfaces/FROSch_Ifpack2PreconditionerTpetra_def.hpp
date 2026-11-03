//@HEADER
// ************************************************************************
//
//               ShyLU: Hybrid preconditioner package
//                 Copyright 2012 Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Alexander Heinlein (alexander.heinlein@uni-koeln.de)
//
// ************************************************************************
//@HEADER

#ifndef _FROSCH_IFPACK2PRECONDITIONERTPETRA_DEF_HPP
#define _FROSCH_IFPACK2PRECONDITIONERTPETRA_DEF_HPP

#include <FROSch_Ifpack2PreconditionerTpetra_decl.hpp>

#include "Ifpack2_Details_getCrsMatrix.hpp"
#include "Ifpack2_RILUK_decl.hpp"
#include "Ifpack2_Chebyshev_decl.hpp"

// VR 2026-11-03: "Local Matrix Format" = "BlockCrs"
#include <Tpetra_BlockCrsMatrix.hpp>
#include <Tpetra_CrsGraph.hpp>
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

#ifdef HAVE_SHYLU_DDFROSCH_ZOLTAN2
#include "Zoltan2_TpetraRowGraphAdapter.hpp"
#include "Zoltan2_OrderingProblem.hpp"
#include "Zoltan2_OrderingSolution.hpp"
#endif

// #define DANE_EIGS_CHECK

namespace FROSch {

    using namespace std;
    using namespace Teuchos;
    using namespace Xpetra;

    template<class SC,class LO,class GO,class NO>
    int Ifpack2PreconditionerTpetra<SC,LO,GO,NO>::initialize()
    {
        FROSCH_TIMER_START_SOLVER(initializeTime,"Ifpack2PreconditionerTpetra::initialize");
        this->IsInitialized_ = true;
        this->IsComputed_ = false;
        Ifpack2Preconditioner_->initialize();
        return 0;
    }

    template<class SC,class LO,class GO,class NO>
    int Ifpack2PreconditionerTpetra<SC,LO,GO,NO>::compute()
    {
        FROSCH_TIMER_START_SOLVER(computeTime,"Ifpack2PreconditionerTpetra::compute");
        FROSCH_ASSERT(this->IsInitialized_,"FROSch::Ifpack2PreconditionerTpetra: !this->IsInitialized_");
        this->IsComputed_ = true;
        if (useBlockCrs_) { // VR 2026-11-03: values (and inverse diagonal) of the point matrix into the blocks
            fillBlockCrsValues();
            if (!blockChecked_) {
                checkBlockCrsApply();
                blockChecked_ = true;
            }
            // Ifpack2 Chebyshev deep-copies the inverse diagonal in setParameters(): hand it over now that it is filled
            // (every compute, so a recompute gets the new values); the full list, as setParameters resets missing ones
            {
                using IST = typename TBlockCrsMatrix::impl_scalar_type;
                using ATS = Kokkos::Details::ArithTraits<IST>;
                auto d = blockInvDiagonal_->getLocalViewHost(Tpetra::Access::ReadOnly);
                LO bad = 0;
                for (size_t k=0; k<d.extent(0); ++k) {
                    if (d(k,0)==ATS::zero() || ATS::isNan(d(k,0)) || ATS::isInf(d(k,0))) ++bad;
                }
                FROSCH_ASSERT(bad==0,"FROSch::Ifpack2PreconditionerTpetra: BlockCrs: "+to_string(bad)+" entries of the inverse diagonal are 0, NaN or Inf.");
            }
            blockIfpack2Params_.set("chebyshev: operator inv diagonal",rcp_implicit_cast<const TVector>(blockInvDiagonal_));
            Ifpack2Preconditioner_->setParameters(blockIfpack2Params_);
        }
        Ifpack2Preconditioner_->compute();
#ifdef DANE_EIGS_CHECK
        // Print eigenvalue estimates with global MPI rank (only if using Chebyshev)
        auto solverName = this->ParameterList_->get("Solver","RILUK");
        std::cout << "[Global Rank " << globalRank_ << "] Solver name: " << solverName << std::endl;
        std::cout << "[Global Rank " << globalRank_ << "] Ifpack2Preconditioner_ type: " << typeid(*Ifpack2Preconditioner_).name() << std::endl;
        std::cout << "[Global Rank " << globalRank_ << "] TRowMatrix type: " << typeid(TRowMatrix).name() << std::endl;
        
        if (solverName == "CHEBYSHEV") {
            std::cout << "[Global Rank " << globalRank_ << "] Attempting Chebyshev cast to Ifpack2::Chebyshev<TRowMatrix>..." << std::endl;
            auto chebPrec = Teuchos::rcp_dynamic_cast<Ifpack2::Chebyshev<TRowMatrix>>(Ifpack2Preconditioner_);
            if (!chebPrec.is_null()) {
                SC lambdaMax = chebPrec->getLambdaMaxForApply();
                std::cout << "[Global Rank " << globalRank_ << "] SUCCESS! Chebyshev max eigenvalue for apply: " << lambdaMax << std::endl;
            } else {
                std::cout << "[Global Rank " << globalRank_ << "] FAILED - Chebyshev cast returned null pointer!" << std::endl;
                std::cout << "[Global Rank " << globalRank_ << "] Trying alternative cast approach..." << std::endl;
                // Try to access through Preconditioner base class
                auto precBase = Ifpack2Preconditioner_;
                if (precBase.is_null()) {
                    std::cout << "[Global Rank " << globalRank_ << "] Preconditioner is null!" << std::endl;
                } else {
                    std::cout << "[Global Rank " << globalRank_ << "] Preconditioner is valid" << std::endl;
                }
            }
        } else {
            std::cout << "[Global Rank " << globalRank_ << "] Not CHEBYSHEV solver, skipping eigenvalue output" << std::endl;
        }
#endif
        return 0;
    }

    template<class SC,class LO,class GO,class NO>
    void Ifpack2PreconditionerTpetra<SC,LO,GO,NO>::apply(const XMultiVector &x,
                                                         XMultiVector &y,
                                                         ETransp mode,
                                                         SC alpha,
                                                         SC beta) const
    {
        FROSCH_TIMER_START_SOLVER(applyTime,"Ifpack2PreconditionerTpetra::apply");
        FROSCH_ASSERT(this->IsComputed_,"FROSch::Ifpack2PreconditionerTpetra: !this->IsComputed_.");

        const TpetraMultiVector<SC,LO,GO,NO> * xTpetraMultiVectorX = dynamic_cast<const TpetraMultiVector<SC,LO,GO,NO> *>(&x);
        FROSCH_ASSERT(xTpetraMultiVectorX,"FROSch::Ifpack2PreconditionerTpetra: dynamic_cast failed.");
        TMultiVectorPtr tpetraMultiVectorX = xTpetraMultiVectorX->getTpetra_MultiVector();

        const TpetraMultiVector<SC,LO,GO,NO> * xTpetraMultiVectorY = dynamic_cast<const TpetraMultiVector<SC,LO,GO,NO> *>(&y);
        FROSCH_ASSERT(xTpetraMultiVectorY,"FROSch::Ifpack2PreconditionerTpetra: dynamic_cast failed.");
        TMultiVectorPtr tpetraMultiVectorY = xTpetraMultiVectorY->getTpetra_MultiVector();

        if (useBlockCrs_) { // VR 2026-11-03: x, y in the node numbering of the BlockCrsMatrix, solve, y back
            FROSCH_ASSERT(tpetraMultiVectorX->getLocalLength()==(size_t) blockInvPerm_.extent(0),"FROSch::Ifpack2PreconditionerTpetra: BlockCrs: x has the wrong length.");
            const size_t numVectors = tpetraMultiVectorX->getNumVectors();
            if (blockIdentity_) { // VR 2026-11-03: already node-ordered: same data, only the block Maps, no copies
                RCP<const TMultiVector> xView = tpetraMultiVectorX->offsetView(blockMatrix_->getDomainMap(),0);
                RCP<TMultiVector> yView = tpetraMultiVectorY->offsetViewNonConst(blockMatrix_->getRangeMap(),0);
                Ifpack2Preconditioner_->apply(*xView,*yView,mode,alpha,beta);
                return;
            }
            if (blockX_.is_null() || blockX_->getNumVectors()!=numVectors) {
                blockX_ = rcp(new TMultiVector(blockMatrix_->getDomainMap(),numVectors));
                blockY_ = rcp(new TMultiVector(blockMatrix_->getRangeMap(),numVectors));
            }
            using HostExec = Kokkos::DefaultHostExecutionSpace;
            const LO n = (LO) blockInvPerm_.extent(0);
            const HostLOView invPerm = blockInvPerm_;
            const bool needY = (beta != ScalarTraits<SC>::zero());
            {
                FROSCH_TIMER_START_SOLVER(blockPermuteTime,"Ifpack2PreconditionerTpetra::apply::blockPermute");
                auto xv = tpetraMultiVectorX->getLocalViewHost(Tpetra::Access::ReadOnly);
                auto bxv = blockX_->getLocalViewHost(Tpetra::Access::OverwriteAll);
                Kokkos::parallel_for("FROSch::BlockCrs::gatherX",Kokkos::RangePolicy<HostExec>(0,n),[=](const LO i) {
                    for (size_t k=0; k<numVectors; ++k) bxv(i,k) = xv(invPerm(i),k);
                });
                if (needY) {
                    auto yv = tpetraMultiVectorY->getLocalViewHost(Tpetra::Access::ReadOnly);
                    auto byv = blockY_->getLocalViewHost(Tpetra::Access::OverwriteAll);
                    Kokkos::parallel_for("FROSch::BlockCrs::gatherY",Kokkos::RangePolicy<HostExec>(0,n),[=](const LO i) {
                        for (size_t k=0; k<numVectors; ++k) byv(i,k) = yv(invPerm(i),k);
                    });
                }
            }
            Ifpack2Preconditioner_->apply(*blockX_,*blockY_,mode,alpha,beta);
            {
                FROSCH_TIMER_START_SOLVER(blockPermuteTime,"Ifpack2PreconditionerTpetra::apply::blockPermute");
                auto byv = blockY_->getLocalViewHost(Tpetra::Access::ReadOnly);
                auto yv = tpetraMultiVectorY->getLocalViewHost(Tpetra::Access::ReadWrite);
                Kokkos::parallel_for("FROSch::BlockCrs::scatterY",Kokkos::RangePolicy<HostExec>(0,n),[=](const LO i) {
                    for (size_t k=0; k<numVectors; ++k) yv(invPerm(i),k) = byv(i,k);
                });
            }
            return;
        }

#ifdef HAVE_SHYLU_DDFROSCH_ZOLTAN2
        if (this->useZoltan2 && this->useRILUK) {
            auto A = Ifpack2Preconditioner_->getMatrix();
            auto filteredA = rcp_dynamic_cast<const TRowMatrixFilterType> (A);

            // make copies of X & Y
            TMultiVector ReorderedX (*tpetraMultiVectorX, Teuchos::Copy);
            TMultiVector ReorderedY (*tpetraMultiVectorY, Teuchos::Copy);

            // permute X & Y
            filteredA->permuteOriginalToReordered (*tpetraMultiVectorX, ReorderedX);
            filteredA->permuteOriginalToReordered (*tpetraMultiVectorY, ReorderedY);

            // solve
            Ifpack2Preconditioner_->apply(ReorderedX,ReorderedY,mode,alpha,beta);

            // permute X back
            filteredA->permuteReorderedToOriginal (ReorderedY, *tpetraMultiVectorY);
        } else
#endif
        {
            Ifpack2Preconditioner_->apply(*tpetraMultiVectorX,*tpetraMultiVectorY,mode,alpha,beta);
        }
    }

    template<class SC,class LO,class GO,class NO>
    int Ifpack2PreconditionerTpetra<SC,LO,GO,NO>::updateMatrix(ConstXMatrixPtr k,
                                                               bool reuseInitialize)
    {
        if (this->useRILUK) {
            const CrsMatrixWrap<SC,LO,GO,NO>& crsOp = dynamic_cast<const CrsMatrixWrap<SC,LO,GO,NO>&>(*this->K_);
            const TpetraCrsMatrix<SC,LO,GO,NO>& xTpetraMat = dynamic_cast<const TpetraCrsMatrix<SC,LO,GO,NO>&>(*crsOp.getCrsMatrix());
            ConstTCrsMatrixPtr tpetraMat = xTpetraMat.getTpetra_CrsMatrix();

            auto RILUPreconditioner = rcp_dynamic_cast<Ifpack2::RILUK<TRowMatrix>>(Ifpack2Preconditioner_);
#ifdef HAVE_SHYLU_DDFROSCH_ZOLTAN2
            if (this->useZoltan2) {
                // if K is replaced, then we need to re-wrap into matrix filter
                auto rowMat = rcp_dynamic_cast<const TRowMatrix>(tpetraMat);
                auto filteredMat = rcp(new TRowMatrixFilterType(rowMat, this->perm, this->revperm));
                RILUPreconditioner->setMatrix(filteredMat);
            } else
#endif
            {
                RILUPreconditioner->setMatrix(tpetraMat);
            }
            return 0;
        }
        FROSCH_ASSERT(false,"FROSch::Ifpack2PreconditionerTpetra: updateMatrix() is not implemented for the Ifpack2PreconditionerTpetra yet.");
    }

    template<class SC,class LO,class GO,class NO>
    Ifpack2PreconditionerTpetra<SC,LO,GO,NO>::Ifpack2PreconditionerTpetra(ConstXMatrixPtr k,
                                                                          ParameterListPtr parameterList,
                                                                          string description) :
    Solver<SC,LO,GO,NO> (k,parameterList,description)
    {
        FROSCH_TIMER_START_SOLVER(Ifpack2PreconditionerTpetraTime,"Ifpack2PreconditionerTpetra::Ifpack2PreconditionerTpetra");
        FROSCH_ASSERT(!this->K_.is_null(),"FROSch::Ifpack2PreconditionerTpetra: K_ is null.");
        FROSCH_ASSERT(this->K_->getRowMap()->lib()==UseTpetra,"FROSch::Ifpack2PreconditionerTpetra: Not compatible with Epetra.")

        const CrsMatrixWrap<SC,LO,GO,NO>& crsOp = dynamic_cast<const CrsMatrixWrap<SC,LO,GO,NO>&>(*this->K_);
        const TpetraCrsMatrix<SC,LO,GO,NO>& xTpetraMat = dynamic_cast<const TpetraCrsMatrix<SC,LO,GO,NO>&>(*crsOp.getCrsMatrix());
        ConstTCrsMatrixPtr tpetraMat = xTpetraMat.getTpetra_CrsMatrix();
        TEUCHOS_TEST_FOR_EXCEPT(tpetraMat.is_null());
#ifdef DANE_EIGS_CHECK
        // Read global MPI rank from parameter list (injected by OverlappingOperator before calling SolverFactory)
        globalRank_ = this->ParameterList_->get("Global MPI Rank", 0);
#endif
        auto solverName = this->ParameterList_->get("Solver","RILUK");
        this->useRILUK = (solverName == "RILUK");
        this->useZoltan2 = this->ParameterList_->get("RILUK: use reordering", false);

        // VR 2026-11-03: "Local Matrix Format" = "Crs" (default, unchanged) | "BlockCrs"
        const string localMatrixFormat = this->ParameterList_->get("Local Matrix Format","Crs");
        FROSCH_ASSERT(localMatrixFormat=="Crs" || localMatrixFormat=="BlockCrs","FROSch::Ifpack2PreconditionerTpetra: \"Local Matrix Format\" must be \"Crs\" or \"BlockCrs\", got \""+localMatrixFormat+"\".");
        useBlockCrs_ = (localMatrixFormat=="BlockCrs");

        Ifpack2::Details::OneLevelFactory<TRowMatrix> ifpack2Factory;
        if (useBlockCrs_) { // VR 2026-11-03
            FROSCH_ASSERT(solverName=="CHEBYSHEV","FROSch::Ifpack2PreconditionerTpetra: \"Local Matrix Format\" = \"BlockCrs\" is only implemented for Solver = CHEBYSHEV.");
            blockSize_ = this->ParameterList_->get("BlockCrs: Block Size",(LO) 3);
            FROSCH_ASSERT(blockSize_>=1,"FROSch::Ifpack2PreconditionerTpetra: \"BlockCrs: Block Size\" must be >= 1.");
            globalRank_ = this->ParameterList_->get("Global MPI Rank",0);
            blockPointMatrix_ = tpetraMat;
            buildBlockCrsStructure();
            Ifpack2Preconditioner_ = ifpack2Factory.create(solverName,rcp_implicit_cast<const TRowMatrix>(blockMatrix_));

            // Ifpack2 Chebyshev would build its diagonal on getRowMap(), which for a BlockCrsMatrix is the node Map
            // (point vs node size), so FROSch gives it the inverse diagonal on the point range Map instead.
            ParameterListPtr ifpack2ParameterList = sublist(this->ParameterList_,"Ifpack2");
            if (ifpack2ParameterList->isSublist(solverName)) {
                ifpack2ParameterList = sublist(ifpack2ParameterList, solverName);
            }
            // Ifpack2 deep-copies that vector in setParameters(): the diagonal is set (and setParameters called again)
            // in compute(), once its values are there; here only the other parameters.
            blockIfpack2Params_ = *ifpack2ParameterList;
            blockIfpack2Params_.setName("Ifpack2");
            FROSCH_ASSERT(!blockIfpack2Params_.isParameter("chebyshev: operator inv diagonal"),"FROSch::Ifpack2PreconditionerTpetra: BlockCrs computes \"chebyshev: operator inv diagonal\" itself; remove it from the list.");
            if (blockIfpack2Params_.isParameter("chebyshev: min diagonal value")) {
                blockMinDiagonal_ = blockIfpack2Params_.get<SC>("chebyshev: min diagonal value");
            }
            Ifpack2Preconditioner_->setParameters(blockIfpack2Params_);
            return;
        }
#ifdef HAVE_SHYLU_DDFROSCH_ZOLTAN2
        if (this->useZoltan2 && this->useRILUK) {
            // pre-ordering matrix, before constructing Ifpack2Preconditioner_ with the matrix
            {
                typedef Tpetra::RowGraph<LO, GO, NO> row_graph_type;
                typedef Zoltan2::TpetraRowGraphAdapter<row_graph_type> z2_adapter_type;
                typedef Zoltan2::OrderingProblem<z2_adapter_type> z2_ordering_problem_type;
                typedef Zoltan2::LocalOrderingSolution<LO> z2_ordering_solution_type;

                auto comm = tpetraMat->getRowMap()->getComm();
                auto constActiveGraph = Teuchos::rcp_const_cast<const row_graph_type>(tpetraMat->getGraph());
                z2_adapter_type Zoltan2Graph (constActiveGraph);

                Teuchos::ParameterList zoltan2_params = this->ParameterList_->sublist ("RILUK: reordering list");
                z2_ordering_problem_type MyOrderingProblem (&Zoltan2Graph, &zoltan2_params, comm);

                MyOrderingProblem.solve ();
                z2_ordering_solution_type sol (*MyOrderingProblem.getLocalOrderingSolution());
                this->perm = sol.getPermutationRCPConst (true);
                this->revperm = sol.getPermutationRCPConst ();
            }

            // wrap into matrix filter
            auto rowMat = rcp_dynamic_cast<const TRowMatrix>(tpetraMat);
            auto filteredMat = rcp(new TRowMatrixFilterType(rowMat, this->perm, this->revperm));
            Ifpack2Preconditioner_ = ifpack2Factory.create(solverName,filteredMat);
        } else
#endif
        {
            Ifpack2Preconditioner_ = ifpack2Factory.create(solverName,tpetraMat);
        }

        // set ifpack2 parameters
        ParameterListPtr ifpack2ParameterList = sublist(this->ParameterList_,"Ifpack2");
        if (ifpack2ParameterList->isSublist(solverName)) {
            ifpack2ParameterList = sublist(ifpack2ParameterList, solverName);
        }
        ifpack2ParameterList->setName("Ifpack2");
        Ifpack2Preconditioner_->setParameters(*ifpack2ParameterList);
    }

    // VR 2026-11-03: mixing function for the row-pattern keys of the BlockCrs setup (splitmix64)
    inline std::uint64_t BlockCrsPatternHash(std::uint64_t z)
    {
        z += 0x9E3779B97F4A7C15ULL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }

    // VR 2026-11-03: node numbering + node graph + BlockCrsMatrix (structure only; values in fillBlockCrsValues)
    template<class SC,class LO,class GO,class NO>
    void Ifpack2PreconditionerTpetra<SC,LO,GO,NO>::buildBlockCrsStructure()
    {
        FROSCH_TIMER_START_SOLVER(blockBuildTime,"Ifpack2PreconditionerTpetra::buildBlockCrsStructure");
        using HostExec = Kokkos::DefaultHostExecutionSpace;
        using std::uint64_t;
        const ConstTCrsMatrixPtr A = blockPointMatrix_;
        const LO b = blockSize_;
        const LO n = (LO) A->getLocalNumRows();
        FROSCH_ASSERT(A->getRowMap()->getComm()->getSize()==1,"FROSch::Ifpack2PreconditionerTpetra: BlockCrs needs the local (one-process) subdomain matrix.");
        FROSCH_ASSERT(A->getDomainMap()->isSameAs(*A->getRowMap()) && A->getRangeMap()->isSameAs(*A->getRowMap()),"FROSch::Ifpack2PreconditionerTpetra: BlockCrs needs domain Map = range Map = row Map.");
        FROSCH_ASSERT(n%b==0,"FROSch::Ifpack2PreconditionerTpetra: BlockCrs: "+to_string(n)+" local rows are not a multiple of \"BlockCrs: Block Size\" "+to_string(b)+".");

        auto lcl = A->getLocalMatrixHost();
        auto rowPtr = lcl.graph.row_map;
        auto cols = lcl.graph.entries;

        // column -> row index (the column Map need not list the GIDs in row order)
        const LO numCols = (LO) A->getColMap()->getLocalNumElements();
        blockColToRow_ = HostLOView("FROSch::BlockCrs::colToRow",numCols);
        LO badCols = 0;
        for (LO lc=0; lc<numCols; ++lc) {
            const LO r = A->getRowMap()->getLocalElement(A->getColMap()->getGlobalElement(lc));
            blockColToRow_(lc) = r;
            if (r==Teuchos::OrdinalTraits<LO>::invalid()) ++badCols;
        }
        FROSCH_ASSERT(badCols==0,"FROSch::Ifpack2PreconditionerTpetra: BlockCrs: "+to_string(badCols)+" columns are not rows of the local matrix.");
        const HostLOView colToRow = blockColToRow_;

        // 1. key per row: length + two hashes of its column set (equal keys = equal column sets)
        Kokkos::View<uint64_t*,Kokkos::HostSpace> key1("FROSch::BlockCrs::key1",n), key2("FROSch::BlockCrs::key2",n);
        Kokkos::parallel_for("FROSch::BlockCrs::rowKeys",Kokkos::RangePolicy<HostExec>(0,n),[=](const LO i) {
            uint64_t s1 = 0, s2 = 0;
            for (auto j=rowPtr(i); j<rowPtr(i+1); ++j) {
                const uint64_t c = (uint64_t) colToRow(cols(j));
                s1 += BlockCrsPatternHash(c);
                s2 += BlockCrsPatternHash(c ^ 0xD6E8FEB86659FD93ULL);
            }
            key1(i) = s1;
            key2(i) = s2;
        });
        auto len = [&](const LO i) {return rowPtr(i+1)-rowPtr(i);};
        auto sameKey = [&](const LO x,const LO y) {return len(x)==len(y) && key1(x)==key1(y) && key2(x)==key2(y);};

        // 2. rows sorted by key, ties by row (rows of one group ascending)
        std::vector<LO> order(n);
        std::iota(order.begin(),order.end(),(LO) 0);
        std::sort(order.begin(),order.end(),[&](const LO x,const LO y) {
            if (len(x)!=len(y)) return len(x)<len(y);
            if (key1(x)!=key1(y)) return key1(x)<key1(y);
            if (key2(x)!=key2(y)) return key2(x)<key2(y);
            return x<y;
        });

        // 3. b-tuples: equal-pattern groups of k*b rows cut in order (rows with one column set in a structurally
        //    symmetric pattern give full blocks in any grouping); single-entry diagonal rows (Dirichlet: identity
        //    rows whose columns are dropped elsewhere) cut in order (diagonal blocks + explicit zeros).
        std::vector<LO> tuples;
        tuples.reserve(n);
        std::vector<LO> dirichlet;
        LO leftover = 0, cutFromLarger = 0;
        for (LO s=0; s<n;) {
            LO e = s+1;
            while (e<n && sameKey(order[e],order[s])) ++e;
            const LO z = e-s;
            const LO r0 = order[s];
            if (z==1 && len(r0)==1 && colToRow(cols(rowPtr(r0)))==r0) {
                dirichlet.push_back(r0);
            } else if (z%b==0) {
                tuples.insert(tuples.end(),order.begin()+s,order.begin()+e);
                if (z>b) cutFromLarger += z/b;
            } else {
                leftover += z;
            }
            s = e;
        }
        std::sort(dirichlet.begin(),dirichlet.end());
        const LO numDirichletTuples = (LO) dirichlet.size()/b;
        leftover += (LO) dirichlet.size()%b;
        FROSCH_ASSERT(leftover==0,"FROSch::Ifpack2PreconditionerTpetra: BlockCrs: "+to_string(leftover)+" rows fit in no "+to_string(b)+"-tuple of rows with one column set (not "+to_string(b)+" dofs per node?). Use \"Local Matrix Format\" = \"Crs\".");
        tuples.insert(tuples.end(),dirichlet.begin(),dirichlet.end());

        // nodes in the order of their first row (keeps the original locality)
        const LO numNodes = n/b;
        std::vector<LO> nodes(numNodes);
        std::iota(nodes.begin(),nodes.end(),(LO) 0);
        std::sort(nodes.begin(),nodes.end(),[&](const LO x,const LO y) {return tuples[x*b]<tuples[y*b];});
        blockPerm_ = HostLOView("FROSch::BlockCrs::perm",n);
        blockInvPerm_ = HostLOView("FROSch::BlockCrs::invPerm",n);
        for (LO k=0; k<numNodes; ++k) {
            for (LO c=0; c<b; ++c) {
                const LO oldRow = tuples[nodes[k]*b+c];
                blockInvPerm_(k*b+c) = oldRow;
                blockPerm_(oldRow) = k*b+c;
            }
        }
        LO moved = 0; // VR 2026-11-03: node-ordered input (e.g. ngs_frosch nodeOrdering) -> identity, apply() without copies
        for (LO i=0; i<n; ++i) if (blockPerm_(i)!=i) ++moved;
        blockIdentity_ = (moved==0);

        // 4. node graph: block columns of a node = nodes of all columns of its b rows (sorted, unique)
        using local_graph_type = typename TCrsGraph::local_graph_device_type;
        typename local_graph_type::row_map_type::non_const_type graphRowPtr("FROSch::BlockCrs::graphRowPtr",numNodes+1);
        auto graphRowPtrHost = Kokkos::create_mirror_view(graphRowPtr);
        std::vector<LO> graphCols;
        graphCols.reserve(A->getLocalNumEntries()/(b*b)+numNodes);
        std::vector<LO> rowCols;
        const HostLOView perm = blockPerm_;
        graphRowPtrHost(0) = 0;
        for (LO k=0; k<numNodes; ++k) {
            rowCols.clear();
            for (LO c=0; c<b; ++c) {
                const LO oldRow = blockInvPerm_(k*b+c);
                for (auto j=rowPtr(oldRow); j<rowPtr(oldRow+1); ++j) rowCols.push_back(perm(colToRow(cols(j)))/b);
            }
            std::sort(rowCols.begin(),rowCols.end());
            rowCols.erase(std::unique(rowCols.begin(),rowCols.end()),rowCols.end());
            graphCols.insert(graphCols.end(),rowCols.begin(),rowCols.end());
            graphRowPtrHost(k+1) = graphCols.size();
        }
        Kokkos::deep_copy(graphRowPtr,graphRowPtrHost);
        typename local_graph_type::entries_type::non_const_type graphColInd("FROSch::BlockCrs::graphCols",graphCols.size());
        auto graphColIndHost = Kokkos::create_mirror_view(graphColInd);
        for (size_t j=0; j<graphCols.size(); ++j) graphColIndHost(j) = graphCols[j];
        Kokkos::deep_copy(graphColInd,graphColIndHost);

        RCP<const TMap> nodeMap = rcp(new TMap((Tpetra::global_size_t) numNodes,(GO) 0,A->getRowMap()->getComm()));
        blockGraph_ = rcp(new TCrsGraph(nodeMap,nodeMap,graphRowPtr,graphColInd));
        blockGraph_->fillComplete(nodeMap,nodeMap);
        blockMatrix_ = rcp(new TBlockCrsMatrix(*blockGraph_,b));
        blockInvDiagonal_ = rcp(new TVector(blockMatrix_->getRangeMap()));

        if (globalRank_==0) {
            const size_t numBlocks = graphCols.size();
            const size_t stored = A->getLocalNumEntries();
            const size_t zeros = numBlocks*b*b-stored;
            std::cout << "FROSch BlockCrs (rank 0): rows " << n << ", nodes " << numNodes << " (block size " << b << "), "
                      << cutFromLarger << " nodes cut from larger equal-pattern groups, " << numDirichletTuples
                      << " Dirichlet tuples, blocks " << numBlocks << ", explicit zeros " << zeros << " ("
                      << 100.0*zeros/stored << " % of stored), permutation "
                      << (blockIdentity_ ? std::string("identity") : "moves "+to_string(moved)+" rows") << std::endl;
        }
    }

    // VR 2026-11-03: values of the point matrix into the blocks + the inverse diagonal (each compute)
    template<class SC,class LO,class GO,class NO>
    void Ifpack2PreconditionerTpetra<SC,LO,GO,NO>::fillBlockCrsValues()
    {
        FROSCH_TIMER_START_SOLVER(blockFillTime,"Ifpack2PreconditionerTpetra::fillBlockCrsValues");
        using HostExec = Kokkos::DefaultHostExecutionSpace;
        using IST = typename TBlockCrsMatrix::impl_scalar_type;
        using ATS = Kokkos::Details::ArithTraits<IST>;
        const LO b = blockSize_;
        const LO n = (LO) blockPerm_.extent(0);
        auto lcl = blockPointMatrix_->getLocalMatrixHost();
        auto rowPtr = lcl.graph.row_map;
        auto cols = lcl.graph.entries;
        auto vals = lcl.values;
        auto bGraph = blockGraph_->getLocalGraphHost();
        auto bRowPtr = bGraph.row_map;
        auto bCols = bGraph.entries;
        auto bVals = blockMatrix_->getValuesHostNonConst();
        Kokkos::deep_copy(bVals,IST(0)); // explicit zeros (Dirichlet blocks)
        const HostLOView perm = blockPerm_;
        const HostLOView invPerm = blockInvPerm_;
        const HostLOView colToRow = blockColToRow_;

        // each point entry has its own slot: no write conflicts
        LO missing = 0;
        Kokkos::parallel_reduce("FROSch::BlockCrs::fillValues",Kokkos::RangePolicy<HostExec>(0,n),[=](const LO i,LO& lmissing) {
            const LO R = perm(i)/b;
            const LO ri = perm(i)%b;
            const LO* begin = bCols.data()+bRowPtr(R);
            const LO* end = bCols.data()+bRowPtr(R+1);
            for (auto j=rowPtr(i); j<rowPtr(i+1); ++j) {
                const LO pc = perm(colToRow(cols(j)));
                const LO C = pc/b;
                const LO ci = pc%b;
                const LO* it = std::lower_bound(begin,end,C);
                if (it==end || *it!=C) {
                    ++lmissing;
                    continue;
                }
                const size_t blk = bRowPtr(R)+(it-begin);
#if defined(TPETRA_ENABLE_BLOCKCRS_LITTLEBLOCK_LAYOUTLEFT)
                bVals(blk*b*b+ci*b+ri) = vals(j);
#else
                bVals(blk*b*b+ri*b+ci) = vals(j);
#endif
            }
        },missing);
        FROSCH_ASSERT(missing==0,"FROSch::Ifpack2PreconditionerTpetra: BlockCrs: "+to_string(missing)+" entries outside the block pattern (local matrix structure changed after the BlockCrs setup?).");

        // inverse diagonal in the node numbering, rule of Ifpack2 Chebyshev: |d| < |min| -> min, else 1/d
        const IST minDiag = blockMinDiagonal_;
        auto d = blockInvDiagonal_->getLocalViewHost(Tpetra::Access::OverwriteAll);
        Kokkos::parallel_for("FROSch::BlockCrs::invDiagonal",Kokkos::RangePolicy<HostExec>(0,n),[=](const LO k) {
            const LO i = invPerm(k);
            IST dii = ATS::zero();
            for (auto j=rowPtr(i); j<rowPtr(i+1); ++j) {
                if (colToRow(cols(j))==i) dii += vals(j);
            }
            d(k,0) = (ATS::abs(dii)<ATS::abs(minDiag)) ? minDiag : ATS::one()/dii;
        });
    }

    // VR 2026-11-03: once per setup: block apply = permuted point apply on a random vector
    template<class SC,class LO,class GO,class NO>
    void Ifpack2PreconditionerTpetra<SC,LO,GO,NO>::checkBlockCrsApply() const
    {
        FROSCH_TIMER_START_SOLVER(blockCheckTime,"Ifpack2PreconditionerTpetra::checkBlockCrsApply");
        using IST = typename TBlockCrsMatrix::impl_scalar_type;
        using ATS = Kokkos::Details::ArithTraits<IST>;
        using Mag = typename ATS::mag_type;
        const LO n = (LO) blockInvPerm_.extent(0);
        TMultiVector x(blockPointMatrix_->getDomainMap(),1);
        TMultiVector y(blockPointMatrix_->getRangeMap(),1);
        TMultiVector bx(blockMatrix_->getDomainMap(),1);
        TMultiVector by(blockMatrix_->getRangeMap(),1);
        x.randomize();
        blockPointMatrix_->apply(x,y);
        {
            auto xv = x.getLocalViewHost(Tpetra::Access::ReadOnly);
            auto bxv = bx.getLocalViewHost(Tpetra::Access::OverwriteAll);
            for (LO k=0; k<n; ++k) bxv(k,0) = xv(blockInvPerm_(k),0);
        }
        blockMatrix_->apply(bx,by);
        Mag maxDiff = 0, maxY = 0;
        {
            auto yv = y.getLocalViewHost(Tpetra::Access::ReadOnly);
            auto byv = by.getLocalViewHost(Tpetra::Access::ReadOnly);
            for (LO k=0; k<n; ++k) {
                maxDiff = std::max(maxDiff,(Mag) ATS::abs(byv(k,0)-yv(blockInvPerm_(k),0)));
                maxY = std::max(maxY,(Mag) ATS::abs(yv(blockInvPerm_(k),0)));
            }
        }
        const Mag rel = (maxY>0) ? maxDiff/maxY : maxDiff;
        if (globalRank_==0) {
            std::cout << "FROSch BlockCrs check (rank 0): max |A x - P^T B P x| / max |A x| = " << rel << std::endl;
        }
        FROSCH_ASSERT(rel<=1.0e-10,"FROSch::Ifpack2PreconditionerTpetra: BlockCrs check failed on global rank "+to_string(globalRank_)+": relative difference "+to_string(rel)+".");
    }

}

#endif
