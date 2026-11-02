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

#ifndef _FROSCH_ALGEBRAICOVERLAPPINGOPERATOR_DEF_HPP
#define _FROSCH_ALGEBRAICOVERLAPPINGOPERATOR_DEF_HPP

#include <FROSch_AlgebraicOverlappingOperator_decl.hpp>

namespace FROSch {

    using namespace std;
    using namespace Teuchos;
    using namespace Xpetra;

    template <class SC,class LO,class GO,class NO>
    AlgebraicOverlappingOperator<SC,LO,GO,NO>::AlgebraicOverlappingOperator(ConstXMatrixPtr k,
                                                                            ParameterListPtr parameterList) :
    OverlappingOperator<SC,LO,GO,NO> (k,parameterList)
    {
        FROSCH_DETAILTIMER_START_LEVELID(algebraicOverlappingOperatorTime,"AlgebraicOverlappingOperator::AlgebraicOverlappingOperator");
        if (!this->ParameterList_->get("Adding Layers Strategy","CrsGraph").compare("CrsGraph")) {
            AddingLayersStrategy_ = LayersFromGraph;
        } else if (!this->ParameterList_->get("Adding Layers Strategy","CrsGraph").compare("CrsMatrix")) {
            AddingLayersStrategy_ = LayersFromMatrix;
        } else if (!this->ParameterList_->get("Adding Layers Strategy","CrsGraph").compare("Old")) {
            AddingLayersStrategy_ = LayersOld;
        } else if (!this->ParameterList_->get("Adding Layers Strategy","CrsGraph").compare("Threaded")) {
            AddingLayersStrategy_ = LayersThreaded; // VR 2026.10.29
        } else {
            FROSCH_ASSERT(false,"FROSch::AlgebraicOverlappingOperator: Specify a valid strategy for adding layers.");
        }
        // VR 2026.11.02: "Extraction Strategy"
        if (!this->ParameterList_->get("Extraction Strategy","Default").compare("Default")) {
            ExtractionStrategy_ = ExtractionDefault;
        } else if (!this->ParameterList_->get("Extraction Strategy","Default").compare("Threaded")) {
            ExtractionStrategy_ = ExtractionThreaded;
        } else {
            FROSCH_ASSERT(false,"FROSch::AlgebraicOverlappingOperator: Specify a valid extraction strategy (Default or Threaded).");
        }
    }

    template <class SC,class LO,class GO,class NO>
    int AlgebraicOverlappingOperator<SC,LO,GO,NO>::initialize(int overlap,
                                                              ConstXMapPtr repeatedMap)
    {
        FROSCH_TIMER_START_LEVELID(initializeTime,"AlgebraicOverlappingOperator::initialize");

        if (this->Verbose_) {
            cout
            << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
            << setw(89) << "-----------------------------------------------------------------------------------------"
            << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
            << "| "
            << left << setw(74) << "AlgebraicOverlappingOperator " << right << setw(8) << "(Level " << setw(2) << this->LevelID_ << ")"
            << " |"
            << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
            << setw(89) << "========================================================================================="
            << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
            << "| " << left << setw(41) << "Adding layers strategy" << right
            << " | " << setw(41) << this->ParameterList_->get("Adding Layers Strategy","CrsGraph")
            << " |"
            // VR 2026.11.02: switches
            << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
            << "| " << left << setw(41) << "Extraction strategy" << right
            << " | " << setw(41) << this->ParameterList_->get("Extraction Strategy","Default")
            << " |"
            << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
            << "| " << left << setw(41) << "Reuse subdomain values" << right
            << " | " << setw(41) << this->ParameterList_->get("Reuse: Subdomain Values",false)
            << " |"
            << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
            << "| " << left << setw(41) << "Combine mode in overlap" << right
            << " | " << setw(41) << this->ParameterList_->get("Combine Values in Overlap","Restricted")
            << " |"
            << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
            << "| " << left << setw(41) << "Solver type" << right
            << " | " << setw(41) << this->ParameterList_->sublist("Solver").get("SolverType","Amesos2")
            << " |"
            << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
            << "| " << left << setw(41) << "Solver" << right
            << " | " << setw(41) << this->ParameterList_->sublist("Solver").get("Solver","Klu")
            << " |"
            << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
            << "| " << left << setw(41) << "Reuse symbolic factorization" << right
            << " | " << setw(41) << this->ParameterList_->get("Reuse: Symbolic Factorization",true)
            << " |"
            << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
            << setw(89) << "-----------------------------------------------------------------------------------------"
            << endl;
        }

        if (repeatedMap.is_null()) repeatedMap = this->K_->getRangeMap();
        this->buildOverlappingMatrices(overlap,repeatedMap);
        this->initializeOverlappingOperator();
        this->updateLocalOverlappingMatrices_Symbolic();
        // VR 2026.10.26: local subdomain matrix statistics
        if (!this->ParameterList_->get("Verbosity","All").compare("All") && !this->localSubdomainMatrix_.is_null()) {
            printLocalSubdomainMatrixStatistics();
        }
        bool reuseSymbolicFactorization = this->ParameterList_->get("Reuse: Symbolic Factorization",true);
        if (this->ExtractLocalSubdomainMatrix_Symbolic_Done_ && reuseSymbolicFactorization) {
            // if reuseSymbolicFactorization=false, we call initializeSubdomainSolver is called during compute
            // Inject global MPI rank into Solver parameters so subdomain solver can use it for diagnostics
            this->ParameterList_->sublist("Solver").set("Global MPI Rank", this->MpiComm_->getRank());
            this->initializeSubdomainSolver(this->localSubdomainMatrix_);
        }

        this->IsInitialized_ = true;
        this->IsComputed_ = false;
        return 0; // RETURN VALUE!!!
    }

    template <class SC,class LO,class GO,class NO>
    int AlgebraicOverlappingOperator<SC,LO,GO,NO>::compute()
    {
        FROSCH_TIMER_START_LEVELID(computeTime,"AlgebraicOverlappingOperator::compute");
        FROSCH_ASSERT(this->IsInitialized_,"ERROR: AlgebraicOverlappingOperator has to be initialized before calling compute()");
        this->computeOverlappingOperator();
        // DL 2026.04.09: For easy benchmarking of local subdomain problems!
#ifdef PRINT_LOCAL_MTX
        int myRank = this->MpiComm_->getRank();
        std::string filename = "local_subdomain_rank_" + std::to_string(myRank) + ".mtx";
        std::cout << "Rank: " << myRank << " writing its local subdomain system to: " << filename << std::endl;
        this->exportLocalMatrix(filename);
        std::cout << "Rank: " << myRank << " finished writing" << std::endl;
        std::fflush(stdout); // ensure output is flushed before exit
        std::exit(EXIT_SUCCESS);
#endif
        return 0; // RETURN VALUE!!!
    }

    template <class SC,class LO,class GO,class NO>
    void AlgebraicOverlappingOperator<SC,LO,GO,NO>::describe(FancyOStream &out,
                                                             const EVerbosityLevel verbLevel) const
    {
        FROSCH_ASSERT(false,"describe() has to be implemented properly...");
    }

    template <class SC,class LO,class GO,class NO>
    string AlgebraicOverlappingOperator<SC,LO,GO,NO>::description() const
    {
        return "Algebraic Overlapping Operator";
    }

    // VR 2026.10.26: rows and nnz of the overlapped local subdomain matrix per rank
    template <class SC,class LO,class GO,class NO>
    void AlgebraicOverlappingOperator<SC,LO,GO,NO>::printLocalSubdomainMatrixStatistics() const
    {
        long long local[2] = {(long long) this->localSubdomainMatrix_->getLocalNumRows(),
                              (long long) this->localSubdomainMatrix_->getLocalNumEntries()};
        long long sum[2] = {0,0}, minVal[2] = {0,0}, maxVal[2] = {0,0};
        reduceAll(*this->MpiComm_,REDUCE_SUM,2,local,sum);
        reduceAll(*this->MpiComm_,REDUCE_MIN,2,local,minVal);
        reduceAll(*this->MpiComm_,REDUCE_MAX,2,local,maxVal);
        if (this->Verbose_) {
            const double numRanks = double(this->MpiComm_->getSize());
            const streamsize oldPrecision = cout.precision();
            cout
            << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
            << setw(89) << "-----------------------------------------------------------------------------------------"
            << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
            << "| "
            << left << setw(74) << "> Local Subdomain Matrix Statistics " << right << setw(8) << "(Level " << setw(2) << this->LevelID_ << ")"
            << " |"
            << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
            << setw(89) << "========================================================================================="
            << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
            << "| " << left << setw(20) << " " << right
            << " | " << setw(10) << "ranks"
            << " | " << setw(10) << "avg"
            << " | " << setw(10) << "min"
            << " | " << setw(10) << "max"
            << " | " << setw(10) << "global sum"
            << " |"
            << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
            << setw(89) << "-----------------------------------------------------------------------------------------"
            << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
            << "| " << left << setw(20) << "Local rows" << right
            << " | " << setw(10) << this->MpiComm_->getSize()
            << " | " << setw(10) << setprecision(8) << sum[0]/numRanks
            << " | " << setw(10) << minVal[0]
            << " | " << setw(10) << maxVal[0]
            << " | " << setw(10) << sum[0]
            << " |"
            << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
            << "| " << left << setw(20) << "Local nnz" << right
            << " | " << setw(10) << this->MpiComm_->getSize()
            << " | " << setw(10) << setprecision(10) << sum[1]/numRanks
            << " | " << setw(10) << minVal[1]
            << " | " << setw(10) << maxVal[1]
            << " | " << setw(10) << sum[1]
            << " |"
            << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
            << setw(89) << "-----------------------------------------------------------------------------------------"
            << setprecision(oldPrecision) << endl;
        }
    }

    template <class SC,class LO,class GO,class NO>
    int AlgebraicOverlappingOperator<SC,LO,GO,NO>::buildOverlappingMatrices(int overlap,
                                                                            ConstXMapPtr repeatedMap)
    {
        FROSCH_DETAILTIMER_START_LEVELID(buildOverlappingMatricesTime,"AlgebraicOverlappingOperator::buildOverlappingMatrices");
        // ====================================================================================
        // AH 08/09/2019: This is just temporary. Implement this properly in all the classes
        Verbosity verbosity = All;
        if (!this->ParameterList_->get("Verbosity","All").compare("None")) {
            verbosity = None;
        } else if (!this->ParameterList_->get("Verbosity","All").compare("All")) {
            verbosity = All;
        } else {
            FROSCH_ASSERT(false,"FROSch::AlgebraicOverlappingOperator: Specify a valid verbosity level.");
        }
        // ====================================================================================

        this->OverlappingMap_ = repeatedMap;
        this->OverlappingMatrix_ = this->K_;

        GO global = 0, sum = 0;
        LO local,minVal,maxVal;
        SC avg;
        if (verbosity==All) {
            FROSCH_DETAILTIMER_START_LEVELID(printStatisticsTime,"print statistics");

            global = this->OverlappingMap_->getMaxAllGlobalIndex();
            if (this->OverlappingMap_->lib()==UseEpetra || this->OverlappingMap_->getGlobalNumElements()>0) {
                global += 1;
            }

            local = (LO) max((LO) this->OverlappingMap_->getLocalNumElements(),(LO) 0);
            reduceAll(*this->MpiComm_,REDUCE_SUM,GO(local),ptr(&sum));
            avg = max(sum/double(this->MpiComm_->getSize()),0.0);
            reduceAll(*this->MpiComm_,REDUCE_MIN,local,ptr(&minVal));
            reduceAll(*this->MpiComm_,REDUCE_MAX,local,ptr(&maxVal));

            if (this->Verbose_) {
                cout
                << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
                << setw(89) << "-----------------------------------------------------------------------------------------"
                << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
                << "| "
                << left << setw(74) << "> Overlapping Subdomains Statistics " << right << setw(8) << "(Level " << setw(2) << this->LevelID_ << ")"
                << " |"
                << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
                << setw(89) << "========================================================================================="
                << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
                << "| " << left << setw(20) << " " << right
                << " | " << setw(10) << "total"
                << " | " << setw(10) << "avg"
                << " | " << setw(10) << "min"
                << " | " << setw(10) << "max"
                << " | " << setw(10) << "global sum"
                << " |"
                << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
                << setw(89) << "-----------------------------------------------------------------------------------------"
                << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
                << "| " << left << setw(20) << "Layer 0" << right
                << " | " << setw(10) << global
                << " | " << setw(10) << setprecision(5) << avg
                << " | " << setw(10) << minVal
                << " | " << setw(10) << maxVal
                << " | " << setw(10) << sum
                << " |";
            }
        }

        // Adding Layers of Elements to the overlapping subdomains
        ConstXCrsGraphPtr overlappingGraph = this->OverlappingMatrix_->getCrsGraph();
        // VR 2026.10.29: "Threaded": all layers in one call; the loop below only prints the statistics
        Array<LO> threadedLayerSizes;
        ConstXMapPtr threadedFinalMap;
        if (AddingLayersStrategy_==LayersThreaded && overlap>0) {
            threadedFinalMap = ExtendOverlapByLayers_Threaded(this->K_->getCrsGraph(),this->OverlappingMap_,overlap,threadedLayerSizes);
        }
        for (int i=0; i<overlap; i++) {
            switch (AddingLayersStrategy_) {
                case LayersFromGraph:
                    ExtendOverlapByOneLayer(overlappingGraph,this->OverlappingMap_,overlappingGraph,this->OverlappingMap_);
                    break;

                case LayersFromMatrix:
                    ExtendOverlapByOneLayer(this->OverlappingMatrix_,this->OverlappingMap_,this->OverlappingMatrix_,this->OverlappingMap_);
                    break;

                case LayersOld:
                    ExtendOverlapByOneLayer_Old(this->OverlappingMatrix_,this->OverlappingMap_,this->OverlappingMatrix_,this->OverlappingMap_);
                    break;

                case LayersThreaded:
                    if (i==overlap-1) this->OverlappingMap_ = threadedFinalMap;
                    break;

                default:
                    FROSCH_ASSERT(false,"FROSch::AlgebraicOverlappingOperator: Specify a valid strategy for adding layers.");
            }
            if (verbosity==All) {
                FROSCH_DETAILTIMER_START_LEVELID(printStatisticsTime,"print statistics");
                local = (AddingLayersStrategy_==LayersThreaded) ? threadedLayerSizes[i]
                                                                : (LO) this->OverlappingMap_->getLocalNumElements();
                local = max(local,(LO) 0);
                reduceAll(*this->MpiComm_,REDUCE_SUM,GO(local),ptr(&sum));
                avg = max(sum/double(this->MpiComm_->getSize()),0.0);
                reduceAll(*this->MpiComm_,REDUCE_MIN,local,ptr(&minVal));
                reduceAll(*this->MpiComm_,REDUCE_MAX,local,ptr(&maxVal));

                if (this->Verbose_) {
                    cout
                    << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
                    << "| " << left << "Layer " << setw(14) << i+1 << right
                    << " | " << setw(10) << global
                    << " | " << setw(10) << setprecision(5) << avg
                    << " | " << setw(10) << minVal
                    << " | " << setw(10) << maxVal
                    << " | " << setw(10) << sum
                    << " |";
                }
            }
        }

        if (verbosity==All) {
            FROSCH_DETAILTIMER_START_LEVELID(printStatisticsTime,"print statistics");
            if (this->Verbose_) {
                cout
                << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
                << setw(89) << "-----------------------------------------------------------------------------------------"
                << endl;
            }
        }

        // AH 08/28/2019 TODO: It seems that ExtendOverlapByOneLayer_Old is currently the fastest method because the map is sorted. This seems to be better for the direct solver. (At least Klu)
        if (this->ParameterList_->get("Sort Overlapping Map",true)) {
            switch (AddingLayersStrategy_) {
                case LayersFromGraph:
                    this->OverlappingMap_ = SortMapByGlobalIndex(this->OverlappingMap_);
                    break;

                case LayersFromMatrix:
                    this->OverlappingMap_ = SortMapByGlobalIndex(this->OverlappingMap_);
                    break;

                case LayersOld:
                    if (this->Verbose_) cout << "FROSch::AlgebraicOverlappingOperator : The overlapping map is already sorted" << endl;
                    break;

                case LayersThreaded:
                    // already sorted; re-built like CrsGraph for identical maps
                    this->OverlappingMap_ = SortMapByGlobalIndex(this->OverlappingMap_);
                    break;

                default:
                    FROSCH_ASSERT(false,"FROSch::AlgebraicOverlappingOperator: Specify a valid strategy for adding layers.");
            }
        }

        return 0;
    }

    template <class SC,class LO,class GO,class NO>
    int AlgebraicOverlappingOperator<SC,LO,GO,NO>::updateLocalOverlappingMatrices()
    {
        FROSCH_DETAILTIMER_START_LEVELID(updateLocalOverlappingMatricesTime,"AlgebraicOverlappingOperator::updateLocalOverlappingMatrices");
        if (this->ExtractLocalSubdomainMatrix_Symbolic_Done_) {
            // using original K_ as input
            // VR 2026.10.31: "Reuse: Subdomain Values": the first compute after the symbolic phase skips the import
            if (ExtractionStrategy_ == ExtractionThreaded) {
                if (!this->SubdomainValuesCurrent_) {
                    ExtractLocalSubdomainMatrix_Compute_Threaded(this->K_, this->OverlappingMap_, this->localSubdomainMatrix_);
                }
            } else {
                ExtractLocalSubdomainMatrix_Compute(this->K_, this->subdomainMatrix_, this->localSubdomainMatrix_, !this->SubdomainValuesCurrent_);
            }
            this->SubdomainValuesCurrent_ = false;
            this->OverlappingMatrix_ = this->localSubdomainMatrix_.getConst();
        } else {
            if (this->IsComputed_) {
                // already computed once and we want to recycle the information. That is why we reset OverlappingMatrix_ to K_, because K_ has been reset at this point
                this->OverlappingMatrix_ = this->K_;
            }
            this->OverlappingMatrix_ = ExtractLocalSubdomainMatrix(this->OverlappingMatrix_, this->OverlappingMap_);
        }
        return 0;
    }


    template <class SC,class LO,class GO,class NO>
    int AlgebraicOverlappingOperator<SC,LO,GO,NO>::updateLocalOverlappingMatrices_Symbolic()
    {
        FROSCH_DETAILTIMER_START_LEVELID(updateLocalOverlappingMatrices_SymbolicTime, "AlgebraicOverlappingOperator::updateLocalOverlappingMatrices_Symbolic");
        this->extractLocalSubdomainMatrix_Symbolic();
        return 0;
    }

    template <class SC,class LO,class GO,class NO>
    void AlgebraicOverlappingOperator<SC,LO,GO,NO>::extractLocalSubdomainMatrix_Symbolic()
    {
        if (this->OverlappingMap_->lib() == UseTpetra) {
            FROSCH_DETAILTIMER_START_LEVELID(AlgebraicOverlappin_extractLocalSubdomainMatrix_SymbolicTime,"AlgebraicOverlappinOperator::extractLocalSubdomainMatrix_Symbolic");
            RCP<Import<LO,GO,NO> > scatter;
            {
                FROSCH_DETAILTIMER_START_LEVELID(buildImportTime,"AlgebraicOverlappingOperator::extractLocalSubdomainMatrix_Symbolic::buildImport");
                scatter = ImportFactory<LO,GO,NO>::Build(this->OverlappingMatrix_->getRowMap(), this->OverlappingMap_);
            }
            RCP<const Comm<LO> > SerialComm = rcp(new MpiComm<LO>(MPI_COMM_SELF));
            RCP<Map<LO,GO,NO> > localSubdomainMap = MapFactory<LO,GO,NO>::Build(this->OverlappingMap_->lib(), this->OverlappingMap_->getLocalNumElements(), 0, SerialComm);
            if (ExtractionStrategy_ == ExtractionThreaded) {
                // VR 2026.11.02: importAndFillComplete, then the local matrix in parallel; the imported matrix is not kept
                RCP<const Matrix<SC,LO,GO,NO> > subdomainMatrix;
                {
                    FROSCH_DETAILTIMER_START_LEVELID(importAndFillCompleteTime,"AlgebraicOverlappingOperator::extractLocalSubdomainMatrix_Symbolic::importAndFillComplete");
                    subdomainMatrix = MatrixFactory<SC,LO,GO,NO>::Build(this->OverlappingMatrix_, *scatter,
                                                                        this->OverlappingMatrix_->getDomainMap(),
                                                                        this->OverlappingMatrix_->getRangeMap());
                }
                ExtractLocalSubdomainMatrix_Symbolic_Threaded(subdomainMatrix, localSubdomainMap.getConst(), this->localSubdomainMatrix_);
                this->subdomainMatrix_ = null;
            } else {
                // buid sudomain matrix
                this->subdomainMatrix_ = MatrixFactory<SC,LO,GO,NO>::Build(this->OverlappingMap_, this->OverlappingMatrix_->getGlobalMaxNumRowEntries());
                {
                    FROSCH_DETAILTIMER_START_LEVELID(doImportTime,"AlgebraicOverlappingOperator::extractLocalSubdomainMatrix_Symbolic::doImport");
                    this->subdomainMatrix_->doImport(*(this->OverlappingMatrix_), *scatter, ADD);
                }

                // build local subdomain matrix
                this->localSubdomainMatrix_ = MatrixFactory<SC,LO,GO,NO>::Build(localSubdomainMap, localSubdomainMap, this->OverlappingMatrix_->getGlobalMaxNumRowEntries());

                // fill in column indexes
                ExtractLocalSubdomainMatrix_Symbolic(this->subdomainMatrix_, // input
                                                     this->localSubdomainMatrix_);   // output
            }

            // turn flag on
            this->ExtractLocalSubdomainMatrix_Symbolic_Done_ = true;

            // VR 2026.10.31: opt-in, valid only if K_ is not changed in place before compute()
            this->SubdomainValuesCurrent_ = this->ParameterList_->get("Reuse: Subdomain Values",false)
                                            && this->OverlappingMatrix_.getRawPtr() == this->K_.getRawPtr();
        }
    }
}

#endif
